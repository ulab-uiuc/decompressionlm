#!/usr/bin/env python3
"""
Perplexity experiment: Measure how well models understand their extracted concepts.

For each concept extracted from the graph, we:
1. Prompt the model to write a Wikipedia article about that concept
2. Measure the perplexity of the generated article
3. Correlate perplexity with concept frequency in the graph
4. Compare across quantization methods and domains

This reveals whether high-frequency concepts are truly "learned" (low perplexity)
or merely "memorized patterns" (high perplexity).
"""

import os
import gc
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import plotext as plt  # Terminal plotting
from tqdm import tqdm
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple

# ----------------------------
# Atomic write helpers
# ----------------------------

def atomic_write_csv(df: pd.DataFrame, final_path: str, **to_csv_kwargs):
    """
    Write CSV atomically:
      1) write to final_path + '.tmp'
      2) fsync
      3) atomic rename (os.replace)

    If interrupted mid-write, you'll only ever see the .tmp file.
    """
    tmp_path = final_path + ".tmp"
    df.to_csv(tmp_path, index=False, **to_csv_kwargs)

    # Ensure bytes hit disk before rename
    with open(tmp_path, "rb") as f:
        os.fsync(f.fileno())

    os.replace(tmp_path, final_path)


# ----------------------------
# Configuration
# ----------------------------
MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]
VARIANTS = ["BF16_BASE", "GPTQ_INT8", "AWQ_4BIT", "GPTQ_INT4", "BNB_4BIT_UNSLOTH"]
TASKS = ["us_law", "git_ver_control", "radiology_imaging", "faa"]
SEQ_LEN = 16  # Focus on seq_len=16 for now

# Task metadata
TASK_INFO = {
    "us_law": {
        "dir": "results/quant_ladder__us_law_16_newlines",
        "display": "US Law (Bar Exam)",
        "domain": "United States law and legal concepts"
    },
    "git_ver_control": {
        "dir": "results/quant_ladder__git_ver_control_16_newlines",
        "display": "Git Version Control",
        "domain": "Git and version control systems"
    },
    "radiology_imaging": {
        "dir": "results/quant_ladder__radiology_imaging_16_newlines",
        "display": "Radiology Imaging",
        "domain": "radiology and medical imaging"
    },
    "faa": {
        "dir": "results/quant_ladder__faa_16_newlines",
        "display": "FAA Aviation",
        "domain": "FAA aviation flight rules and airspace"
    }
}

# Model repos
MODEL_REPOS = {
    "Qwen2.5-7B-Instruct": {
        "BF16_BASE": ("Qwen/Qwen2.5-7B-Instruct", torch.bfloat16),
        "GPTQ_INT8": ("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8", "auto"),
        "AWQ_4BIT": ("Qwen/Qwen2.5-7B-Instruct-AWQ", "auto"),
        "GPTQ_INT4": ("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", "auto"),
        "BNB_4BIT_UNSLOTH": ("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", "auto"),
    },
    "Llama-3.1-8B-Instruct": {
        "BF16_BASE": ("meta-llama/Llama-3.1-8B-Instruct", torch.bfloat16),
        "GPTQ_INT8": ("abdo-Mansour/Meta-Llama-3.1-8B-Instruct-GPTQ-8bit", "auto"),
        "AWQ_4BIT": ("hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", "auto"),
        "GPTQ_INT4": ("hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4", "auto"),
        "BNB_4BIT_UNSLOTH": ("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", "auto"),
    }
}

# Sampling parameters
SAMPLE_SIZE = 200  # Number of concepts to sample per variant
WIKI_MAX_TOKENS = 128  # Length of generated Wikipedia article
TEMPERATURE = 0.7
TOP_P = 0.9

# Output directory
OUTPUT_DIR = "results/perplexity_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Helper Functions
# ----------------------------

def load_concepts_from_file(file_path: str) -> List[Tuple[str, int]]:
    """Load concepts and their frequencies from a .delm.parquet file."""
    try:
        table = pq.read_table(file_path)
        metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}

        if 'graph_concept_frequencies' not in metadata:
            print(f"  Warning: No graph_concept_frequencies in {file_path}")
            return []

        concept_freqs = json.loads(metadata['graph_concept_frequencies'])
        return concept_freqs  # List of [concept, frequency] pairs
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return []


def sample_concepts_by_frequency(concept_freqs: List[Tuple[str, int]],
                                 n_samples: int) -> List[Tuple[str, int]]:
    """
    Sample concepts with stratified sampling across frequency ranges.

    This ensures we get:
    - High frequency concepts (likely learned)
    - Medium frequency concepts
    - Low frequency concepts (potentially noise)
    """
    if len(concept_freqs) <= n_samples:
        return concept_freqs

    # Sort by frequency (already sorted from extraction, but ensure)
    sorted_concepts = sorted(concept_freqs, key=lambda x: x[1], reverse=True)

    # Stratified sampling
    total = len(sorted_concepts)

    n_high = int(n_samples * 0.25)
    n_mid = int(n_samples * 0.50)
    n_low = n_samples - n_high - n_mid

    high_idx = int(total * 0.25)
    mid_idx = int(total * 0.75)

    sampled = []
    sampled.extend(sorted_concepts[:high_idx][:n_high])  # Top quartile
    sampled.extend(
        sorted_concepts[high_idx:mid_idx][::max(1, (mid_idx - high_idx) // max(1, n_mid))][:n_mid]
    )  # Middle
    sampled.extend(
        sorted_concepts[mid_idx:][::max(1, (total - mid_idx) // max(1, n_low))][:n_low]
    )  # Bottom

    return sampled[:n_samples]


def load_model_and_tokenizer(model_name: str, variant: str):
    """Load model and tokenizer for a specific variant."""
    repo, dtype = MODEL_REPOS[model_name][variant]

    print(f"  Loading model: {repo}")
    tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Special handling for problematic GPTQ models
    if repo == "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4":
        config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
        qc = getattr(config, "quantization_config", None)
        if qc:
            qc["desc_act"] = False
            config.quantization_config = qc
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            config=config,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    return model, tokenizer


def create_wikipedia_prompt(concept: str, domain: str) -> str:
    """Create a prompt asking the model to write a Wikipedia article."""
    return f"""Write a concise Wikipedia article about the following concept in the context of {domain}.

Concept: {concept}

Article:"""


def measure_perplexity(model, tokenizer, text: str, device: str = "cuda") -> float:
    """
    Measure perplexity of generated text.

    Perplexity = exp(average negative log-likelihood)
    Lower perplexity = better understanding
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()

    return float(np.exp(loss))


def generate_and_measure(model, tokenizer, concept: str, domain: str,
                         max_tokens: int = 128) -> Tuple[str, float]:
    """Generate Wikipedia article and measure its perplexity."""
    prompt = create_wikipedia_prompt(concept, domain)

    # Apply chat template if available
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (exclude prompt)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Measure perplexity on the generated text
    if generated_text.strip():
        perplexity = measure_perplexity(model, tokenizer, generated_text, device=model.device)
    else:
        perplexity = float('inf')

    return generated_text, perplexity


def process_variant(model_name: str, variant: str, task: str) -> pd.DataFrame:
    """Process one model variant for one task."""
    print(f"\n{'='*80}")
    print(f"Processing: {model_name} - {variant} - {task}")
    print(f"{'='*80}")

    # Construct file path
    label = f"{model_name}__{variant}"
    task_stem = f"__{task}_bar_meta"
    filename = f"{label}{task_stem}__L{SEQ_LEN}__T1.0.delm.parquet"
    file_path = os.path.join(TASK_INFO[task]["dir"], filename)

    if not os.path.exists(file_path):
        print(f"  ⚠️  File not found: {file_path}")
        return None

    # Load concepts
    print(f"  Loading concepts from {filename}...")
    concept_freqs = load_concepts_from_file(file_path)

    if not concept_freqs:
        print(f"  ⚠️  No concepts found")
        return None

    print(f"  Total concepts: {len(concept_freqs)}")

    # Sample concepts
    sampled_concepts = sample_concepts_by_frequency(concept_freqs, SAMPLE_SIZE)
    print(f"  Sampled: {len(sampled_concepts)} concepts")

    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer(model_name, variant)
    except Exception as e:
        print(f"  ⚠️  Error loading model: {e}")
        return None

    # Process each concept
    results = []
    domain = TASK_INFO[task]["domain"]

    print(f"  Generating Wikipedia articles and measuring perplexity...")
    for concept, frequency in tqdm(sampled_concepts, desc="  Concepts"):
        try:
            generated_text, perplexity = generate_and_measure(
                model, tokenizer, concept, domain, max_tokens=WIKI_MAX_TOKENS
            )

            results.append({
                'model': model_name,
                'variant': variant,
                'task': task,
                'concept': concept,
                'frequency': frequency,
                'perplexity': perplexity,
                'generated_length': len(generated_text),
                'generated_text': generated_text[:200]  # Store first 200 chars
            })
        except Exception as e:
            print(f"    ⚠️  Error processing '{concept}': {e}")
            continue

    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    df = pd.DataFrame(results)

    if len(df) > 0:
        print(f"\n  Results:")
        print(f"    Concepts processed: {len(df)}")
        print(f"    Perplexity - Min: {df['perplexity'].min():.2f}")
        print(f"    Perplexity - Max: {df['perplexity'].max():.2f}")
        print(f"    Perplexity - Mean: {df['perplexity'].mean():.2f}")
        print(f"    Perplexity - Median: {df['perplexity'].median():.2f}")

    return df


def print_variant_stats(df: pd.DataFrame, model_name: str, variant: str, task: str):
    """
    When loading cached CSV, print the same stats as process_variant() would.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {model_name} - {variant} - {task}")
    print(f"{'='*80}")

    if df is None or len(df) == 0:
        print("  ⚠️  No rows in cached result")
        return

    # Filter to just this combo (safety if caller passes a larger df)
    need = (df["model"] == model_name) & (df["variant"] == variant) & (df["task"] == task)
    df = df[need]
    if len(df) == 0:
        print("  ⚠️  No matching rows in cached result for this (model, variant, task)")
        return

    print(f"\n  Results:")
    print(f"    Concepts processed: {len(df)}")
    print(f"    Perplexity - Min: {df['perplexity'].min():.2f}")
    print(f"    Perplexity - Max: {df['perplexity'].max():.2f}")
    print(f"    Perplexity - Mean: {df['perplexity'].mean():.2f}")
    print(f"    Perplexity - Median: {df['perplexity'].median():.2f}")


def plot_frequency_vs_perplexity_terminal(df: pd.DataFrame, model_name: str):
    """Create terminal-based scatter plots of frequency vs perplexity using plotext."""
    print(f"\n{'='*80}")
    print(f"FREQUENCY vs PERPLEXITY PLOTS - {model_name}")
    print(f"{'='*80}\n")

    for task in TASKS:
        task_df = df[df['task'] == task]

        if len(task_df) == 0:
            continue

        print(f"\n{TASK_INFO[task]['display']}")
        print("-" * 80)

        variants = sorted(task_df['variant'].unique())

        for variant in variants:
            variant_df = task_df[task_df['variant'] == variant]
            if len(variant_df) == 0:
                continue

            plt.clf()

            x = variant_df['frequency'].values
            y = np.log10(variant_df['perplexity'].values)

            plt.scatter(x, y, marker='dot')

            median_freq = np.median(x)
            median_ppl = np.median(y)

            plt.vline(median_freq, 'red')
            plt.hline(median_ppl, 'red')

            plt.title(f"{variant}")
            plt.xlabel("Frequency")
            plt.ylabel("log10(Perplexity)")
            plt.plotsize(100, 25)

            plt.show()

            real_ppl = variant_df['perplexity'].values
            print(f"\n  {variant} Stats:")
            print(f"    N concepts: {len(variant_df)}")
            print(f"    Perplexity: min={real_ppl.min():.1f}, median={np.median(real_ppl):.1f}, max={real_ppl.max():.1f}")
            print(f"    Frequency:  min={x.min():.0f}, median={median_freq:.0f}, max={x.max():.0f}")

            corr = np.corrcoef(x, real_ppl)[0, 1]
            print(f"    Freq-Ppl Correlation: {corr:+.3f}")
            print()


def plot_variant_comparison_terminal(df: pd.DataFrame, model_name: str):
    """Create terminal-based box plots comparing perplexity across variants."""
    print(f"\n{'='*80}")
    print(f"VARIANT COMPARISON - {model_name}")
    print(f"{'='*80}\n")

    for task in TASKS:
        task_df = df[df['task'] == task]
        if len(task_df) == 0:
            continue

        print(f"\n{TASK_INFO[task]['display']}")
        print("-" * 80)

        plt.clf()

        variants_order = ['BF16_BASE', 'GPTQ_INT8', 'AWQ_4BIT', 'BNB_4BIT_UNSLOTH', 'GPTQ_INT4']
        variants_present = [v for v in variants_order if v in task_df['variant'].unique()]

        data_to_plot = []
        labels = []

        for variant in variants_present:
            variant_df = task_df[task_df['variant'] == variant]
            if len(variant_df) > 0:
                ppl_values = np.log10(variant_df['perplexity'].values)
                data_to_plot.append(ppl_values)
                labels.append(variant[:10])

        if not data_to_plot:
            continue

        plt.box(data_to_plot, labels=labels)
        plt.title(f"{TASK_INFO[task]['display']} - Perplexity Distribution")
        plt.ylabel("log10(Perplexity)")
        plt.plotsize(100, 25)
        plt.show()

        print("\nDetailed Statistics:")
        for variant in variants_present:
            variant_df = task_df[task_df['variant'] == variant]
            real_ppl = variant_df['perplexity'].values

            print(f"  {variant:15s}: N={len(variant_df):3d}, "
                  f"Mean={real_ppl.mean():7.1f}, "
                  f"Median={np.median(real_ppl):7.1f}, "
                  f"Std={real_ppl.std():7.1f}")
        print()


def generate_summary_stats(df: pd.DataFrame, output_path: str):
    """Generate summary statistics table."""
    summary_data = []

    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name]

        for task in TASKS:
            task_df = model_df[model_df['task'] == task]

            for variant in task_df['variant'].unique():
                variant_df = task_df[task_df['variant'] == variant]
                if len(variant_df) == 0:
                    continue

                corr = variant_df[['frequency', 'perplexity']].corr().iloc[0, 1]

                summary_data.append({
                    'Model': model_name,
                    'Task': TASK_INFO[task]['display'],
                    'Variant': variant,
                    'N Concepts': len(variant_df),
                    'Perplexity Min': variant_df['perplexity'].min(),
                    'Perplexity Max': variant_df['perplexity'].max(),
                    'Perplexity Mean': variant_df['perplexity'].mean(),
                    'Perplexity Median': variant_df['perplexity'].median(),
                    'Perplexity Std': variant_df['perplexity'].std(),
                    'Freq-Ppl Correlation': corr,
                })

    summary_df = pd.DataFrame(summary_data)
    atomic_write_csv(summary_df, output_path, float_format='%.3f')
    print(f"\n✓ Saved summary (atomic): {output_path}")
    return summary_df


def print_summary_table(summary_df: pd.DataFrame):
    """Print summary statistics as formatted table."""
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    for model_name in summary_df['Model'].unique():
        model_data = summary_df[summary_df['Model'] == model_name]

        print(f"\n{model_name}")
        print("=" * 80)

        for task in model_data['Task'].unique():
            task_data = model_data[model_data['Task'] == task]

            print(f"\n{task}")
            print("-" * 80)
            print(f"{'Variant':<15} {'N':>5} {'Mean':>8} {'Median':>8} {'Std':>8} {'Corr':>7}")
            print("-" * 80)

            for _, row in task_data.iterrows():
                print(f"{row['Variant']:<15} "
                      f"{row['N Concepts']:>5.0f} "
                      f"{row['Perplexity Mean']:>8.1f} "
                      f"{row['Perplexity Median']:>8.1f} "
                      f"{row['Perplexity Std']:>8.1f} "
                      f"{row['Freq-Ppl Correlation']:>+7.3f}")
            print()


# ----------------------------
# Main Experiment
# ----------------------------

def main():
    print("="*80)
    print("PERPLEXITY EXPERIMENT")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Models: {MODELS}")
    print(f"  Variants: {VARIANTS}")
    print(f"  Tasks: {TASKS}")
    print(f"  Sample size: {SAMPLE_SIZE} concepts per variant")
    print(f"  Wiki article length: {WIKI_MAX_TOKENS} tokens")
    print(f"  Output directory: {OUTPUT_DIR}")

    all_results = []

    for model_name in MODELS:
        for task in TASKS:
            for variant in VARIANTS:
                result_file = os.path.join(
                    OUTPUT_DIR,
                    f"{model_name}__{variant}__{task}__perplexity.csv"
                )

                # If cached, load and print the same stats as a fresh run
                if os.path.exists(result_file):
                    print(f"\n✓ Already processed: {model_name} - {variant} - {task}")
                    df = pd.read_csv(result_file)
                    print_variant_stats(df, model_name, variant, task)
                    all_results.append(df)
                    continue

                df = process_variant(model_name, variant, task)

                if df is not None and len(df) > 0:
                    atomic_write_csv(df, result_file)
                    print(f"  ✓ Saved (atomic): {result_file}")
                    all_results.append(df)

    if not all_results:
        print("\n⚠️  No results to analyze")
        return

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_file = os.path.join(OUTPUT_DIR, "all_perplexity_results.csv")
    atomic_write_csv(combined_df, combined_file)
    print(f"\n✓ Saved combined results (atomic): {combined_file}")

    print("\n" + "="*80)
    print("GENERATING TERMINAL VISUALIZATIONS")
    print("="*80)

    for model_name in MODELS:
        model_df = combined_df[combined_df['model'] == model_name]
        if len(model_df) == 0:
            continue

        plot_frequency_vs_perplexity_terminal(model_df, model_name)
        plot_variant_comparison_terminal(model_df, model_name)

    print("\n" + "="*80)
    print("GENERATING SUMMARY STATISTICS")
    print("="*80)

    summary_path = os.path.join(OUTPUT_DIR, "perplexity_summary_stats.csv")
    summary_df = generate_summary_stats(combined_df, summary_path)
    print_summary_table(summary_df)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - Combined data: all_perplexity_results.csv")
    print(f"  - Summary stats: perplexity_summary_stats.csv")
    print(f"  - Individual CSVs: *__perplexity.csv")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    for model_name in MODELS:
        model_df = combined_df[combined_df['model'] == model_name]
        if len(model_df) == 0:
            continue

        print(f"\n{model_name}:")
        for task in TASKS:
            task_df = model_df[model_df['task'] == task]
            if len(task_df) == 0:
                continue

            print(f"\n  {TASK_INFO[task]['display']}:")
            for variant in ['BF16_BASE', 'AWQ_4BIT', 'GPTQ_INT4']:
                variant_df = task_df[task_df['variant'] == variant]
                if len(variant_df) == 0:
                    continue

                corr = variant_df[['frequency', 'perplexity']].corr().iloc[0, 1]
                mean_ppl = variant_df['perplexity'].mean()

                print(f"    {variant:15s}: Mean PPL={mean_ppl:7.2f}, Freq-PPL Corr={corr:+.3f}")


if __name__ == "__main__":
    main()
