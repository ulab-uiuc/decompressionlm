#!/usr/bin/env python3
"""
Complete perplexity experiment on MMLU-Pro Law ranked models.

Part 1: Extract concept graphs from models (if not already done)
Part 2: Measure perplexity on sampled concepts
Part 3: Analyze correlation with MMLU-Pro Law performance

This tests: Do better MMLU performers have lower perplexity on their own concepts?
"""

import os
import gc
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import plotext as plt
from tqdm import tqdm
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
from src.bin_entropy import estimate_prefix_mass

# ----------------------------
# Configuration
# ----------------------------

# Models ranked by MMLU-Pro Law performance (top to bottom)
MODELS = [
    # "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-Small-Instruct-2409",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen2-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "microsoft/Phi-3-mini-128k-instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "abacusai/Llama-3-Smaug-8B",
    "ibm-granite/granite-3.1-8b-instruct",
    "mistralai/Ministral-8B-Instruct-2410",
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    "ibm-granite/granite-3.1-2b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "m-a-p/neo_7b_instruct_v0.1",
    "ibm-granite/granite-3.1-3b-a800m-instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct",
    "deepseek-ai/deepseek-math-7b-instruct",
    "ibm-granite/granite-3.1-1b-a400m-instruct",
]

# Concept extraction parameters
EXTRACTION_PROMPT = """Generate United States bar exam legal concepts as keywords.

Please output ONE concept per line.
Each concept can be multiple words if needed.
Do not include explanations or extra text.
Please begin from any random concept.
Please use English.
"""

EXTRACTION_DIR = "results/mmlu_pro_law_concept_extraction_16"
MAX_LEN = 16
PREFIX_LEN = 16
PROB_THRESHOLD = 1.0  # Disabled
MAX_SAMPLES = 8192
BATCH_SIZE = 64 # 256
DISPLAY_INTERVAL = BATCH_SIZE

# Task metadata
TASK_INFO = {
    "domain": "United States law and legal concepts",
    "display": "US Law (Bar Exam)"
}

# Perplexity sampling parameters
SAMPLE_SIZE = 200  # Number of concepts to sample per model
WIKI_MAX_TOKENS = 128  # Length of generated Wikipedia article
TEMPERATURE = 0.7
TOP_P = 0.9

# Output directory
OUTPUT_DIR = "results/perplexity_analysis_mmlu_law"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXTRACTION_DIR, exist_ok=True)

# ----------------------------
# Part 1: Concept Extraction
# ----------------------------

def extract_concepts_for_model(model_name: str) -> bool:
    """Extract concept graph for one model."""
    print(f"\n{'='*80}")
    print(f"CONCEPT EXTRACTION: {model_name}")
    print(f"{'='*80}")
    
    safe_name = model_name.replace("/", "_")
    save_name = f"{safe_name}__us_law_concepts__L{PREFIX_LEN}__T{PROB_THRESHOLD}.delm.parquet"
    save_path = os.path.join(EXTRACTION_DIR, save_name)
    
    # Check if already extracted
    if os.path.exists(save_path):
        print(f"✓ Concepts already extracted: {save_path}")
        return True
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    
    trust_remote = True
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote,
            use_fast=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=trust_remote,
        )
        
        # Extract concepts
        print("Extracting concepts...")
        if "gemma" in model_name.lower():
            model.config.use_cache = False

        results = estimate_prefix_mass(
            model=model,
            tokenizer=tokenizer,
            prefix=EXTRACTION_PROMPT,
            prefix_len=PREFIX_LEN,
            prob_threshold=PROB_THRESHOLD,
            max_samples=MAX_SAMPLES,
            max_len=MAX_LEN,
            use_chat_template=True,
            batch_size=BATCH_SIZE,
            display_interval=DISPLAY_INTERVAL,
            save_path=save_path,
            model_name=model_name,
            enable_graph_analysis=True
        )
        
        print(f"\nExtraction complete:")
        print(f"  Unique prefixes: {results['unique_prefixes']}")
        print(f"  Samples done: {results['samples_done']}")
        print(f"  Effective set size: {results['effective_set_size']}")
        print(f"  Avg tokens/seq: {results['effective_set_stats']['avg_tokens']:.1f}")
        print(f"  Saved to: {save_path}")
        
        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"⚠️  Error extracting concepts: {e}")
        import traceback
        traceback.print_exc()
        return False


# ----------------------------
# Part 2: Perplexity Measurement
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
    
    Distribution:
    - Top 25%: high frequency (50 concepts) - likely learned
    - Middle 50%: medium frequency (100 concepts) - specialized knowledge
    - Bottom 25%: low frequency (50 concepts) - noise or rare concepts
    """
    if len(concept_freqs) <= n_samples:
        return concept_freqs
    
    # Sort by frequency (descending)
    sorted_concepts = sorted(concept_freqs, key=lambda x: x[1], reverse=True)
    
    # Calculate sample sizes for each stratum
    n_high = int(n_samples * 0.25)      # 50 high-frequency
    n_mid = int(n_samples * 0.50)       # 100 mid-frequency
    n_low = n_samples - n_high - n_mid  # 50 low-frequency
    
    # Define stratum boundaries
    total = len(sorted_concepts)
    high_idx = int(total * 0.25)  # Top 25% boundary
    mid_idx = int(total * 0.75)   # Bottom 25% boundary
    
    sampled = []
    
    # High frequency stratum: take first 50 from top 25%
    high_stratum = sorted_concepts[:high_idx]
    sampled.extend(high_stratum[:n_high])
    
    # Medium frequency stratum: evenly sample 100 from middle 50%
    mid_stratum = sorted_concepts[high_idx:mid_idx]
    if len(mid_stratum) > 0:
        step = max(1, len(mid_stratum) // n_mid)
        sampled.extend(mid_stratum[::step][:n_mid])
    
    # Low frequency stratum: evenly sample 50 from bottom 25%
    low_stratum = sorted_concepts[mid_idx:]
    if len(low_stratum) > 0:
        step = max(1, len(low_stratum) // n_low)
        sampled.extend(low_stratum[::step][:n_low])
    
    return sampled[:n_samples]


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer (no quantization)."""
    print(f"  Loading model: {model_name}")
    
    # Special trust_remote_code for certain models
    trust_remote = True
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote,
        use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote,
    )
    
    return model, tokenizer


def create_wikipedia_prompt(concept: str, domain: str) -> str:
    """Create a prompt asking the model to write a Wikipedia article."""
    prompt = f"""Write a concise Wikipedia article about the following concept in the context of {domain}.

Concept: {concept}

Article:"""
    return prompt


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
    
    perplexity = np.exp(loss)
    return perplexity


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
    
    # Generate
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
    if len(generated_text.strip()) > 0:
        perplexity = measure_perplexity(model, tokenizer, generated_text, device=model.device)
    else:
        perplexity = float('inf')  # Failed to generate
    
    return generated_text, perplexity


def measure_perplexity_for_model(model_name: str) -> pd.DataFrame:
    """Measure perplexity for one model's concepts."""
    print(f"\n{'='*80}")
    print(f"PERPLEXITY MEASUREMENT: {model_name}")
    print(f"{'='*80}")
    
    # Construct file path
    safe_name = model_name.replace("/", "_")
    concept_file = f"{safe_name}__us_law_concepts__L{PREFIX_LEN}__T{PROB_THRESHOLD}.delm.parquet"
    concept_path = os.path.join(EXTRACTION_DIR, concept_file)
    
    if not os.path.exists(concept_path):
        print(f"  ⚠️  Concept file not found: {concept_path}")
        return None
    
    # Load concepts
    print(f"  Loading concepts from {concept_file}...")
    concept_freqs = load_concepts_from_file(concept_path)
    
    if not concept_freqs:
        print(f"  ⚠️  No concepts found")
        return None
    
    print(f"  Total concepts: {len(concept_freqs)}")
    
    # Sample concepts
    sampled_concepts = sample_concepts_by_frequency(concept_freqs, SAMPLE_SIZE)
    print(f"  Sampled: {len(sampled_concepts)} concepts")
    
    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer(model_name)
    except Exception as e:
        print(f"  ⚠️  Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Process each concept
    results = []
    domain = TASK_INFO["domain"]
    
    print(f"  Generating Wikipedia articles and measuring perplexity...")
    for concept, frequency in tqdm(sampled_concepts, desc="  Concepts"):
        try:
            generated_text, perplexity = generate_and_measure(
                model, tokenizer, concept, domain, max_tokens=WIKI_MAX_TOKENS
            )
            
            results.append({
                'model': model_name,
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
        
        # Calculate correlation
        corr = df[['frequency', 'perplexity']].corr().iloc[0, 1]
        print(f"    Freq-Ppl Correlation: {corr:+.3f}")
    
    return df


# ----------------------------
# Part 3: Analysis & Visualization
# ----------------------------

def plot_model_comparison_terminal(df: pd.DataFrame):
    """Create terminal-based plots comparing models."""
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON - PERPLEXITY vs MMLU-Pro Law Rank")
    print(f"{'='*80}\n")
    
    # Calculate stats per model
    model_stats = []
    for i, model_name in enumerate(MODELS):
        model_df = df[df['model'] == model_name]
        
        if len(model_df) == 0:
            continue
        
        model_stats.append({
            'model': model_name.split('/')[-1][:20],  # Short name
            'rank': i + 1,  # MMLU rank (1 = best)
            'mean_ppl': model_df['perplexity'].mean(),
            'median_ppl': model_df['perplexity'].median(),
            'corr': model_df[['frequency', 'perplexity']].corr().iloc[0, 1],
            'n_concepts': len(model_df)
        })
    
    if not model_stats:
        print("No data to plot")
        return
    
    stats_df = pd.DataFrame(model_stats)
    
    # Plot 1: Mean perplexity vs MMLU rank
    print("\n1. Mean Perplexity vs MMLU-Pro Law Rank")
    print("-" * 80)
    
    plt.clf()
    plt.scatter(
        stats_df['rank'].values,
        stats_df['mean_ppl'].values,
        marker='dot'
    )
    
    plt.xlabel("MMLU-Pro Law Rank (1=best)")
    plt.ylabel("Mean Perplexity")
    plt.title("Lower rank → Lower perplexity?")
    plt.plotsize(100, 25)
    plt.show()
    
    # Calculate rank correlation
    rank_corr = np.corrcoef(stats_df['rank'], stats_df['mean_ppl'])[0, 1]
    print(f"\nRank-Perplexity Correlation: {rank_corr:+.3f}")
    print("(Negative = better MMLU → lower perplexity)")
    
    # Plot 2: Frequency-Perplexity correlation by model
    print("\n\n2. Frequency-Perplexity Correlation by Model")
    print("-" * 80)
    
    plt.clf()
    plt.scatter(
        stats_df['rank'].values,
        stats_df['corr'].values,
        marker='dot'
    )
    
    plt.xlabel("MMLU-Pro Law Rank (1=best)")
    plt.ylabel("Freq-Ppl Correlation")
    plt.title("Do better models show stronger frequency-perplexity relationship?")
    plt.plotsize(100, 25)
    plt.show()
    
    # Print detailed stats
    print("\n\n3. Detailed Model Statistics")
    print("-" * 80)
    print(f"{'Rank':<5} {'Model':<22} {'N':>5} {'Mean PPL':>9} {'Med PPL':>9} {'Freq-Corr':>10}")
    print("-" * 80)
    
    for _, row in stats_df.iterrows():
        print(f"{row['rank']:>4} {row['model']:<22} "
              f"{row['n_concepts']:>5.0f} "
              f"{row['mean_ppl']:>9.1f} "
              f"{row['median_ppl']:>9.1f} "
              f"{row['corr']:>+10.3f}")


def plot_top_vs_bottom_terminal(df: pd.DataFrame):
    """Compare top 5 vs bottom 5 models."""
    print(f"\n{'='*80}")
    print(f"TOP 5 vs BOTTOM 5 MODELS")
    print(f"{'='*80}\n")
    
    top_5_models = MODELS[:5]
    bottom_5_models = MODELS[-5:]
    
    top_df = df[df['model'].isin(top_5_models)]
    bottom_df = df[df['model'].isin(bottom_5_models)]
    
    if len(top_df) == 0 or len(bottom_df) == 0:
        print("Insufficient data")
        return
    
    print("Top 5 models (best MMLU-Pro Law):")
    for model in top_5_models:
        print(f"  - {model}")
    
    print("\nBottom 5 models (worst MMLU-Pro Law):")
    for model in bottom_5_models:
        print(f"  - {model}")
    
    # Box plot comparison
    print("\nPerplexity Distribution Comparison:")
    print("-" * 80)
    
    plt.clf()
    
    top_ppl = np.log10(top_df['perplexity'].values)
    bottom_ppl = np.log10(bottom_df['perplexity'].values)
    
    plt.box([top_ppl, bottom_ppl], labels=['Top 5', 'Bottom 5'])
    plt.ylabel("log10(Perplexity)")
    plt.title("Do top performers have lower perplexity?")
    plt.plotsize(100, 25)
    plt.show()
    
    # Statistical comparison
    print("\nStatistics:")
    print(f"  Top 5    - Mean: {10**top_ppl.mean():7.1f}, Median: {10**np.median(top_ppl):7.1f}")
    print(f"  Bottom 5 - Mean: {10**bottom_ppl.mean():7.1f}, Median: {10**np.median(bottom_ppl):7.1f}")
    
    # t-test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(top_ppl, bottom_ppl)
    print(f"\n  t-test: t={t_stat:.3f}, p={p_value:.4f}")
    if p_value < 0.05:
        print(f"  ✓ Significant difference (p < 0.05)")
    else:
        print(f"  ✗ No significant difference (p >= 0.05)")


def generate_summary_stats(df: pd.DataFrame, output_path: str):
    """Generate summary statistics table."""
    summary_data = []
    
    for i, model_name in enumerate(MODELS):
        model_df = df[df['model'] == model_name]
        
        if len(model_df) == 0:
            continue
        
        # Calculate correlation
        corr = model_df[['frequency', 'perplexity']].corr().iloc[0, 1]
        
        summary_data.append({
            'MMLU_Rank': i + 1,
            'Model': model_name,
            'N_Concepts': len(model_df),
            'Perplexity_Min': model_df['perplexity'].min(),
            'Perplexity_Max': model_df['perplexity'].max(),
            'Perplexity_Mean': model_df['perplexity'].mean(),
            'Perplexity_Median': model_df['perplexity'].median(),
            'Perplexity_Std': model_df['perplexity'].std(),
            'Freq_Ppl_Correlation': corr,
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False, float_format='%.3f')
    print(f"\n✓ Saved summary: {output_path}")
    
    return summary_df


# ----------------------------
# Main Pipeline
# ----------------------------

def main():
    print("="*80)
    print("COMPLETE PERPLEXITY EXPERIMENT - MMLU-Pro Law Models")
    print("="*80)
    print(f"\nPhase 1: Concept Extraction")
    print(f"Phase 2: Perplexity Measurement")
    print(f"Phase 3: Analysis & Visualization")
    print(f"\nModels: {len(MODELS)} models ranked by MMLU-Pro Law")
    print(f"Sample size: {SAMPLE_SIZE} concepts per model")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Phase 1: Extract concepts for all models
    print("\n" + "="*80)
    print("PHASE 1: CONCEPT EXTRACTION")
    print("="*80)
    
    for model_name in MODELS:
        extract_concepts_for_model(model_name)
    
    # Phase 2: Measure perplexity for all models
    print("\n" + "="*80)
    print("PHASE 2: PERPLEXITY MEASUREMENT")
    print("="*80)
    
    all_results = []
    
    for model_name in MODELS:
        # Check if already processed
        safe_name = model_name.replace("/", "_")
        result_file = os.path.join(OUTPUT_DIR, f"{safe_name}__perplexity.csv")
        
        if os.path.exists(result_file):
            print(f"\n✓ Already measured: {model_name}")
            df = pd.read_csv(result_file)
            all_results.append(df)
            continue
        
        # Measure perplexity
        df = measure_perplexity_for_model(model_name)
        
        if df is not None and len(df) > 0:
            # Save individual result
            df.to_csv(result_file, index=False)
            print(f"  ✓ Saved: {result_file}")
            all_results.append(df)
    
    # Phase 3: Analysis
    print("\n" + "="*80)
    print("PHASE 3: ANALYSIS & VISUALIZATION")
    print("="*80)
    
    if not all_results:
        print("\n⚠️  No results to analyze")
        return
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_file = os.path.join(OUTPUT_DIR, "all_perplexity_results.csv")
    combined_df.to_csv(combined_file, index=False)
    print(f"\n✓ Saved combined results: {combined_file}")
    
    # Generate visualizations
    plot_model_comparison_terminal(combined_df)
    plot_top_vs_bottom_terminal(combined_df)
    
    # Generate summary statistics
    summary_path = os.path.join(OUTPUT_DIR, "perplexity_summary_stats.csv")
    summary_df = generate_summary_stats(combined_df, summary_path)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - Combined data: all_perplexity_results.csv")
    print(f"  - Summary stats: perplexity_summary_stats.csv")
    print(f"  - Individual CSVs: *__perplexity.csv")
    print(f"\nConcept extractions saved to: {EXTRACTION_DIR}")
    
    print("\n" + "="*80)
    print("KEY HYPOTHESIS TEST")
    print("="*80)
    print("\nDo models with better MMLU-Pro Law scores have:")
    print("  1. Lower mean perplexity on their own concepts?")
    print("  2. Stronger frequency-perplexity correlation?")
    print("\nCheck the analysis above for answers!")


if __name__ == "__main__":
    main()