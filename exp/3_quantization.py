#!/usr/bin/env python3
"""
Experiment 3: Quantization Ladder

Test decompressionLM with different quantization methods across 4 domains.
Each config runs once with concept_threshold.

Models: Llama-3.1-8B, Qwen2.5-7B
Quantizations: BF16, GPTQ_INT8, AWQ_4BIT, GPTQ_INT4, BNB_4BIT
Domains: US law, git, radiology, FAA
"""

import os
import sys
import time
import csv
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from exp.prompt_templates import flat_concept_list
from src.concept_sampling import sample_concepts
from src.profiling import measure_model_memory


# ============================================================================
# Configuration
# ============================================================================

MODELS = {
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
}

QUANTIZATIONS = {
    "BF16": {"load_in_4bit": False, "load_in_8bit": False, "torch_dtype": torch.bfloat16},
    "GPTQ_INT8": {"load_in_8bit": True, "torch_dtype": torch.float16},
    "AWQ_4BIT": {"load_in_4bit": True, "bnb_4bit_use_double_quant": False, "bnb_4bit_quant_type": "nf4"},
    "GPTQ_INT4": {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16},
    "BNB_4BIT": {"load_in_4bit": True, "bnb_4bit_use_double_quant": True, "bnb_4bit_quant_type": "nf4"},
}

DOMAINS = {
    "us_law": "US law and bar exam",
    "git": "Git version control",
    "radiology": "Radiology and medical imaging",
    "faa": "FAA regulations and aviation",
}

# Experiment config
CONCEPT_THRESHOLD = 4096
MAX_LEN = 32
BATCH_SIZE = 16
SAMPLING_METHOD = "vdc"


# ============================================================================
# Helper Functions
# ============================================================================

def load_model_with_timing(model_name: str, quant_config: dict):
    """Load model and return (model, tokenizer, load_time)."""
    print(f"\n{'='*80}")
    print(f"Loading {model_name} with {quant_config}...")
    print(f"{'='*80}")
    
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        **quant_config
    )
    
    load_time = time.time() - start
    
    # Measure static memory
    model_mem = measure_model_memory()
    
    print(f"✓ Loaded in {load_time:.2f}s")
    print(f"✓ GPU memory: {model_mem.get('model_memory_allocated_gb', 0):.2f} GB")
    
    return model, tokenizer, load_time, model_mem


def run_experiment(
    model,
    tokenizer,
    model_name: str,
    quant_name: str,
    domain_name: str,
    domain_prompt: str,
    save_dir: str,
):
    """Run single experiment configuration."""
    
    # Save path
    save_name = f"{model_name}__{quant_name}__{domain_name}"
    save_path = os.path.join(save_dir, save_name)
    
    print(f"\n{'='*80}")
    print(f"Running: {save_name}")
    print(f"{'='*80}")
    print(f"Threshold: {CONCEPT_THRESHOLD}")
    
    # Run sampling
    results = sample_concepts(
        model=model,
        tokenizer=tokenizer,
        prompt_fn=lambda: flat_concept_list(domain_prompt),
        concept_threshold=CONCEPT_THRESHOLD,
        max_samples=100000,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        sampling_method=SAMPLING_METHOD,
        save_path=save_path,
        model_name=model_name,
    )
    
    # Extract metrics
    metrics = extract_metrics(results, model_name, quant_name, domain_name)
    
    return metrics


def extract_metrics(results: dict, model_name: str, quant_name: str, domain_name: str) -> dict:
    """Extract all metrics from results."""
    
    prof = results.get('profiling', {})
    elapsed = results['elapsed_time']
    samples_done = results['samples_done']
    
    # Rate metrics
    tokens_total = sum(len(seq['tokens']) for seq in results['sequences'])
    tokens_per_sec = tokens_total / elapsed if elapsed > 0 else 0
    
    concepts_per_sec = results['valid_concepts'] / elapsed if elapsed > 0 else 0
    
    # Invalid concept ratio (raw, before merging)
    total_raw_concepts = results['valid_concepts'] + results['invalid_concepts']
    invalid_ratio = results['invalid_concepts'] / total_raw_concepts if total_raw_concepts > 0 else 0
    
    metrics = {
        # Identifiers
        'model': model_name,
        'quantization': quant_name,
        'domain': domain_name,
        
        # Core counts
        'samples': samples_done,
        'valid_concepts': results['valid_concepts'],
        'invalid_concepts': results['invalid_concepts'],
        'invalid_ratio': f"{invalid_ratio:.4f}",
        'threshold_reached': results['threshold_reached'],
        
        # Rate metrics
        'tokens_per_sec': f"{tokens_per_sec:.2f}",
        'concepts_per_sec': f"{concepts_per_sec:.4f}",
        
        # Time metrics
        'total_time': f"{elapsed:.2f}",
        'time_to_converge': f"{elapsed:.2f}" if results['threshold_reached'] else "N/A",
        
        # Profiling
        'sampling_time_avg': f"{prof.get('time_sampling_avg', 0):.4f}",
        'extraction_time_avg': f"{prof.get('time_concept_extraction_avg', 0):.4f}",
        
        # GPU memory
        'gpu_allocated_avg': f"{prof.get('gpu_sampling_end_allocated_avg', 0):.2f}",
        'gpu_allocated_max': f"{prof.get('gpu_sampling_end_allocated_max', 0):.2f}",
        'gpu_reserved_avg': f"{prof.get('gpu_sampling_end_reserved_avg', 0):.2f}",
        
        # Token stats
        'total_tokens': tokens_total,
        'avg_tokens_per_seq': f"{tokens_total / samples_done if samples_done > 0 else 0:.2f}",
    }
    
    return metrics


def print_metrics_table(all_metrics: list):
    """Print formatted metrics table."""
    
    if not all_metrics:
        print("No metrics to display")
        return
    
    print(f"\n{'='*120}")
    print("EXPERIMENT 3: QUANTIZATION LADDER - RESULTS")
    print(f"{'='*120}\n")
    
    # Header
    print(f"{'Model':<15} | {'Quant':<10} | {'Domain':<10} | {'Samples':>7} | "
          f"{'Valid':>6} | {'Invalid':>7} | {'Ratio':>6} | {'Conv':>5} | "
          f"{'Time':>7} | {'Tok/s':>7} | {'Con/s':>7}")
    print("-" * 120)
    
    # Rows
    for m in all_metrics:
        converged = "YES" if m['threshold_reached'] else "NO"
        print(f"{m['model']:<15} | {m['quantization']:<10} | {m['domain']:<10} | "
              f"{m['samples']:>7} | {m['valid_concepts']:>6} | {m['invalid_concepts']:>7} | "
              f"{m['invalid_ratio']:>6} | {converged:>5} | "
              f"{m['total_time']:>7}s | {m['tokens_per_sec']:>7} | {m['concepts_per_sec']:>7}")
    
    print(f"\n{'='*120}\n")


def save_to_csv(all_metrics: list, output_path: str):
    """Save metrics to CSV."""
    
    if not all_metrics:
        return
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = list(all_metrics[0].keys())
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)
    
    print(f"✓ Saved CSV to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run experiment 3."""
    
    save_dir = "results/exp3_quantization"
    csv_path = f"{save_dir}/metrics.csv"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*80}")
    print("EXPERIMENT 3: QUANTIZATION LADDER")
    print(f"{'='*80}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Quantizations: {list(QUANTIZATIONS.keys())}")
    print(f"Domains: {list(DOMAINS.keys())}")
    print(f"Concept threshold: {CONCEPT_THRESHOLD}")
    print(f"Total experiments: {len(MODELS) * len(QUANTIZATIONS) * len(DOMAINS)}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*80}\n")
    
    all_metrics = []
    
    # Iterate through all combinations
    for model_short, model_path in MODELS.items():
        for quant_name, quant_config in QUANTIZATIONS.items():
            
            # Load model once per quantization
            model, tokenizer, load_time, model_mem = load_model_with_timing(model_path, quant_config)
            
            for domain_name, domain_prompt in DOMAINS.items():
                
                try:
                    metrics = run_experiment(
                        model, tokenizer,
                        model_short, quant_name,
                        domain_name, domain_prompt,
                        save_dir
                    )
                    
                    all_metrics.append(metrics)
                    
                    # Print after each run
                    print(f"\n✓ Completed: {model_short}/{quant_name}/{domain_name}")
                    print(f"  Samples: {metrics['samples']}, Valid: {metrics['valid_concepts']}, "
                          f"Invalid: {metrics['invalid_concepts']}, Time: {metrics['total_time']}s")
                    
                except Exception as e:
                    print(f"\n✗ Failed: {model_short}/{quant_name}/{domain_name}")
                    print(f"  Error: {e}")
            
            # Clean up model
            del model
            torch.cuda.empty_cache()
    
    # Print final table
    print_metrics_table(all_metrics)
    
    # Save CSV
    save_to_csv(all_metrics, csv_path)
    
    print(f"\n✓ Experiment 3 complete!")
    print(f"✓ Results saved to {save_dir}/")


if __name__ == "__main__":
    main()
