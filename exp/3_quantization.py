#!/usr/bin/env python3
"""
Experiment 3: Quantization Ladder

Test decompressionLM with different quantization methods across 4 domains.
Each config runs twice:
1. concept_threshold=100 (or configured), max_seq_count=None
2. concept_threshold=0 (infinity), max_seq_count=4096

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
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import sample_concepts
from src.profiling import measure_model_memory


# Configuration
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

# Experiment configs
RUNS = [
    {"name": "threshold", "concept_threshold": 4096, "max_seq_count": None},
    {"name": "full_gen", "concept_threshold": 0, "max_seq_count": 4096},  # 0 = infinity
]

# Shared params
MAX_LEN = 32
BATCH_SIZE = 16
SAMPLING_METHOD = "vdc"


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
    run_config: dict,
    save_dir: str,
):
    """Run single experiment configuration."""
    
    concept_threshold = run_config["concept_threshold"]
    max_seq_count = run_config["max_seq_count"]
    run_name = run_config["name"]
    
    # Build prompt - STRICT FORMAT
    if domain_name == "us_law":
        prefix = """Generate United States bar exam legal concepts as keywords.
Please output ONE concept per line.
Each concept can be multiple words if needed.
Do not include explanations or extra text.
Please begin from any random concept.
Please use English.
"""
    else:
        # For other domains, adapt the format
        prefix = f"""Generate {domain_prompt} concepts as keywords.
Please output ONE concept per line.
Each concept can be multiple words if needed.
Do not include explanations or extra text.
Please begin from any random concept.
Please use English.
"""
    
    # Save path
    save_name = f"{model_name}__{quant_name}__{domain_name}__{run_name}"
    save_path = os.path.join(save_dir, save_name)
    
    print(f"\n{'='*80}")
    print(f"Running: {save_name}")
    print(f"{'='*80}")
    print(f"Threshold: {concept_threshold if concept_threshold > 0 else 'INFINITY (disabled)'}")
    print(f"Max sequences: {max_seq_count if max_seq_count else 'None'}")
    
    # Run sampling (model loading time NOT counted)
    # If threshold=0, it means infinity - just use max_seq_count as stopping condition
    results = sample_concepts(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        concept_threshold=concept_threshold,  # Pass 0 directly, let sample_concepts handle it
        max_samples=max_seq_count if max_seq_count else 100000,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        sampling_method=SAMPLING_METHOD,
        save_path=save_path,
        model_name=model_name,
    )
    
    # Extract metrics
    metrics = extract_metrics(results, model_name, quant_name, domain_name, run_name)
    
    return metrics


def extract_metrics(results: dict, model_name: str, quant_name: str, domain_name: str, run_name: str) -> dict:
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
    
    # Graph metrics (if available)
    # Note: We need to compute graph from sequences
    # For now, placeholder - will add graph computation
    
    metrics = {
        # Identifiers
        'model': model_name,
        'quantization': quant_name,
        'domain': domain_name,
        'run': run_name,
        
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
    print("EXPERIMENT 1: QUANTIZATION LADDER - RESULTS")
    print(f"{'='*120}\n")
    
    # Group by run type
    for run_name in ["threshold", "full_gen"]:
        run_metrics = [m for m in all_metrics if m['run'] == run_name]
        
        if not run_metrics:
            continue
        
        print(f"\n{run_name.upper().replace('_', ' ')}")
        print("-" * 120)
        
        # Header
        print(f"{'Model':<15} | {'Quant':<10} | {'Domain':<10} | {'Samples':>7} | "
              f"{'Valid':>6} | {'Invalid':>7} | {'Ratio':>6} | {'Conv':>5} | "
              f"{'Time':>7} | {'Tok/s':>7} | {'Con/s':>7}")
        print("-" * 120)
        
        # Rows
        for m in run_metrics:
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


def main():
    """Run experiment 3."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/exp3_quantization_{timestamp}"
    csv_path = f"{save_dir}/metrics.csv"
    
    print(f"{'='*80}")
    print("EXPERIMENT 3: QUANTIZATION LADDER")
    print(f"{'='*80}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Quantizations: {list(QUANTIZATIONS.keys())}")
    print(f"Domains: {list(DOMAINS.keys())}")
    print(f"Runs per config: {len(RUNS)}")
    print(f"Total experiments: {len(MODELS) * len(QUANTIZATIONS) * len(DOMAINS) * len(RUNS)}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*80}\n")
    
    all_metrics = []
    
    # Iterate through all combinations
    for model_short, model_path in MODELS.items():
        for quant_name, quant_config in QUANTIZATIONS.items():
            
            # Load model once per quantization
            model, tokenizer, load_time, model_mem = load_model_with_timing(model_path, quant_config)
            
            for domain_name, domain_prompt in DOMAINS.items():
                for run_config in RUNS:
                    
                    try:
                        metrics = run_experiment(
                            model, tokenizer,
                            model_short, quant_name,
                            domain_name, domain_prompt,
                            run_config, save_dir
                        )
                        
                        all_metrics.append(metrics)
                        
                        # Print after each run
                        print(f"\n✓ Completed: {model_short}/{quant_name}/{domain_name}/{run_config['name']}")
                        print(f"  Samples: {metrics['samples']}, Valid: {metrics['valid_concepts']}, "
                              f"Invalid: {metrics['invalid_concepts']}, Time: {metrics['total_time']}s")
                        
                    except Exception as e:
                        print(f"\n✗ Failed: {model_short}/{quant_name}/{domain_name}/{run_config['name']}")
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