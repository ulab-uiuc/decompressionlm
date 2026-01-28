#!/usr/bin/env python3
"""
Experiment 3: Quantization Ladder (UNIFIED)

Test decompressionLM with different quantization methods across 4 domains.

Models: Llama-3.1-8B, Qwen2.5-7B
Quantizations: BF16, GPTQ_INT8, AWQ_4BIT, GPTQ_INT4, BNB_4BIT
Domains: US law, Git, Radiology, FAA
Threshold: 4096 concepts
Seed: 42
"""

import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from exp.prompt_templates import flat_concept_list
from src.concept_sampling import sample_concepts

# Import unified utilities
sys.path.insert(0, str(Path(__file__).parent))
from exp_common_utils import (
    load_model_with_profiling,
    extract_metrics_from_fresh_results,
    save_metrics_to_csv,
    print_metrics_table,
)


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

CONCEPT_THRESHOLD = 4096
MAX_LEN = 32
BATCH_SIZE = 16
SAMPLING_METHOD = "vdc"


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(
    model, tokenizer, model_short: str, quant_name: str,
    domain_key: str, domain_prompt: str,
    gpu_profiling: dict, save_dir: str
):
    """Run single quantization + domain experiment."""
    
    save_name = f"{model_short}__{quant_name}__{domain_key}"
    save_path = os.path.join(save_dir, save_name)
    json_path = save_path + ".delm.json"
    
    # Check if exists
    if os.path.exists(json_path):
        print(f"✓ Result exists: {os.path.basename(json_path)}")
        
        # Load and extract metrics
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        metrics = extract_metrics_from_fresh_results(
            data, model_short, quant_name, domain_key, CONCEPT_THRESHOLD, gpu_profiling
        )
        metrics['quantization'] = quant_name
        return metrics
    
    # Run new experiment
    print(f"\n{'='*80}")
    print(f"{model_short} | {quant_name} | {domain_key}")
    print(f"{'='*80}")
    
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
        model_name=model_short,
    )
    
    # Extract metrics
    metrics = extract_metrics_from_fresh_results(
        results, model_short, quant_name, domain_key, CONCEPT_THRESHOLD, gpu_profiling
    )
    metrics['quantization'] = quant_name
    
    print(f"✓ Completed: {metrics['samples']} samples, {metrics['valid_concepts']} concepts")
    
    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    save_dir = "results/exp3_quantization"
    csv_path = f"{save_dir}/metrics.csv"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    total_experiments = len(MODELS) * len(QUANTIZATIONS) * len(DOMAINS)
    
    print(f"{'='*80}")
    print("EXPERIMENT 3: QUANTIZATION LADDER (UNIFIED)")
    print(f"{'='*80}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Quantizations: {list(QUANTIZATIONS.keys())}")
    print(f"Domains: {list(DOMAINS.keys())}")
    print(f"Concept threshold: {CONCEPT_THRESHOLD}")
    print(f"Total experiments: {total_experiments}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*80}\n")
    
    all_metrics = []
    
    # Iterate through all combinations
    for model_short, model_path in MODELS.items():
        for quant_name, quant_config in QUANTIZATIONS.items():
            
            print(f"\n{'#'*80}")
            print(f"# {model_short} | {quant_name}")
            print(f"{'#'*80}")
            
            # Load model with quantization
            model, tokenizer, gpu_profiling = load_model_with_profiling(
                model_path, model_short, quant_config=quant_config
            )
            
            for domain_key, domain_prompt in DOMAINS.items():
                
                try:
                    metrics = run_experiment(
                        model, tokenizer, model_short, quant_name,
                        domain_key, domain_prompt,
                        gpu_profiling, save_dir
                    )
                    
                    all_metrics.append(metrics)
                    
                except Exception as e:
                    print(f"\n✗ Failed: {model_short}/{quant_name}/{domain_key}")
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Clean up model
            del model
            torch.cuda.empty_cache()
    
    # Print and save results
    print_metrics_table(
        all_metrics,
        "EXPERIMENT 3: QUANTIZATION LADDER - RESULTS",
        show_gpu=True
    )
    
    save_metrics_to_csv(all_metrics, csv_path)
    
    print(f"\n✓ Experiment 3 complete!")
    print(f"✓ Results: {save_dir}/")
    print(f"✓ CSV: {csv_path}")
    print(f"✓ Total experiments: {len(all_metrics)}")


if __name__ == "__main__":
    main()
