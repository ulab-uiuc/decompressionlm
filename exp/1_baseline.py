#!/usr/bin/env python3
"""
Experiment 1: Baseline Comparison (UNIFIED)

Compare 5 sampling methods across 4 domains and 2 models.

Methods:
  1. deLM+VdC: Flat sampling with Van der Corput (offset=0)
  2. deLM+Random: Flat sampling with random sampling  
  3. GraphExplore+BeamHigh: Hierarchical graph with beam search (temp=1.5)
  4. GraphExplore+BeamLow: Hierarchical graph with beam search (temp=0.5)
  5. GraphExplore+Greedy: Hierarchical graph with greedy sampling

Domains: US law, Git, Radiology, FAA
Models: Llama-3.1-8B, Qwen2.5-7B
Threshold: 1024 concepts
Seed: 42
"""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from exp.prompt_templates import graph_root_prompt, graph_child_prompt, flat_concept_list
from src.concept_sampling import sample_concepts
from src.exploration_sampling import explore_graph

# Import unified utilities
sys.path.insert(0, str(Path(__file__).parent))
from exp_common_utils import (
    load_model_with_profiling,
    create_unified_json_schema,
    save_unified_json,
    load_unified_json,
    extract_metrics_from_json,
    extract_metrics_from_fresh_results,
    save_metrics_to_csv,
    print_metrics_table,
)


# ============================================================================
# Configuration
# ============================================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

MODELS = {
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
}

DOMAINS = {
    "us_law": "US law and bar exam",
    "git": "Git version control",
    "radiology": "Radiology and medical imaging",
    "faa": "FAA regulations and aviation",
}

CONCEPT_THRESHOLD = 1024
MAX_LEN = 32
BATCH_SIZE = 64

METHODS = {
    "delm_vdc": {
        "name": "deLM+VdC",
        "type": "flat",
        "sampling_method": "vdc",
    },
    "delm_random": {
        "name": "deLM+Random",
        "type": "flat",
        "sampling_method": "random",
    },
    "graph_greedy": {
        "name": "GraphExplore+Greedy",
        "type": "exploration",
        "max_depth": 64,
        "sequences_per_node": 4,
        "sampling_method": "greedy",
        "node_parallel_batch_size": BATCH_SIZE,
    },
    "graph_beam_high": {
        "name": "GraphExplore+BeamHigh",
        "type": "exploration",
        "max_depth": 64,
        "sequences_per_node": 4,
        "sampling_method": "beam_high",
        "node_parallel_batch_size": BATCH_SIZE,
    },
    "graph_beam_low": {
        "name": "GraphExplore+BeamLow",
        "type": "exploration",
        "max_depth": 64,
        "sequences_per_node": 4,
        "sampling_method": "beam_low",
        "node_parallel_batch_size": BATCH_SIZE,
    },
}


# ============================================================================
# Experiment Runners
# ============================================================================

def run_graph_exploration(
    model, tokenizer, model_short: str, domain_key: str, domain_prompt: str,
    method_config: dict, gpu_profiling: dict, save_dir: str
) -> dict:
    """Run graph exploration method."""
    
    method_name = method_config['name']
    sampling_method = method_config['sampling_method']
    
    # Create save path
    save_name = f"{model_short}__{domain_key}__graph_{sampling_method}"
    save_path = os.path.join(save_dir, save_name)
    json_path = save_path + ".delm.json"
    
    # Check if already exists
    if os.path.exists(json_path):
        print(f"✓ Result exists: {os.path.basename(json_path)}")
        data = load_unified_json(json_path)
        return extract_metrics_from_json(data, model_short, method_name, domain_key, CONCEPT_THRESHOLD)
    
    # Run new exploration
    print(f"\n{'='*80}")
    print(f"Method: {method_name} | Domain: {domain_key}")
    print(f"{'='*80}")
    
    start = time.time()
    
    root, concept_map, graph_stats = explore_graph(
        model=model,
        tokenizer=tokenizer,
        domain=domain_prompt,
        root_prompt_fn=graph_root_prompt,
        child_prompt_fn=graph_child_prompt,
        max_concepts=CONCEPT_THRESHOLD,
        max_depth=method_config['max_depth'],
        sequences_per_node=method_config['sequences_per_node'],
        max_len=MAX_LEN,
        sampling_method=sampling_method,
        node_parallel_batch_size=method_config.get("node_parallel_batch_size", 1),
    )
    
    elapsed = time.time() - start
    
    # Create unified JSON schema
    schema = create_unified_json_schema(
        model_name=model_short,
        method_name=method_name,
        method_type='exploration',
        domain_key=domain_key,
        domain_prompt=domain_prompt,
        threshold=CONCEPT_THRESHOLD,
        sampling_method=sampling_method,
        elapsed_time=elapsed,
        samples_done=graph_stats['total_sequences_generated'],
        valid_concepts_count=graph_stats['concepts_discovered'],
        invalid_concepts_count=0,
        threshold_reached=graph_stats['concepts_discovered'] >= CONCEPT_THRESHOLD,
        sequences=[],  # Not stored for graph methods
        valid_concepts=[],  # Concept list stored separately
        invalid_concepts=[],
        gpu_profiling=gpu_profiling,
        method_config=method_config,
        graph_stats=graph_stats,
        concept_list=list(concept_map.keys()),
    )
    
    save_unified_json(save_path, schema)
    
    return extract_metrics_from_json(schema, model_short, method_name, domain_key, CONCEPT_THRESHOLD)


def run_flat_sampling(
    model, tokenizer, model_short: str, domain_key: str, domain_prompt: str,
    method_config: dict, gpu_profiling: dict, save_dir: str
) -> dict:
    """Run flat sampling method."""
    
    sampling_method = method_config['sampling_method']
    method_name = method_config['name']
    
    # Create save path
    save_name = f"{model_short}__{domain_key}__{sampling_method}"
    save_path = os.path.join(save_dir, save_name)
    json_path = save_path + ".delm.json"
    
    # Check if exists
    if os.path.exists(json_path):
        print(f"✓ Result exists: {os.path.basename(json_path)}")
        data = load_unified_json(json_path)
        return extract_metrics_from_json(data, model_short, method_name, domain_key, CONCEPT_THRESHOLD)
    
    # Run sampling
    print(f"\n{'='*80}")
    print(f"Method: {method_name} | Domain: {domain_key}")
    print(f"{'='*80}")
    
    sampling_params = {}
    if sampling_method == "vdc":
        sampling_params['offset'] = 0.0
    
    results = sample_concepts(
        model=model,
        tokenizer=tokenizer,
        prompt_fn=lambda: flat_concept_list(domain_prompt),
        concept_threshold=CONCEPT_THRESHOLD,
        max_samples=100000,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        sampling_method=sampling_method,
        sampling_params=sampling_params,
        save_path=save_path,
        model_name=model_short,
    )
    
    return extract_metrics_from_fresh_results(
        results, model_short, method_name, domain_key, CONCEPT_THRESHOLD, gpu_profiling
    )


# ============================================================================
# Main
# ============================================================================

def main():
    save_dir = "results/exp1_baselines"
    csv_path = f"{save_dir}/metrics.csv"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*80}")
    print("EXPERIMENT 1: BASELINE COMPARISON (UNIFIED)")
    print(f"{'='*80}")
    print(f"Seed: {SEED}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Domains: {list(DOMAINS.keys())}")
    print(f"Methods: {[m['name'] for m in METHODS.values()]}")
    print(f"Concept threshold: {CONCEPT_THRESHOLD}")
    print(f"Total experiments: {len(MODELS) * len(DOMAINS) * len(METHODS)}")
    print(f"{'='*80}\n")
    
    all_metrics = []
    
    for model_short, model_path in MODELS.items():
        # Load model with profiling (setup padding for graph methods)
        model, tokenizer, gpu_profiling = load_model_with_profiling(
            model_path, model_short, setup_padding=True
        )
        
        for domain_key, domain_prompt in DOMAINS.items():
            print(f"\n{'#'*80}")
            print(f"# {model_short} | {domain_key}")
            print(f"{'#'*80}")
            
            for method_key, method_config in METHODS.items():
                try:
                    method_type = method_config['type']
                    
                    if method_type == 'exploration':
                        metrics = run_graph_exploration(
                            model, tokenizer, model_short, domain_key, domain_prompt,
                            method_config, gpu_profiling, save_dir
                        )
                    elif method_type == 'flat':
                        metrics = run_flat_sampling(
                            model, tokenizer, model_short, domain_key, domain_prompt,
                            method_config, gpu_profiling, save_dir
                        )
                    else:
                        raise ValueError(f"Unknown type: {method_type}")
                    
                    all_metrics.append(metrics)
                    
                    print(f"\n✓ Completed: {model_short}/{domain_key}/{method_config['name']}")
                    print(f"  Samples: {metrics['samples']}, Valid: {metrics['valid_concepts']}, "
                          f"Time: {metrics['total_time']}s")
                    
                except Exception as e:
                    print(f"\n✗ Failed: {model_short}/{domain_key}/{method_config['name']}")
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
        
        del model
        torch.cuda.empty_cache()
    
    # Print and save results
    print_metrics_table(all_metrics, "EXPERIMENT 1: BASELINE COMPARISON - RESULTS", show_gpu=True)
    save_metrics_to_csv(all_metrics, csv_path)
    
    print(f"\n✓ Experiment 1 complete!")
    print(f"✓ Results: {save_dir}/")
    print(f"✓ CSV: {csv_path}")
    print(f"✓ Total experiments: {len(all_metrics)}")


if __name__ == "__main__":
    main()
