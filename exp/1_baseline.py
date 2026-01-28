#!/usr/bin/env python3
"""
Experiment 1: Baseline Comparison

Compare different sampling methods on US law domain.

Models: Llama-3.1-8B, Qwen2.5-7B
Domain: US law & Bar Exam
Methods:
  1: Beam search + single YAML graph
  2: Graph exploration (BFS-style with reconnections)
  3: deLM + beam high temp
  4: deLM + beam low temp
  5: deLM + uniform random sample
"""

import os
import sys
import time
import csv
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from exp.prompt_templates import beam_graph_prompt, graph_root_prompt, graph_child_prompt, flat_concept_list
from src.concept_sampling import sample_concepts
from src.exploration_sampling import explore_graph
from src.profiling import measure_model_memory


# ============================================================================
# Configuration
# ============================================================================

MODELS = {
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
}

DOMAIN = "US law and bar exam"

METHODS = {
    "beam_graph": {
        "name": "Beam+Graph",
        "type": "graph_gen",
        "sampling_method": "beam_low",
    },
    "graph_explore": {
        "name": "GraphExplore",
        "type": "exploration",
        "max_depth": 3,
        "sequences_per_node": 4,
    },
    "delm_beam_high": {
        "name": "deLM+BeamHigh",
        "type": "flat",
        "sampling_method": "beam_high",
    },
    "delm_beam_low": {
        "name": "deLM+BeamLow",
        "type": "flat",
        "sampling_method": "beam_low",
    },
    "delm_random": {
        "name": "deLM+Random",
        "type": "flat",
        "sampling_method": "random",
    },
}

CONCEPT_THRESHOLD = 4096
MAX_SEQ_COUNT = 0
MAX_LEN = 32
BATCH_SIZE = 16


# ============================================================================
# Helper Functions
# ============================================================================

def load_model_with_timing(model_name: str):
    """Load model and return (model, tokenizer, load_time)."""
    print(f"\n{'='*80}")
    print(f"Loading {model_name}...")
    print(f"{'='*80}")
    
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    load_time = time.time() - start
    model_mem = measure_model_memory()
    
    print(f"✓ Loaded in {load_time:.2f}s")
    print(f"✓ GPU memory: {model_mem.get('model_memory_allocated_gb', 0):.2f} GB")
    
    return model, tokenizer, load_time, model_mem


def run_graph_generation(model, tokenizer, model_name: str, save_dir: str):
    """Method: Beam search + generate graph in YAML format (single sequence)."""
    
    save_path = os.path.join(save_dir, f"{model_name}__beam_graph")
    
    print(f"\n{'='*80}")
    print(f"Method: Beam+Graph (single YAML sequence)")
    print(f"{'='*80}")
    
    start = time.time()
    
    # Get prompt
    prompt = beam_graph_prompt(DOMAIN)
    
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        prefix_ids,
        max_new_tokens=512,
        num_beams=4,
        temperature=0.7,
        do_sample=False,
    )
    
    elapsed = time.time() - start
    
    generated_ids = outputs[0][prefix_ids.shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Parse concepts
    import re
    concepts = re.findall(r'-\s*concept:\s*([^\n]+)', text)
    concepts = [c.strip() for c in concepts]
    
    # Save
    save_file = f"{save_path}.yaml"
    with open(save_file, 'w') as f:
        f.write(text)
    
    print(f"✓ Generated graph with {len(concepts)} concepts in {elapsed:.2f}s")
    print(f"✓ Saved to {save_file}")
    
    return {
        'model': model_name,
        'method': 'Beam+Graph',
        'samples': 1,
        'valid_concepts': len(concepts),
        'invalid_concepts': 0,
        'invalid_ratio': "0.0000",
        'threshold_reached': len(concepts) >= CONCEPT_THRESHOLD,
        'tokens_per_sec': f"{len(generated_ids) / elapsed:.2f}",
        'concepts_per_sec': f"{len(concepts) / elapsed:.4f}",
        'total_time': f"{elapsed:.2f}",
        'time_to_converge': f"{elapsed:.2f}",
        'total_tokens': len(generated_ids),
        'avg_tokens_per_seq': f"{len(generated_ids):.2f}",
        'gpu_allocated_max': "N/A",
    }


def run_graph_exploration(model, tokenizer, model_name: str, method_config: dict, save_dir: str):
    """Method: Graph exploration with reconnections."""
    
    save_path = os.path.join(save_dir, f"{model_name}__graph_explore")
    
    print(f"\n{'='*80}")
    print(f"Method: GraphExplore (graph with reconnections)")
    print(f"{'='*80}")
    
    start = time.time()
    
    root, concept_map, stats = explore_graph(
        model=model,
        tokenizer=tokenizer,
        domain=DOMAIN,
        root_prompt_fn=graph_root_prompt,
        child_prompt_fn=graph_child_prompt,
        max_concepts=CONCEPT_THRESHOLD,
        max_depth=method_config['max_depth'],
        sequences_per_node=method_config['sequences_per_node'],
        max_len=MAX_LEN,
        sampling_method="vdc",
    )
    
    elapsed = time.time() - start
    
    total_tokens = stats['total_sequences_generated'] * MAX_LEN
    
    return {
        'model': model_name,
        'method': 'GraphExplore',
        'samples': stats['total_sequences_generated'],
        'valid_concepts': stats['concepts_discovered'],
        'invalid_concepts': 0,
        'invalid_ratio': "N/A",
        'threshold_reached': stats['concepts_discovered'] >= CONCEPT_THRESHOLD,
        'tokens_per_sec': f"{total_tokens / elapsed:.2f}",
        'concepts_per_sec': f"{stats['concepts_discovered'] / elapsed:.4f}",
        'total_time': f"{elapsed:.2f}",
        'time_to_converge': f"{elapsed:.2f}",
        'total_tokens': total_tokens,
        'avg_tokens_per_seq': f"{MAX_LEN:.2f}",
        'gpu_allocated_max': "N/A",
        'max_depth': stats['max_depth_reached'],
        'nodes_explored': stats['total_nodes_explored'],
        'reconnections': stats['reconnections'],
    }


def run_flat_sampling(model, tokenizer, model_name: str, method_config: dict, save_dir: str):
    """Methods: Flat sampling with different methods."""
    
    sampling_method = method_config['sampling_method']
    method_name = method_config['name']
    
    save_path = os.path.join(save_dir, f"{model_name}__{sampling_method}")
    
    print(f"\n{'='*80}")
    print(f"Method: {method_name}")
    print(f"{'='*80}")
    
    results = sample_concepts(
        model=model,
        tokenizer=tokenizer,
        prompt_fn=lambda: flat_concept_list(DOMAIN),
        concept_threshold=CONCEPT_THRESHOLD,
        max_samples=MAX_SEQ_COUNT if MAX_SEQ_COUNT > 0 else 100000,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        sampling_method=sampling_method,
        save_path=save_path,
        model_name=model_name,
    )
    
    return extract_metrics(results, model_name, method_name)


def extract_metrics(results: dict, model_name: str, method_name: str) -> dict:
    """Extract metrics from flat sampling results."""
    
    prof = results.get('profiling', {})
    elapsed = results['elapsed_time']
    samples_done = results['samples_done']
    
    tokens_total = sum(len(seq['tokens']) for seq in results['sequences'])
    tokens_per_sec = tokens_total / elapsed if elapsed > 0 else 0
    concepts_per_sec = results['valid_concepts'] / elapsed if elapsed > 0 else 0
    
    total_raw = results['valid_concepts'] + results['invalid_concepts']
    invalid_ratio = results['invalid_concepts'] / total_raw if total_raw > 0 else 0
    
    return {
        'model': model_name,
        'method': method_name,
        'samples': samples_done,
        'valid_concepts': results['valid_concepts'],
        'invalid_concepts': results['invalid_concepts'],
        'invalid_ratio': f"{invalid_ratio:.4f}",
        'threshold_reached': results['threshold_reached'],
        'tokens_per_sec': f"{tokens_per_sec:.2f}",
        'concepts_per_sec': f"{concepts_per_sec:.4f}",
        'total_time': f"{elapsed:.2f}",
        'time_to_converge': f"{elapsed:.2f}" if results['threshold_reached'] else "N/A",
        'total_tokens': tokens_total,
        'avg_tokens_per_seq': f"{tokens_total / samples_done if samples_done > 0 else 0:.2f}",
        'sampling_time_avg': f"{prof.get('time_sampling_avg', 0):.4f}",
        'extraction_time_avg': f"{prof.get('time_concept_extraction_avg', 0):.4f}",
        'gpu_allocated_avg': f"{prof.get('gpu_sampling_end_allocated_avg', 0):.2f}",
        'gpu_allocated_max': f"{prof.get('gpu_sampling_end_allocated_max', 0):.2f}",
    }


def print_metrics_table(all_metrics: list):
    """Print formatted comparison table."""
    
    print(f"\n{'='*120}")
    print("EXPERIMENT 1: BASELINE COMPARISON - RESULTS")
    print(f"Domain: {DOMAIN}")
    print(f"{'='*120}\n")
    
    print(f"{'Model':<15} | {'Method':<15} | {'Samples':>7} | {'Valid':>6} | {'Invalid':>7} | "
          f"{'Ratio':>6} | {'Conv':>5} | {'Time':>7} | {'Tok/s':>8} | {'Con/s':>8}")
    print("-" * 120)
    
    for m in all_metrics:
        converged = "YES" if m['threshold_reached'] else "NO"
        print(f"{m['model']:<15} | {m['method']:<15} | {m['samples']:>7} | "
              f"{m['valid_concepts']:>6} | {m['invalid_concepts']:>7} | "
              f"{m['invalid_ratio']:>6} | {converged:>5} | "
              f"{m['total_time']:>7}s | {m['tokens_per_sec']:>8} | {m['concepts_per_sec']:>8}")
    
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
    """Run experiment 1."""
    
    save_dir = "results/exp1_baselines"
    csv_path = f"{save_dir}/metrics.csv"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*80}")
    print("EXPERIMENT 1: BASELINE COMPARISON")
    print(f"{'='*80}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Domain: {DOMAIN}")
    print(f"Methods: {[m['name'] for m in METHODS.values()]}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*80}\n")
    
    all_metrics = []
    
    for model_short, model_path in MODELS.items():
        
        model, tokenizer, load_time, model_mem = load_model_with_timing(model_path)
        
        for method_key, method_config in METHODS.items():
            
            try:
                method_type = method_config['type']
                
                if method_type == 'graph_gen':
                    metrics = run_graph_generation(model, tokenizer, model_short, save_dir)
                elif method_type == 'exploration':
                    metrics = run_graph_exploration(model, tokenizer, model_short, method_config, save_dir)
                elif method_type == 'flat':
                    metrics = run_flat_sampling(model, tokenizer, model_short, method_config, save_dir)
                else:
                    raise ValueError(f"Unknown method type: {method_type}")
                
                all_metrics.append(metrics)
                
                print(f"\n✓ Completed: {model_short}/{method_config['name']}")
                print(f"  Samples: {metrics['samples']}, Valid: {metrics['valid_concepts']}, Time: {metrics['total_time']}s")
                
            except Exception as e:
                print(f"\n✗ Failed: {model_short}/{method_config['name']}")
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
        
        del model
        torch.cuda.empty_cache()
    
    print_metrics_table(all_metrics)
    save_to_csv(all_metrics, csv_path)
    
    print(f"\n✓ Experiment 1 complete!")
    print(f"✓ Results saved to {save_dir}/")


if __name__ == "__main__":
    main()
