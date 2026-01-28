#!/usr/bin/env python3
"""
Experiment 4: Hallucination Metrics (UNIFIED)

Run models on US law domain with different sampling methods.
Reduced to 128 concepts for manual fact-checking.

Models: Llama-3.1-8B, Qwen2.5-7B, Mistral-7B, Phi-3-14B
Domain: US law & Bar Exam
Methods: deLM+VdC, deLM+Random, GraphExplore, Beam+Graph
Threshold: 128 concepts (for manual verification)
Seed: 42
"""

import os
import sys
import re
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from exp.prompt_templates import (
    beam_graph_prompt,
    graph_root_prompt,
    graph_child_prompt,
    flat_concept_list,
)
from src.concept_sampling import sample_concepts
from src.exploration_sampling import explore_graph

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
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Phi-3-14B": "microsoft/Phi-3-medium-128k-instruct",
}

DOMAIN = "US law and bar exam"

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
    "graph_explore": {
        "name": "GraphExplore",
        "type": "exploration",
        "max_depth": 3,
        "sequences_per_node": 4,
    },
    "beam_graph": {
        "name": "Beam+Graph",
        "type": "graph_gen",
        "sampling_method": "beam_low",
    },
}

CONCEPT_THRESHOLD = 128
MAX_LEN = 32
BATCH_SIZE = 16


# ============================================================================
# Method Runners
# ============================================================================

def run_graph_generation(model, tokenizer, model_short: str, gpu_profiling: dict, save_dir: str):
    """Beam search + generate graph in YAML format."""
    
    save_path = os.path.join(save_dir, f"{model_short}__beam_graph")
    json_path = save_path + ".yaml"
    
    # Check if exists
    if os.path.exists(json_path):
        print(f"✓ Result exists: {os.path.basename(json_path)}")
        
        # Load and parse
        with open(json_path, 'r') as f:
            text = f.read()
        
        concepts = re.findall(r'-\s*concept:\s*([^\n]+)', text)
        concepts = [c.strip() for c in concepts]
        
        return {
            'model': model_short,
            'method': 'Beam+Graph',
            'samples': 1,
            'valid_concepts': len(concepts),
            'invalid_concepts': 0,
            'invalid_ratio': "0.0000",
            'threshold_reached': len(concepts) >= CONCEPT_THRESHOLD,
            'total_time': "0.00",
            'tokens_per_sec': "0.00",
            'concepts_per_sec': "0.0000",
            'total_tokens': 0,
        }
    
    # Generate new
    print(f"\n{'='*80}")
    print(f"{model_short} | Beam+Graph")
    print(f"{'='*80}")
    
    import time
    start = time.time()
    
    prompt = beam_graph_prompt(DOMAIN)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(formatted, return_tensors="pt").to(model.device)
    
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
    concepts = re.findall(r'-\s*concept:\s*([^\n]+)', text)
    concepts = [c.strip() for c in concepts]
    
    # Save
    with open(json_path, 'w') as f:
        f.write(text)
    
    print(f"✓ Generated {len(concepts)} concepts in {elapsed:.2f}s")
    
    return {
        'model': model_short,
        'method': 'Beam+Graph',
        'samples': 1,
        'valid_concepts': len(concepts),
        'invalid_concepts': 0,
        'invalid_ratio': "0.0000",
        'threshold_reached': len(concepts) >= CONCEPT_THRESHOLD,
        'total_time': f"{elapsed:.2f}",
        'tokens_per_sec': f"{len(generated_ids) / elapsed:.2f}",
        'concepts_per_sec': f"{len(concepts) / elapsed:.4f}",
        'total_tokens': len(generated_ids),
    }


def run_graph_exploration(model, tokenizer, model_short: str, method_config: dict, 
                         gpu_profiling: dict, save_dir: str):
    """Graph exploration method."""
    
    save_path = os.path.join(save_dir, f"{model_short}__graph_explore")
    json_path = save_path + ".delm.json"
    
    # Check if exists
    if os.path.exists(json_path):
        print(f"✓ Result exists: {os.path.basename(json_path)}")
        
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        stats = data.get('graph_stats', {})
        metadata = data.get('metadata', {})
        
        return {
            'model': model_short,
            'method': 'GraphExplore',
            'samples': stats.get('total_sequences_generated', 0),
            'valid_concepts': stats.get('concepts_discovered', 0),
            'invalid_concepts': 0,
            'invalid_ratio': "N/A",
            'threshold_reached': metadata.get('threshold_reached', False),
            'total_time': f"{metadata.get('elapsed_time', 0.0):.2f}",
            'tokens_per_sec': "N/A",
            'concepts_per_sec': "N/A",
            'total_tokens': stats.get('total_sequences_generated', 0) * MAX_LEN,
        }
    
    # Run new
    print(f"\n{'='*80}")
    print(f"{model_short} | GraphExplore")
    print(f"{'='*80}")
    
    import time
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
    )
    
    elapsed = time.time() - start
    total_tokens = stats['total_sequences_generated'] * MAX_LEN
    
    # Save
    import json
    save_data = {
        'metadata': {
            'model': model_short,
            'method': 'GraphExplore',
            'elapsed_time': elapsed,
            'threshold_reached': stats['concepts_discovered'] >= CONCEPT_THRESHOLD,
        },
        'graph_stats': stats,
        'concept_list': list(concept_map.keys()),
    }
    
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    return {
        'model': model_short,
        'method': 'GraphExplore',
        'samples': stats['total_sequences_generated'],
        'valid_concepts': stats['concepts_discovered'],
        'invalid_concepts': 0,
        'invalid_ratio': "N/A",
        'threshold_reached': stats['concepts_discovered'] >= CONCEPT_THRESHOLD,
        'total_time': f"{elapsed:.2f}",
        'tokens_per_sec': f"{total_tokens / elapsed:.2f}",
        'concepts_per_sec': f"{stats['concepts_discovered'] / elapsed:.4f}",
        'total_tokens': total_tokens,
    }


def run_flat_sampling(model, tokenizer, model_short: str, method_config: dict,
                     gpu_profiling: dict, save_dir: str):
    """Flat sampling methods."""
    
    sampling_method = method_config['sampling_method']
    method_name = method_config['name']
    
    save_path = os.path.join(save_dir, f"{model_short}__{sampling_method}")
    json_path = save_path + ".delm.json"
    
    # Check if exists
    if os.path.exists(json_path):
        print(f"✓ Result exists: {os.path.basename(json_path)}")
        
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return extract_metrics_from_fresh_results(
            data, model_short, method_name, None, CONCEPT_THRESHOLD, gpu_profiling
        )
    
    # Run new
    print(f"\n{'='*80}")
    print(f"{model_short} | {method_name}")
    print(f"{'='*80}")
    
    results = sample_concepts(
        model=model,
        tokenizer=tokenizer,
        prompt_fn=lambda: flat_concept_list(DOMAIN),
        concept_threshold=CONCEPT_THRESHOLD,
        max_samples=100000,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        sampling_method=sampling_method,
        save_path=save_path,
        model_name=model_short,
    )
    
    return extract_metrics_from_fresh_results(
        results, model_short, method_name, None, CONCEPT_THRESHOLD, gpu_profiling
    )


# ============================================================================
# Main
# ============================================================================

def main():
    save_dir = "results/exp4_hallucination"
    csv_path = f"{save_dir}/metrics.csv"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*80}")
    print("EXPERIMENT 4: HALLUCINATION METRICS (UNIFIED)")
    print(f"{'='*80}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Domain: {DOMAIN}")
    print(f"Target concepts: {CONCEPT_THRESHOLD} (for manual verification)")
    print(f"Methods: {[m['name'] for m in METHODS.values()]}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*80}\n")
    
    all_metrics = []
    
    for model_short, model_path in MODELS.items():
        
        try:
            # Load model (try bfloat16, fallback to float16)
            try:
                model, tokenizer, gpu_profiling = load_model_with_profiling(
                    model_path, model_short
                )
            except:
                print(f"⚠ BF16 failed, trying FP16...")
                model, tokenizer, gpu_profiling = load_model_with_profiling(
                    model_path, model_short,
                    quant_config={"torch_dtype": torch.float16}
                )
            
            for method_key, method_config in METHODS.items():
                
                try:
                    method_type = method_config['type']
                    
                    if method_type == 'graph_gen':
                        metrics = run_graph_generation(model, tokenizer, model_short, 
                                                      gpu_profiling, save_dir)
                    elif method_type == 'exploration':
                        metrics = run_graph_exploration(model, tokenizer, model_short,
                                                       method_config, gpu_profiling, save_dir)
                    elif method_type == 'flat':
                        metrics = run_flat_sampling(model, tokenizer, model_short,
                                                   method_config, gpu_profiling, save_dir)
                    
                    all_metrics.append(metrics)
                    
                    print(f"✓ Completed: {model_short}/{method_config['name']}")
                    
                except Exception as e:
                    print(f"✗ Failed: {model_short}/{method_config['name']}: {e}")
                    import traceback
                    traceback.print_exc()
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"✗ Failed to load {model_short}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print and save results
    print_metrics_table(
        all_metrics,
        f"EXPERIMENT 4: HALLUCINATION METRICS - RESULTS\n"
        f"Domain: {DOMAIN} | Target: {CONCEPT_THRESHOLD} concepts (manual fact-checking)",
        show_gpu=True
    )
    
    save_metrics_to_csv(all_metrics, csv_path)
    
    print(f"\n✓ Experiment 4 complete!")
    print(f"✓ Results: {save_dir}/")
    print(f"✓ CSV: {csv_path}")
    print(f"\n📋 Next steps:")
    print(f"  1. Review {csv_path}")
    print(f"  2. Manually fact-check concepts for hallucination assessment")
    print(f"  3. Compare invalid_ratio across models and methods")


if __name__ == "__main__":
    main()
