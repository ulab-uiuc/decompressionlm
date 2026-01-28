#!/usr/bin/env python3
"""
Experiment 4: Hallucination Metrics (MMLU Pro Law)

Run same models from MMLU script on US law domain with different sampling methods.
Same table structure as Exp 3 - I'll manually fact-check results.

Models: Open source instruct ≤16B (from your MMLU script)
Domain: US law & Bar Exam
Methods: Same as Exp 2
"""

import os
import sys
import time
import csv
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import sample_concepts, explore_bfs
from src.profiling import measure_model_memory


# Configuration - Models from your MMLU script (adjust as needed)
MODELS = {
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Phi-3-14B": "microsoft/Phi-3-medium-128k-instruct",
    # Add more models from your MMLU script here
}

DOMAIN = "US law and bar exam"

# Same methods as Exp 2
METHODS = {
    "beam_graph": {
        "name": "Beam+Graph",
        "type": "graph_gen",
        "sampling_method": "beam_low",
    },
    "bfs": {
        "name": "BFS",
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

# Shared params
CONCEPT_THRESHOLD = 100
MAX_SEQ_COUNT = 10000
MAX_LEN = 32
BATCH_SIZE = 16


def load_model_with_timing(model_name: str):
    """Load model and return (model, tokenizer, load_time)."""
    print(f"\n{'='*80}")
    print(f"Loading {model_name}...")
    print(f"{'='*80}")
    
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Auto-detect best dtype and device map
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    except:
        # Fallback to float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    
    load_time = time.time() - start
    model_mem = measure_model_memory()
    
    print(f"✓ Loaded in {load_time:.2f}s")
    print(f"✓ GPU memory: {model_mem.get('model_memory_allocated_gb', 0):.2f} GB")
    
    return model, tokenizer, load_time, model_mem


def run_graph_generation(model, tokenizer, model_name: str, save_dir: str):
    """Method: Beam search + generate graph in YAML format."""
    
    prefix = """Generate United States bar exam legal concepts as keywords.
Please output ONE concept per line.
Each concept can be multiple words if needed.
Do not include explanations or extra text.
Please begin from any random concept.
Please use English.
"""
    
    save_path = os.path.join(save_dir, f"{model_name}__beam_graph")
    
    print(f"\n{'='*80}")
    print(f"{model_name} | Beam+Graph")
    print(f"{'='*80}")
    
    start = time.time()
    
    messages = [{"role": "user", "content": prefix}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
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
    with open(f"{save_path}.yaml", 'w') as f:
        f.write(text)
    
    print(f"✓ Generated {len(concepts)} concepts in {elapsed:.2f}s")
    
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
        'total_tokens': len(generated_ids),
    }


def run_bfs_exploration(model, tokenizer, model_name: str, method_config: dict, save_dir: str):
    """Method: BFS hierarchical exploration."""
    
    save_path = os.path.join(save_dir, f"{model_name}__bfs")
    
    print(f"\n{'='*80}")
    print(f"{model_name} | BFS")
    print(f"{'='*80}")
    
    start = time.time()
    
    root, concept_map, stats = explore_bfs(
        model=model,
        tokenizer=tokenizer,
        domain=DOMAIN,
        max_concepts=CONCEPT_THRESHOLD,
        max_depth=method_config['max_depth'],
        sequences_per_node=method_config['sequences_per_node'],
        max_len=MAX_LEN,
    )
    
    elapsed = time.time() - start
    
    total_tokens = stats['total_sequences_generated'] * MAX_LEN
    
    return {
        'model': model_name,
        'method': 'BFS',
        'samples': stats['total_sequences_generated'],
        'valid_concepts': stats['concepts_discovered'],
        'invalid_concepts': 0,
        'invalid_ratio': "N/A",
        'threshold_reached': stats['concepts_discovered'] >= CONCEPT_THRESHOLD,
        'tokens_per_sec': f"{total_tokens / elapsed:.2f}",
        'concepts_per_sec': f"{stats['concepts_discovered'] / elapsed:.4f}",
        'total_time': f"{elapsed:.2f}",
        'total_tokens': total_tokens,
    }


def run_flat_sampling(model, tokenizer, model_name: str, method_config: dict, save_dir: str):
    """Methods: Flat sampling with different methods."""
    
    sampling_method = method_config['sampling_method']
    method_name = method_config['name']
    
    prefix = """Generate United States bar exam legal concepts as keywords.
Please output ONE concept per line.
Each concept can be multiple words if needed.
Do not include explanations or extra text.
Please begin from any random concept.
Please use English.
"""
    
    save_path = os.path.join(save_dir, f"{model_name}__{sampling_method}")
    
    print(f"\n{'='*80}")
    print(f"{model_name} | {method_name}")
    print(f"{'='*80}")
    
    results = sample_concepts(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        concept_threshold=CONCEPT_THRESHOLD,
        max_samples=MAX_SEQ_COUNT,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        sampling_method=sampling_method,
        save_path=save_path,
        model_name=model_name,
    )
    
    prof = results.get('profiling', {})
    elapsed = results['elapsed_time']
    samples_done = results['samples_done']
    
    tokens_total = sum(len(seq['tokens']) for seq in results['sequences'])
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
        'tokens_per_sec': f"{tokens_total / elapsed:.2f}",
        'concepts_per_sec': f"{results['valid_concepts'] / elapsed:.4f}",
        'total_time': f"{elapsed:.2f}",
        'total_tokens': tokens_total,
        'gpu_allocated_max': f"{prof.get('gpu_sampling_end_allocated_max', 0):.2f}",
    }


def print_metrics_table(all_metrics: list):
    """Print formatted comparison table."""
    
    print(f"\n{'='*120}")
    print("EXPERIMENT 3: HALLUCINATION METRICS - RESULTS")
    print(f"Domain: {DOMAIN}")
    print(f"NOTE: Manually fact-check invalid_ratio to assess hallucination")
    print(f"{'='*120}\n")
    
    # Header
    print(f"{'Model':<20} | {'Method':<15} | {'Samples':>7} | {'Valid':>6} | {'Invalid':>7} | "
          f"{'Ratio':>6} | {'Conv':>5} | {'Time':>7} | {'Tok/s':>8} | {'Con/s':>8}")
    print("-" * 120)
    
    # Rows
    for m in all_metrics:
        converged = "YES" if m['threshold_reached'] else "NO"
        print(f"{m['model']:<20} | {m['method']:<15} | {m['samples']:>7} | "
              f"{m['valid_concepts']:>6} | {m['invalid_concepts']:>7} | "
              f"{m['invalid_ratio']:>6} | {converged:>5} | "
              f"{m['total_time']:>7}s | {m['tokens_per_sec']:>8} | {m['concepts_per_sec']:>8}")
    
    print(f"\n{'='*120}")
    print("NOTE: High invalid_ratio suggests potential hallucination")
    print("      Manually verify concept validity for final assessment")
    print(f"{'='*120}\n")


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
    save_dir = f"results/exp4_hallucination_{timestamp}"
    csv_path = f"{save_dir}/metrics.csv"
    
    print(f"{'='*80}")
    print("EXPERIMENT 4: HALLUCINATION METRICS")
    print(f"{'='*80}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Domain: {DOMAIN}")
    print(f"Methods: {[m['name'] for m in METHODS.values()]}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*80}\n")
    
    all_metrics = []
    
    for model_short, model_path in MODELS.items():
        
        try:
            # Load model
            model, tokenizer, load_time, model_mem = load_model_with_timing(model_path)
            
            for method_key, method_config in METHODS.items():
                
                try:
                    method_type = method_config['type']
                    
                    if method_type == 'graph_gen':
                        metrics = run_graph_generation(model, tokenizer, model_short, save_dir)
                    elif method_type == 'exploration':
                        metrics = run_bfs_exploration(model, tokenizer, model_short, method_config, save_dir)
                    elif method_type == 'flat':
                        metrics = run_flat_sampling(model, tokenizer, model_short, method_config, save_dir)
                    
                    all_metrics.append(metrics)
                    
                    print(f"✓ Completed: {model_short}/{method_config['name']}")
                    
                except Exception as e:
                    print(f"✗ Failed: {model_short}/{method_config['name']}: {e}")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"✗ Failed to load {model_short}: {e}")
            continue
    
    # Print final table
    print_metrics_table(all_metrics)
    
    # Save CSV
    save_to_csv(all_metrics, csv_path)
    
    print(f"\n✓ Experiment 4 complete!")
    print(f"✓ Results saved to {save_dir}/")
    print(f"\n📋 Next steps:")
    print(f"  1. Review {csv_path}")
    print(f"  2. Manually fact-check concepts for hallucination assessment")
    print(f"  3. Compare invalid_ratio across models and methods")


if __name__ == "__main__":
    main()