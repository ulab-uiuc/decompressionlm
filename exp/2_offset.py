#!/usr/bin/env python3
"""
Experiment 2: VdC Offset Comparison (New Codebase)

Test VdC sampling with different offsets to measure:
1. Concept overlap between 8 random offsets
2. Correlation between sampling efficiency and offset=0 baseline

Models: Llama-3.1-8B, Qwen2.5-7B
Target: 1024 concepts
Domain: US law & bar exam
"""

import os
import sys
import json
import time
import struct
import random
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import sample_concepts
from src.profiling import measure_model_memory


# ============================================================================
# Configuration
# ============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Models to test
MODELS = {
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
}

# Experiment parameters
CONCEPT_THRESHOLD = 1024
MAX_SEQ_COUNT = 10000
MAX_LEN = 32
BATCH_SIZE = 16
NUM_OFFSET_RUNS = 8  # Number of random offsets to test

# Prompt
PROMPT = """Generate United States bar exam legal concepts as keywords.
Please output ONE concept per line.
Each concept can be multiple words if needed.
Do not include explanations or extra text.
Please begin from any random concept.
Please use English.
"""

# Output directories (NO DATE in path for persistence)
OUTPUT_DIR = "results/exp2_vdc_offsets"
VDC_OFFSETS_BINARY = os.path.join(OUTPUT_DIR, "vdc_offsets.bin")
VDC_OFFSETS_META = os.path.join(OUTPUT_DIR, "vdc_offsets_meta.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# VdC Offset Generation (Binary Format)
# ============================================================================

def generate_vdc_offsets_binary(num_offsets, seed=42):
    """Generate random VdC offsets in [0, 1) and save in binary format."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate offsets as numpy float64
    # Include offset=0 as first, then random offsets
    offsets_np = np.zeros(num_offsets + 1, dtype=np.float64)
    offsets_np[0] = 0.0  # Baseline
    
    for i in range(1, num_offsets + 1):
        offsets_np[i] = np.random.uniform(0.0, 1.0)
    
    print(f"\nGenerated {len(offsets_np)} VdC offsets (seed={seed}):")
    print(f"  NumPy dtype: {offsets_np.dtype}")
    
    for i, offset in enumerate(offsets_np):
        offset_str = np.format_float_positional(offset, precision=17, unique=True, fractional=True, trim='k')
        label = "BASELINE" if i == 0 else f"Random {i}"
        print(f"  Offset {i}: {offset_str:24} ({label})")
    
    # Save binary data
    with open(VDC_OFFSETS_BINARY, 'wb') as f:
        f.write(b'VDC1')  # Magic number
        f.write(struct.pack('I', 1))  # Version
        f.write(struct.pack('Q', len(offsets_np)))  # Count
        
        dtype_str = str(offsets_np.dtype).encode('utf-8')
        f.write(struct.pack('I', len(dtype_str)))
        f.write(dtype_str)
        
        f.write(offsets_np.tobytes())
    
    print(f"\n✓ Saved binary offsets to: {VDC_OFFSETS_BINARY}")
    print(f"  File size: {os.path.getsize(VDC_OFFSETS_BINARY)} bytes")
    
    # Save metadata
    metadata = {
        "seed": seed,
        "num_offsets": len(offsets_np),
        "includes_baseline": True,
        "baseline_index": 0,
        "description": "VdC offsets for offset comparison experiment",
        "numpy_dtype": str(offsets_np.dtype),
        "binary_file": VDC_OFFSETS_BINARY,
        "created_at": datetime.now().isoformat(),
    }
    
    with open(VDC_OFFSETS_META, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved metadata to: {VDC_OFFSETS_META}")
    
    return offsets_np


def load_vdc_offsets_binary():
    """Load VdC offsets from binary file."""
    if not os.path.exists(VDC_OFFSETS_BINARY):
        return None
    
    try:
        with open(VDC_OFFSETS_BINARY, 'rb') as f:
            magic = f.read(4)
            if magic != b'VDC1':
                return None
            
            version = struct.unpack('I', f.read(4))[0]
            if version != 1:
                return None
            
            num_offsets = struct.unpack('Q', f.read(8))[0]
            
            dtype_len = struct.unpack('I', f.read(4))[0]
            dtype_str = f.read(dtype_len).decode('utf-8')
            
            expected_bytes = num_offsets * np.dtype(dtype_str).itemsize
            data_bytes = f.read(expected_bytes)
            
            if len(data_bytes) != expected_bytes:
                return None
            
            offsets_np = np.frombuffer(data_bytes, dtype=dtype_str)
            
            if len(offsets_np) != num_offsets:
                return None
        
        print(f"\n✓ Loaded {len(offsets_np)} VdC offsets from binary file:")
        for i, offset in enumerate(offsets_np):
            offset_str = np.format_float_positional(offset, precision=17, unique=True, fractional=True, trim='k')
            label = "BASELINE" if i == 0 else f"Random {i}"
            print(f"  Offset {i}: {offset_str:24} ({label})")
        
        return offsets_np
        
    except Exception as e:
        print(f"❌ Error loading VdC offsets: {e}")
        return None


# ============================================================================
# Experiment Runner
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
    
    return model, tokenizer, load_time


def run_vdc_offset_experiment(
    model,
    tokenizer,
    model_short: str,
    offset: float,
    offset_idx: int,
    save_dir: str,
):
    """Run single offset experiment."""
    
    # Format offset for filename (no dots, use 'p' for decimal point)
    offset_str = f"{offset:.17f}".replace('.', 'p')
    
    save_name = f"{model_short}__offset{offset_idx}__vdc{offset_str}"
    save_path = os.path.join(save_dir, save_name)
    
    # Check if already exists
    json_path = save_path + ".delm.json"
    if os.path.exists(json_path):
        print(f"✓ Result exists: {os.path.basename(json_path)}")
        print(f"  Loading existing result...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return {
            'offset': offset,
            'offset_idx': offset_idx,
            'samples': data['metadata']['samples_done'],
            'valid_concepts': data['concept_stats']['valid_concepts_count'],
            'invalid_concepts': data['concept_stats']['invalid_concepts_count'],
            'time': data['metadata']['elapsed_time'],
            'threshold_reached': data['metadata']['threshold_reached'],
            'save_path': json_path,
        }
    
    # Run new experiment
    print(f"\n{'='*80}")
    print(f"Running: {model_short} | Offset {offset_idx} | VdC={offset:.6f}")
    print(f"{'='*80}")
    
    results = sample_concepts(
        model=model,
        tokenizer=tokenizer,
        prefix=PROMPT,
        concept_threshold=CONCEPT_THRESHOLD,
        max_samples=MAX_SEQ_COUNT,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        sampling_method="vdc",
        sampling_params={"offset": float(offset)},
        save_path=save_path,
        model_name=model_short,
    )
    
    print(f"✓ Completed: {results['samples_done']} samples, {results['valid_concepts']} concepts")
    
    return {
        'offset': offset,
        'offset_idx': offset_idx,
        'samples': results['samples_done'],
        'valid_concepts': results['valid_concepts'],
        'invalid_concepts': results['invalid_concepts'],
        'time': results['elapsed_time'],
        'threshold_reached': results['threshold_reached'],
        'save_path': save_path + ".delm.json",
    }


# ============================================================================
# Analysis Functions
# ============================================================================

def load_concepts_from_json(json_path):
    """Load concepts from saved JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    concepts = set()
    for item in data['valid_concepts']:
        concepts.add(item['concept'])
    
    return concepts


def analyze_overlap(all_results, model_short):
    """Analyze concept overlap between different offsets."""
    
    print(f"\n{'='*80}")
    print(f"OVERLAP ANALYSIS: {model_short}")
    print(f"{'='*80}")
    
    # Load concepts for each offset
    offset_concepts = {}
    for result in all_results:
        offset_idx = result['offset_idx']
        concepts = load_concepts_from_json(result['save_path'])
        offset_concepts[offset_idx] = {
            'offset': result['offset'],
            'concepts': concepts,
            'samples': result['samples'],
        }
    
    # Print individual stats
    print(f"\n📊 INDIVIDUAL OFFSET STATS:")
    print()
    print("| Offset | VdC Value | Samples | Concepts | Status |")
    print("| ------ | --------- | ------: | -------: | ------ |")
    
    for idx in sorted(offset_concepts.keys()):
        data = offset_concepts[idx]
        label = "BASELINE" if idx == 0 else f"Random {idx}"
        status = "✓" if len(data['concepts']) >= CONCEPT_THRESHOLD else "✗"
        print(f"| {idx:6d} | {data['offset']:9.6f} | {data['samples']:7d} | {len(data['concepts']):8d} | {status:6} |")
    
    # Pairwise overlap
    print(f"\n🔄 PAIRWISE CONCEPT OVERLAP:")
    print()
    
    indices = sorted(offset_concepts.keys())
    
    # Calculate overlaps with baseline (offset 0)
    if 0 in offset_concepts:
        baseline_concepts = offset_concepts[0]['concepts']
        
        print("Overlap with BASELINE (Offset 0):")
        print()
        print("| Offset | Overlap | % of Offset | % of Baseline | Jaccard |")
        print("| ------ | ------: | ----------: | ------------: | ------: |")
        
        for idx in indices[1:]:  # Skip baseline itself
            concepts = offset_concepts[idx]['concepts']
            overlap = baseline_concepts.intersection(concepts)
            union = baseline_concepts.union(concepts)
            
            overlap_count = len(overlap)
            pct_offset = (overlap_count / len(concepts) * 100) if concepts else 0
            pct_baseline = (overlap_count / len(baseline_concepts) * 100) if baseline_concepts else 0
            jaccard = (overlap_count / len(union) * 100) if union else 0
            
            print(f"| {idx:6d} | {overlap_count:7d} | {pct_offset:11.1f}% | {pct_baseline:13.1f}% | {jaccard:7.1f}% |")
    
    # All pairwise comparisons
    print(f"\n\nAll Pairwise Overlaps (Jaccard similarity %):")
    print()
    
    # Header
    header_row = "| Offset |"
    for idx in indices:
        header_row += f" {idx:6d} |"
    print(header_row)
    
    sep_row = "| ------ |"
    for _ in indices:
        sep_row += " ------: |"
    print(sep_row)
    
    # Data rows
    for idx_a in indices:
        row = f"| {idx_a:6d} |"
        concepts_a = offset_concepts[idx_a]['concepts']
        
        for idx_b in indices:
            if idx_a == idx_b:
                row += "    — |"
            else:
                concepts_b = offset_concepts[idx_b]['concepts']
                overlap = concepts_a.intersection(concepts_b)
                union = concepts_a.union(concepts_b)
                jaccard = (len(overlap) / len(union) * 100) if union else 0
                row += f" {jaccard:6.1f} |"
        
        print(row)
    
    # Shared by all
    all_concept_sets = [data['concepts'] for data in offset_concepts.values()]
    shared_by_all = set.intersection(*all_concept_sets) if all_concept_sets else set()
    
    print(f"\n🌟 CONCEPTS SHARED BY ALL {len(offset_concepts)} OFFSETS:")
    print(f"  Count: {len(shared_by_all):,}")
    
    if shared_by_all:
        all_concepts_union = set.union(*all_concept_sets)
        shared_pct = (len(shared_by_all) / len(all_concepts_union) * 100) if all_concepts_union else 0
        print(f"  Percentage of total unique: {shared_pct:.1f}%")
        
        # Show examples
        shared_list = sorted(list(shared_by_all))[:10]
        if shared_list:
            print(f"  Examples: {', '.join(shared_list)}...")


def analyze_efficiency_ranking(all_results, model_short):
    """Analyze if sampling efficiency ranking matches offset=0 baseline."""
    
    print(f"\n{'='*80}")
    print(f"EFFICIENCY RANKING ANALYSIS: {model_short}")
    print(f"{'='*80}")
    
    # Sort by samples needed (ascending = more efficient)
    sorted_by_samples = sorted(all_results, key=lambda x: x['samples'])
    
    print(f"\n📊 RANKING BY SAMPLING EFFICIENCY:")
    print(f"   (Fewer samples = more efficient)")
    print()
    print("| Rank | Offset | VdC Value | Samples | Concepts | Efficiency Score |")
    print("| ---- | ------ | --------- | ------: | -------: | ---------------: |")
    
    baseline_samples = None
    baseline_offset = None
    
    for rank, result in enumerate(sorted_by_samples, 1):
        offset_idx = result['offset_idx']
        label = "BASELINE" if offset_idx == 0 else f"Random {offset_idx}"
        
        # Calculate efficiency score (concepts per sample)
        efficiency = result['valid_concepts'] / result['samples'] if result['samples'] > 0 else 0
        
        if offset_idx == 0:
            baseline_samples = result['samples']
            baseline_offset = result['offset']
        
        marker = " ★" if offset_idx == 0 else ""
        print(f"| {rank:4d} | {offset_idx:6d} | {result['offset']:9.6f} | {result['samples']:7d} | "
              f"{result['valid_concepts']:8d} | {efficiency:16.4f}{marker} |")
    
    # Calculate baseline rank
    baseline_rank = None
    for rank, result in enumerate(sorted_by_samples, 1):
        if result['offset_idx'] == 0:
            baseline_rank = rank
            break
    
    print(f"\n📈 BASELINE ANALYSIS:")
    print(f"  Baseline (offset=0) rank: {baseline_rank}/{len(sorted_by_samples)}")
    
    if baseline_rank == 1:
        print(f"  ✓ Baseline is MOST efficient")
    elif baseline_rank == len(sorted_by_samples):
        print(f"  ✗ Baseline is LEAST efficient")
    elif baseline_rank <= len(sorted_by_samples) // 2:
        print(f"  → Baseline is in TOP HALF (above average efficiency)")
    else:
        print(f"  → Baseline is in BOTTOM HALF (below average efficiency)")
    
    # Compare samples needed
    print(f"\n📊 SAMPLES COMPARISON TO BASELINE:")
    print()
    print("| Offset | Samples | Diff from Baseline | % Difference |")
    print("| ------ | ------: | -----------------: | -----------: |")
    
    for result in sorted_by_samples:
        if baseline_samples:
            diff = result['samples'] - baseline_samples
            pct_diff = (diff / baseline_samples * 100) if baseline_samples > 0 else 0
            marker = " ★" if result['offset_idx'] == 0 else ""
            
            print(f"| {result['offset_idx']:6d} | {result['samples']:7d} | {diff:18+d} | {pct_diff:12.1f}%{marker} |")


def print_metrics_table(all_metrics):
    """Print combined metrics table for all models."""
    
    print(f"\n{'='*120}")
    print("EXPERIMENT 4: VDC OFFSET COMPARISON - COMBINED RESULTS")
    print(f"{'='*120}\n")
    
    print(f"{'Model':<15} | {'Offset':>6} | {'VdC':>9} | {'Samples':>7} | {'Valid':>6} | "
          f"{'Invalid':>7} | {'Conv':>5} | {'Time':>7} | {'Efficiency':>10}")
    print("-" * 120)
    
    for m in all_metrics:
        converged = "YES" if m['threshold_reached'] else "NO"
        efficiency = m['valid_concepts'] / m['samples'] if m['samples'] > 0 else 0
        marker = " ★" if m['offset_idx'] == 0 else ""
        
        print(f"{m['model']:<15} | {m['offset_idx']:6d} | {m['offset']:9.6f} | "
              f"{m['samples']:7d} | {m['valid_concepts']:6d} | {m['invalid_concepts']:7d} | "
              f"{converged:5} | {m['time']:7.1f}s | {efficiency:10.4f}{marker}")
    
    print(f"\n{'='*120}\n")


def save_summary_csv(all_metrics, output_path):
    """Save summary to CSV."""
    import csv
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['model', 'offset_idx', 'offset', 'samples', 'valid_concepts', 
                  'invalid_concepts', 'threshold_reached', 'time', 'efficiency']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for m in all_metrics:
            efficiency = m['valid_concepts'] / m['samples'] if m['samples'] > 0 else 0
            writer.writerow({
                'model': m['model'],
                'offset_idx': m['offset_idx'],
                'offset': m['offset'],
                'samples': m['samples'],
                'valid_concepts': m['valid_concepts'],
                'invalid_concepts': m['invalid_concepts'],
                'threshold_reached': m['threshold_reached'],
                'time': m['time'],
                'efficiency': efficiency,
            })
    
    print(f"✓ Saved summary CSV to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*80)
    print("EXPERIMENT 2: VDC OFFSET COMPARISON")
    print("="*80)
    print(f"Seed: {SEED}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Concept threshold: {CONCEPT_THRESHOLD}")
    print(f"Number of offsets: {NUM_OFFSET_RUNS + 1} (including baseline)")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)
    
    # Generate or load VdC offsets
    vdc_offsets = load_vdc_offsets_binary()
    if vdc_offsets is None or len(vdc_offsets) != NUM_OFFSET_RUNS + 1:
        print(f"\nGenerating new VdC offsets (seed={SEED})...")
        vdc_offsets = generate_vdc_offsets_binary(NUM_OFFSET_RUNS, seed=SEED)
    else:
        print("\n✓ Using existing VdC offsets")
    
    all_metrics = []
    
    # Run experiments for each model
    for model_short, model_path in MODELS.items():
        print(f"\n{'='*80}")
        print(f"TESTING MODEL: {model_short}")
        print(f"{'='*80}")
        
        # Load model
        model, tokenizer, load_time = load_model_with_timing(model_path)
        
        # Run each offset
        model_results = []
        for offset_idx, offset in enumerate(vdc_offsets):
            result = run_vdc_offset_experiment(
                model, tokenizer, model_short,
                offset, offset_idx, OUTPUT_DIR
            )
            
            result['model'] = model_short
            model_results.append(result)
            all_metrics.append(result)
        
        # Analyze this model
        analyze_overlap(model_results, model_short)
        analyze_efficiency_ranking(model_results, model_short)
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Print combined table
    print_metrics_table(all_metrics)
    
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "exp2_summary.csv")
    save_summary_csv(all_metrics, csv_path)
    
    print("\n" + "="*80)
    print("EXPERIMENT 2 COMPLETE!")
    print("="*80)
    
    print(f"\n✓ All results saved to: {OUTPUT_DIR}/")
    print(f"✓ Summary CSV: {csv_path}")
    print(f"✓ VdC offsets: {VDC_OFFSETS_BINARY}")


if __name__ == "__main__":
    main()