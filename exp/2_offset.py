#!/usr/bin/env python3
"""
Experiment 2: VdC Offset Comparison (UNIFIED)

Test VdC sampling with different offsets to measure:
1. Concept overlap between 8 random offsets
2. Correlation between sampling efficiency and offset=0 baseline

Models: Llama-3.1-8B, Qwen2.5-7B
Target: 1024 concepts
Domain: US law & bar exam
Seed: 42
"""

import os
import sys
import json
import struct
import random
import torch
import numpy as np
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
)


# ============================================================================
# Configuration
# ============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODELS = {
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
}

DOMAIN = "US law and bar exam"

CONCEPT_THRESHOLD = 1024
MAX_LEN = 32
BATCH_SIZE = 16
NUM_OFFSET_RUNS = 8

OUTPUT_DIR = "results/exp2_vdc_offsets"
VDC_OFFSETS_BINARY = os.path.join(OUTPUT_DIR, "vdc_offsets.bin")
VDC_OFFSETS_META = os.path.join(OUTPUT_DIR, "vdc_offsets_meta.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# VdC Offset Binary Storage
# ============================================================================

def generate_vdc_offsets_binary(num_offsets, seed=42):
    """Generate random VdC offsets and save in binary format."""
    random.seed(seed)
    np.random.seed(seed)
    
    offsets_np = np.zeros(num_offsets + 1, dtype=np.float64)
    offsets_np[0] = 0.0  # Baseline
    
    for i in range(1, num_offsets + 1):
        offsets_np[i] = np.random.uniform(0.0, 1.0)
    
    print(f"\nGenerated {len(offsets_np)} VdC offsets (seed={seed}):")
    for i, offset in enumerate(offsets_np):
        label = "BASELINE" if i == 0 else f"Random {i}"
        print(f"  Offset {i}: {offset:.17f} ({label})")
    
    # Save binary
    with open(VDC_OFFSETS_BINARY, 'wb') as f:
        f.write(b'VDC1')
        f.write(struct.pack('I', 1))
        f.write(struct.pack('Q', len(offsets_np)))
        
        dtype_str = str(offsets_np.dtype).encode('utf-8')
        f.write(struct.pack('I', len(dtype_str)))
        f.write(dtype_str)
        f.write(offsets_np.tobytes())
    
    print(f"\n✓ Saved to {VDC_OFFSETS_BINARY} ({os.path.getsize(VDC_OFFSETS_BINARY)} bytes)")
    
    # Save metadata
    metadata = {
        "seed": seed,
        "num_offsets": len(offsets_np),
        "includes_baseline": True,
        "baseline_index": 0,
        "numpy_dtype": str(offsets_np.dtype),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(VDC_OFFSETS_META, 'w') as f:
        json.dump(metadata, f, indent=2)
    
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
        
        print(f"\n✓ Loaded {len(offsets_np)} VdC offsets from binary")
        return offsets_np
        
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return None


# ============================================================================
# Experiment Runner
# ============================================================================

def run_vdc_offset_experiment(
    model, tokenizer, model_short: str, offset: float, offset_idx: int,
    gpu_profiling: dict, save_dir: str
):
    """Run single offset experiment."""
    
    offset_str = f"{offset:.17f}".replace('.', 'p')
    save_name = f"{model_short}__offset{offset_idx}__vdc{offset_str}"
    save_path = os.path.join(save_dir, save_name)
    json_path = save_path + ".delm.json"
    
    # Check if exists
    if os.path.exists(json_path):
        print(f"✓ Result exists: {os.path.basename(json_path)}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return {
            'model': model_short,
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
    print(f"{model_short} | Offset {offset_idx} | VdC={offset:.6f}")
    print(f"{'='*80}")
    
    results = sample_concepts(
        model=model,
        tokenizer=tokenizer,
        prompt_fn=lambda: flat_concept_list(DOMAIN),
        concept_threshold=CONCEPT_THRESHOLD,
        max_samples=100000,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        sampling_method="vdc",
        sampling_params={"offset": float(offset)},
        save_path=save_path,
        model_name=model_short,
    )
    
    print(f"✓ Completed: {results['samples_done']} samples, {results['valid_concepts']} concepts")
    
    return {
        'model': model_short,
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
    """Load concepts from saved JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    concepts = set()
    for item in data['valid_concepts']:
        concepts.add(item['concept'])
    
    return concepts


def analyze_overlap(all_results, model_short):
    """Analyze concept overlap between offsets."""
    
    print(f"\n{'='*80}")
    print(f"OVERLAP ANALYSIS: {model_short}")
    print(f"{'='*80}")
    
    # Load concepts
    offset_concepts = {}
    for result in all_results:
        idx = result['offset_idx']
        concepts = load_concepts_from_json(result['save_path'])
        offset_concepts[idx] = {
            'offset': result['offset'],
            'concepts': concepts,
            'samples': result['samples'],
        }
    
    # Individual stats
    print(f"\n📊 INDIVIDUAL STATS:")
    print(f"| Offset | VdC Value | Samples | Concepts | Status |")
    print(f"|--------|-----------|---------|----------|--------|")
    
    for idx in sorted(offset_concepts.keys()):
        data = offset_concepts[idx]
        status = "✓" if len(data['concepts']) >= CONCEPT_THRESHOLD else "✗"
        print(f"| {idx:6d} | {data['offset']:9.6f} | {data['samples']:7d} | "
              f"{len(data['concepts']):8d} | {status:6} |")
    
    # Pairwise with baseline
    if 0 in offset_concepts:
        baseline = offset_concepts[0]['concepts']
        
        print(f"\n🔄 OVERLAP WITH BASELINE:")
        print(f"| Offset | Overlap | % of Offset | % of Base | Jaccard |")
        print(f"|--------|---------|-------------|-----------|---------|")
        
        for idx in sorted(offset_concepts.keys())[1:]:
            concepts = offset_concepts[idx]['concepts']
            overlap = baseline.intersection(concepts)
            union = baseline.union(concepts)
            
            pct_offset = len(overlap) / len(concepts) * 100 if concepts else 0
            pct_base = len(overlap) / len(baseline) * 100 if baseline else 0
            jaccard = len(overlap) / len(union) * 100 if union else 0
            
            print(f"| {idx:6d} | {len(overlap):7d} | {pct_offset:11.1f}% | "
                  f"{pct_base:9.1f}% | {jaccard:7.1f}% |")


def analyze_efficiency_ranking(all_results, model_short):
    """Analyze efficiency ranking."""
    
    print(f"\n{'='*80}")
    print(f"EFFICIENCY RANKING: {model_short}")
    print(f"{'='*80}")
    
    sorted_by_samples = sorted(all_results, key=lambda x: x['samples'])
    
    print(f"\n📊 RANKING (Fewer samples = more efficient):")
    print(f"| Rank | Offset | VdC Value | Samples | Concepts | Efficiency |")
    print(f"|------|--------|-----------|---------|----------|------------|")
    
    baseline_samples = None
    
    for rank, result in enumerate(sorted_by_samples, 1):
        idx = result['offset_idx']
        efficiency = result['valid_concepts'] / result['samples'] if result['samples'] > 0 else 0
        
        if idx == 0:
            baseline_samples = result['samples']
        
        marker = " ★" if idx == 0 else ""
        print(f"| {rank:4d} | {idx:6d} | {result['offset']:9.6f} | {result['samples']:7d} | "
              f"{result['valid_concepts']:8d} | {efficiency:10.4f}{marker} |")
    
    # Baseline analysis
    baseline_rank = next((r for r, res in enumerate(sorted_by_samples, 1) 
                         if res['offset_idx'] == 0), None)
    
    print(f"\n📈 BASELINE ANALYSIS:")
    print(f"  Baseline rank: {baseline_rank}/{len(sorted_by_samples)}")
    
    if baseline_rank == 1:
        print(f"  ✓ Baseline is MOST efficient")
    elif baseline_rank <= len(sorted_by_samples) // 2:
        print(f"  → Baseline in TOP HALF")
    else:
        print(f"  → Baseline in BOTTOM HALF")


def print_combined_table(all_metrics):
    """Print combined results table."""
    
    print(f"\n{'='*120}")
    print("EXPERIMENT 2: VDC OFFSET COMPARISON - COMBINED RESULTS")
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


# ============================================================================
# Main
# ============================================================================

def main():
    import time
    
    print(f"{'='*80}")
    print("EXPERIMENT 2: VDC OFFSET COMPARISON (UNIFIED)")
    print(f"{'='*80}")
    print(f"Seed: {SEED}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Concept threshold: {CONCEPT_THRESHOLD}")
    print(f"Number of offsets: {NUM_OFFSET_RUNS + 1} (including baseline)")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*80}")
    
    # Load or generate offsets
    vdc_offsets = load_vdc_offsets_binary()
    if vdc_offsets is None or len(vdc_offsets) != NUM_OFFSET_RUNS + 1:
        print(f"\nGenerating new VdC offsets...")
        vdc_offsets = generate_vdc_offsets_binary(NUM_OFFSET_RUNS, seed=SEED)
    
    all_metrics = []
    
    # Run experiments
    for model_short, model_path in MODELS.items():
        print(f"\n{'='*80}")
        print(f"TESTING MODEL: {model_short}")
        print(f"{'='*80}")
        
        # Load model
        model, tokenizer, gpu_profiling = load_model_with_profiling(model_path, model_short)
        
        # Run each offset
        model_results = []
        for offset_idx, offset in enumerate(vdc_offsets):
            result = run_vdc_offset_experiment(
                model, tokenizer, model_short, offset, offset_idx,
                gpu_profiling, OUTPUT_DIR
            )
            
            model_results.append(result)
            all_metrics.append(result)
        
        # Analyze
        analyze_overlap(model_results, model_short)
        analyze_efficiency_ranking(model_results, model_short)
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Print and save
    print_combined_table(all_metrics)
    
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    save_metrics_to_csv(all_metrics, csv_path)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT 2 COMPLETE!")
    print(f"{'='*80}")
    print(f"✓ Results: {OUTPUT_DIR}/")
    print(f"✓ CSV: {csv_path}")
    print(f"✓ VdC offsets: {VDC_OFFSETS_BINARY}")


if __name__ == "__main__":
    main()
