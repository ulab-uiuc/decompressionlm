#!/usr/bin/env python3
"""
Utility to analyze saved .delm.parquet results.

Load and inspect effective support sets from mass-based sampling.
"""

import pyarrow.parquet as pq
import sys
from pathlib import Path
from transformers import AutoTokenizer


def load_results(filepath: str):
    """Load results from .delm.parquet file."""
    table = pq.read_table(filepath)
    metadata = table.schema.metadata
    
    # Decode metadata
    meta = {k.decode(): v.decode() for k, v in metadata.items()}
    
    # Convert to dict
    data = {
        'codes': table['code'].to_pylist(),
        'sequences': table['sequence'].to_pylist(),
        'lengths': table['sequence_length'].to_pylist(),
        'terminated': table['terminated'].to_pylist(),
        'in_effective_set': table['in_effective_set'].to_pylist(),
    }
    
    return data, meta


def analyze_results(filepath: str, show_examples: int = 5):
    """
    Analyze and display results from a saved file.
    
    Args:
        filepath: Path to .delm.parquet file
        show_examples: Number of example sequences to display
    """
    print(f"Loading results from {filepath}...")
    data, meta = load_results(filepath)
    
    # Display metadata
    print(f"\n{'='*80}")
    print("METADATA")
    print(f"{'='*80}")
    
    important_keys = [
        'model_name',
        'actual_prompt',
        'prefix_len',
        'prob_threshold',
        'similarity_threshold',
        'total_sequences',
        'effective_set_size',
        'unique_prefixes_discovered',
        'final_prefix_mass',
        'effective_set_min_tokens',
        'effective_set_max_tokens',
        'effective_set_avg_tokens',
        'effective_set_total_tokens',
        'elapsed_time',
    ]
    
    for key in important_keys:
        if key in meta:
            print(f"{key:30s}: {meta[key]}")
    
    # Effective set analysis
    print(f"\n{'='*80}")
    print("EFFECTIVE SUPPORT SET ANALYSIS")
    print(f"{'='*80}")
    
    effective_indices = [i for i, flag in enumerate(data['in_effective_set']) if flag]
    effective_sequences = [data['sequences'][i] for i in effective_indices]
    
    print(f"Total sequences sampled: {len(data['sequences'])}")
    print(f"Effective set size: {len(effective_sequences)}")
    print(f"Reduction: {100 * (1 - len(effective_sequences)/len(data['sequences'])):.1f}%")
    
    # Load tokenizer if model name is available
    if 'model_name' in meta and show_examples > 0:
        print(f"\n{'='*80}")
        print(f"EXAMPLE SEQUENCES (first {show_examples} from effective set)")
        print(f"{'='*80}")
        
        try:
            tokenizer_name = meta.get('tokenizer_name', meta['model_name'])
            print(f"Loading tokenizer: {tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            for i, idx in enumerate(effective_indices[:show_examples]):
                seq = data['sequences'][idx]
                text = tokenizer.decode(seq, skip_special_tokens=True)
                print(f"\n[{i+1}] Sequence {idx} ({len(seq)} tokens):")
                print(f"  {text[:200]}...")
                
        except Exception as e:
            print(f"Could not load tokenizer: {e}")
            print("Showing raw token IDs instead:")
            
            for i, idx in enumerate(effective_indices[:show_examples]):
                seq = data['sequences'][idx]
                print(f"\n[{i+1}] Sequence {idx} ({len(seq)} tokens):")
                print(f"  {seq[:20]}...")
    
    # Sequence length distribution
    print(f"\n{'='*80}")
    print("LENGTH DISTRIBUTION (effective set)")
    print(f"{'='*80}")
    
    eff_lengths = [len(data['sequences'][i]) for i in effective_indices]
    
    if eff_lengths:
        import numpy as np
        print(f"Min: {min(eff_lengths)}")
        print(f"25%: {np.percentile(eff_lengths, 25):.1f}")
        print(f"50%: {np.percentile(eff_lengths, 50):.1f}")
        print(f"75%: {np.percentile(eff_lengths, 75):.1f}")
        print(f"Max: {max(eff_lengths)}")
        print(f"Mean: {np.mean(eff_lengths):.1f}")
        print(f"Std: {np.std(eff_lengths):.1f}")


def compare_results(filepath1: str, filepath2: str):
    """Compare two result files."""
    print(f"Loading {filepath1}...")
    data1, meta1 = load_results(filepath1)
    
    print(f"Loading {filepath2}...")
    data2, meta2 = load_results(filepath2)
    
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    
    eff1 = sum(data1['in_effective_set'])
    eff2 = sum(data2['in_effective_set'])
    
    print(f"\nFile 1: {filepath1}")
    print(f"  Effective set: {eff1}")
    print(f"  Total tokens: {meta1.get('effective_set_total_tokens', 'N/A')}")
    print(f"  Prompt: {meta1.get('actual_prompt', 'N/A')[:60]}...")
    
    print(f"\nFile 2: {filepath2}")
    print(f"  Effective set: {eff2}")
    print(f"  Total tokens: {meta2.get('effective_set_total_tokens', 'N/A')}")
    print(f"  Prompt: {meta2.get('actual_prompt', 'N/A')[:60]}...")
    
    print(f"\nDifference:")
    print(f"  Effective set: {eff2 - eff1:+d} ({100*(eff2/eff1 - 1):+.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Analyze single file:")
        print("    python analyze_results.py results/file.delm.parquet")
        print("  Compare two files:")
        print("    python analyze_results.py results/file1.delm.parquet results/file2.delm.parquet")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        analyze_results(sys.argv[1])
    elif len(sys.argv) == 3:
        compare_results(sys.argv[1], sys.argv[2])
    else:
        print("Too many arguments")
        sys.exit(1)
