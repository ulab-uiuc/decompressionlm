"""
Read and analyze DecompressionLM bin entropy results from parquet files.

Expected schema:
- code: float64
- sequence: list<uint32>   (token IDs)
- sequence_length: int32

Metadata includes:
- bin_prefix_lens: List of tracked bin lengths
- For each bin length N:
  - lenN_converged: Whether it converged
  - lenN_convergence_sample: Sample number where it converged
  - lenN_final_entropy_bits: Final entropy in bits
  - lenN_n_unique_bins: Number of unique bins observed
"""

import sys
import pyarrow.parquet as pq
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from transformers import AutoTokenizer


def load_results(path: str):
    """Load bin entropy results from parquet file."""
    table = pq.read_table(path)

    # Extract metadata (bytes -> str)
    metadata = table.schema.metadata or {}
    metadata = {k.decode(): v.decode() for k, v in metadata.items()}

    data = table.to_pydict()

    # Strict: sequences must be list-of-ints (Arrow list<uint32>)
    sequences = data.get("sequence", None)
    if sequences is None:
        raise ValueError("Missing required column: 'sequence'")

    if len(sequences) > 0 and not isinstance(sequences[0], list):
        raise TypeError(
            f"Expected 'sequence' to be a list column (list<uint32>). "
            f"Got element type: {type(sequences[0])}. "
            f"Old format is not supported; regenerate results."
        )

    # Strict: token IDs must be non-negative
    for i, seq in enumerate(sequences[:10]):  # quick sanity check on first few
        if any(int(t) < 0 for t in seq):
            raise ValueError(f"Negative token id found in sample {i}: {seq[:20]}")

    return {
        "codes": data["code"],
        "sequences": sequences,  # list[list[int]]
        "sequence_lengths": data["sequence_length"],
        "metadata": metadata,
    }


def extract_bin_prefix(tokens: List[int], bin_len: int, eos_token_id: int) -> tuple:
    """
    Extract bin prefix of specified length, padding with EOS if needed.
    (Same logic as in bin_entropy.py)
    """
    if len(tokens) >= bin_len:
        return tuple(tokens[:bin_len])
    else:
        # Pad with EOS tokens
        return tuple(tokens + [eos_token_id] * (bin_len - len(tokens)))


def compute_bin_statistics(
    sequences: List[List[int]], 
    bin_len: int, 
    eos_token_id: int
) -> Dict:
    """Compute statistics for a specific bin length."""
    bin_counts = defaultdict(int)
    bin_to_sequences = defaultdict(list)  # Track which sequences belong to each bin
    
    for seq_idx, seq in enumerate(sequences):
        bin_tuple = extract_bin_prefix(seq, bin_len, eos_token_id)
        bin_counts[bin_tuple] += 1
        bin_to_sequences[bin_tuple].append(seq_idx)
    
    # Compute empirical entropy
    total = len(sequences)
    entropy = 0.0
    for count in bin_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    
    # Get top bins by frequency
    sorted_bins = sorted(bin_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'n_unique': len(bin_counts),
        'entropy_bits': entropy,
        'total_samples': total,
        'bin_counts': dict(bin_counts),
        'bin_to_sequences': dict(bin_to_sequences),
        'top_bins': sorted_bins[:10],  # Top 10 most frequent
    }


def analyze_results(
    results, 
    decode_sequences: bool = True, 
    show_sample_bins: int = 5,
    examples_per_bin: int = 3,
    max_chars_per_example: int = 150
):
    """Print comprehensive analysis of bin entropy results."""
    meta = results["metadata"]

    print("=" * 80)
    print("BIN ENTROPY RESULTS SUMMARY")
    print("=" * 80)
    print(f"Model: {meta.get('model_name', 'N/A')}")
    print(f"Prefix: {meta.get('prefix', 'N/A')}")
    print(f"Total samples: {meta.get('samples', 'N/A')}")
    print(f"All converged: {meta.get('all_converged', 'N/A')}")
    print(f"Variance threshold: {meta.get('variance_threshold', 'N/A')}")
    print(f"Elapsed time: {meta.get('elapsed_time', 'N/A')} seconds")

    # Parse bin_prefix_lens from metadata
    bin_prefix_lens_str = meta.get('bin_prefix_lens', '[]')
    bin_prefix_lens = eval(bin_prefix_lens_str)  # Safe since we control the format
    
    print(f"\nTracked bin lengths: {bin_prefix_lens}")

    # Print convergence info for each bin length
    print("\n" + "=" * 80)
    print("CONVERGENCE STATUS PER BIN LENGTH")
    print("=" * 80)
    
    for bin_len in bin_prefix_lens:
        converged = meta.get(f'len{bin_len}_converged', 'N/A')
        conv_sample = meta.get(f'len{bin_len}_convergence_sample', 'N/A')
        final_entropy = meta.get(f'len{bin_len}_final_entropy_bits', 'N/A')
        n_unique = meta.get(f'len{bin_len}_n_unique_bins', 'N/A')
        
        status = "✓" if converged == "True" else "✗"
        print(f"Bin length {bin_len:2d}: {status} converged at sample {conv_sample}")
        print(f"  Final entropy: {final_entropy} bits")
        print(f"  Unique bins: {n_unique}")
        print()

    # Sequence statistics
    print("=" * 80)
    print("SEQUENCE STATISTICS")
    print("=" * 80)

    lengths = results["sequence_lengths"]
    print(f"Avg length: {np.mean(lengths):.1f} tokens")
    print(f"Min length: {min(lengths)} tokens")
    print(f"Max length: {max(lengths)} tokens")
    print(f"Std length: {np.std(lengths):.1f} tokens")

    # Load tokenizer if needed
    tokenizer = None
    eos_token_id = None
    if decode_sequences:
        model_name = meta.get("model_name", None)
        if not model_name:
            print("\nWarning: decode_sequences=True but parquet metadata missing 'model_name'")
            print("Skipping decoding...")
        else:
            print(f"\nLoading tokenizer for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            eos_token_id = tokenizer.eos_token_id

    # Analyze bins for each length
    sequences = results["sequences"]
    
    if eos_token_id is None:
        print("\nWarning: Cannot compute bin statistics without EOS token ID")
        print("Skipping bin analysis...")
        return

    for bin_len in bin_prefix_lens:
        print("\n" + "=" * 80)
        print(f"BIN ANALYSIS: LENGTH {bin_len}")
        print("=" * 80)
        
        bin_stats = compute_bin_statistics(sequences, bin_len, eos_token_id)
        
        print(f"Unique bins: {bin_stats['n_unique']}")
        print(f"Empirical entropy: {bin_stats['entropy_bits']:.4f} bits")
        print(f"Total samples: {bin_stats['total_samples']}")
        
        # Show top bins with example sequences
        print(f"\nTop {min(show_sample_bins, len(bin_stats['top_bins']))} most frequent bins:")
        print("-" * 80)
        
        for i, (bin_tuple, count) in enumerate(bin_stats['top_bins'][:show_sample_bins]):
            prob = count / bin_stats['total_samples']
            print(f"\nBin #{i+1} (count={count}, prob={prob:.4f}):")
            
            # Decode the bin itself
            if tokenizer is not None:
                bin_text = tokenizer.decode(list(bin_tuple), skip_special_tokens=False)
                print(f"  Bin: {repr(bin_text)}")
            
            # Show example sequences from this bin
            seq_indices = bin_stats['bin_to_sequences'][bin_tuple]
            n_examples = min(examples_per_bin, len(seq_indices))
            print(f"\n  Example sequences from this bin ({n_examples}/{len(seq_indices)} shown):")
            
            for j, seq_idx in enumerate(seq_indices[:n_examples]):
                full_seq = sequences[seq_idx]
                
                if tokenizer is not None:
                    full_text = tokenizer.decode(full_seq, skip_special_tokens=True)
                    if len(full_text) > max_chars_per_example:
                        full_text = full_text[:max_chars_per_example] + "..."
                    print(f"    {j+1}. {repr(full_text)}")
                else:
                    print(f"    {j+1}. [tokens: {full_seq[:20]}{'...' if len(full_seq) > 20 else ''}]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.read_results <path_to_parquet>")
        print("Example: python -m src.read_results results/factual_question.delm.parquet")
        sys.exit(1)
    
    path = sys.argv[1]
    print(f"Loading results from {path}...")
    results = load_results(path)
    analyze_results(results, decode_sequences=True, show_sample_bins=5, examples_per_bin=3)