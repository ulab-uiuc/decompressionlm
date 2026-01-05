"""
Read and analyze DecompressionLM results from parquet files.

Expected schema (strict, no legacy support):
- code: float64
- log_prob: float64
- sequence_length: int32
- sequence: list<uint32>   (token IDs)
Parquet file metadata should include: model_name, prefix, entropy_bits, etc.
"""

import sys
import pyarrow.parquet as pq
import numpy as np
from transformers import AutoTokenizer


def load_results(path: str):
    """Load results from parquet file (strict schema)."""
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
            f"Old string-based format is not supported; regenerate results."
        )

    # Strict: token IDs must be non-negative
    for i, seq in enumerate(sequences[:10]):  # quick sanity check on first few
        if any(int(t) < 0 for t in seq):
            raise ValueError(f"Negative token id found in sample {i}: {seq[:20]}")

    return {
        "codes": data["code"],
        "log_probs": data["log_prob"],
        "sequence_lengths": data["sequence_length"],
        "sequences": sequences,  # list[list[int]]
        "metadata": metadata,
    }


def analyze_results(results, decode_sequences: bool = True, show_k: int = 5, max_chars: int = 500):
    """Print analysis of results. Prints both token IDs and decoded text."""
    meta = results["metadata"]

    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {meta.get('model_name', 'N/A')}")
    print(f"Prefix: {meta.get('prefix', 'N/A')}")
    print(f"Samples: {meta.get('n_samples', 'N/A')}")
    print(f"Entropy: {meta.get('entropy', 'N/A')} nats")
    print(f"Entropy: {meta.get('entropy_bits', 'N/A')} bits")
    print(f"EOS rate: {meta.get('eos_rate', 'N/A')}")
    print(f"Converged: {meta.get('converged', 'N/A')}")
    print(f"Offset: {meta.get('offset', '0.0')}")

    print("\n" + "=" * 60)
    print("SEQUENCE STATISTICS")
    print("=" * 60)

    lengths = results["sequence_lengths"]
    log_probs = results["log_probs"]

    print(f"Avg length: {np.mean(lengths):.1f} tokens")
    print(f"Min length: {min(lengths)} tokens")
    print(f"Max length: {max(lengths)} tokens")
    print(f"Std length: {np.std(lengths):.1f} tokens")

    print(f"\nAvg log prob: {np.mean(log_probs):.2f}")
    print(f"Min log prob: {min(log_probs):.2f}")
    print(f"Max log prob: {max(log_probs):.2f}")

    # Per-token entropy (approx): H / E[length]
    try:
        entropy_bits = float(meta.get("entropy_bits", 0.0))
        avg_len = float(np.mean(lengths))
        if avg_len > 0:
            print(f"\nPer-token entropy (H/E[L]): {entropy_bits / avg_len:.2f} bits/token")
    except Exception:
        pass

    # Tokenizer for decoding
    tokenizer = None
    if decode_sequences:
        model_name = meta.get("model_name", None)
        if not model_name:
            raise ValueError("decode_sequences=True but parquet metadata missing 'model_name'")
        print("\nLoading tokenizer for {0}...".format(model_name))
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("\n" + "=" * 60)
    print(f"SAMPLE SEQUENCES (first {min(show_k, len(results['sequences']))})")
    print("=" * 60)

    sequences = results["sequences"]

    for i in range(min(show_k, len(sequences))):
        token_ids = [int(t) for t in sequences[i]]  # ensure plain ints
        if any(t < 0 for t in token_ids):
            raise ValueError(f"Negative token id found in sample {i}: {token_ids[:20]}")

        print(f"\nSample {i+1} (len={lengths[i]}, logp={log_probs[i]:.2f}):")
        print(f"Token IDs: {token_ids[:80]}{' ...' if len(token_ids) > 80 else ''}")

        if tokenizer is not None:
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            print("Decoded:")
            print(text)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/gas_question.parquet"
    print(f"Loading results from {path}...")
    results = load_results(path)
    analyze_results(results, decode_sequences=True)
