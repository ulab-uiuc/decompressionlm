"""
DecompressionLM: Estimate entropy and mutual information for language models.

Main API for computing H(X|prefix) using QMC sampling with arithmetic coding.
"""

import torch
torch.set_float32_matmul_precision('high')
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Optional, List, Dict
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.entropy import estimate_conditional_entropy


def estimate_entropy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str = "",
    max_samples: int = 1000,
    max_len: int = 100,
    use_chat_template: bool = True,
    save_path: Optional[str] = None,
    model_name: Optional[str] = None,
    device: str = "cuda",
    use_cache: bool = True,
    variance_threshold: float = 1e-4,
    offset: float = 0.0
) -> Dict:
    """
    Estimate H(X | prefix) using QMC sampling with early stopping.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prefix: Text prefix to condition on (empty string "" = BOS-only unconditional)
        max_samples: Maximum number of samples (hard stop)
        max_len: Maximum generation length per sample
        use_chat_template: Whether to format with chat template (ignored if prefix="")
        save_path: Optional path to save results as parquet
        model_name: Model name/identifier to save in metadata
        device: Device to run on
        use_cache: Whether to use KV cache (highly recommended for speed)
        variance_threshold: Stop when variance of entropy in second half < threshold
        offset: Cranley-Patterson rotation offset in [0, 1)
        
    Returns:
        Dict with entropy, log_probs, sequences, etc.
    """
    entropy, info = estimate_conditional_entropy(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        max_samples=max_samples,
        max_len=max_len,
        use_chat_template=use_chat_template if prefix != "" else False,
        device=device,
        use_cache=use_cache,
        variance_threshold=variance_threshold,
        offset=offset
    )
    
    # Convert to bits
    entropy_bits = entropy / torch.log(torch.tensor(2.0)).item()
    
    # Package results
    results = {
        'entropy': entropy,
        'entropy_bits': entropy_bits,
        'eos_rate': info['eos_rate'],
        'n_samples': info['n_samples'],
        'prefix': info['prefix'],
        'log_probs': info['log_probs'],
        'sequences': info['sequences'],
        'codes': info['codes'],
        'entropy_history': info.get('entropy_history', []),
        'converged': info.get('converged', False),
        'offset': offset,
    }
    
    # Save if requested
    if save_path:
        save_results(results, save_path, model_name=model_name)
        print(f"Saved results to {save_path}")
    
    return results


def save_results(results: Dict, path: str, model_name: Optional[str] = None):
    """Save results to parquet file using compact uint32 token IDs."""
    
    # Convert sequences to Arrow list<uint32>
    sequence_array = pa.array(
        results['sequences'],
        type=pa.list_(pa.uint32())
    )

    data = {
        'code': pa.array(results['codes'], type=pa.float64()),
        'log_prob': pa.array(results['log_probs'], type=pa.float64()),
        'sequence_length': pa.array(
            [len(seq) for seq in results['sequences']],
            type=pa.int32()
        ),
        'sequence': sequence_array,
    }
    
    # Metadata (unchanged)
    metadata = {
        'entropy': str(results['entropy']),
        'entropy_bits': str(results['entropy_bits']),
        'eos_rate': str(results['eos_rate']),
        'n_samples': str(results['n_samples']),
        'prefix': results['prefix'],
        'converged': str(results['converged']),
        'offset': str(results['offset']),
    }
    
    if model_name:
        metadata['model_name'] = model_name
    
    table = pa.Table.from_pydict(data)
    table = table.replace_schema_metadata(metadata)
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


if __name__ == "__main__":
    # Example usage
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading model {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )
    
    # Optional: compile model for speed (can be unstable on some setups)
    # Uncomment if your GPU/driver supports it well
    # print("Compiling model...")
    # model = torch.compile(model)
    
    # # Example 1: Factual vs Reasoning
    # print("\n" + "="*60)
    # print("EXAMPLE 1: Factual < Reasoning Entropy")
    # print("="*60)
    
    # factual = estimate_entropy(
    #     model=model,
    #     tokenizer=tokenizer,
    #     prefix="What is the capital of France?",
    #     max_samples=500,
    #     max_len=32,
    #     model_name=model_name,
    #     variance_threshold=1e-2,
    #     offset=0.0
    # )
    
    # reasoning = estimate_entropy(
    #     model=model,
    #     tokenizer=tokenizer,
    #     prefix="What are common ways to solve a coding challenge?",
    #     max_samples=1000,
    #     max_len=32,
    #     model_name=model_name,
    #     variance_threshold=1e-2,
    #     offset=0.0
    # )
    
    # print(f"\nFactual entropy:   {factual['entropy_bits']:.2f} bits (EOS rate: {factual['eos_rate']:.1%}, samples: {factual['n_samples']})")
    # print(f"Reasoning entropy: {reasoning['entropy_bits']:.2f} bits (EOS rate: {reasoning['eos_rate']:.1%}, samples: {reasoning['n_samples']})")
    # print(f"Difference:        {reasoning['entropy_bits'] - factual['entropy_bits']:.2f} bits")
    
    # if reasoning['entropy_bits'] > factual['entropy_bits']:
    #     print("\n✓ HYPOTHESIS CONFIRMED: Reasoning has higher entropy than factual!")
    # else:
    #     print("\n✗ HYPOTHESIS REJECTED")
    
    # Example 2: Technical Question Entropy
    print("\n" + "="*60)
    print("EXAMPLE 2: Technical Question Entropy")
    print("="*60)
    
    technical = estimate_entropy(
        model=model,
        tokenizer=tokenizer,
        prefix="What are some concepts that are important for the GNU assembler (GAS)?",
        max_samples=1000,
        max_len=1024,
        save_path="results/gas_question.parquet",
        model_name=model_name,
        variance_threshold=1e-2,
        offset=0.0
    )

    print()
    print(f"Technical question entropy: {technical['entropy_bits']:.2f} bits")
    print(f"EOS rate                  : {technical['eos_rate']:.1%}")
    print(f"Samples used              : {technical['n_samples']}")
    print(f"Converged?                : {technical['converged']}")
    
    # Example 3: Compare question specificity
    print("\n" + "="*60)
    print("EXAMPLE 3: Question Specificity Comparison")
    print("="*60)
    
    vague = estimate_entropy(
        model=model,
        tokenizer=tokenizer,
        prefix="What is programming?",
        max_samples=500,
        max_len=50,
        model_name=model_name,
        variance_threshold=1e-2,
        offset=0.0
    )
    
    specific = estimate_entropy(
        model=model,
        tokenizer=tokenizer,
        prefix="What is the time complexity of quicksort?",
        max_samples=500,
        max_len=64,
        model_name=model_name,
        variance_threshold=1e-2,
        offset=0.0
    )
    
    print(f"\nVague question entropy:    {vague['entropy_bits']:.2f} bits (EOS: {vague['eos_rate']:.1%}, samples: {vague['n_samples']})")
    print(f"Specific question entropy: {specific['entropy_bits']:.2f} bits (EOS: {specific['eos_rate']:.1%}, samples: {specific['n_samples']})")
    print(f"Difference: {vague['entropy_bits'] - specific['entropy_bits']:.2f} bits")
    
    if vague['entropy_bits'] > specific['entropy_bits']:
        print("\n✓ Vague questions have higher entropy (more diverse answers)")
    else:
        print("\n✗ Unexpected: Specific question has higher entropy")