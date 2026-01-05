"""
Example script showing how to use decompressionLM to estimate entropy.
Replicates the examples from decompress.py's main function with parallel batching.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.decompress import estimate_entropy


def main():
    # Load model
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
    
    # Example 1: Factual vs Reasoning
    print("\n" + "="*60)
    print("EXAMPLE 1: Factual < Reasoning Entropy")
    print("="*60)
    
    factual = estimate_entropy(
        model=model,
        tokenizer=tokenizer,
        prefix="What is the capital of France?",
        max_samples=131072,
        max_len=32,
        model_name=model_name,
        variance_threshold=5e-4,
        offset=0.0,
        batch_size=128,  # Smaller batch for short sequences
        display_interval=128,  # Update every 2 batches
    )
    
    reasoning = estimate_entropy(
        model=model,
        tokenizer=tokenizer,
        prefix="What are common ways to solve a coding challenge?",
        max_samples=131072,
        max_len=32,
        model_name=model_name,
        variance_threshold=5e-4,
        offset=0.0,
        batch_size=128,
        display_interval=128,
    )
    
    print(f"\nFactual entropy:   {factual['entropy_bits']:.2f} bits (EOS rate: {factual['eos_rate']:.1%}, samples: {factual['n_samples']})")
    print(f"Reasoning entropy: {reasoning['entropy_bits']:.2f} bits (EOS rate: {reasoning['eos_rate']:.1%}, samples: {reasoning['n_samples']})")
    print(f"Difference:        {reasoning['entropy_bits'] - factual['entropy_bits']:.2f} bits")
    
    if reasoning['entropy_bits'] > factual['entropy_bits']:
        print("\n✓ HYPOTHESIS CONFIRMED: Reasoning has higher entropy than factual!")
    else:
        print("\n✗ HYPOTHESIS REJECTED")
    
    # Example 2: Technical Question Entropy
    print("\n" + "="*60)
    print("EXAMPLE 2: Technical Question Entropy")
    print("="*60)
    
    technical = estimate_entropy(
        model=model,
        tokenizer=tokenizer,
        prefix="What are some concepts that are important for the GNU assembler (GAS)?",
        max_samples=131072,
        max_len=64,
        save_path="results/gas_question.parquet",
        model_name=model_name,
        variance_threshold=5e-4,
        offset=0.0,
        batch_size=128,  # Larger batch for medium sequences
        display_interval=256,  # Update every 2 batches
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
        max_samples=131072,
        max_len=64,
        model_name=model_name,
        variance_threshold=5e-4,
        offset=0.0,
        batch_size=128,
        display_interval=128,
    )
    
    specific = estimate_entropy(
        model=model,
        tokenizer=tokenizer,
        prefix="What is the time complexity of quicksort?",
        max_samples=131072,
        max_len=64,
        model_name=model_name,
        variance_threshold=5e-4,
        offset=0.0,
        batch_size=128,
        display_interval=128,
    )
    
    print(f"\nVague question entropy:    {vague['entropy_bits']:.2f} bits (EOS: {vague['eos_rate']:.1%}, samples: {vague['n_samples']})")
    print(f"Specific question entropy: {specific['entropy_bits']:.2f} bits (EOS: {specific['eos_rate']:.1%}, samples: {specific['n_samples']})")
    print(f"Difference: {vague['entropy_bits'] - specific['entropy_bits']:.2f} bits")
    
    if vague['entropy_bits'] > specific['entropy_bits']:
        print("\n✓ Vague questions have higher entropy (more diverse answers)")
    else:
        print("\n✗ Unexpected: Specific question has higher entropy")


if __name__ == "__main__":
    main()