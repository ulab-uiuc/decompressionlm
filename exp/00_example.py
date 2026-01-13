"""
Example script demonstrating bin entropy estimation.

This shows how to use the bin entropy API to detect memorization vs. learning
by measuring the empirical entropy of token prefix distributions.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.bin_entropy import estimate_bin_entropy


def main():
    print("="*80)
    print("BIN ENTROPY ESTIMATION EXAMPLE")
    print("="*80)
    print("\nThis example demonstrates how bin entropy can reveal memorization.")
    print("- Low entropy at short bin lengths → model memorized common patterns")
    print("- High entropy → model is generating diverse, creative responses")
    print()
    
    # Load model
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )
    
    print("Model loaded!\n")
    
    # Example 1: Factual question (expect low entropy - memorized answer)
    print("\n" + "="*80)
    print("EXAMPLE 1: Factual Question (Expected: LOW entropy)")
    print("="*80)
    print("Question: 'What is the capital of France?'")
    print("Hypothesis: Model has memorized this fact, should have low bin entropy\n")
    
    factual_results = estimate_bin_entropy(
        model=model,
        tokenizer=tokenizer,
        prefix="What is the capital of France?",
        bin_prefix_lens=[1, 2, 4, 8, 16],
        max_samples=262144,
        max_len=32,
        batch_size=128,
        display_interval=256,
        variance_threshold=1e-2,
        save_path="results/factual_question.delm.parquet",
        model_name=model_name,
    )
    
    # Example 2: Open-ended question (expect higher entropy - diverse answers)
    print("\n" + "="*80)
    print("EXAMPLE 2: Open-ended Question (Expected: HIGHER entropy)")
    print("="*80)
    print("Question: 'What are some interesting facts about space?'")
    print("Hypothesis: Many valid answers, should have higher bin entropy\n")
    
    creative_results = estimate_bin_entropy(
        model=model,
        tokenizer=tokenizer,
        prefix="What are some interesting facts about space?",
        bin_prefix_lens=[1, 2, 4, 8, 16, 32],
        max_samples=262144,
        max_len=64,
        batch_size=128,
        display_interval=256,
        variance_threshold=1e-2,
        save_path="results/creative_question.delm.parquet",
        model_name=model_name,
    )
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    print("\nFactual Question Entropies:")
    for bin_len in factual_results['bin_prefix_lens']:
        h = factual_results['bin_data'][bin_len]['entropy_history'][-1]
        print(f"  Bin length {bin_len}: {h:.3f} bits")
    
    print("\nOpen-ended Question Entropies:")
    for bin_len in creative_results['bin_prefix_lens']:
        h = creative_results['bin_data'][bin_len]['entropy_history'][-1]
        print(f"  Bin length {bin_len}: {h:.3f} bits")
    
    print("\nEntropy Differences (Open-ended - Factual):")
    for bin_len in factual_results['bin_prefix_lens']:
        h_fact = factual_results['bin_data'][bin_len]['entropy_history'][-1]
        h_open = creative_results['bin_data'][bin_len]['entropy_history'][-1]
        diff = h_open - h_fact
        print(f"  Bin length {bin_len}: {diff:+.3f} bits")
    
    # Check hypothesis
    print("\n" + "="*80)
    print("HYPOTHESIS CHECK")
    print("="*80)
    
    # Compare at bin length 2 (good discriminator)
    h_fact_2 = factual_results['bin_data'][2]['entropy_history'][-1]
    h_open_2 = creative_results['bin_data'][2]['entropy_history'][-1]
    
    if h_open_2 > h_fact_2:
        print("✓ HYPOTHESIS CONFIRMED!")
        print(f"  Open-ended question has {h_open_2 - h_fact_2:.3f} bits MORE entropy at bin length 2")
        print("  This suggests the model generates more diverse responses for open-ended questions.")
    else:
        print("✗ HYPOTHESIS REJECTED")
        print("  Unexpected result - factual question has equal or higher entropy")
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETE")
    print("="*80)
    print("\nKey Takeaway:")
    print("Bin entropy reveals what the model has memorized (low entropy)")
    print("vs. what it generates creatively (high entropy).")
    print("\nResults saved to results/ directory as .delm.parquet files.")


if __name__ == "__main__":
    main()