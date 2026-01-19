#!/usr/bin/env python3
"""
Example script for prefix mass-based sampling.

This demonstrates the new algorithm:
1. Sample sequences until cumulative prefix probability mass > threshold
2. Cluster sequences by similarity to get effective support set
3. Report statistics about the support set
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.bin_entropy import estimate_prefix_mass


def main():
    # Configuration
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Test prompts
    prompts = [
        "How does LEDs work?",
        "What are the differences between Minecraft Java and Bedrock editions?",
        # "",  # Unconditional (BOS-only)
    ]
    
    # Sampling parameters
    max_len = 256 # Max seqence length
    prefix_len = 32  # Track first 4 tokens as prefix pattern
    prob_threshold = 0.1  # Stop when 90% of prefix mass is discovered
    similarity_threshold = 0.85  # More than 85% similarity will be considered deduplication
    max_samples = 262144
    batch_size = 512
    
    # Load model once
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )
    
    # Run for each prompt
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*80}")
        print(f"PROMPT {i+1}/{len(prompts)}")
        print(f"{'='*80}")
        
        if prompt == "":
            save_name = "unconditional"
            use_chat = False
        else:
            save_name = prompt[:30].replace(" ", "_").replace("?", "")
            use_chat = True
        
        results = estimate_prefix_mass(
            model=model,
            tokenizer=tokenizer,
            prefix=prompt,
            prefix_len=prefix_len,
            prob_threshold=prob_threshold,
            similarity_threshold=similarity_threshold,
            max_samples=max_samples,  # Hard limit
            max_len=max_len,
            use_chat_template=use_chat,
            batch_size=batch_size,
            display_interval=256,
            save_path=f"results/{save_name}.delm.parquet",
            model_name=model_name,
        )
        
        print(f"\nSummary for '{prompt[:50]}...':")
        print(f"  Discovered {results['unique_prefixes']} unique prefix patterns")
        print(f"  Sampled {results['samples_done']} total sequences")
        print(f"  Effective support: {results['effective_set_size']} sequences")
        print(f"  Avg tokens/seq: {results['effective_set_stats']['avg_tokens']:.1f}")
        print(f"  Total tokens: {results['effective_set_stats']['total_tokens']}")


if __name__ == "__main__":
    main()
