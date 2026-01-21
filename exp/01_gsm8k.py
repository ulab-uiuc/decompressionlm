#!/usr/bin/env python3
"""
GSM8K meta-question prefix-mass sampling experiment.
"""
import os
import torch
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from src.bin_entropy import estimate_prefix_mass


def main():
    models = [
        # "google/gemma-3-27b-it",
        # "Qwen/Qwen2.5-14B-Instruct",
        # "google/gemma-3-12b-it",
        # "Qwen/Qwen2.5-7B-Instruct",
        # "google/gemma-3-4b-it",
        "Qwen/Qwen2.5-Omni-7B",
        # "microsoft/phi-4-mini-instruct",
        # "microsoft/Phi-3.5-mini-instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "ibm/granite-3.3-8b-instruct",
        # "meta-llama/Llama-3.2-3B-Instruct",
        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it",
        # "google/gemma-3-1b-it",
    ]

    prompt = """GSM8K is a collection of short, grade-school word problems whose arithmetic is trivial but whose solution requires constructing and executing a small, implicit program from natural language. These problems are easy to compute but hard to interpret. Describe, in concrete terms, what a person or model must already know about the world, quantities, and language in order to correctly solve every GSM8K problem. For each piece of knowledge, explain what would go wrong if it were missing. Do not use labels like “state tracking” or “entity binding” unless you also explain exactly what information is being represented."""
    
    max_len = 64
    prefix_len = 64
    prob_threshold = 0.0000001
    max_samples = 262144
    batch_size = 64
    display_interval = batch_size

    os.makedirs("results/gsm8k", exist_ok=True)

    for model_name in models:
        save_name = (
            model_name.replace("/", "_")
            + "__gsm8k_meta"
            + f"__L{prefix_len}"
            + f"__T{prob_threshold}"
            + ".delm.parquet"
        )

        save_path = os.path.join("results/gsm8k", save_name)

        print(f"\n{'='*80}")
        print(f"Processing {model_name}")
        print(f"{'='*80}")
        
        # Load model and tokenizer
        print(f"Loading model and tokenizer...")
        if model_name == "Qwen/Qwen2.5-Omni-7B":
            processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="auto"
            )
            tokenizer = processor.tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="auto"
            )

        try:
            # Run experiment (with automatic caching/validation)
            results = estimate_prefix_mass(
                model=model,
                tokenizer=tokenizer,
                prefix=prompt,
                prefix_len=prefix_len,
                prob_threshold=prob_threshold,
                max_samples=max_samples,
                max_len=max_len,
                use_chat_template=True,
                batch_size=batch_size,
                display_interval=display_interval,
                save_path=save_path,
                model_name=model_name,
            )

            print(f"\n{'='*80}")
            print(f"Results for {model_name}")
            print(f"{'='*80}")
            print(f"Unique prefixes     : {results['unique_prefixes']}")
            print(f"Samples done        : {results['samples_done']}")
            print(f"Effective set size  : {results['effective_set_size']}")
            print(f"Avg tokens/seq      : {results['effective_set_stats']['avg_tokens']:.1f}")
            print(f"Total tokens        : {results['effective_set_stats']['total_tokens']}")
            print(f"Saved to            : {save_path}")
            
        except Exception as e:
            print(f"\n⚠️  Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up memory
            print(f"\nCleaning up memory for {model_name}...")
            del model
            del tokenizer
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            print(f"✓ Memory cleaned for {model_name}\n")


if __name__ == "__main__":
    main()