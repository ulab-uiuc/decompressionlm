"""Baseline sampling methods for comparison with arithmetic sampling."""

import torch
import random
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


def random_sample_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix_ids: torch.Tensor,
    num_sequences: int,
    max_len: int = 100,
    seed: int = 42,
) -> List[Tuple[List[int], dict]]:
    """
    Sample using random sampling with fixed seed for reproducibility.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prefix_ids: Prefix token IDs [1, prefix_len]
        num_sequences: Number of sequences to generate
        max_len: Maximum sequence length
        seed: Random seed for reproducibility
        
    Returns:
        List of (tokens, info) tuples
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    model.eval()
    device = next(model.parameters()).device
    
    # Check if model needs special handling
    model_type = (getattr(model, "config", None) and getattr(model.config, "model_type", "")) or ""
    is_gemma = "gemma" in str(model_type).lower() or "deepseek" in str(model_type).lower()
    
    prefix_ids = prefix_ids.to(device)
    input_ids = prefix_ids.repeat(num_sequences, 1)
    
    gemma_full_ids = input_ids
    
    # Track state
    active_mask = torch.ones(num_sequences, dtype=torch.bool, device=device)
    generated_tokens = [[] for _ in range(num_sequences)]
    
    with torch.no_grad():
        if is_gemma:
            outputs = model(gemma_full_ids, use_cache=False)
            past_key_values = None
        else:
            outputs = model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
        
        logits = outputs.logits[:, -1, :]
        
        for _step in range(max_len):
            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update sequences
            for i in range(num_sequences):
                if active_mask[i]:
                    tid = int(token_ids[i].item())
                    
                    if tid == tokenizer.eos_token_id:
                        active_mask[i] = False
                    else:
                        generated_tokens[i].append(tid)
            
            if not active_mask.any():
                break
            
            next_tokens = token_ids.unsqueeze(1)
            
            if is_gemma:
                next_tokens = next_tokens.clone()
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                next_tokens[~active_mask] = pad_id
                gemma_full_ids = torch.cat([gemma_full_ids, next_tokens], dim=1)
                outputs = model(gemma_full_ids, use_cache=False)
            else:
                outputs = model(next_tokens, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
            
            logits = outputs.logits[:, -1, :]
            logits[~active_mask] = 0.0
    
    # Build results
    results = []
    for i in range(num_sequences):
        info = {
            "num_tokens": len(generated_tokens[i]),
            "terminated_with_eos": (len(generated_tokens[i]) == 0) or (not active_mask[i].item()),
        }
        results.append((generated_tokens[i], info))
    
    return results


def beam_search_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix_ids: torch.Tensor,
    num_sequences: int,
    max_len: int = 100,
    temperature: float = 1.0,
    num_beams: int = 4,
) -> List[Tuple[List[int], dict]]:
    """
    Sample using beam search with temperature.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prefix_ids: Prefix token IDs [1, prefix_len]
        num_sequences: Number of sequences to generate (must be divisible by num_beams)
        max_len: Maximum sequence length
        temperature: Sampling temperature
        num_beams: Number of beams for beam search
        
    Returns:
        List of (tokens, info) tuples
    """
    model.eval()
    device = next(model.parameters()).device
    
    prefix_ids = prefix_ids.to(device)
    
    # Generate multiple batches if needed
    num_batches = num_sequences // num_beams
    all_results = []
    
    with torch.no_grad():
        for _ in range(num_batches):
            outputs = model.generate(
                prefix_ids,
                max_new_tokens=max_len,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                temperature=temperature,
                do_sample=False,  # Beam search is deterministic
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            
            # Extract generated tokens (remove prefix)
            prefix_len = prefix_ids.shape[1]
            for seq in outputs:
                generated = seq[prefix_len:].tolist()
                
                # Remove EOS if present
                if generated and generated[-1] == tokenizer.eos_token_id:
                    terminated = True
                    generated = generated[:-1]
                else:
                    terminated = False
                
                info = {
                    "num_tokens": len(generated),
                    "terminated_with_eos": terminated,
                }
                all_results.append((generated, info))
    
    return all_results[:num_sequences]


if __name__ == "__main__":
    # Quick test
    print("Testing baseline sampling methods...")
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    prefix = "List some concepts: "
    messages = [{"role": "user", "content": prefix}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    print("\n1. Random sampling (seed=42):")
    results = random_sample_batch(model, tokenizer, prefix_ids, num_sequences=2, max_len=16, seed=42)
    for i, (tokens, info) in enumerate(results):
        text = tokenizer.decode(tokens)
        print(f"  Seq {i+1}: {text[:60]}...")
    
    print("\n2. Beam search (temp=0.5, beams=2):")
    results = beam_search_batch(model, tokenizer, prefix_ids, num_sequences=2, max_len=16, temperature=0.5, num_beams=2)
    for i, (tokens, info) in enumerate(results):
        text = tokenizer.decode(tokens)
        print(f"  Seq {i+1}: {text[:60]}...")
    
    print("\n3. Beam search (temp=1.5, beams=2):")
    results = beam_search_batch(model, tokenizer, prefix_ids, num_sequences=2, max_len=16, temperature=1.5, num_beams=2)
    for i, (tokens, info) in enumerate(results):
        text = tokenizer.decode(tokens)
        print(f"  Seq {i+1}: {text[:60]}...")
    
    print("\n✓ Baseline methods working!")
