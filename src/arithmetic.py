"""Arithmetic sampling for language models using deterministic codes."""

import torch
from typing import Tuple, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer


def arithmetic_sample_token(logits: torch.Tensor, code: float) -> Tuple[int, float, float]:
    """
    Sample a single token using arithmetic coding with a deterministic code.
    
    Uses numerically stable computation:
    - log_softmax in fp32 for stability
    - cumsum in fp64 for accurate CDF
    
    Args:
        logits: Logits over vocab [vocab_size]
        code: Deterministic code in [0, 1) from VdC sequence
        
    Returns:
        token_id: Sampled token index
        log_prob: Log probability of sampled token
        rescaled_code: Code rescaled to the subinterval for next token
    """
    # Clamp code to [0, 1) to prevent OOB from searchsorted
    # VdC should give [0, 1), but rescaling can introduce rounding errors
    code = min(max(float(code), 0.0), 1.0 - 1e-16)
    
    # Numerically stable: log_softmax in fp32
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    
    # Accurate CDF computation in fp64
    cdf = torch.cumsum(probs.double(), dim=0)
    cdf[-1] = 1.0  # Fix floating point errors
    
    # Find which interval the code falls into
    token_id = torch.searchsorted(cdf, code).item()
    token_logp = log_probs[token_id].item()
    
    # Get interval bounds [m, M) for the sampled token
    m = 0.0 if token_id == 0 else cdf[token_id - 1].item()
    M = cdf[token_id].item()
    
    # Rescale code to subinterval: c_{t+1} = (c_t - m) / (M - m)
    # Guard against pathological M == m case (can happen with fp16 underflow)
    denom = M - m
    rescaled_code = 0.0 if denom == 0 else (code - m) / denom
    
    return token_id, token_logp, rescaled_code


def arithmetic_sample_sequence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix_ids: torch.Tensor,
    code: float,
    max_len: int = 100,
    device: str = "cuda",
    use_cache: bool = True
) -> Tuple[list[int], float, dict]:
    """
    Sample a complete sequence autoregressively using arithmetic coding.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prefix_ids: Input token IDs [1, seq_len]
        code: Deterministic code in [0, 1) from VdC sequence
        max_len: Maximum generation length
        device: Device to run on
        use_cache: Whether to use KV cache (highly recommended for speed)
        
    Returns:
        tokens: List of generated token IDs (excluding EOS)
        total_log_prob: Sum of log probabilities INCLUDING EOS if encountered
        info: Dict with 'per_token_log_probs', 'num_tokens', 'terminated_with_eos'
    """
    model.eval()
    input_ids = prefix_ids.to(device)
    generated_tokens = []
    total_log_prob = 0.0
    per_token_log_probs = []
    
    # Start with initial code, will be rescaled at each step
    current_code = code
    terminated_with_eos = False
    
    with torch.no_grad():
        if use_cache:
            # First pass: get initial logits and cache
            outputs = model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :].squeeze(0)  # [vocab_size]
            
            for step in range(max_len):
                # Sample using arithmetic coding
                token_id, log_prob, current_code = arithmetic_sample_token(
                    next_token_logits, current_code
                )
                
                # Always include log_prob for proper sequence probability
                per_token_log_probs.append(log_prob)
                total_log_prob += log_prob
                
                # Check for EOS - include its probability but don't add to output
                if token_id == tokenizer.eos_token_id:
                    terminated_with_eos = True
                    break
                
                # Add non-EOS tokens to output
                generated_tokens.append(token_id)
                
                # Generate next token using cache (O(1) instead of O(L))
                outputs = model(
                    torch.tensor([[token_id]], device=device),
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :].squeeze(0)
        else:
            # Fallback: no cache (slower O(L²) but simpler)
            for step in range(max_len):
                # Get logits for next token
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :].squeeze(0)  # [vocab_size]
                
                # Sample using arithmetic coding and get rescaled code for next step
                token_id, log_prob, current_code = arithmetic_sample_token(
                    next_token_logits, current_code
                )
                
                # Always include log_prob for proper sequence probability
                per_token_log_probs.append(log_prob)
                total_log_prob += log_prob
                
                # Check for EOS - include its probability but don't add to output
                if token_id == tokenizer.eos_token_id:
                    terminated_with_eos = True
                    break
                
                # Add non-EOS tokens to output
                generated_tokens.append(token_id)
                
                # Update input for next iteration
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[token_id]], device=device)
                ], dim=1)
    
    info = {
        'per_token_log_probs': per_token_log_probs,
        'num_tokens': len(generated_tokens),
        'terminated_with_eos': terminated_with_eos
    }
    
    return generated_tokens, total_log_prob, info


if __name__ == "__main__":
    # Test with a small model
    print("Loading Qwen2.5-1.5B-Instruct...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )
    
    # Test prefix
    prefix = "Please answer this question: what are some concept that is important for the gnu assembler (GAS)?"
    messages = [{"role": "user", "content": prefix}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"\nPrefix: {prefix}")
    print(f"Formatted prompt:\n{prompt}\n")
    
    # Encode prefix
    prefix_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Test with a few different codes
    from src.vdc import generate_vdc_sequence
    
    codes = generate_vdc_sequence(100)
    print("Sampling with different VdC codes:\n")
    
    for i, code in enumerate(codes[:5]):  # Just show first 5 for testing
        tokens, log_prob, info = arithmetic_sample_sequence(
            model, tokenizer, prefix_ids, code, max_len=128
        )
        
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"Code {i+1} ({code:.4f}):")
        print(f"  Generated: {text}")
        print(f"  Total log prob (including EOS): {log_prob:.4f}")
        print(f"  Num tokens: {info['num_tokens']}")
        print(f"  EOS encountered: {info['terminated_with_eos']}")
        print()
