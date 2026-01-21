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


def arithmetic_sample_token_batch(logits: torch.Tensor, codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample tokens for a batch using arithmetic coding with deterministic codes.
    
    Args:
        logits: Logits over vocab [batch_size, vocab_size]
        codes: Deterministic codes in [0, 1) [batch_size]
        
    Returns:
        token_ids: Sampled token indices [batch_size]
        log_probs: Log probabilities of sampled tokens [batch_size]
        rescaled_codes: Codes rescaled to subintervals [batch_size]
    """
    batch_size = logits.shape[0]
    device = logits.device
    
    # Clamp codes to [0, 1)
    codes = torch.clamp(codes, min=0.0, max=1.0 - 1e-16)
    
    # Numerically stable: log_softmax in fp32
    log_probs_all = torch.log_softmax(logits.float(), dim=-1)  # [B, V]
    probs = log_probs_all.exp()
    
    # Accurate CDF computation in fp64
    cdf = torch.cumsum(probs.double(), dim=-1)  # [B, V]
    cdf[:, -1] = 1.0  # Fix floating point errors
    
    # Find which interval each code falls into
    token_ids = torch.searchsorted(cdf, codes.unsqueeze(1).double()).squeeze(1)  # [B]
    
    # Gather log probs for sampled tokens
    log_probs = log_probs_all.gather(1, token_ids.unsqueeze(1)).squeeze(1)  # [B]
    
    # Get interval bounds [m, M) for sampled tokens
    # m: left boundary (0 for first token, otherwise cdf[i-1])
    m = torch.zeros(batch_size, dtype=torch.float64, device=device)
    valid_mask = token_ids > 0
    if valid_mask.any():
        m[valid_mask] = cdf[valid_mask, token_ids[valid_mask] - 1]
    
    # M: right boundary
    M = cdf.gather(1, token_ids.unsqueeze(1)).squeeze(1)  # [B]
    
    # Rescale codes to subintervals
    denom = M - m
    rescaled_codes = torch.where(
        denom == 0,
        torch.zeros_like(codes),
        (codes.double() - m) / denom
    ).float()
    
    return token_ids, log_probs, rescaled_codes


def parallel_arithmetic_sample_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix_ids: torch.Tensor,
    codes: List[float],
    max_len: int = 100,
    device: str = "cuda",
) -> List[Tuple[List[int], float, dict, List[float]]]:
    """
    Sample multiple sequences in parallel using batched forward passes.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prefix_ids: Input token IDs [1, seq_len]
        codes: List of deterministic codes in [0, 1)
        max_len: Maximum generation length per sample
        device: Device to run on
        
    Returns:
        List of (tokens, total_log_prob, info, per_token_log_probs) for each sequence
    """
    batch_size = len(codes)
    model.eval()
    
    # Expand prefix for batch [B, prefix_len]
    input_ids = prefix_ids.repeat(batch_size, 1).to(device)
    
    # Track state for each sequence
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    current_codes = torch.tensor(codes, dtype=torch.float32, device=device)
    generated_tokens = [[] for _ in range(batch_size)]
    total_log_probs = torch.zeros(batch_size, device=device)
    per_token_log_probs_list = [[] for _ in range(batch_size)]  # Track per-token log probs
    
    with torch.no_grad():
        # Initial forward pass with shared prefix
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]  # [B, vocab_size]
        
        for step in range(max_len):
            # Sample for all active sequences using batched operation
            token_ids_batch, log_probs_batch, new_codes = arithmetic_sample_token_batch(
                logits, current_codes
            )
            
            # Update state for each sequence
            for i in range(batch_size):
                if active_mask[i]:
                    tid = token_ids_batch[i].item()
                    lp = log_probs_batch[i].item()
                    
                    per_token_log_probs_list[i].append(lp)  # Store per-token log prob
                    total_log_probs[i] += lp
                    
                    if tid == tokenizer.eos_token_id:
                        active_mask[i] = False
                    else:
                        generated_tokens[i].append(tid)
            
            current_codes = new_codes
            
            # Early exit if all sequences terminated
            if not active_mask.any():
                break
            
            # Prepare next tokens for ALL sequences (use padding for inactive)
            # This is simpler and avoids cache slicing issues
            next_tokens = token_ids_batch.unsqueeze(1)  # [B, 1]
            
            # Forward pass with full batch
            outputs = model(
                next_tokens,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # [B, vocab_size]
            
            # Zero out logits for inactive sequences (they won't be used anyway)
            logits[~active_mask] = 0.0
    
    # Package results
    results = []
    for i in range(batch_size):
        info = {
            'num_tokens': len(generated_tokens[i]),
            'terminated_with_eos': len(generated_tokens[i]) == 0 or not active_mask[i]
        }
        results.append((
            generated_tokens[i],
            total_log_probs[i].item(),
            info,
            per_token_log_probs_list[i]  # NEW: include per-token log probs as 4th element
        ))
    
    return results


def arithmetic_sample_sequence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix_ids: torch.Tensor,
    code: float,
    max_len: int = 100,
    device: str = "cuda",
    use_cache: bool = True
) -> Tuple[list[int], float, dict, List[float]]:
    """
    Sample a complete sequence autoregressively using arithmetic coding.
    
    NOTE: This is the legacy sequential version. Use parallel_arithmetic_sample_batch
    for better GPU utilization.
    
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
        info: Dict with 'num_tokens', 'terminated_with_eos'
        per_token_log_probs: List of log probabilities for each token
    """
    # Just use the batched version with batch_size=1
    results = parallel_arithmetic_sample_batch(
        model=model,
        tokenizer=tokenizer,
        prefix_ids=prefix_ids,
        codes=[code],
        max_len=max_len,
        device=device
    )
    
    return results[0]


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
    
    # Test parallel batching
    from src.vdc import generate_vdc_sequence
    
    codes = generate_vdc_sequence(8)  # Test with small batch
    print("Testing parallel sampling with 8 sequences:\n")
    
    results = parallel_arithmetic_sample_batch(
        model, tokenizer, prefix_ids, codes, max_len=32
    )
    
    for i, (tokens, log_prob, info, per_token_log_probs) in enumerate(results):
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"Sequence {i+1}:")
        print(f"  Generated: {text[:100]}...")
        print(f"  Total log prob: {log_prob:.4f}")
        print(f"  Num tokens: {info['num_tokens']}")
        print(f"  Per-token log probs: {per_token_log_probs[:5]}...")  # Show first 5
        print(f"  EOS: {info['terminated_with_eos']}")
        print()