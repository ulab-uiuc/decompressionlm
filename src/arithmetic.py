"""Arithmetic sampling for language models using deterministic codes."""
import os; os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from typing import Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer


def arithmetic_sample_token(logits: torch.Tensor, code: float) -> Tuple[int, float, float]:
    """
    Sample a single token using arithmetic coding with a deterministic code.

    Uses numerically stable computation:
    - log_softmax in fp32 for stability
    - cumsum in fp64 for accurate CDF
    """
    code = min(max(float(code), 0.0), 1.0 - 1e-16)

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()

    cdf = torch.cumsum(probs.double(), dim=0)
    cdf[-1] = 1.0

    token_id = torch.searchsorted(cdf, code).item()
    token_logp = log_probs[token_id].item()

    m = 0.0 if token_id == 0 else cdf[token_id - 1].item()
    M = cdf[token_id].item()

    denom = M - m
    rescaled_code = 0.0 if denom == 0 else (code - m) / denom

    return token_id, token_logp, rescaled_code


def arithmetic_sample_token_batch(
    logits: torch.Tensor, codes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample tokens for a batch using arithmetic coding with deterministic codes.
    """
    batch_size = logits.shape[0]
    # Use the device of the input logits to ensure consistency
    device = logits.device

    codes = torch.clamp(codes, min=0.0, max=1.0 - 1e-16).to(device)

    log_probs_all = torch.log_softmax(logits.float(), dim=-1)  # [B, V]
    probs = log_probs_all.exp()

    cdf = torch.cumsum(probs.double(), dim=-1)  # [B, V]
    cdf[:, -1] = 1.0

    token_ids = torch.searchsorted(cdf, codes.unsqueeze(1).double()).squeeze(1)  # [B]
    log_probs = log_probs_all.gather(1, token_ids.unsqueeze(1)).squeeze(1)  # [B]

    m = torch.zeros(batch_size, dtype=torch.float64, device=device)
    valid_mask = token_ids > 0
    if valid_mask.any():
        m[valid_mask] = cdf[valid_mask, token_ids[valid_mask] - 1]

    M = cdf.gather(1, token_ids.unsqueeze(1)).squeeze(1)  # [B]

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
) -> List[Tuple[List[int], float, dict, List[float]]]:
    """
    Sample multiple sequences in parallel. 
    Fix: Automatically detects the model's device to avoid CPU/GPU mismatch.
    """
    batch_size = len(codes)
    model.eval()

    # CRITICAL FIX: Get the device from the model itself
    # This handles device_map="auto" where layers might be split.
    device = next(model.parameters()).device

    # Here gemma refers to models that we had to disable kv cache in order to run
    # for deepseek-coder there should be better, faster fixes though
    model_type = (getattr(model, "config", None) and getattr(model.config, "model_type", "")) or ""
    is_gemma = "gemma" in str(model_type).lower() or "deepseek" in str(model_type).lower()

    # Move input to the correct device
    prefix_ids = prefix_ids.to(device)
    input_ids = prefix_ids.repeat(batch_size, 1)  # [B, prefix_len]

    gemma_full_ids = input_ids

    # Track state on the correct device
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    current_codes = torch.tensor(codes, dtype=torch.float32, device=device)
    generated_tokens = [[] for _ in range(batch_size)]
    total_log_probs = torch.zeros(batch_size, device=device)
    per_token_log_probs_list = [[] for _ in range(batch_size)]

    with torch.no_grad():
        if is_gemma:
            outputs = model(gemma_full_ids, use_cache=False)
            past_key_values = None
        else:
            outputs = model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values

        logits = outputs.logits[:, -1, :]  # [B, V]

        for _step in range(max_len):
            token_ids_batch, log_probs_batch, new_codes = arithmetic_sample_token_batch(
                logits, current_codes
            )

            for i in range(batch_size):
                if active_mask[i]:
                    tid = int(token_ids_batch[i].item())
                    lp = float(log_probs_batch[i].item())
                    per_token_log_probs_list[i].append(lp)
                    total_log_probs[i] += lp

                    if tid == tokenizer.eos_token_id:
                        active_mask[i] = False
                    else:
                        generated_tokens[i].append(tid)

            current_codes = new_codes
            if not active_mask.any():
                break

            next_tokens = token_ids_batch.unsqueeze(1)  # [B, 1]

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

    results = []
    for i in range(batch_size):
        info = {
            "num_tokens": len(generated_tokens[i]),
            "terminated_with_eos": (len(generated_tokens[i]) == 0) or (not active_mask[i].item()),
        }
        results.append((generated_tokens[i], float(total_log_probs[i].item()), info, per_token_log_probs_list[i]))

    return results

# Helper for testing if you don't have the external vdc file
def generate_vdc_sequence(n: int) -> List[float]:
    """Van der Corput sequence (base 2) for uniform deterministic codes."""
    codes = []
    for i in range(1, n + 1):
        res = 0
        f = 0.5
        while i > 0:
            res += f * (i % 2)
            i //= 2
            f /= 2
        codes.append(res)
    return codes

if __name__ == "__main__":
    print("Loading Model...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    prefix = "Explain the core concept of the GNU Assembler."
    messages = [{"role": "user", "content": prefix}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(prompt, return_tensors="pt")

    codes = generate_vdc_sequence(4)
    results = parallel_arithmetic_sample_batch(model, tokenizer, prefix_ids, codes, max_len=32)

    for i, (tokens, log_prob, info, _) in enumerate(results):
        print(f"Result {i+1}: {tokenizer.decode(tokens)}")


def arithmetic_sample_sequence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix_ids: torch.Tensor,
    code: float,
    max_len: int = 100,
) -> Tuple[list[int], float, dict, List[float]]:
    """
    Legacy sequential interface. Internally uses batch_size=1.
    """
    return parallel_arithmetic_sample_batch(
        model=model,
        tokenizer=tokenizer,
        prefix_ids=prefix_ids,
        codes=[code],
        max_len=max_len,
    )[0]


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

    prefix = "Please answer this question: what are some concept that is important for the gnu assembler (GAS)?"
    messages = [{"role": "user", "content": prefix}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"\nPrefix: {prefix}")
    print(f"Formatted prompt:\n{prompt}\n")

    prefix_ids = tokenizer.encode(prompt, return_tensors="pt")

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
