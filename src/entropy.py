"""Entropy estimation for language models using quasi-Monte Carlo sampling."""

import time
import os

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

import plotext as plt

from src.vdc import generate_vdc_sequence
from src.arithmetic import arithmetic_sample_sequence


def clear_lines(n):
    """Clear n lines from terminal by moving cursor up and clearing."""
    for _ in range(n):
        print('\033[F\033[K', end='')


def estimate_conditional_entropy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    max_samples: int = 1000,
    max_len: int = 100,
    use_chat_template: bool = True,
    device: str = "cuda",
    use_cache: bool = True,
    variance_threshold: float = 1e-4,
    offset: float = 0.0  # Cranley-Patterson rotation
) -> Tuple[float, Dict]:
    """
    Estimate H(X | prefix) using quasi-Monte Carlo sampling with early stopping.
    
    Uses Van der Corput sequence for deterministic low-discrepancy sampling
    and arithmetic coding to get exact probabilities.
    
    Early stopping: Stops when the variance of entropy estimates in the second half
    of samples falls below variance_threshold, or when max_samples is reached.
    
    Distribution definition:
    - X includes the EOS token probability (proper termination)
    - Sequences that hit max_len without EOS are truncated (no special handling)
    
    Special case: If prefix is "" (empty string), uses BOS-only unconditional mode.
    
    Args:
        model: Language model
        tokenizer: Tokenizer  
        prefix: Text prefix to condition on (empty string "" = BOS-only unconditional)
        max_samples: Maximum number of samples (hard stop)
        max_len: Maximum generation length per sample
        use_chat_template: Whether to format with chat template (ignored if prefix="")
        device: Device to run on
        use_cache: Whether to use KV cache (highly recommended)
        variance_threshold: Stop when variance of entropy in second half < threshold
        offset: Cranley-Patterson rotation offset in [0, 1)
        
    Returns:
        entropy: Estimated H(X | prefix) in nats
        info: Dict with detailed sampling information
    """
    
    # Validate offset
    if not (0.0 <= offset < 1.0):
        raise ValueError("offset must be in [0, 1)")

    # Prepare prefix
    if prefix == "":
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            prefix_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long)
            print(f"Using BOS-only unconditional mode (token_id={tokenizer.bos_token_id})")
        elif hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            prefix_ids = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long)
            print(f"WARNING: No BOS token, using PAD token (token_id={tokenizer.pad_token_id})")
        else:
            prefix_ids = tokenizer.encode("", return_tensors="pt", add_special_tokens=True)
            if prefix_ids.shape[1] == 0:
                prefix_ids = tokenizer.encode(" ", return_tensors="pt")
                print("WARNING: No BOS/PAD token, using space as starting token")
            else:
                print("Using encoded empty string as unconditional start")
    else:
        if use_chat_template:
            messages = [{"role": "user", "content": prefix}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = prefix
        
        prefix_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate VdC sequence up to max_samples
    codes = generate_vdc_sequence(max_samples)
    
    # Apply Cranley-Patterson rotation if offset is non-zero
    if offset > 0.0:
        codes = [(code + offset) % 1.0 for code in codes]

    log_probs = []
    sequences = []
    entropy_history = []
    eos_count = 0
    sum_log_probs = 0.0  # Running sum for O(N) entropy computation

    if prefix == "":
        print(f"Sampling sequences unconditionally (BOS-only) with early stopping")
    else:
        print(f"Sampling sequences for prefix: '{prefix}' with early stopping")
    print(f"Max samples       : {max_samples}")
    print(f"Variance threshold: {variance_threshold}")
    print(f"Offset            : {offset}")
    print(f"Max Seq Length    : {max_len}")

    start_time = time.time()
    last_plot_lines = 0
    converged = False
    samples_done = 0

    for i, code in enumerate(codes):
        tokens, log_prob, sample_info = arithmetic_sample_sequence(
            model=model,
            tokenizer=tokenizer,
            prefix_ids=prefix_ids,
            code=code,
            max_len=max_len,
            device=device,
            use_cache=use_cache
        )

        log_probs.append(log_prob)
        sequences.append(tokens)
        sum_log_probs += log_prob  # O(1) update
        
        if sample_info['terminated_with_eos']:
            eos_count += 1

        samples_done = i + 1
        current_entropy = -sum_log_probs / samples_done  # O(1) instead of O(N)
        current_entropy_bits = current_entropy / torch.log(torch.tensor(2.0)).item()
        entropy_history.append(current_entropy_bits)

        # Check early stopping condition (need at least 10 samples in second half)
        if samples_done >= 20:
            # Calculate variance of entropy in second half
            half_point = (samples_done - 1) // 2 + 1
            second_half_entropy = entropy_history[half_point:]
            
            if len(second_half_entropy) > 1:
                entropy_variance = np.var(second_half_entropy)
                
                if entropy_variance < variance_threshold:
                    converged = True
                    break

        elapsed = time.time() - start_time
        samples_per_sec = samples_done / elapsed if elapsed > 0 else 0
        samples_left = max_samples - samples_done
        max_eta_sec = samples_left / samples_per_sec if samples_per_sec > 0 else 0

        if samples_done % 10 == 0:
            if last_plot_lines > 0:
                clear_lines(last_plot_lines + 1)

            plt.clf()
            plt.plot(range(1, len(entropy_history) + 1), entropy_history)
            plt.title("QMC Entropy Convergence")
            plt.xlabel("Sample")
            plt.ylabel("Entropy (bits)")
            plt.plotsize(100, 20)
            plt.show()

            last_plot_lines = 20

            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            max_eta_min = int(max_eta_sec // 60)
            max_eta_sec_remainder = int(max_eta_sec % 60)
            eos_rate = eos_count / samples_done
            
            # Show variance status
            if samples_done >= 20:
                half_point = (samples_done - 1) // 2 + 1
                second_half_entropy = entropy_history[half_point:]
                if len(second_half_entropy) > 1:
                    entropy_variance = np.var(second_half_entropy)
                    var_status = f"Var: {entropy_variance:.6f}"
                else:
                    var_status = "Var: computing..."
            else:
                var_status = f"Need {20 - samples_done} more"
            
            print(
                f"[{samples_done}/{max_samples}] "
                f"H={current_entropy_bits:.3f}b | "
                f"EOS: {eos_rate:.1%} | "
                f"{samples_per_sec:.1f} samp/s | "
                f"{elapsed_min}m{elapsed_sec:02d}s | "
                f"MaxETA: {max_eta_min}m{max_eta_sec_remainder:02d}s | "
                f"{var_status}"
            )

    # Display final plot
    if last_plot_lines > 0:
        clear_lines(last_plot_lines + 1)
    
    plt.clf()
    plt.plot(range(1, len(entropy_history) + 1), entropy_history)
    plt.title("QMC Entropy Convergence - FINAL")
    plt.xlabel("Sample")
    plt.ylabel("Entropy (bits)")
    plt.plotsize(100, 20)
    plt.show()
    
    # Print final status in same format as progress lines
    elapsed = time.time() - start_time
    elapsed_min = int(elapsed // 60)
    elapsed_sec = int(elapsed % 60)
    eos_rate = eos_count / samples_done
    samples_per_sec = samples_done / elapsed if elapsed > 0 else 0
    
    half_point = (samples_done - 1) // 2 + 1
    second_half_entropy = entropy_history[half_point:]
    entropy_variance = np.var(second_half_entropy)
    var_status = f"Var: {entropy_variance:.6f}"

    if converged:
        status = "CONVERGENCE"
    else:
        status = "NO CONVERGENCE"
    
    print(
        f"[{status}] [{samples_done}/{max_samples}] "
        f"H={current_entropy_bits:.3f}b | "
        f"EOS: {eos_rate:.1%} | "
        f"{samples_per_sec:.1f} samp/s | "
        f"Total: {elapsed_min}m{elapsed_sec:02d}s | "
        f"{var_status}"
    )

    entropy = -sum(log_probs) / samples_done

    result_info = {
        'log_probs': log_probs,
        'sequences': sequences,
        'codes': codes[:samples_done],
        'n_samples': samples_done,
        'prefix': prefix if prefix != "" else "[BOS-only unconditional]",
        'entropy_history': entropy_history,
        'eos_rate': eos_count / samples_done,
        'converged': converged,
    }

    return entropy, result_info


def batch_estimate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefixes: list[str],
    max_samples: int = 1000,
    max_len: int = 100,
    use_chat_template: bool = True,
    device: str = "cuda",
    use_cache: bool = True,
    variance_threshold: float = 1e-4,
    offset: float = 0.0
) -> Dict[str, Tuple[float, Dict]]:
    """
    Estimate entropy for multiple prefixes with early stopping.
    """
    results = {}

    for prefix in prefixes:
        print(f"\n{'='*60}")
        entropy, info = estimate_conditional_entropy(
            model=model,
            tokenizer=tokenizer,
            prefix=prefix,
            max_samples=max_samples,
            max_len=max_len,
            use_chat_template=use_chat_template,
            device=device,
            use_cache=use_cache,
            variance_threshold=variance_threshold,
            offset=offset
        )
        results[prefix] = (entropy, info)
        print(f"Entropy: {entropy:.4f} nats ({entropy/torch.log(torch.tensor(2.0)):.4f} bits)")
        print(f"EOS rate: {info['eos_rate']:.1%}")
        print(f"Samples used: {info['n_samples']}")
        print(f"Converged: {info['converged']}")

    return results


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
    
    print(f"\nTesting early stopping with variance threshold 1e-4...")
    
    entropy, info = estimate_conditional_entropy(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        max_samples=1000,
        max_len=128,
        variance_threshold=1e-4
    )
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Estimated H(X | prefix): {entropy:.4f} nats")
    print(f"                         {entropy/torch.log(torch.tensor(2.0)):.4f} bits")
    print(f"Total samples: {info['n_samples']}")
    print(f"Converged: {info['converged']}")
    print(f"EOS rate: {info['eos_rate']:.1%}")