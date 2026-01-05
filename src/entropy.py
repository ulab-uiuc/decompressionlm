"""Entropy estimation for language models using quasi-Monte Carlo sampling."""

import time
import os

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

import plotext as plt

from src.vdc import generate_vdc_sequence
from src.arithmetic import parallel_arithmetic_sample_batch


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
    use_cache: bool = True,  # Deprecated, always uses cache in parallel mode
    variance_threshold: float = 1e-4,
    offset: float = 0.0,
    batch_size: int = 128,
    display_interval: int = 128,
) -> Tuple[float, Dict]:
    """
    Estimate H(X | prefix) using quasi-Monte Carlo sampling with early stopping.
    
    Uses Van der Corput sequence for deterministic low-discrepancy sampling
    and arithmetic coding to get exact probabilities. Now with parallel batching!
    
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
        use_cache: Deprecated (always True in parallel mode)
        variance_threshold: Stop when variance of entropy in second half < threshold
        offset: Cranley-Patterson rotation offset in [0, 1)
        batch_size: Number of sequences to generate in parallel (default: 128)
        display_interval: Update display every N samples (must be 0 or multiple of batch_size)
        
    Returns:
        entropy: Estimated H(X | prefix) in nats
        info: Dict with detailed sampling information
    """
    
    # Validate parameters
    if not (0.0 <= offset < 1.0):
        raise ValueError("offset must be in [0, 1)")
    
    if display_interval < 0:
        raise ValueError("display_interval must be >= 0")
    
    if display_interval > 0:
        if display_interval < batch_size:
            raise ValueError(f"display_interval ({display_interval}) must be >= batch_size ({batch_size})")
        if display_interval % batch_size != 0:
            raise ValueError(f"display_interval ({display_interval}) must be a multiple of batch_size ({batch_size})")

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
    total_tokens = 0  # Track total tokens generated

    if prefix == "":
        print(f"Sampling sequences unconditionally (BOS-only) with parallel batching")
    else:
        print(f"Sampling sequences for prefix: '{prefix}' with parallel batching")
    print(f"Max Seq Length    : {max_len}")
    print(f"Max samples       : {max_samples}")
    print(f"Batch size        : {batch_size}")
    print(f"Display interval  : {display_interval if display_interval > 0 else 'disabled (final only)'}")
    print(f"Variance threshold: {variance_threshold}")
    print(f"Offset            : {offset}")

    start_time = time.time()
    last_plot_lines = 0
    converged = False
    samples_done = 0

    # Process in batches
    for batch_start in range(0, max_samples, batch_size):
        batch_end = min(batch_start + batch_size, max_samples)
        batch_codes = codes[batch_start:batch_end]
        
        # Generate batch in parallel
        batch_results = parallel_arithmetic_sample_batch(
            model=model,
            tokenizer=tokenizer,
            prefix_ids=prefix_ids,
            codes=batch_codes,
            max_len=max_len,
            device=device
        )
        
        # Collect results from batch
        for tokens, log_prob, sample_info in batch_results:
            log_probs.append(log_prob)
            sequences.append(tokens)
            sum_log_probs += log_prob
            total_tokens += sample_info['num_tokens']
            
            if sample_info['terminated_with_eos']:
                eos_count += 1
            
            samples_done += 1
            current_entropy = -sum_log_probs / samples_done
            current_entropy_bits = current_entropy / torch.log(torch.tensor(2.0)).item()
            entropy_history.append(current_entropy_bits)
        
        # Check early stopping condition (need at least 20 samples total)
        if samples_done >= 20:
            half_point = (samples_done - 1) // 2 + 1
            second_half_entropy = entropy_history[half_point:]
            
            if len(second_half_entropy) > 1:
                entropy_variance = np.var(second_half_entropy)
                
                if entropy_variance < variance_threshold:
                    converged = True
                    # Still display final stats even if converged
                    break
        
        # Display update
        should_display = (display_interval > 0 and samples_done % display_interval == 0)
        
        if should_display:
            elapsed = time.time() - start_time
            samples_per_sec = samples_done / elapsed if elapsed > 0 else 0
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            samples_left = max_samples - samples_done
            max_eta_sec = samples_left / samples_per_sec if samples_per_sec > 0 else 0
            
            # Clear previous display (plot + 2 stat lines)
            if last_plot_lines > 0:
                clear_lines(last_plot_lines + 2)
            
            # Plot
            plt.clf()
            plt.plot(range(1, len(entropy_history) + 1), entropy_history)
            plt.title("QMC Entropy Convergence")
            plt.xlabel("Sample")
            plt.ylabel("Entropy (bits)")
            plt.plotsize(100, 20)
            plt.show()
            
            last_plot_lines = 20
            
            # Stats lines
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            max_eta_min = int(max_eta_sec // 60)
            max_eta_sec_remainder = int(max_eta_sec % 60)
            eos_rate = eos_count / samples_done
            
            # Variance status
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
            
            # Line 1: Progress, entropy, EOS rate, variance
            print(
                f"[RUNNING] [{samples_done}/{max_samples}] "
                f"H={current_entropy_bits:.3f}b | "
                f"EOS: {eos_rate:.1%} | "
                f"{var_status}"
            )
            
            # Line 2: Speed metrics and ETA
            print(
                f"| Speed: {samples_per_sec:.1f} samp/s, {tokens_per_sec:.1f} tok/s | "
                f"Elapsed: {elapsed_min}m{elapsed_sec:02d}s | "
                f"MaxETA: {max_eta_min}m{max_eta_sec_remainder:02d}s"
            )

    # Display final plot and stats
    elapsed = time.time() - start_time
    samples_per_sec = samples_done / elapsed if elapsed > 0 else 0
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    eos_rate = eos_count / samples_done
    current_entropy = -sum_log_probs / samples_done
    current_entropy_bits = current_entropy / torch.log(torch.tensor(2.0)).item()
    
    # Clear previous display if it exists (only if display_interval > 0)
    if display_interval > 0 and last_plot_lines > 0:
        clear_lines(last_plot_lines + 2)
    
    # Final plot
    plt.clf()
    plt.plot(range(1, len(entropy_history) + 1), entropy_history)
    plt.title("QMC Entropy Convergence - FINAL")
    plt.xlabel("Sample")
    plt.ylabel("Entropy (bits)")
    plt.plotsize(100, 20)
    plt.show()
    
    # Final stats
    elapsed_min = int(elapsed // 60)
    elapsed_sec = int(elapsed % 60)
    
    if samples_done >= 20:
        half_point = (samples_done - 1) // 2 + 1
        second_half_entropy = entropy_history[half_point:]
        entropy_variance = np.var(second_half_entropy)
        var_status = f"Var: {entropy_variance:.6f}"
    else:
        var_status = "N/A (< 20 samples)"
    
    status = "CONVERGED" if converged else "COMPLETED"
    
    # Line 1: Final status, progress, entropy, EOS rate, variance
    print(
        f"[{status}] [{samples_done}/{max_samples}] "
        f"H={current_entropy_bits:.3f}b | "
        f"EOS: {eos_rate:.1%} | "
        f"{var_status}"
    )
    
    # Line 2: Speed metrics and total time
    print(
        f"| Speed: {samples_per_sec:.1f} samp/s, {tokens_per_sec:.1f} tok/s | "
        f"Total: {elapsed_min}m{elapsed_sec:02d}s"
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
    offset: float = 0.0,
    batch_size: int = 128,
    display_interval: int = 128,
) -> Dict[str, Tuple[float, Dict]]:
    """
    Estimate entropy for multiple prefixes with early stopping and parallel batching.
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
            offset=offset,
            batch_size=batch_size,
            display_interval=display_interval,
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
    
    print(f"\nTesting parallel batching with batch_size=128...")
    
    entropy, info = estimate_conditional_entropy(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        max_samples=131072,
        max_len=128,
        variance_threshold=5e-4,
        batch_size=128,
        display_interval=128
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