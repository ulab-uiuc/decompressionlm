"""
Example script comparing QMC vs IID sampling for entropy estimation.
Shows convergence speed differences between deterministic low-discrepancy and random sampling.
"""
import torch
import numpy as np
import time
from typing import Dict, Tuple, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotext as plt

from src.decompress import estimate_entropy
from src.vdc import generate_vdc_sequence
from src.arithmetic import arithmetic_sample_sequence


def clear_lines(n):
    """Clear n lines from terminal by moving cursor up and clearing."""
    for _ in range(n):
        print('\033[F\033[K', end='')


def estimate_entropy_iid(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    max_samples: int = 1000,
    max_len: int = 100,
    use_chat_template: bool = True,
    device: str = "cuda",
    use_cache: bool = True,
    variance_threshold: float = 1e-4,
) -> Dict:
    """
    Estimate H(X | prefix) using IID (random) sampling with early stopping.
    Same logic as QMC version but with random codes instead of Van der Corput.
    """
    
    # Prepare prefix
    if prefix == "":
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            prefix_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long)
        else:
            prefix_ids = tokenizer.encode("", return_tensors="pt", add_special_tokens=True)
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

    # Generate random codes for IID sampling
    codes = np.random.random(max_samples).tolist()
    sampling_method = "IID (Random)"

    log_probs = []
    sequences = []
    entropy_history = []
    eos_count = 0
    sum_log_probs = 0.0

    print(f"Sampling sequences for prefix: '{prefix}' with early stopping [{sampling_method}]")
    print(f"Max samples       : {max_samples}")
    print(f"Variance threshold: {variance_threshold}")

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
        sum_log_probs += log_prob
        
        if sample_info['terminated_with_eos']:
            eos_count += 1

        samples_done = i + 1
        current_entropy = -sum_log_probs / samples_done
        current_entropy_bits = current_entropy / torch.log(torch.tensor(2.0)).item()
        entropy_history.append(current_entropy_bits)

        # Check early stopping condition
        if samples_done >= 20:
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
            plt.title(f"{sampling_method} Entropy Convergence")
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
    plt.title(f"{sampling_method} Entropy Convergence - FINAL")
    plt.xlabel("Sample")
    plt.ylabel("Entropy (bits)")
    plt.plotsize(100, 20)
    plt.show()
    
    # Print final status
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
    entropy_bits = entropy / torch.log(torch.tensor(2.0)).item()

    return {
        'entropy': entropy,
        'entropy_bits': entropy_bits,
        'eos_rate': eos_count / samples_done,
        'n_samples': samples_done,
        'prefix': prefix,
        'log_probs': log_probs,
        'sequences': sequences,
        'codes': codes[:samples_done],
        'entropy_history': entropy_history,
        'converged': converged,
    }


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
    
    # Set random seed for reproducible IID sampling
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Example 1: Factual Question
    print("\n" + "="*60)
    print("EXAMPLE 1: Factual Question - QMC vs IID")
    print("="*60)
    
    prefix = "Where is the capital of France?"
    
    print("\n--- QMC Sampling (Van der Corput) ---")
    factual_qmc = estimate_entropy(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        max_samples=10000,
        max_len=32,
        model_name=model_name,
        variance_threshold=5e-4,
        offset=0.0
    )
    
    print("\n--- IID Sampling (Random) ---")
    factual_iid = estimate_entropy_iid(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        max_samples=10000,
        max_len=32,
        variance_threshold=5e-4,
    )
    
    print("\n" + "="*60)
    print("COMPARISON: Factual Question")
    print("="*60)
    print(f"QMC: {factual_qmc['entropy_bits']:.2f} bits (samples: {factual_qmc['n_samples']}, converged: {factual_qmc['converged']})")
    print(f"IID: {factual_iid['entropy_bits']:.2f} bits (samples: {factual_iid['n_samples']}, converged: {factual_iid['converged']})")
    print(f"Sample ratio (IID/QMC): {factual_iid['n_samples']/factual_qmc['n_samples']:.2f}x")
    
    # Example 2: Reasoning Question
    print("\n" + "="*60)
    print("EXAMPLE 2: Reasoning Question - QMC vs IID")
    print("="*60)
    
    prefix = "What are the common ways to solve a coding challenge?"
    
    print("\n--- QMC Sampling (Van der Corput) ---")
    reasoning_qmc = estimate_entropy(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        max_samples=10000,
        max_len=32,
        model_name=model_name,
        variance_threshold=1e-2,
        offset=0.0
    )
    
    print("\n--- IID Sampling (Random) ---")
    reasoning_iid = estimate_entropy_iid(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        max_samples=10000,
        max_len=32,
        variance_threshold=1e-2,
    )
    
    print("\n" + "="*60)
    print("COMPARISON: Reasoning Question")
    print("="*60)
    print(f"QMC: {reasoning_qmc['entropy_bits']:.2f} bits (samples: {reasoning_qmc['n_samples']}, converged: {reasoning_qmc['converged']})")
    print(f"IID: {reasoning_iid['entropy_bits']:.2f} bits (samples: {reasoning_iid['n_samples']}, converged: {reasoning_iid['converged']})")
    print(f"Sample ratio (IID/QMC): {reasoning_iid['n_samples']/reasoning_qmc['n_samples']:.2f}x")
    
    # Example 3: Technical Question
    print("\n" + "="*60)
    print("EXAMPLE 3: Technical Question - QMC vs IID")
    print("="*60)
    
    prefix = "What are some concepts that are important for the GNU assembler (GAS)?"
    
    print("\n--- QMC Sampling (Van der Corput) ---")
    technical_qmc = estimate_entropy(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        max_samples=10000,
        max_len=64,
        model_name=model_name,
        variance_threshold=1e-2,
        offset=0.0
    )
    
    print("\n--- IID Sampling (Random) ---")
    technical_iid = estimate_entropy_iid(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        max_samples=10000,
        max_len=64,
        variance_threshold=1e-2,
    )
    
    print("\n" + "="*60)
    print("COMPARISON: Technical Question")
    print("="*60)
    print(f"QMC: {technical_qmc['entropy_bits']:.2f} bits (samples: {technical_qmc['n_samples']}, converged: {technical_qmc['converged']})")
    print(f"IID: {technical_iid['entropy_bits']:.2f} bits (samples: {technical_iid['n_samples']}, converged: {technical_iid['converged']})")
    print(f"Sample ratio (IID/QMC): {technical_iid['n_samples']/technical_qmc['n_samples']:.2f}x")
    
    # Final Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY: QMC vs IID CONVERGENCE SPEED")
    print("="*60)
    
    all_qmc = [factual_qmc, reasoning_qmc, technical_qmc]
    all_iid = [factual_iid, reasoning_iid, technical_iid]
    
    qmc_avg_samples = sum(r['n_samples'] for r in all_qmc) / len(all_qmc)
    iid_avg_samples = sum(r['n_samples'] for r in all_iid) / len(all_iid)
    
    qmc_converged = sum(1 for r in all_qmc if r['converged'])
    iid_converged = sum(1 for r in all_iid if r['converged'])
    
    print(f"\nAverage samples to convergence:")
    print(f"  QMC: {qmc_avg_samples:.1f}")
    print(f"  IID: {iid_avg_samples:.1f}")
    print(f"  Speedup: {iid_avg_samples/qmc_avg_samples:.2f}x (QMC is faster)")
    
    print(f"\nConvergence rate:")
    print(f"  QMC: {qmc_converged}/{len(all_qmc)}")
    print(f"  IID: {iid_converged}/{len(all_iid)}")


if __name__ == "__main__":
    main()