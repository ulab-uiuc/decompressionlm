"""
Bin-based entropy estimation for detecting memorization vs. learning.

Instead of measuring H(X|prefix) = -E[log P(X)], we measure the empirical
entropy of bin distributions where bins are defined by the first N tokens
of generation.
"""

import time
import torch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotext as plt
import transformers

from src.vdc import generate_vdc_sequence
from src.arithmetic import parallel_arithmetic_sample_batch
from src.plot_utils import get_bin_colors


def clear_lines(n):
    """Clear n lines from terminal by moving cursor up and clearing."""
    for _ in range(n):
        print('\033[F\033[K', end='')


def extract_bin_prefix(tokens: List[int], bin_len: int, eos_token_id: int) -> tuple:
    """
    Extract bin prefix of specified length, padding with EOS if needed.
    
    Args:
        tokens: Generated token IDs (excluding EOS at end if terminated)
        bin_len: Length of bin prefix
        eos_token_id: EOS token ID for padding
        
    Returns:
        Tuple of exactly bin_len token IDs
        
    Examples:
        >>> extract_bin_prefix([123, 456], 4, 999)
        (123, 456, 999, 999)
        >>> extract_bin_prefix([1, 2, 3, 4, 5], 3, 999)
        (1, 2, 3)
    """
    if len(tokens) >= bin_len:
        return tuple(tokens[:bin_len])
    else:
        # Pad with EOS tokens
        return tuple(tokens + [eos_token_id] * (bin_len - len(tokens)))


class IncrementalEntropyTracker:
    """
    Track empirical entropy incrementally in O(1) per update.
    
    Uses the identity:
    H = log2(N) - (1/N) * Σ c_i * log2(c_i)
    
    Maintains S = Σ c_i * log2(c_i) and updates only affected bin.
    """
    
    def __init__(self):
        self.N = 0  # Total samples
        self.S = 0.0  # Σ c_i * log2(c_i)
        self.counts = defaultdict(int)
        self.entropy_history = []
    
    def update(self, bin_tuple: tuple) -> float:
        """
        Update with a new sample and return current entropy.
        
        Args:
            bin_tuple: The bin for this sample
            
        Returns:
            Current entropy in bits
        """
        old_count = self.counts[bin_tuple]
        new_count = old_count + 1
        
        # Update S by removing old contribution and adding new
        if old_count > 0:
            self.S -= old_count * np.log2(old_count)
        self.S += new_count * np.log2(new_count)
        
        # Update count and total
        self.counts[bin_tuple] = new_count
        self.N += 1
        
        # Compute entropy: H = log2(N) - S/N
        if self.N > 0:
            entropy = np.log2(self.N) - self.S / self.N
        else:
            entropy = 0.0
        
        self.entropy_history.append(entropy)
        return entropy


def save_bin_results(
    results: Dict,
    sequences: List[List[int]],
    codes: List[float],
    terminated: List[bool],
    save_path: str,
    model_name: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    actual_prompt: Optional[str] = None,
    max_len: Optional[int] = None,
    batch_size: Optional[int] = None,
    display_interval: Optional[int] = None,
    use_chat_template: Optional[bool] = None,
    offset: Optional[float] = None,
    torch_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
):
    """
    Save bin entropy results to a single .delm.parquet file.
    
    Saves raw sequences in sampling order so any bin length can be reconstructed.
    Also saves metadata about convergence for each bin length tracked.
    
    Args:
        results: Results dict from estimate_bin_entropy()
        sequences: List of raw token sequences in sampling order
        codes: List of VdC codes used for sampling
        terminated: List of whether each sequence hit EOS
        save_path: Path to save file (should end in .delm.parquet)
        model_name: Model name/identifier
        tokenizer_name: Tokenizer name (may differ from model)
        actual_prompt: The actual prompt after chat template
        max_len: Maximum generation length
        batch_size: Batch size used
        display_interval: Display interval used
        use_chat_template: Whether chat template was used
        offset: VdC offset used
        torch_dtype: Torch dtype string (e.g., "torch.float16")
        device_map: Device map string (e.g., "auto")
    """
    # Parse the path
    path = Path(save_path)
    
    # Check if it ends with .delm.parquet
    if not save_path.endswith('.delm.parquet'):
        # Auto-add the extension if missing
        save_path = save_path + '.delm.parquet'
        path = Path(save_path)
        print(f"Note: Auto-adding .delm.parquet extension")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert sequences to Arrow list<uint32>
    sequence_array = pa.array(sequences, type=pa.list_(pa.uint32()))
    
    # Create data table
    data = {
        'code': pa.array(codes, type=pa.float64()),
        'sequence': sequence_array,
        'sequence_length': pa.array([len(seq) for seq in sequences], type=pa.int32()),
        'terminated': pa.array(terminated, type=pa.bool_()),
    }
    
    # Build convergence info for metadata
    convergence_info = {}
    final_entropies = {}
    for bin_len in results['bin_prefix_lens']:
        bin_data = results['bin_data'][bin_len]
        convergence_info[f'len{bin_len}_converged'] = str(bin_data['converged'])
        convergence_info[f'len{bin_len}_convergence_sample'] = str(bin_data['convergence_sample'] or 'N/A')
        final_entropy = bin_data['entropy_history'][-1] if bin_data['entropy_history'] else 0.0
        final_entropies[f'len{bin_len}_final_entropy_bits'] = str(final_entropy)
        convergence_info[f'len{bin_len}_n_unique_bins'] = str(len(bin_data['counts']))
    
    # Metadata - core info
    metadata = {
        'prefix': results['prefix'],
        'bin_prefix_lens': str(results['bin_prefix_lens']),
        'samples': str(results['samples_done']),
        'all_converged': str(results['all_converged']),
        'variance_threshold': str(results['variance_threshold']),
        'elapsed_time': str(results['elapsed_time']),
    }
    
    # Sampling parameters
    if max_len is not None:
        metadata['max_len'] = str(max_len)
    if batch_size is not None:
        metadata['batch_size'] = str(batch_size)
    if display_interval is not None:
        metadata['display_interval'] = str(display_interval)
    if use_chat_template is not None:
        metadata['use_chat_template'] = str(use_chat_template)
    if offset is not None:
        metadata['offset'] = str(offset)
    
    # Model/tokenizer info
    if model_name:
        metadata['model_name'] = model_name
    if tokenizer_name:
        metadata['tokenizer_name'] = tokenizer_name
    if actual_prompt:
        metadata['actual_prompt'] = actual_prompt
    if torch_dtype:
        metadata['torch_dtype'] = torch_dtype
    if device_map:
        metadata['device_map'] = device_map
    
    # Environment info
    metadata['transformers_version'] = transformers.__version__
    metadata['torch_version'] = torch.__version__
    
    # Add convergence info and final entropies
    metadata.update(convergence_info)
    metadata.update(final_entropies)
    
    table = pa.Table.from_pydict(data)
    table = table.replace_schema_metadata(metadata)
    
    pq.write_table(table, str(path))
    print(f"Saved results to {path}")
    print(f"  - {len(sequences)} sequences saved")
    print(f"  - Can reconstruct bins at any length from raw sequences")


def estimate_bin_entropy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    bin_prefix_lens: List[int],
    max_samples: int = 10000,
    max_len: int = 100,
    use_chat_template: bool = True,
    device: str = "cuda",
    variance_threshold: float = 1e-4,
    offset: float = 0.0,
    batch_size: int = 128,
    display_interval: int = 128,
    save_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict:
    """
    Estimate empirical bin entropy for multiple bin prefix lengths.
    
    For each bin length, we:
    1. Extract first N tokens from each generation (padding with EOS if needed)
    2. Count frequency of each unique bin
    3. Compute H = -Σ (count/total) * log2(count/total)
    4. Track convergence and stop when variance < threshold
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prefix: Text prefix to condition on (empty string "" = BOS-only)
        bin_prefix_lens: List of bin lengths to track (e.g., [1, 2, 4, 8])
        max_samples: Maximum number of samples (hard stop)
        max_len: Maximum generation length per sample
        use_chat_template: Whether to format with chat template
        device: Device to run on
        variance_threshold: Stop when variance of entropy in second half < threshold
        offset: Cranley-Patterson rotation offset in [0, 1)
        batch_size: Number of sequences to generate in parallel
        display_interval: Update display every N samples (0 to disable)
        save_path: Optional path for saving results (should end in .delm.parquet)
                   Saves raw sequences so any bin length can be reconstructed
        model_name: Model name/identifier to save in metadata
        
    Returns:
        Dict with results for each bin length including entropy history,
        convergence status, and final bin distributions
    """
    
    # Validate parameters
    if not (0.0 <= offset < 1.0):
        raise ValueError("offset must be in [0, 1)")
    
    if display_interval > 0:
        if display_interval < batch_size:
            raise ValueError(f"display_interval ({display_interval}) must be >= batch_size ({batch_size})")
        if display_interval % batch_size != 0:
            raise ValueError(f"display_interval ({display_interval}) must be a multiple of batch_size ({batch_size})")
    
    if not bin_prefix_lens:
        raise ValueError("bin_prefix_lens cannot be empty")
    
    if any(l <= 0 for l in bin_prefix_lens):
        raise ValueError("All bin_prefix_lens must be positive")
    
    # Prepare prefix
    actual_prompt = None
    if prefix == "":
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            prefix_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long)
            actual_prompt = f"<BOS:{tokenizer.bos_token_id}>"
            print(f"Using BOS-only unconditional mode (token_id={tokenizer.bos_token_id})")
        elif hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            prefix_ids = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long)
            actual_prompt = f"<PAD:{tokenizer.pad_token_id}>"
            print(f"WARNING: No BOS token, using PAD token (token_id={tokenizer.pad_token_id})")
        else:
            prefix_ids = tokenizer.encode("", return_tensors="pt", add_special_tokens=True)
            if prefix_ids.shape[1] == 0:
                prefix_ids = tokenizer.encode(" ", return_tensors="pt")
                actual_prompt = " "
                print("WARNING: No BOS/PAD token, using space as starting token")
            else:
                actual_prompt = "<empty_with_special_tokens>"
                print("Using encoded empty string as unconditional start")
    else:
        if use_chat_template:
            messages = [{"role": "user", "content": prefix}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            actual_prompt = prompt
        else:
            prompt = prefix
            actual_prompt = prefix
        
        prefix_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Get EOS token ID
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token ID")
    
    # Generate VdC sequence
    codes = generate_vdc_sequence(max_samples)
    
    # Apply Cranley-Patterson rotation if offset is non-zero
    if offset > 0.0:
        codes = [(code + offset) % 1.0 for code in codes]
    
    # Initialize incremental entropy tracking for each bin length
    bin_data = {}
    for bin_len in bin_prefix_lens:
        bin_data[bin_len] = {
            'tracker': IncrementalEntropyTracker(),
            'converged': False,
            'convergence_sample': None,
        }
    
    # Store all sequences in order for saving
    all_sequences = []
    used_codes = []
    terminated_flags = []
    
    # Get colors for each bin length (consistent throughout)
    bin_colors = get_bin_colors(len(bin_prefix_lens))
    
    # Print header
    if prefix == "":
        print(f"\nSampling sequences unconditionally (BOS-only) with bin entropy estimation")
    else:
        print(f"\nSampling sequences for prefix: '{prefix[:50]}...'")
    print(f"Bin prefix lengths: {bin_prefix_lens}")
    print(f"Max Seq Length    : {max_len}")
    print(f"Max samples       : {max_samples}")
    print(f"Batch size        : {batch_size}")
    print(f"Display interval  : {display_interval if display_interval > 0 else 'disabled (final only)'}")
    print(f"Variance threshold: {variance_threshold}")
    print(f"Offset            : {offset}")
    if save_path:
        print(f"Save path         : {save_path}")
    
    start_time = time.time()
    last_plot_lines = 0
    samples_done = 0
    all_converged = False
    update_counter = 0  # Counter for cycling draw order
    
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
        
        # Process each result and update bin counts
        for i, (tokens, log_prob, sample_info) in enumerate(batch_results):
            samples_done += 1
            
            # Store sequence and code
            all_sequences.append(tokens)
            used_codes.append(batch_codes[i])
            terminated_flags.append(sample_info['terminated_with_eos'])
            
            # Extract bins for each length and update entropy incrementally
            for bin_len in bin_prefix_lens:
                bin_tuple = extract_bin_prefix(tokens, bin_len, eos_token_id)
                tracker = bin_data[bin_len]['tracker']
                tracker.update(bin_tuple)
        
        # Check convergence for each bin length
        if samples_done >= 20:
            all_converged = True
            for bin_len in bin_prefix_lens:
                if not bin_data[bin_len]['converged']:
                    tracker = bin_data[bin_len]['tracker']
                    half_point = samples_done // 2
                    second_half = tracker.entropy_history[half_point:]
                    
                    if len(second_half) > 1:
                        variance = np.var(second_half)
                        if variance < variance_threshold:
                            bin_data[bin_len]['converged'] = True
                            bin_data[bin_len]['convergence_sample'] = samples_done
                        else:
                            all_converged = False
                    else:
                        all_converged = False
            
            if all_converged:
                break
        else:
            all_converged = False
        
        # Display update
        should_display = (display_interval > 0 and samples_done % display_interval == 0)
        
        if should_display:
            elapsed = time.time() - start_time
            samples_per_sec = samples_done / elapsed if elapsed > 0 else 0
            
            # Clear previous display
            if last_plot_lines > 0:
                clear_lines(last_plot_lines)
            
            # Plot combined view with cycled draw order but consistent colors
            plt.clf()
            
            # Cycle the drawing order to show overlapping lines better
            n_lens = len(bin_prefix_lens)
            draw_order = [(i + update_counter) % n_lens for i in range(n_lens)]
            
            for idx in draw_order:
                bin_len = bin_prefix_lens[idx]
                color = bin_colors[idx]  # Color stays with the bin length, not draw order
                tracker = bin_data[bin_len]['tracker']
                history = tracker.entropy_history
                plt.plot(range(1, len(history) + 1), history, label=f"len={bin_len}", color=color)
            
            plt.title("Bin Entropy Convergence - All Lengths")
            plt.xlabel("Sample")
            plt.ylabel("Entropy (bits)")
            plt.plotsize(100, 20)
            plt.show()
            
            last_plot_lines = 20
            
            # Stats line
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            samples_left = max_samples - samples_done
            eta_sec = samples_left / samples_per_sec if samples_per_sec > 0 else 0
            eta_min = int(eta_sec // 60)
            eta_sec_remainder = int(eta_sec % 60)
            
            converged_status = []
            for bin_len in bin_prefix_lens:
                status = "✓" if bin_data[bin_len]['converged'] else "..."
                converged_status.append(f"L{bin_len}:{status}")
            
            print(f"[RUNNING] [{samples_done}/{max_samples}] | " +
                  " ".join(converged_status) +
                  f" | {samples_per_sec:.1f} samp/s | " +
                  f"Elapsed: {elapsed_min}m{elapsed_sec:02d}s | " +
                  f"ETA: {eta_min}m{eta_sec_remainder:02d}s")
            last_plot_lines += 1
            
            update_counter += 1  # Increment for next update
    
    # Final display
    elapsed = time.time() - start_time
    samples_per_sec = samples_done / elapsed if elapsed > 0 else 0
    
    # Clear previous display if it exists
    if display_interval > 0 and last_plot_lines > 0:
        clear_lines(last_plot_lines)
    
    # Final combined plot - USE DEFAULT ORDER
    print("\n" + "="*80)
    print("COMBINED VIEW")
    print("="*80)
    plt.clf()
    for idx, bin_len in enumerate(bin_prefix_lens):
        color = bin_colors[idx]
        tracker = bin_data[bin_len]['tracker']
        history = tracker.entropy_history
        plt.plot(range(1, len(history) + 1), history, label=f"len={bin_len}", color=color)
    plt.title("Bin Entropy Convergence - All Lengths")
    plt.xlabel("Sample")
    plt.ylabel("Entropy (bits)")
    plt.plotsize(100, 20)
    plt.show()
    
    # Final stats
    elapsed_min = int(elapsed // 60)
    elapsed_sec = int(elapsed % 60)
    
    status = "ALL CONVERGED" if all_converged else "COMPLETED"
    print(f"\n[{status}] Samples: {samples_done}/{max_samples} | " +
          f"Speed: {samples_per_sec:.1f} samp/s | " +
          f"Total: {elapsed_min}m{elapsed_sec:02d}s")
    
    print("\nFinal Entropies:")
    for bin_len in bin_prefix_lens:
        tracker = bin_data[bin_len]['tracker']
        final_entropy = tracker.entropy_history[-1] if tracker.entropy_history else 0.0
        n_unique = len(tracker.counts)
        converged = "✓" if bin_data[bin_len]['converged'] else "✗"
        conv_sample = bin_data[bin_len]['convergence_sample'] or "N/A"
        print(f"  len={bin_len:2d}: H={final_entropy:8.3f} bits | " +
              f"Unique bins: {n_unique:6d} | " +
              f"Converged: {converged} (at sample {conv_sample})")
    
    # Package results - convert tracker data to dict format
    results_bin_data = {}
    for bin_len in bin_prefix_lens:
        tracker = bin_data[bin_len]['tracker']
        results_bin_data[bin_len] = {
            'counts': dict(tracker.counts),
            'entropy_history': tracker.entropy_history,
            'converged': bin_data[bin_len]['converged'],
            'convergence_sample': bin_data[bin_len]['convergence_sample'],
        }
    
    results = {
        'prefix': prefix if prefix != "" else "[BOS-only unconditional]",
        'bin_prefix_lens': bin_prefix_lens,
        'samples_done': samples_done,
        'all_converged': all_converged,
        'bin_data': results_bin_data,
        'elapsed_time': elapsed,
        'variance_threshold': variance_threshold,
    }
    
    # Save if requested with full metadata
    if save_path:
        # Get dtype string
        dtype_str = str(next(model.parameters()).dtype) if hasattr(model, 'parameters') else None
        
        # Get device map if it was set
        device_map_str = getattr(model, 'hf_device_map', None)
        if device_map_str is not None:
            device_map_str = str(device_map_str)
        
        save_bin_results(
            results=results,
            sequences=all_sequences,
            codes=used_codes,
            terminated=terminated_flags,
            save_path=save_path,
            model_name=model_name,
            tokenizer_name=tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else None,
            actual_prompt=actual_prompt,
            max_len=max_len,
            batch_size=batch_size,
            display_interval=display_interval,
            use_chat_template=use_chat_template,
            offset=offset,
            torch_dtype=dtype_str,
            device_map=device_map_str,
        )
    
    return results


if __name__ == "__main__":
    # Test with a small model
    print("Loading Qwen2.5-1.5B-Instruct...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Test with multiple bin lengths and save results
    results = estimate_bin_entropy(
        model=model,
        tokenizer=tokenizer,
        prefix="What is the capital of France?",
        bin_prefix_lens=[1, 2, 4, 8],
        max_samples=10000,
        max_len=32,
        batch_size=128,
        display_interval=256,
        variance_threshold=1e-3,
        save_path="results/capital_of_france.delm.parquet",
        model_name=model_name,
    )
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)