"""
Prefix mass-based sampling for estimating effective support size.

Sample sequences until cumulative probability mass of discovered prefix patterns
exceeds a threshold, then compute effective support set after deduplication.
"""

import time
import torch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotext as plt
import transformers
import Levenshtein  # pip install python-Levenshtein
import tqdm

from src.vdc import generate_vdc_sequence
from src.arithmetic import parallel_arithmetic_sample_batch
from src.plot_utils import get_bin_colors


def clear_lines(n):
    """Clear n lines from terminal by moving cursor up and clearing."""
    for _ in range(n):
        print('\033[F\033[K', end='')


def extract_prefix(tokens: List[int], prefix_len: int, eos_token_id: int) -> tuple:
    """
    Extract prefix of specified length, padding with EOS if needed.
    
    Args:
        tokens: Generated token IDs (excluding EOS at end if terminated)
        prefix_len: Length of prefix
        eos_token_id: EOS token ID for padding
        
    Returns:
        Tuple of exactly prefix_len token IDs
    """
    if len(tokens) >= prefix_len:
        return tuple(tokens[:prefix_len])
    else:
        # Pad with EOS tokens
        return tuple(tokens + [eos_token_id] * (prefix_len - len(tokens)))


def compute_prefix_probabilities(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    prefix_patterns: Set[tuple],
    device: str = "cuda",
    max_batch_size: int = 128,
) -> Dict[tuple, float]:
    """
    Compute exact probabilities for a set of prefix patterns.
    Now correctly handles each pattern's own history in parallel.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt_ids: Prompt token IDs [1, seq_len]
        prefix_patterns: Set of prefix tuples to compute probs for
        device: Device to run on
        max_batch_size: Maximum batch size for processing (to control memory)
        
    Returns:
        Dict mapping prefix tuple -> probability
    """
    if not prefix_patterns:
        return {}
    
    model.eval()
    prefix_probs = {}
    
    # Group prefixes by length for efficiency
    by_length = defaultdict(list)
    for pattern in prefix_patterns:
        by_length[len(pattern)].append(pattern)
    
    # Process each length group
    for length, patterns in by_length.items():
        # Process in chunks if needed to control memory
        for chunk_start in range(0, len(patterns), max_batch_size):
            chunk_end = min(chunk_start + max_batch_size, len(patterns))
            chunk_patterns = patterns[chunk_start:chunk_end]
            batch_size = len(chunk_patterns)
            
            with torch.no_grad():
                # Repeat prompt for batch - each row will follow its own pattern
                input_ids = prompt_ids.repeat(batch_size, 1).to(device)  # [B, prompt_len]
                
                # Initialize log probs accumulator for each pattern
                log_probs_accum = torch.zeros(batch_size, device=device)
                
                # For each position in the prefix
                for pos in range(length):
                    # Forward pass for entire batch
                    outputs = model(input_ids)
                    logits = outputs.logits[:, -1, :]  # [B, vocab_size]
                    log_probs = torch.log_softmax(logits, dim=-1)  # [B, vocab_size]
                    
                    # Gather log prob for each pattern's token at this position
                    token_ids = torch.tensor(
                        [pattern[pos] for pattern in chunk_patterns],
                        device=device
                    )  # [B]
                    
                    # Get log prob for each pattern's specific token
                    token_log_probs = log_probs[torch.arange(batch_size, device=device), token_ids]  # [B]
                    log_probs_accum += token_log_probs
                    
                    # Append each pattern's own token for next iteration
                    # This is the KEY FIX: each row gets its own token, not pattern[0]'s token
                    if pos < length - 1:
                        input_ids = torch.cat([
                            input_ids,
                            token_ids.unsqueeze(1)
                        ], dim=1)  # [B, prompt_len + pos + 1]
            
            # Convert log probs to probs and store
            probs = torch.exp(log_probs_accum).cpu().numpy()
            for i, pattern in enumerate(chunk_patterns):
                prefix_probs[pattern] = float(probs[i])
    
    return prefix_probs


def tokens_to_string(tokens: List[int], tokenizer: AutoTokenizer) -> str:
    """Convert token IDs to string for comparison."""
    return tokenizer.decode(tokens, skip_special_tokens=True)


def compute_sequence_similarity(seq1: List[int], seq2: List[int], tokenizer: AutoTokenizer) -> float:
    """
    Compute similarity between two sequences using normalized edit distance.
    
    Returns:
        Similarity score in [0, 1] where 1 is identical
    """
    # Convert to strings for comparison
    str1 = tokens_to_string(seq1, tokenizer)
    str2 = tokens_to_string(seq2, tokenizer)
    
    # Levenshtein distance
    dist = Levenshtein.distance(str1, str2)
    max_len = max(len(str1), len(str2))
    
    if max_len == 0:
        return 1.0
    
    # Normalize to [0, 1] similarity
    similarity = 1.0 - (dist / max_len)
    return similarity


def cluster_sequences(
    sequences: List[List[int]],
    tokenizer: AutoTokenizer,
    similarity_threshold: float = 0.85,
) -> List[int]:
    """
    Cluster similar sequences and return representative indices.
    
    Uses greedy clustering: iterate through sequences, add to cluster if
    sufficiently similar to any existing representative, otherwise start new cluster.
    
    Args:
        sequences: List of token ID sequences
        tokenizer: Tokenizer for decoding
        similarity_threshold: Minimum similarity to be considered duplicate
        
    Returns:
        List of indices into sequences representing unique clusters
    """
    if not sequences:
        return []
    
    representatives = [0]  # Start with first sequence
    
    for i in tqdm.tqdm(range(1, len(sequences)), desc="Deduplicating", unit="seq"):
        is_duplicate = False
        
        # Check against all representatives
        for rep_idx in representatives:
            sim = compute_sequence_similarity(sequences[i], sequences[rep_idx], tokenizer)
            if sim >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            representatives.append(i)
    
    return representatives


def save_mass_results(
    sequences: List[List[int]],
    codes: List[float],
    terminated: List[bool],
    prefix_probs: Dict[tuple, float],
    discovered_prefixes: List[tuple],
    mass_history: List[float],
    effective_set_indices: List[int],
    save_path: str,
    model_name: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    actual_prompt: Optional[str] = None,
    prefix_len: Optional[int] = None,
    prob_threshold: Optional[float] = None,
    similarity_threshold: Optional[float] = None,
    max_len: Optional[int] = None,
    batch_size: Optional[int] = None,
    offset: Optional[float] = None,
    total_mass: Optional[float] = None,
    n_unique_prefixes: Optional[int] = None,
    elapsed_time: Optional[float] = None,
):
    """
    Save mass-based sampling results to .delm.parquet file.
    
    Saves all sequences with a flag indicating if they're in the effective support set.
    """
    path = Path(save_path)
    
    if not save_path.endswith('.delm.parquet'):
        save_path = save_path + '.delm.parquet'
        path = Path(save_path)
        print(f"Note: Auto-adding .delm.parquet extension")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create effective set flags
    effective_set_flags = [False] * len(sequences)
    for idx in effective_set_indices:
        effective_set_flags[idx] = True
    
    # Convert sequences to Arrow
    sequence_array = pa.array(sequences, type=pa.list_(pa.uint32()))
    
    # Create data table
    data = {
        'code': pa.array(codes, type=pa.float64()),
        'sequence': sequence_array,
        'sequence_length': pa.array([len(seq) for seq in sequences], type=pa.int32()),
        'terminated': pa.array(terminated, type=pa.bool_()),
        'in_effective_set': pa.array(effective_set_flags, type=pa.bool_()),
    }
    
    # Metadata
    metadata = {
        'sampling_mode': 'prefix_mass',
        'total_sequences': str(len(sequences)),
        'effective_set_size': str(len(effective_set_indices)),
    }
    
    if prefix_len is not None:
        metadata['prefix_len'] = str(prefix_len)
    if prob_threshold is not None:
        metadata['prob_threshold'] = str(prob_threshold)
    if similarity_threshold is not None:
        metadata['similarity_threshold'] = str(similarity_threshold)
    if total_mass is not None:
        metadata['final_prefix_mass'] = str(total_mass)
    if n_unique_prefixes is not None:
        metadata['unique_prefixes_discovered'] = str(n_unique_prefixes)
    if elapsed_time is not None:
        metadata['elapsed_time'] = str(elapsed_time)
    
    # Sampling parameters
    if max_len is not None:
        metadata['max_len'] = str(max_len)
    if batch_size is not None:
        metadata['batch_size'] = str(batch_size)
    if offset is not None:
        metadata['offset'] = str(offset)
    
    # Model info
    if model_name:
        metadata['model_name'] = model_name
    if tokenizer_name:
        metadata['tokenizer_name'] = tokenizer_name
    if actual_prompt:
        metadata['actual_prompt'] = actual_prompt
    
    # Effective set statistics
    if effective_set_indices:
        eff_sequences = [sequences[i] for i in effective_set_indices]
        eff_lengths = [len(seq) for seq in eff_sequences]
        metadata['effective_set_min_tokens'] = str(min(eff_lengths))
        metadata['effective_set_max_tokens'] = str(max(eff_lengths))
        metadata['effective_set_avg_tokens'] = str(np.mean(eff_lengths))
        metadata['effective_set_total_tokens'] = str(sum(eff_lengths))
    
    # Environment
    metadata['transformers_version'] = transformers.__version__
    metadata['torch_version'] = torch.__version__
    
    table = pa.Table.from_pydict(data)
    table = table.replace_schema_metadata(metadata)
    
    pq.write_table(table, str(path))
    print(f"Saved results to {path}")


def estimate_prefix_mass(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    prefix_len: int,
    prob_threshold: float = 0.9,
    similarity_threshold: float = 0.85,
    max_samples: int = 100000,
    max_len: int = 100,
    use_chat_template: bool = True,
    device: str = "cuda",
    offset: float = 0.0,
    batch_size: int = 128,
    display_interval: int = 64,
    save_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict:
    """
    Sample sequences until cumulative prefix probability mass exceeds threshold.
    
    Algorithm:
    1. Sample sequences using VdC codes
    2. Extract prefix of length prefix_len from each sequence
    3. Track discovered prefixes and their probabilities
    4. Stop when sum of discovered prefix probs > prob_threshold
    5. Cluster sequences by similarity to get effective support set
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prefix: Text prefix to condition on
        prefix_len: Length of prefix pattern to track
        prob_threshold: Stop when cumulative prefix mass > this (e.g., 0.9)
        similarity_threshold: Similarity threshold for deduplication (e.g., 0.85)
        max_samples: Hard stop limit
        max_len: Maximum generation length per sample
        use_chat_template: Whether to format with chat template
        device: Device to run on
        offset: Cranley-Patterson rotation offset
        batch_size: Batch size for parallel sampling
        display_interval: Update display every N samples (default 64)
        save_path: Optional path to save results
        model_name: Model name for metadata
        
    Returns:
        Dict with results including effective support set statistics
    """
    # Validate
    if not (0.0 <= offset < 1.0):
        raise ValueError("offset must be in [0, 1)")
    if not (0.0 < prob_threshold <= 1.0):
        raise ValueError("prob_threshold must be in (0, 1]")
    if not (0.0 < similarity_threshold <= 1.0):
        raise ValueError("similarity_threshold must be in (0, 1]")
    if prefix_len <= 0:
        raise ValueError("prefix_len must be positive")
    
    # Prepare prompt
    actual_prompt = None
    if prefix == "":
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            prefix_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long)
            actual_prompt = f"<BOS:{tokenizer.bos_token_id}>"
        else:
            raise ValueError("Empty prefix requires BOS token")
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
    
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token ID")
    
    # Generate VdC codes
    codes = generate_vdc_sequence(max_samples)
    if offset > 0.0:
        codes = [(code + offset) % 1.0 for code in codes]
    
    # Tracking
    discovered_prefixes: Set[tuple] = set()
    prefix_first_seen: Dict[tuple, int] = {}  # Map prefix -> sample index
    all_sequences: List[List[int]] = []
    used_codes: List[float] = []
    terminated_flags: List[bool] = []
    mass_history: List[float] = [(0, 0.0)]
    
    # Print header
    print(f"\n{'='*80}")
    print(f"PREFIX MASS SAMPLING")
    print(f"{'='*80}")
    print(f"Prompt: '{prefix[:50]}...'")
    print(f"Prefix length    : {prefix_len}")
    print(f"Prob threshold   : {prob_threshold}")
    print(f"Similarity thresh: {similarity_threshold}")
    print(f"Max samples      : {max_samples}")
    print(f"Max seq length   : {max_len}")
    print(f"Batch size       : {batch_size}")
    print(f"Display interval : {display_interval}")
    print(f"Offset           : {offset}")
    if save_path:
        print(f"Save path        : {save_path}")
    print()
    
    start_time = time.time()
    samples_done = 0
    current_mass = 0.0
    threshold_reached = False
    last_display_lines = 0
    
    # Process in batches
    for batch_start in range(0, max_samples, batch_size):
        batch_end = min(batch_start + batch_size, max_samples)
        batch_codes = codes[batch_start:batch_end]
        
        # Generate batch
        batch_results = parallel_arithmetic_sample_batch(
            model=model,
            tokenizer=tokenizer,
            prefix_ids=prefix_ids,
            codes=batch_codes,
            max_len=max_len,
            device=device
        )
        
        # Process results
        newly_discovered = []
        for i, (tokens, log_prob, sample_info) in enumerate(batch_results):
            samples_done += 1
            
            # Store sequence
            all_sequences.append(tokens)
            used_codes.append(batch_codes[i])
            terminated_flags.append(sample_info['terminated_with_eos'])
            
            # Extract prefix
            prefix_tuple = extract_prefix(tokens, prefix_len, eos_token_id)
            
            # Track if new
            if prefix_tuple not in discovered_prefixes:
                discovered_prefixes.add(prefix_tuple)
                prefix_first_seen[prefix_tuple] = samples_done
                newly_discovered.append(prefix_tuple)
        
        # Update mass if we found new prefixes
        if newly_discovered:
            # Compute exact probabilities for newly discovered prefixes
            new_probs = compute_prefix_probabilities(
                model, tokenizer, prefix_ids, set(newly_discovered), device
            )
            
            # Update total mass
            for pattern in newly_discovered:
                current_mass += new_probs.get(pattern, 0.0)
        
        mass_history.append((samples_done, current_mass))
        
        # Display update - CHECK THIS FIRST (before threshold check)
        if display_interval > 0 and samples_done % display_interval == 0:
            elapsed = time.time() - start_time
            samples_per_sec = samples_done / elapsed if elapsed > 0 else 0
            
            if last_display_lines > 0:
                clear_lines(last_display_lines)
            
            # Plot mass convergence
            plt.clf()
            x_vals = [x for x, _ in mass_history]
            y_vals = [y for _, y in mass_history]
            
            # If we're still below half the threshold, plot the smallest 1/2^k threshold above current max
            y_max = max(y_vals) if y_vals else 0.0
            half_thresh = prob_threshold / 2.0
            
            if y_max < half_thresh:
                # Show ONLY the 1/2**k guide line
                if y_max > 0.0:
                    # Choose k so that threshold/2**k is the smallest value strictly > y_max
                    k = int(np.floor(np.log2(prob_threshold / y_max)))
                    if (prob_threshold / (2 ** k)) <= y_max:
                        k -= 1
                    k = max(k, 1)
                else:
                    # If y_max == 0, just use 1/2**1 as a sane first guide
                    k = 1
            
                guide = prob_threshold / (2 ** k)
                plt.plot(
                    x_vals,
                    [guide] * len(x_vals),
                    color='yellow',
                    label=f"1/2**{k} threshold"
                )
            else:
                # Show ONLY the full threshold line
                plt.plot(
                    x_vals,
                    [prob_threshold] * len(x_vals),
                    color='red',
                    label='threshold'
                )
            
            # Mass curve LAST (on top)
            plt.plot(
                x_vals,
                y_vals,
                color='cyan',
                label='cumulative mass'
            )
            
            plt.title(f"Cumulative Prefix Mass (len={prefix_len})")
            plt.xlabel("Sample")
            plt.ylabel("Probability Mass")
            plt.plotsize(100, 20)
            plt.show()
            
            last_display_lines = 20
            
            # Status line with ETA predictions
            status_lines = []
            status_lines.append(
                f"[SAMPLING] {samples_done}/{max_samples} | " +
                f"Mass: {current_mass:.4f}/{prob_threshold} | " +
                f"Unique: {len(discovered_prefixes)} | " +
                f"{samples_per_sec:.1f} samp/s"
            )
            
            # Predict ETA to threshold based on mass accumulation rate
            if len(mass_history) >= 3 and current_mass > 0:
                # Calculate mass accumulation rate from recent history
                recent_window = min(5, len(mass_history))
                recent_samples = [mass_history[i][0] for i in range(-recent_window, 0)]
                recent_masses = [mass_history[i][1] for i in range(-recent_window, 0)]
                
                if len(recent_masses) >= 2:
                    sample_diff = recent_samples[-1] - recent_samples[0]
                    mass_diff = recent_masses[-1] - recent_masses[0]
                    
                    if mass_diff > 0 and sample_diff > 0:
                        mass_per_sample = mass_diff / sample_diff
                        remaining_mass = prob_threshold - current_mass
                        predicted_samples = remaining_mass / mass_per_sample
                        predicted_time = predicted_samples / samples_per_sec if samples_per_sec > 0 else float('inf')
                        
                        pred_eta_min = int(predicted_time // 60)
                        pred_eta_sec = int(predicted_time % 60)
                        
                        # Check if we'll exceed max_samples
                        total_predicted_samples = samples_done + predicted_samples
                        
                        if total_predicted_samples <= max_samples:
                            status_lines.append(
                                f"  Predicted: {int(predicted_samples)} more samples needed | " +
                                f"ETA to threshold: {pred_eta_min}m{pred_eta_sec:02d}s"
                            )
                        else:
                            samples_at_max = max_samples - samples_done
                            time_at_max = samples_at_max / samples_per_sec if samples_per_sec > 0 else 0
                            max_eta_min = int(time_at_max // 60)
                            max_eta_sec = int(time_at_max % 60)
                            
                            predicted_mass_at_max = current_mass + (mass_per_sample * samples_at_max)
                            
                            status_lines.append(
                                f"  ⚠ Predicted: {int(predicted_samples)} samples needed but max is {max_samples}"
                            )
                            status_lines.append(
                                f"  At max_samples: mass ≈ {predicted_mass_at_max:.4f} | " +
                                f"ETA to max: {max_eta_min}m{max_eta_sec:02d}s"
                            )
                            status_lines.append(
                                f"  Would need {int(predicted_samples)} samples | " +
                                f"Would take ≈ {pred_eta_min}m{pred_eta_sec:02d}s total"
                            )
            
            for line in status_lines:
                print(line)
            last_display_lines += len(status_lines)
        
        # Check threshold - CHECK THIS SECOND (after display)
        if current_mass >= prob_threshold:
            threshold_reached = True
            
            # Clear the last display before showing completion message
            # if last_display_lines > 0:
            #     clear_lines(last_display_lines)
            #     last_display_lines = 0
            
            print(f"\n{'='*80}")
            print(f"THRESHOLD REACHED at sample {samples_done}")
            print(f"Cumulative prefix mass: {current_mass:.6f} >= {prob_threshold}")
            print(f"{'='*80}\n")
            break
    
    elapsed = time.time() - start_time
    
    # Compute exact probabilities for all discovered prefixes
    print(f"Computing exact probabilities for {len(discovered_prefixes)} unique prefixes...")
    prefix_probs = compute_prefix_probabilities(
        model, tokenizer, prefix_ids, discovered_prefixes, device
    )
    
    # Recompute final mass from exact probabilities
    final_mass = sum(prefix_probs.values())
    
    # Cluster sequences to get effective support set
    print(f"\nClustering {len(all_sequences)} sequences (similarity >= {similarity_threshold})...")
    effective_set_indices = cluster_sequences(all_sequences, tokenizer, similarity_threshold)
    
    # Compute statistics
    effective_sequences = [all_sequences[i] for i in effective_set_indices]
    effective_lengths = [len(seq) for seq in effective_sequences]
    
    # Final display (no need to clear lines here since we already did)
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Status: {'THRESHOLD REACHED' if threshold_reached else 'MAX SAMPLES'}")
    print(f"Total samples: {samples_done}")
    print(f"Unique prefixes (len={prefix_len}): {len(discovered_prefixes)}")
    print(f"Final prefix mass: {final_mass:.6f}")
    print(f"\nEffective Support Set:")
    print(f"  Size: {len(effective_set_indices)} sequences")
    print(f"  Total tokens: {sum(effective_lengths)}")
    print(f"  Min tokens: {min(effective_lengths)}")
    print(f"  Max tokens: {max(effective_lengths)}")
    print(f"  Avg tokens: {np.mean(effective_lengths):.1f}")
    print(f"\nTime: {elapsed:.1f}s ({samples_done/elapsed:.1f} samp/s)")
    print(f"{'='*80}\n")
    
    # Package results
    results = {
        'prefix': prefix,
        'prefix_len': prefix_len,
        'prob_threshold': prob_threshold,
        'similarity_threshold': similarity_threshold,
        'samples_done': samples_done,
        'threshold_reached': threshold_reached,
        'unique_prefixes': len(discovered_prefixes),
        'final_mass': final_mass,
        'effective_set_size': len(effective_set_indices),
        'effective_set_indices': effective_set_indices,
        'effective_set_stats': {
            'total_tokens': sum(effective_lengths),
            'min_tokens': min(effective_lengths) if effective_lengths else 0,
            'max_tokens': max(effective_lengths) if effective_lengths else 0,
            'avg_tokens': np.mean(effective_lengths) if effective_lengths else 0,
        },
        'elapsed_time': elapsed,
    }
    
    # Save if requested
    if save_path:
        save_mass_results(
            sequences=all_sequences,
            codes=used_codes,
            terminated=terminated_flags,
            prefix_probs=prefix_probs,
            discovered_prefixes=list(discovered_prefixes),
            mass_history=mass_history,
            effective_set_indices=effective_set_indices,
            save_path=save_path,
            model_name=model_name,
            tokenizer_name=tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else None,
            actual_prompt=actual_prompt,
            prefix_len=prefix_len,
            prob_threshold=prob_threshold,
            similarity_threshold=similarity_threshold,
            max_len=max_len,
            batch_size=batch_size,
            offset=offset,
            total_mass=final_mass,
            n_unique_prefixes=len(discovered_prefixes),
            elapsed_time=elapsed,
        )
    
    return results


if __name__ == "__main__":
    # Test
    print("Loading Qwen2.5-1.5B-Instruct...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )
    
    # Test prefix mass sampling
    results = estimate_prefix_mass(
        model=model,
        tokenizer=tokenizer,
        prefix="What is the capital of France?",
        prefix_len=4,
        prob_threshold=0.9,
        similarity_threshold=0.85,
        max_samples=10000,
        max_len=32,
        batch_size=128,
        display_interval=128,
        save_path="results/capital_mass.delm.parquet",
        model_name=model_name,
    )
    
    print("\nTEST COMPLETE")