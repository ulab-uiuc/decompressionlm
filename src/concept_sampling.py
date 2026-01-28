"""
Concept-based sampling for knowledge extraction from LLMs.

Samples sequences until the number of unique valid concepts reaches a threshold.
Valid concepts must be ASCII alphabetic with length > 1.
"""

import time
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Callable
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotext as plt

from src.concept_utils import extract_concepts_from_sequence, track_concepts, normalize_concept
from src.profiling import ProfileStats, profile_section, format_time, measure_model_memory


def clear_lines(n):
    """Clear n lines from terminal by moving cursor up and clearing."""
    for _ in range(n):
        print("\033[F\033[K", end="")


def sample_concepts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_fn: Callable[[], str],
    concept_threshold: int,
    max_samples: int = 100000,
    max_len: int = 100,
    use_chat_template: bool = True,
    batch_size: int = 128,
    display_interval: int = 64,
    sampling_method: str = "vdc",
    sampling_params: Optional[Dict] = None,
    save_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict:
    """
    Sample sequences until unique valid concept count reaches threshold.
    
    Args:
        model: Language model for sampling
        tokenizer: Tokenizer
        prompt_fn: Function that returns the prompt string
        concept_threshold: Stop when this many unique valid concepts are discovered
        max_samples: Maximum number of sequences to sample
        max_len: Maximum sequence length
        use_chat_template: Whether to use chat template for prompt
        batch_size: Batch size for parallel sampling
        display_interval: Update display every N samples
        sampling_method: "vdc", "random", "beam_low", "beam_high"
        sampling_params: Additional parameters for sampling method
        save_path: Path to save results
        model_name: Model name for metadata
        
    Returns:
        Dict with results and statistics
    """
    # Initialize profiling
    stats = ProfileStats()
    
    # Record model memory (static cost)
    model_mem = measure_model_memory()
    
    # Infer model name if not provided
    if model_name is None and hasattr(tokenizer, "name_or_path"):
        model_name = tokenizer.name_or_path
    
    # Get prompt from function
    prefix = prompt_fn()
    
    # Prepare prefix
    if use_chat_template:
        messages = [{"role": "user", "content": prefix}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        actual_prompt = prompt
    else:
        prompt = prefix
        actual_prompt = prefix
    
    prefix_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token ID")
    
    # Initialize tracking
    valid_concepts = set()
    invalid_concepts = set()
    valid_freq = {}
    invalid_freq = {}
    
    all_sequences = []
    concept_history = [(0, 0)]
    
    # Initialize sampling method
    if sampling_method == "vdc":
        from src.vdc import generate_vdc_sequence
        offset = sampling_params.get("offset", 0.0) if sampling_params else 0.0
        codes = generate_vdc_sequence(max_samples)
        if offset > 0.0:
            codes = [(code + offset) % 1.0 for code in codes]
        
        from src.arithmetic import parallel_arithmetic_sample_batch
        sample_fn = lambda batch_codes: parallel_arithmetic_sample_batch(
            model, tokenizer, prefix_ids, batch_codes, max_len
        )
        
    elif sampling_method == "random":
        from src.baseline_sampling import random_sample_batch
        seed = sampling_params.get("seed", 42) if sampling_params else 42
        codes = list(range(max_samples))
        
        sample_fn = lambda batch_codes: random_sample_batch(
            model, tokenizer, prefix_ids, len(batch_codes), max_len, seed=seed + batch_codes[0]
        )
        
    elif sampling_method in ["beam_low", "beam_high"]:
        from src.baseline_sampling import beam_search_batch
        temp = 0.5 if sampling_method == "beam_low" else 1.5
        num_beams = sampling_params.get("num_beams", 4) if sampling_params else 4
        codes = list(range(max_samples))
        
        sample_fn = lambda batch_codes: beam_search_batch(
            model, tokenizer, prefix_ids, len(batch_codes), max_len, 
            temperature=temp, num_beams=num_beams
        )
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    # Display header
    print(f"\n{'='*80}")
    print("CONCEPT-BASED SAMPLING")
    print(f"{'='*80}")
    print(f"Method           : {sampling_method}")
    print(f"Prompt           : '{prefix[:50]}...'")
    threshold_display = "INFINITY (disabled)" if concept_threshold == 0 else str(concept_threshold)
    print(f"Concept threshold: {threshold_display}")
    print(f"Max samples      : {max_samples}")
    print(f"Max seq length   : {max_len}")
    print(f"Batch size       : {batch_size}")
    print(f"Display interval : {display_interval}")
    if save_path:
        print(f"Save path        : {save_path}")
    print()
    
    # Initial plot
    plt.clf()
    plt.plot([0], [0], color="cyan", label="valid concepts")
    if concept_threshold > 0:
        plt.plot([0, 0.5], [concept_threshold, concept_threshold], color="red", label="threshold")
    plt.title("Valid Concept Discovery")
    plt.xlabel("Sample")
    plt.ylabel("Unique Valid Concepts")
    plt.plotsize(100, 20)
    plt.show()
    
    start_time = time.time()
    samples_done = 0
    threshold_reached = False
    last_display_lines = 20
    cutoff_index = None
    
    # Sample in batches
    for batch_start in range(0, max_samples, batch_size):
        batch_end = min(batch_start + batch_size, max_samples)
        batch_codes = codes[batch_start:batch_end]
        
        with profile_section(stats, "sampling"):
            batch_results = sample_fn(batch_codes)
        
        with profile_section(stats, "concept_extraction"):
            batch_should_break = False
            
            for i, result in enumerate(batch_results):
                if len(result) >= 2:
                    tokens = result[0]
                    info = result[-2] if len(result) >= 3 else result[1]
                else:
                    tokens, info = result, {}
                
                samples_done += 1
                
                valid, invalid = extract_concepts_from_sequence(tokens, tokenizer, eos_token_id)
                
                new_valid, new_invalid = track_concepts(
                    valid, invalid,
                    valid_concepts, invalid_concepts,
                    valid_freq, invalid_freq
                )
                
                if new_valid > 0:
                    with profile_section(stats, "new_concept_discovery"):
                        pass
                
                all_sequences.append({
                    "tokens": tokens,
                    "valid_concepts": valid,
                    "invalid_concepts": invalid,
                    "terminated": info.get("terminated_with_eos", False),
                })
                
                current_count = len(valid_concepts)
                if concept_threshold > 0 and current_count >= concept_threshold and cutoff_index is None:
                    cutoff_index = samples_done - 1
                    threshold_reached = True
                    batch_should_break = True
                    break
        
        concept_history.append((samples_done, len(valid_concepts)))
        
        if batch_should_break:
            break
        
        # Periodic display
        if display_interval > 0 and samples_done % display_interval == 0:
            elapsed = time.time() - start_time
            samples_per_sec = samples_done / elapsed if elapsed > 0 else 0.0
            
            if last_display_lines > 0:
                clear_lines(last_display_lines)
            
            plt.clf()
            x_vals = [x for x, _ in concept_history]
            y_vals = [y for _, y in concept_history]
            
            if concept_threshold > 0:
                plt.plot(x_vals, [concept_threshold] * len(x_vals), color="red", label="threshold")
            plt.plot(x_vals, y_vals, color="cyan", label="valid concepts")
            
            plt.title("Valid Concept Discovery")
            plt.xlabel("Sample")
            plt.ylabel("Unique Valid Concepts")
            plt.plotsize(100, 20)
            plt.show()
            
            status_lines = []
            threshold_display = "INFINITY" if concept_threshold == 0 else str(concept_threshold)
            status_lines.append(
                f"[SAMPLING] {samples_done}/{max_samples} | "
                f"Valid: {len(valid_concepts)}/{threshold_display} | "
                f"Invalid: {len(invalid_concepts)} | "
                f"{samples_per_sec:.1f} samp/s | "
                f"Elapsed: {format_time(elapsed)}"
            )
            
            if concept_threshold > 0 and len(valid_concepts) < concept_threshold and len(concept_history) >= 3:
                recent_window = min(5, len(concept_history))
                recent_samples = [concept_history[i][0] for i in range(-recent_window, 0)]
                recent_counts = [concept_history[i][1] for i in range(-recent_window, 0)]
                
                if len(recent_counts) >= 2:
                    sample_diff = recent_samples[-1] - recent_samples[0]
                    count_diff = recent_counts[-1] - recent_counts[0]
                    
                    if count_diff > 0 and sample_diff > 0:
                        concepts_per_sample = count_diff / sample_diff
                        remaining_concepts = concept_threshold - len(valid_concepts)
                        predicted_samples = remaining_concepts / concepts_per_sample
                        predicted_time = predicted_samples / samples_per_sec if samples_per_sec > 0 else float("inf")
                        
                        total_predicted = samples_done + predicted_samples
                        
                        if total_predicted <= max_samples:
                            status_lines.append(
                                f"  Predicted: {int(predicted_samples)} more samples | "
                                f"ETA: {format_time(predicted_time)}"
                            )
                        else:
                            status_lines.append(
                                f"  ⚠ Predicted: {int(predicted_samples)} samples needed but max is {max_samples}"
                            )
            
            while len(status_lines) < 4:
                status_lines.append("")
            
            for line in status_lines[:4]:
                print(line)
            
            last_display_lines = 24
    
    if cutoff_index is not None:
        all_sequences = all_sequences[:cutoff_index + 1]
        samples_done = len(all_sequences)
    
    elapsed = time.time() - start_time
    samples_per_sec = samples_done / elapsed if elapsed > 0 else 0.0
    
    if last_display_lines > 0:
        clear_lines(last_display_lines)
    
    # Final plot
    plt.clf()
    x_vals = [x for x, _ in concept_history]
    y_vals = [y for _, y in concept_history]
    
    if x_vals:
        plt.plot(x_vals, y_vals, color="cyan", label="valid concepts")
    else:
        plt.plot([0], [0], color="cyan", label="valid concepts")
    
    plt.title("Valid Concept Discovery")
    plt.xlabel("Sample")
    plt.ylabel("Unique Valid Concepts")
    plt.plotsize(100, 20)
    plt.show()
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    threshold_status = "THRESHOLD REACHED" if threshold_reached else ("MAX SAMPLES" if max_samples else "COMPLETE")
    print(f"Status: {threshold_status}")
    print(f"Total samples: {samples_done}")
    threshold_display = "INFINITY (disabled)" if concept_threshold == 0 else str(concept_threshold)
    print(f"Valid concepts: {len(valid_concepts)} (threshold: {threshold_display})")
    print(f"Invalid concepts: {len(invalid_concepts)}")
    print(f"\nValid concept frequencies:")
    print(f"  Total occurrences: {sum(valid_freq.values())}")
    print(f"  Avg occurrences: {sum(valid_freq.values()) / len(valid_freq) if valid_freq else 0:.2f}")
    print(f"\nInvalid concept frequencies:")
    print(f"  Total occurrences: {sum(invalid_freq.values())}")
    print(f"  Avg occurrences: {sum(invalid_freq.values()) / len(invalid_freq) if invalid_freq else 0:.2f}")
    print(f"\nTime: {format_time(elapsed)} ({samples_per_sec:.1f} samp/s)")
    print(f"{'='*80}\n")
    
    stats.print_summary()
    
    results = {
        "sampling_method": sampling_method,
        "sampling_params": sampling_params or {},
        "prefix": prefix,
        "concept_threshold": concept_threshold,
        "samples_done": samples_done,
        "threshold_reached": threshold_reached,
        "cutoff_index": cutoff_index,
        "valid_concepts": len(valid_concepts),
        "invalid_concepts": len(invalid_concepts),
        "valid_freq": valid_freq,
        "invalid_freq": invalid_freq,
        "concept_history": concept_history,
        "sequences": all_sequences,
        "elapsed_time": elapsed,
        "profiling": stats.get_summary(),
        "model_memory": model_mem,
    }
    
    if save_path:
        save_concept_results(
            results,
            save_path,
            model_name=model_name,
            tokenizer_name=tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else None,
            actual_prompt=actual_prompt,
        )
    
    return results


def save_concept_results(
    results: Dict,
    save_path: str,
    model_name: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    actual_prompt: Optional[str] = None,
):
    """Save results to an indented JSON file."""
    path = Path(save_path)
    if not save_path.endswith(".delm.json"):
        save_path = save_path + ".delm.json"
        path = Path(save_path)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        "metadata": {
            "model": model_name,
            "tokenizer": tokenizer_name,
            "prompt": actual_prompt,
            "sampling_method": results['sampling_method'],
            "sampling_params": results['sampling_params'],
            "concept_threshold": results['concept_threshold'],
            "samples_done": results['samples_done'],
            "threshold_reached": results['threshold_reached'],
            "cutoff_index": results['cutoff_index'],
            "elapsed_time": results['elapsed_time'],
        },
        "concept_stats": {
            "valid_concepts_count": results['valid_concepts'],
            "invalid_concepts_count": results['invalid_concepts'],
            "valid_total_occurrences": sum(results['valid_freq'].values()),
            "invalid_total_occurrences": sum(results['invalid_freq'].values()),
        },
        "valid_concepts": [
            {"concept": concept, "frequency": freq}
            for concept, freq in sorted(results['valid_freq'].items(), key=lambda x: x[1], reverse=True)
        ],
        "invalid_concepts": [
            {"concept": concept, "frequency": freq}
            for concept, freq in sorted(results['invalid_freq'].items(), key=lambda x: x[1], reverse=True)[:64]
        ],
        "profiling": results.get('profiling', {}),
        "model_memory": results.get('model_memory', {}),
        "concept_history": [
            {"sample": s, "concepts": c}
            for s, c in results['concept_history']
        ],
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Saved results to {path}")
