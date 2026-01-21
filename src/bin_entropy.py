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
import math

from src.vdc import generate_vdc_sequence
from src.arithmetic import parallel_arithmetic_sample_batch
from src.plot_utils import get_bin_colors


def clear_lines(n):
    """Clear n lines from terminal by moving cursor up and clearing."""
    for _ in range(n):
        print("\033[F\033[K", end="")


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


def load_and_validate_results(
    save_path: str,
    prefix: str,
    prefix_len: int,
    prob_threshold: float,
    max_samples: int,
    max_len: int,
    use_chat_template: bool,
    offset: float,
    batch_size: int,
    model_name: Optional[str] = None,
) -> Optional[Tuple[Dict, Dict, List]]:
    """
    Load existing results and validate parameters match.

    Returns:
        (data_dict, metadata_dict, mass_history) if valid, None if file doesn't exist,
        raises ValueError if parameters don't match
    """
    path = Path(save_path)
    if not save_path.endswith(".delm.parquet"):
        path = Path(save_path + ".delm.parquet")

    if not path.exists():
        return None

    print(f"Found existing results at {path}")
    print("Validating parameters...")

    table = pq.read_table(str(path))
    metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}

    data = {
        "codes": table["code"].to_pylist(),
        "sequences": table["sequence"].to_pylist(),
        "lengths": table["sequence_length"].to_pylist(),
        "terminated": table["terminated"].to_pylist(),
        "in_effective_set": table["in_effective_set"].to_pylist(),
    }

    import json

    mass_history = []
    if "mass_history_samples" in metadata and "mass_history_masses" in metadata:
        samples = json.loads(metadata["mass_history_samples"])
        masses = json.loads(metadata["mass_history_masses"])
        mass_history = list(zip(samples, masses))

    mismatches = []

    def check_param(name, expected, metadata_key=None):
        if metadata_key is None:
            metadata_key = name
        if metadata_key in metadata:
            stored = metadata[metadata_key]
            if isinstance(expected, bool):
                stored_val = stored.lower() == "true"
            elif isinstance(expected, (int, float)):
                stored_val = type(expected)(stored)
            else:
                stored_val = stored
            if stored_val != expected:
                mismatches.append(f"  {name}: expected {expected}, got {stored_val}")

    check_param("prefix_len", prefix_len)
    check_param("prob_threshold", prob_threshold)
    check_param("max_samples", max_samples)
    check_param("max_len", max_len)
    check_param("use_chat_template", use_chat_template)
    check_param("offset", offset)
    check_param("batch_size", batch_size)

    if model_name and "model_name" in metadata:
        if metadata["model_name"] != model_name:
            mismatches.append(f"  model_name: expected {model_name}, got {metadata['model_name']}")

    if "actual_prompt" in metadata:
        if prefix == "" and not metadata["actual_prompt"].startswith("<BOS:"):
            mismatches.append(f"  prompt: expected BOS-only, got {metadata['actual_prompt'][:50]}...")
        elif prefix != "" and use_chat_template:
            if prefix not in metadata["actual_prompt"]:
                mismatches.append(
                    f"  prompt: expected to contain '{prefix[:30]}...', got '{metadata['actual_prompt'][:50]}...'"
                )

    if mismatches:
        print("\n" + "=" * 80)
        print("PARAMETER MISMATCH ERROR")
        print("=" * 80)
        print("The existing results file has different parameters:")
        for m in mismatches:
            print(m)
        print("\nStored metadata:")
        for key in [
            "prefix_len",
            "prob_threshold",
            "max_samples",
            "max_len",
            "use_chat_template",
            "offset",
            "batch_size",
            "model_name",
            "actual_prompt",
        ]:
            if key in metadata:
                val = metadata[key]
                if key == "actual_prompt" and len(val) > 60:
                    val = val[:60] + "..."
                print(f"  {key}: {val}")
        print("=" * 80)
        raise ValueError("Parameter mismatch with existing results file")

    print("✓ All parameters match!")
    return data, metadata, mass_history


def display_results_from_file(
    data: Dict,
    metadata: Dict,
    mass_history: List[Tuple[int, float]],
):
    """
    Display results in the same format as the live sampling output.
    """
    print(f"\n{'='*80}")
    print("LOADED RESULTS FROM FILE")
    print(f"{'='*80}")

    if mass_history:
        plt.clf()
        x_vals = [x for x, _ in mass_history]
        y_vals = [y for _, y in mass_history]

        prob_threshold = float(metadata.get("prob_threshold", 0.9))
        prefix_len = int(metadata.get("prefix_len", 4))

        plt.plot(x_vals, [prob_threshold] * len(x_vals), color="red", label="threshold")
        plt.plot(x_vals, y_vals, color="cyan", label="cumulative mass")

        plt.title(f"Cumulative Prefix Mass (len={prefix_len})")
        plt.xlabel("Sample")
        plt.ylabel("Probability Mass")
        plt.plotsize(100, 20)
        plt.show()

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    threshold_reached = metadata.get("threshold_reached", "Unknown").lower() == "true"
    print(f"Status: {'THRESHOLD REACHED' if threshold_reached else 'MAX SAMPLES'}")
    print(f"Total samples: {metadata.get('total_sequences', 'N/A')}")
    print(f"Unique prefixes (len={metadata.get('prefix_len', 'N/A')}): {metadata.get('unique_prefixes_discovered', 'N/A')}")
    print(f"Final prefix mass: {metadata.get('final_prefix_mass', 'N/A')}")

    print(f"\nEffective Support Set:")
    print(f"  Size: {metadata.get('effective_set_size', 'N/A')} sequences")
    print(f"  Total tokens: {metadata.get('effective_set_total_tokens', 'N/A')}")
    print(f"  Min tokens: {metadata.get('effective_set_min_tokens', 'N/A')}")
    print(f"  Max tokens: {metadata.get('effective_set_max_tokens', 'N/A')}")
    print(f"  Avg tokens: {metadata.get('effective_set_avg_tokens', 'N/A')}")

    elapsed = metadata.get("elapsed_time", None)
    if elapsed:
        elapsed_val = float(elapsed)
        samples_done = int(metadata.get("total_sequences", 0))
        print(f"\nTime: {elapsed_val:.1f}s ({samples_done/elapsed_val:.1f} samp/s)")

    print(f"{'='*80}\n")


def save_mass_results(
    sequences: List[List[int]],
    codes: List[float],
    terminated: List[bool],
    prefix_probs: Dict[tuple, float],
    discovered_prefixes: List[tuple],
    mass_history: List[Tuple[int, float]],
    effective_set_indices: List[int],
    save_path: str,
    model_name: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    actual_prompt: Optional[str] = None,
    prefix_len: Optional[int] = None,
    prob_threshold: Optional[float] = None,
    max_len: Optional[int] = None,
    batch_size: Optional[int] = None,
    offset: Optional[float] = None,
    total_mass: Optional[float] = None,
    n_unique_prefixes: Optional[int] = None,
    elapsed_time: Optional[float] = None,
    use_chat_template: Optional[bool] = None,
    threshold_reached: Optional[bool] = None,
    display_interval: Optional[int] = None,
    max_samples: Optional[int] = None,
):
    """
    Save mass-based sampling results to .delm.parquet file.

    Saves all sequences with a flag indicating if they're in the effective support set.
    Also saves mass_history as separate arrays for reconstruction.
    """
    path = Path(save_path)

    if not save_path.endswith(".delm.parquet"):
        save_path = save_path + ".delm.parquet"
        path = Path(save_path)
        print("Note: Auto-adding .delm.parquet extension")

    path.parent.mkdir(parents=True, exist_ok=True)

    effective_set_flags = [False] * len(sequences)
    for idx in effective_set_indices:
        effective_set_flags[idx] = True

    sequence_array = pa.array(sequences, type=pa.list_(pa.uint32()))

    if mass_history:
        mass_history_samples = [x for x, _ in mass_history]
        mass_history_masses = [y for _, y in mass_history]
    else:
        mass_history_samples = []
        mass_history_masses = []

    data = {
        "code": pa.array(codes, type=pa.float64()),
        "sequence": sequence_array,
        "sequence_length": pa.array([len(seq) for seq in sequences], type=pa.int32()),
        "terminated": pa.array(terminated, type=pa.bool_()),
        "in_effective_set": pa.array(effective_set_flags, type=pa.bool_()),
    }

    metadata = {
        "sampling_mode": "prefix_mass",
        "total_sequences": str(len(sequences)),
        "effective_set_size": str(len(effective_set_indices)),
    }

    if prefix_len is not None:
        metadata["prefix_len"] = str(prefix_len)
    if prob_threshold is not None:
        metadata["prob_threshold"] = str(prob_threshold)
    if max_len is not None:
        metadata["max_len"] = str(max_len)
    if batch_size is not None:
        metadata["batch_size"] = str(batch_size)
    if offset is not None:
        metadata["offset"] = str(offset)
    if max_samples is not None:
        metadata["max_samples"] = str(max_samples)
    if display_interval is not None:
        metadata["display_interval"] = str(display_interval)
    if use_chat_template is not None:
        metadata["use_chat_template"] = str(use_chat_template)

    if total_mass is not None:
        metadata["final_prefix_mass"] = str(total_mass)
    if n_unique_prefixes is not None:
        metadata["unique_prefixes_discovered"] = str(n_unique_prefixes)
    if elapsed_time is not None:
        metadata["elapsed_time"] = str(elapsed_time)
    if threshold_reached is not None:
        metadata["threshold_reached"] = str(threshold_reached)

    if model_name:
        metadata["model_name"] = model_name
    if tokenizer_name:
        metadata["tokenizer_name"] = tokenizer_name
    if actual_prompt:
        metadata["actual_prompt"] = actual_prompt

    if effective_set_indices:
        eff_sequences = [sequences[i] for i in effective_set_indices]
        eff_lengths = [len(seq) for seq in eff_sequences]
        metadata["effective_set_min_tokens"] = str(min(eff_lengths))
        metadata["effective_set_max_tokens"] = str(max(eff_lengths))
        metadata["effective_set_avg_tokens"] = str(np.mean(eff_lengths))
        metadata["effective_set_total_tokens"] = str(sum(eff_lengths))

    import json

    metadata["mass_history_samples"] = json.dumps(mass_history_samples)
    metadata["mass_history_masses"] = json.dumps(mass_history_masses)

    metadata["transformers_version"] = transformers.__version__
    metadata["torch_version"] = torch.__version__

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

    If save_path exists and parameters match, loads and returns existing results.
    If parameters don't match, raises ValueError.
    """
    if save_path:
        existing = load_and_validate_results(
            save_path=save_path,
            prefix=prefix,
            prefix_len=prefix_len,
            prob_threshold=prob_threshold,
            max_samples=max_samples,
            max_len=max_len,
            use_chat_template=use_chat_template,
            offset=offset,
            batch_size=batch_size,
            model_name=model_name,
        )

        if existing is not None:
            data, metadata, mass_history = existing
            display_results_from_file(data, metadata, mass_history)

            return {
                "prefix": prefix,
                "prefix_len": int(metadata.get("prefix_len", prefix_len)),
                "prob_threshold": float(metadata.get("prob_threshold", prob_threshold)),
                "samples_done": int(metadata.get("total_sequences", 0)),
                "threshold_reached": metadata.get("threshold_reached", "False").lower() == "true",
                "unique_prefixes": int(metadata.get("unique_prefixes_discovered", 0)),
                "final_mass": float(metadata.get("final_prefix_mass", 0.0)),
                "effective_set_size": int(metadata.get("effective_set_size", 0)),
                "effective_set_indices": [i for i, flag in enumerate(data["in_effective_set"]) if flag],
                "effective_set_stats": {
                    "total_tokens": int(metadata.get("effective_set_total_tokens", 0)),
                    "min_tokens": int(metadata.get("effective_set_min_tokens", 0)),
                    "max_tokens": int(metadata.get("effective_set_max_tokens", 0)),
                    "avg_tokens": float(metadata.get("effective_set_avg_tokens", 0.0)),
                },
                "elapsed_time": float(metadata.get("elapsed_time", 0.0)),
            }

    if not (0.0 <= offset < 1.0):
        raise ValueError("offset must be in [0, 1)")
    if not (0.0 < prob_threshold <= 1.0):
        raise ValueError("prob_threshold must be in (0, 1]")
    if prefix_len <= 0:
        raise ValueError("prefix_len must be positive")

    actual_prompt = None
    if prefix == "":
        if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            prefix_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long)
            actual_prompt = f"<BOS:{tokenizer.bos_token_id}>"
        else:
            raise ValueError("Empty prefix requires BOS token")
    else:
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

    codes = generate_vdc_sequence(max_samples)
    if offset > 0.0:
        codes = [(code + offset) % 1.0 for code in codes]

    discovered_prefixes: Set[tuple] = set()
    prefix_first_seen: Dict[tuple, int] = {}
    all_sequences: List[List[int]] = []
    used_codes: List[float] = []
    terminated_flags: List[bool] = []
    mass_history: List[Tuple[int, float]] = [(0, 0.0)]

    prefix_probs: Dict[tuple, float] = {}

    print(f"\n{'='*80}")
    print("PREFIX MASS SAMPLING")
    print(f"{'='*80}")
    print(f"Prompt: '{prefix[:50]}...'")
    print(f"Prefix length    : {prefix_len}")
    print(f"Prob threshold   : {prob_threshold}")
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

    # Initial plot
    plt.clf()
    plt.plot([0], [0.0], color="cyan", label="cumulative mass")
    plt.plot([0, 0.5], [prob_threshold, prob_threshold], color="red", label="threshold")
    plt.title(f"Cumulative Prefix Mass (len={prefix_len})")
    plt.xlabel("Sample")
    plt.ylabel("Probability Mass")
    plt.plotsize(100, 20)
    plt.show()
    last_display_lines = 20

    # Process in batches
    for batch_start in range(0, max_samples, batch_size):
        batch_end = min(batch_start + batch_size, max_samples)
        batch_codes = codes[batch_start:batch_end]

        batch_results = parallel_arithmetic_sample_batch(
            model=model,
            tokenizer=tokenizer,
            prefix_ids=prefix_ids,
            codes=batch_codes,
            max_len=max_len,
            device=device,
        )

        batch_should_break = False

        for i, result in enumerate(batch_results):
            # Handle both old format (3-tuple) and new format (4-tuple)
            if len(result) == 4:
                tokens, log_prob, sample_info, per_token_log_probs = result
            else:
                tokens, log_prob, sample_info = result
                per_token_log_probs = None

            samples_done += 1

            all_sequences.append(tokens)
            used_codes.append(batch_codes[i])
            terminated_flags.append(sample_info["terminated_with_eos"])

            prefix_tuple = extract_prefix(tokens, prefix_len, eos_token_id)

            if prefix_tuple not in discovered_prefixes:
                discovered_prefixes.add(prefix_tuple)
                prefix_first_seen[prefix_tuple] = samples_done
            
                # OPTIMIZED: Use per-token log probs from sampling!
                if per_token_log_probs is not None:
                    # Take up to prefix_len log probs (or all if sequence shorter)
                    # After EOS, padding is deterministic (log_prob = 0), so it doesn't affect sum
                    actual_prefix_len = min(len(per_token_log_probs), prefix_len)
                    prefix_log_prob = sum(per_token_log_probs[:actual_prefix_len])
                    
                    # Note: If sequence ended before prefix_len, the padding EOS tokens
                    # have probability 1.0 (deterministic), contributing log(1.0) = 0.0
                    # So no adjustment needed!
                    
                    prefix_prob = math.exp(prefix_log_prob)
                    prefix_probs[prefix_tuple] = prefix_prob
                    current_mass += prefix_prob

                    if current_mass >= prob_threshold:
                        threshold_reached = True
                        batch_should_break = True
                        break
                else:
                    # Fallback: shouldn't happen with updated arithmetic.py
                    print("WARNING: per_token_log_probs not available, skipping this prefix")

        mass_history.append((samples_done, current_mass))

        # Periodic display update
        if display_interval > 0 and samples_done % display_interval == 0:
            elapsed = time.time() - start_time
            samples_per_sec = samples_done / elapsed if elapsed > 0 else 0.0

            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            elapsed_str = f"{elapsed_min}m{elapsed_sec:02d}s"

            if last_display_lines > 0:
                clear_lines(last_display_lines)

            plt.clf()
            x_vals = [x for x, _ in mass_history]
            y_vals = [y for _, y in mass_history]

            y_max = max(y_vals) if y_vals else 0.0
            half_thresh = prob_threshold / 2.0

            if y_max < half_thresh:
                if y_max > 0.0:
                    k = int(np.floor(np.log2(prob_threshold / y_max)))
                    if (prob_threshold / (2**k)) <= y_max:
                        k -= 1
                    k = max(k, 1)
                else:
                    k = 1

                guide = prob_threshold / (2**k)
                plt.plot(x_vals, [guide] * len(x_vals), color="yellow", label=f"1/2**{k} threshold")
            else:
                plt.plot(x_vals, [prob_threshold] * len(x_vals), color="red", label="threshold")

            plt.plot(x_vals, y_vals, color="cyan", label="cumulative mass")

            plt.title(f"Cumulative Prefix Mass (len={prefix_len})")
            plt.xlabel("Sample")
            plt.ylabel("Probability Mass")
            plt.plotsize(100, 20)
            plt.show()

            status_lines = []
            status_lines.append(
                f"[SAMPLING] {samples_done}/{max_samples} | "
                f"Mass: {current_mass}/{prob_threshold} | "
                f"Unique: {len(discovered_prefixes)} | "
                f"{samples_per_sec:.1f} samp/s | "
                f"Elapsed: {elapsed_str}"
            )

            if len(mass_history) >= 3 and current_mass > 0:
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
                        predicted_time = predicted_samples / samples_per_sec if samples_per_sec > 0 else float("inf")

                        pred_eta_min = int(predicted_time // 60)
                        pred_eta_sec = int(predicted_time % 60)

                        total_predicted_samples = samples_done + predicted_samples

                        if total_predicted_samples <= max_samples:
                            status_lines.append(
                                f"  Predicted: {int(predicted_samples)} more samples needed | "
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
                                f"  At max_samples: mass ≈ {predicted_mass_at_max} | "
                                f"ETA to max: {max_eta_min}m{max_eta_sec:02d}s"
                            )
                            status_lines.append(
                                f"  Would need {int(predicted_samples)} samples | "
                                f"Would take ≈ {pred_eta_min}m{pred_eta_sec:02d}s total"
                            )

            # Pad to exactly 4 lines
            while len(status_lines) < 4:
                status_lines.append("")

            # Print exactly 4 lines
            for line in status_lines[:4]:
                print(line)

            # Always 20 (plot) + 4 (status) = 24 lines total
            last_display_lines = 24

    elapsed = time.time() - start_time
    final_mass = current_mass

    # -------------------------------------------------------------------------
    # CLEAN MODE: Always print ONE final plot once after the loop finishes.
    # -------------------------------------------------------------------------
    if last_display_lines > 0:
        clear_lines(last_display_lines)

    plt.clf()
    x_vals = [x for x, _ in mass_history]
    y_vals = [y for _, y in mass_history]

    if x_vals:
        plt.plot(x_vals, [prob_threshold] * len(x_vals), color="red", label="threshold")
        plt.plot(x_vals, y_vals, color="cyan", label="cumulative mass")
    else:
        plt.plot([0], [prob_threshold], color="red", label="threshold")
        plt.plot([0], [0.0], color="cyan", label="cumulative mass")

    plt.title(f"Cumulative Prefix Mass (len={prefix_len})")
    plt.xlabel("Sample")
    plt.ylabel("Probability Mass")
    plt.plotsize(100, 20)
    plt.show()
    # Done with dynamic display updates, moving to final text output
    last_display_lines = 0

    # Effective set now = all sequences (no deduplication)
    effective_set_indices = list(range(len(all_sequences)))

    effective_sequences = [all_sequences[i] for i in effective_set_indices]
    effective_lengths = [len(seq) for seq in effective_sequences]

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Status: {'THRESHOLD REACHED' if threshold_reached else 'MAX SAMPLES'}")
    print(f"Total samples: {samples_done}")
    print(f"Unique prefixes (len={prefix_len}): {len(discovered_prefixes)}")
    print(f"Final prefix mass: {final_mass}")
    print("\nEffective Support Set:")
    print(f"  Size: {len(effective_set_indices)} sequences")
    print(f"  Total tokens: {sum(effective_lengths)}")
    print(f"  Min tokens: {min(effective_lengths) if effective_lengths else 0}")
    print(f"  Max tokens: {max(effective_lengths) if effective_lengths else 0}")
    print(f"  Avg tokens: {np.mean(effective_lengths) if effective_lengths else 0:.1f}")
    print(f"\nTime: {elapsed:.1f}s ({samples_done/elapsed:.1f} samp/s)")
    print(f"{'='*80}\n")

    results = {
        "prefix": prefix,
        "prefix_len": prefix_len,
        "prob_threshold": prob_threshold,
        "samples_done": samples_done,
        "threshold_reached": threshold_reached,
        "unique_prefixes": len(discovered_prefixes),
        "final_mass": final_mass,
        "effective_set_size": len(effective_set_indices),
        "effective_set_indices": effective_set_indices,
        "effective_set_stats": {
            "total_tokens": sum(effective_lengths),
            "min_tokens": min(effective_lengths) if effective_lengths else 0,
            "max_tokens": max(effective_lengths) if effective_lengths else 0,
            "avg_tokens": float(np.mean(effective_lengths)) if effective_lengths else 0.0,
        },
        "elapsed_time": elapsed,
    }

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
            tokenizer_name=tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else None,
            actual_prompt=actual_prompt,
            prefix_len=prefix_len,
            prob_threshold=prob_threshold,
            max_len=max_len,
            batch_size=batch_size,
            offset=offset,
            total_mass=final_mass,
            n_unique_prefixes=len(discovered_prefixes),
            elapsed_time=elapsed,
            use_chat_template=use_chat_template,
            threshold_reached=threshold_reached,
            display_interval=display_interval,
            max_samples=max_samples,
        )

    return results


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING PREFIX MASS SAMPLING WITH VALIDATION")
    print("=" * 80)

    print("\nLoading Qwen2.5-1.5B-Instruct...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print("\n" + "=" * 80)
    print("TEST 1: First run (will compute)")
    print("=" * 80)

    results = estimate_prefix_mass(
        model=model,
        tokenizer=tokenizer,
        prefix="What is the capital of France?",
        prefix_len=8,
        prob_threshold=0.9,
        max_samples=10000,
        max_len=32,
        batch_size=16,
        display_interval=128,
        save_path="results/test_capital_mass.delm.parquet",
        model_name=model_name,
    )

    print("\n✓ Test 1 complete")
    print(f"  Unique prefixes: {results['unique_prefixes']}")
    print(f"  Effective set: {results['effective_set_size']}")
    print(f"  Final mass: {results['final_mass']:.6f}")

    print("\n" + "=" * 80)
    print("TEST 2: Second run with SAME parameters (should load from cache)")
    print("=" * 80)
    input("\nPress Enter to continue...")

    results2 = estimate_prefix_mass(
        model=model,
        tokenizer=tokenizer,
        prefix="What is the capital of France?",
        prefix_len=8,
        prob_threshold=0.9,
        max_samples=10000,
        max_len=32,
        batch_size=16,
        display_interval=128,
        save_path="results/test_capital_mass.delm.parquet",
        model_name=model_name,
    )

    print("\n✓ Test 2 complete - results loaded from cache!")
    print(f"  Unique prefixes: {results2['unique_prefixes']}")
    print(f"  Effective set: {results2['effective_set_size']}")
    print(f"  Final mass: {results2['final_mass']:.6f}")

    print("\n" + "=" * 80)
    print("TEST 3: Different parameters (should raise validation error)")
    print("=" * 80)
    input("\nPress Enter to continue...")

    try:
        results3 = estimate_prefix_mass(
            model=model,
            tokenizer=tokenizer,
            prefix="What is the capital of France?",
            prefix_len=32,  # Changed!
            prob_threshold=0.9,
            max_samples=10000,
            max_len=32,
            batch_size=16,
            display_interval=128,
            save_path="results/test_capital_mass.delm.parquet",
            model_name=model_name,
        )
        print("\n⚠️ ERROR: Should have raised ValueError but didn't!")
    except ValueError as e:
        print("\n✓ Test 3 complete - validation error correctly raised!")
        print(f"  Error: {str(e)[:100]}...")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
    print("\nYou can view the results with:")
    print("  python tools/view_results.py results/test_capital_mass.delm.parquet")
    print("\nClean up test file:")
    print("  rm results/test_capital_mass.delm.parquet")