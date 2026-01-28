"""
Common utilities for all decompressionLM experiments.

Provides unified functions for:
- Model loading with GPU profiling
- JSON schema (consistent across all experiments)
- CSV writing
- Metrics extraction
"""

import os
import csv
import json
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.profiling import measure_model_memory


# ============================================================================
# Model Loading (Unified)
# ============================================================================

def load_model_with_profiling(
    model_path: str,
    model_short: str,
    quant_config: Optional[Dict] = None,
    setup_padding: bool = False,
) -> tuple:
    """
    Load model with full GPU profiling.
    
    Returns:
        (model, tokenizer, load_metrics)
        
    load_metrics contains:
        - load_time: float
        - gpu_memory_allocated_gb: float
        - gpu_memory_reserved_gb: float
    """
    print(f"\n{'='*80}")
    print(f"Loading {model_short}...")
    if quant_config:
        print(f"Quantization: {quant_config}")
    print(f"{'='*80}")
    
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model with optional quantization
    if quant_config:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            **quant_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    
    # Setup padding if requested (for exp1 graph methods)
    if setup_padding:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "left"
    
    load_time = time.time() - start
    model_mem = measure_model_memory()
    
    load_metrics = {
        'load_time': load_time,
        'gpu_memory_allocated_gb': model_mem.get('model_memory_allocated_gb', 0.0),
        'gpu_memory_reserved_gb': model_mem.get('model_memory_reserved_gb', 0.0),
    }
    
    print(f"✓ Loaded in {load_time:.2f}s")
    print(f"✓ GPU memory: {load_metrics['gpu_memory_allocated_gb']:.2f} GB allocated, "
          f"{load_metrics['gpu_memory_reserved_gb']:.2f} GB reserved")
    
    return model, tokenizer, load_metrics


# ============================================================================
# JSON Schema (Unified)
# ============================================================================

def create_unified_json_schema(
    model_name: str,
    method_name: str,
    method_type: str,  # 'flat', 'exploration', 'graph_gen'
    domain_key: str,
    domain_prompt: str,
    threshold: int,
    sampling_method: str,
    elapsed_time: float,
    samples_done: int,
    valid_concepts_count: int,
    invalid_concepts_count: int,
    threshold_reached: bool,
    sequences: List[Dict],  # List of {tokens, num_tokens, etc}
    valid_concepts: List[Dict],  # List of {concept, frequency, etc}
    invalid_concepts: List[Dict],
    gpu_profiling: Dict,
    method_config: Optional[Dict] = None,
    graph_stats: Optional[Dict] = None,
    concept_list: Optional[List[str]] = None,
) -> Dict:
    """
    Create unified JSON schema for all experiments.
    
    This ensures consistency across flat sampling, graph exploration, etc.
    """
    
    # Base metadata (common to all)
    metadata = {
        'model': model_name,
        'method': method_name,
        'method_type': method_type,
        'domain': domain_key,
        'domain_prompt': domain_prompt,
        'threshold': threshold,
        'sampling_method': sampling_method,
        'elapsed_time': elapsed_time,
        'samples_done': samples_done,
        'threshold_reached': threshold_reached,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    # Add method-specific config
    if method_config:
        metadata['method_config'] = method_config
    
    # Concept statistics (common to all)
    concept_stats = {
        'valid_concepts_count': valid_concepts_count,
        'invalid_concepts_count': invalid_concepts_count,
        'total_concepts': valid_concepts_count + invalid_concepts_count,
        'invalid_ratio': (invalid_concepts_count / (valid_concepts_count + invalid_concepts_count)
                         if (valid_concepts_count + invalid_concepts_count) > 0 else 0.0),
    }
    
    # GPU profiling (common to all)
    gpu_stats = {
        'load_time': gpu_profiling.get('load_time', 0.0),
        'gpu_memory_allocated_gb': gpu_profiling.get('gpu_memory_allocated_gb', 0.0),
        'gpu_memory_reserved_gb': gpu_profiling.get('gpu_memory_reserved_gb', 0.0),
        
        # Sampling GPU stats (if available from profiling dict)
        'gpu_sampling_allocated_avg_gb': gpu_profiling.get('gpu_sampling_end_allocated_avg', 0.0),
        'gpu_sampling_allocated_max_gb': gpu_profiling.get('gpu_sampling_end_allocated_max', 0.0),
        'gpu_sampling_reserved_avg_gb': gpu_profiling.get('gpu_sampling_end_reserved_avg', 0.0),
        'gpu_sampling_reserved_max_gb': gpu_profiling.get('gpu_sampling_end_reserved_max', 0.0),
        
        # Timing breakdown (if available)
        'time_sampling_avg': gpu_profiling.get('time_sampling_avg', 0.0),
        'time_extraction_avg': gpu_profiling.get('time_concept_extraction_avg', 0.0),
        'time_new_concept_discovery_avg': gpu_profiling.get('time_new_concept_discovery_avg', 0.0),
    }
    
    # Build unified schema
    schema = {
        'metadata': metadata,
        'concept_stats': concept_stats,
        'gpu_stats': gpu_stats,
        'sequences': sequences,
        'valid_concepts': valid_concepts,
        'invalid_concepts': invalid_concepts,
    }
    
    # Add graph-specific data if present
    if graph_stats:
        schema['graph_stats'] = graph_stats
    
    if concept_list:
        schema['concept_list'] = concept_list
    
    return schema


def save_unified_json(save_path: str, schema: Dict):
    """Save unified JSON schema to file."""
    json_path = save_path if save_path.endswith('.json') else save_path + '.delm.json'
    
    with open(json_path, 'w') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved to {json_path}")
    return json_path


def load_unified_json(json_path: str) -> Dict:
    """Load unified JSON schema from file."""
    with open(json_path, 'r') as f:
        return json.load(f)


# ============================================================================
# CSV Writing (Unified)
# ============================================================================

def save_metrics_to_csv(all_metrics: List[Dict], csv_path: str):
    """
    Save metrics to CSV with consistent column ordering.
    
    Works for ALL experiments (1-4) with different schemas.
    """
    if not all_metrics:
        print(f"⚠ No metrics to write to {csv_path}")
        return
    
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Union of all keys across metrics
    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())
    
    # Preferred column order (human-friendly)
    preferred_columns = [
        # Identifiers
        'model', 'domain', 'method', 'method_type', 'threshold',
        'quantization', 'offset_idx', 'offset',
        
        # Core metrics
        'samples', 'valid_concepts', 'invalid_concepts', 'invalid_ratio',
        'threshold_reached', 'converged',
        
        # Performance
        'total_time', 'time_to_converge',
        'tokens_per_sec', 'concepts_per_sec',
        'total_tokens', 'avg_tokens_per_seq',
        
        # Graph-specific
        'max_depth', 'nodes_explored', 'reconnections',
        'parallel_sequences_per_node', 'average_parallel_sequences_per_node',
        
        # GPU stats
        'gpu_load_time', 'gpu_memory_allocated_gb', 'gpu_memory_reserved_gb',
        'gpu_sampling_allocated_avg_gb', 'gpu_sampling_allocated_max_gb',
        'gpu_sampling_reserved_avg_gb', 'gpu_sampling_reserved_max_gb',
        
        # Timing breakdown
        'time_sampling_avg', 'time_extraction_avg', 'time_new_concept_discovery_avg',
    ]
    
    # Build final fieldnames: preferred first, then remaining sorted
    fieldnames = [k for k in preferred_columns if k in all_keys] + \
                 sorted(all_keys - set(preferred_columns))
    
    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for m in all_metrics:
            # Fill missing keys with None
            row = {k: m.get(k, None) for k in fieldnames}
            writer.writerow(row)
    
    print(f"✓ Saved {len(all_metrics)} rows to {csv_path}")


# ============================================================================
# Metrics Extraction (Unified)
# ============================================================================

def extract_metrics_from_json(
    json_data: Dict,
    model_name: str,
    method_name: str,
    domain_key: Optional[str] = None,
    threshold: Optional[int] = None,
    additional_fields: Optional[Dict] = None,
) -> Dict:
    """
    Extract metrics from unified JSON schema.
    
    Works for both loaded JSON and fresh results with consistent output.
    """
    metadata = json_data.get('metadata', {})
    concept_stats = json_data.get('concept_stats', {})
    gpu_stats = json_data.get('gpu_stats', {})
    graph_stats = json_data.get('graph_stats', {})
    
    elapsed = metadata.get('elapsed_time', 0.0)
    samples = metadata.get('samples_done', 0)
    
    # Calculate total tokens
    sequences = json_data.get('sequences', [])
    total_tokens = sum(seq.get('num_tokens', 0) for seq in sequences)
    
    # Build metrics dict
    metrics = {
        'model': model_name,
        'method': method_name,
        'method_type': metadata.get('method_type', 'unknown'),
        'sampling_method': metadata.get('sampling_method', 'unknown'),
        
        # Core counts
        'samples': samples,
        'valid_concepts': concept_stats.get('valid_concepts_count', 0),
        'invalid_concepts': concept_stats.get('invalid_concepts_count', 0),
        'invalid_ratio': f"{concept_stats.get('invalid_ratio', 0.0):.4f}",
        'threshold_reached': metadata.get('threshold_reached', False),
        
        # Performance metrics
        'total_time': f"{elapsed:.2f}",
        'time_to_converge': f"{elapsed:.2f}" if metadata.get('threshold_reached') else "N/A",
        'tokens_per_sec': f"{total_tokens / elapsed:.2f}" if elapsed > 0 else "0.00",
        'concepts_per_sec': f"{concept_stats.get('valid_concepts_count', 0) / elapsed:.4f}" if elapsed > 0 else "0.0000",
        'total_tokens': total_tokens,
        'avg_tokens_per_seq': f"{total_tokens / samples:.2f}" if samples > 0 else "0.00",
        
        # GPU stats
        'gpu_load_time': f"{gpu_stats.get('load_time', 0.0):.2f}",
        'gpu_memory_allocated_gb': f"{gpu_stats.get('gpu_memory_allocated_gb', 0.0):.2f}",
        'gpu_memory_reserved_gb': f"{gpu_stats.get('gpu_memory_reserved_gb', 0.0):.2f}",
        'gpu_sampling_allocated_avg_gb': f"{gpu_stats.get('gpu_sampling_allocated_avg_gb', 0.0):.2f}",
        'gpu_sampling_allocated_max_gb': f"{gpu_stats.get('gpu_sampling_allocated_max_gb', 0.0):.2f}",
        
        # Timing breakdown
        'time_sampling_avg': f"{gpu_stats.get('time_sampling_avg', 0.0):.4f}",
        'time_extraction_avg': f"{gpu_stats.get('time_extraction_avg', 0.0):.4f}",
    }
    
    # Add domain if provided
    if domain_key:
        metrics['domain'] = domain_key
    elif 'domain' in metadata:
        metrics['domain'] = metadata['domain']
    
    # Add threshold if provided
    if threshold:
        metrics['threshold'] = threshold
    elif 'threshold' in metadata:
        metrics['threshold'] = metadata['threshold']
    
    # Add graph-specific metrics if present
    if graph_stats:
        metrics.update({
            'max_depth': graph_stats.get('max_depth_reached', 0),
            'nodes_explored': graph_stats.get('total_nodes_explored', 0),
            'reconnections': graph_stats.get('reconnections', 0),
            'average_parallel_sequences_per_node': f"{graph_stats.get('average_parallel_sequences_per_node', 0.0):.3f}",
        })
    
    # Add any additional fields
    if additional_fields:
        metrics.update(additional_fields)
    
    return metrics


def extract_metrics_from_fresh_results(
    results: Dict,
    model_name: str,
    method_name: str,
    domain_key: Optional[str] = None,
    threshold: Optional[int] = None,
    gpu_profiling: Optional[Dict] = None,
    additional_fields: Optional[Dict] = None,
) -> Dict:
    """
    Extract metrics from fresh sampling results (not yet saved to JSON).
    """
    elapsed = results.get('elapsed_time', 0.0)
    samples = results.get('samples_done', 0)
    valid = results.get('valid_concepts', 0)
    invalid = results.get('invalid_concepts', 0)
    
    # Calculate tokens
    sequences = results.get('sequences', [])
    if sequences and isinstance(sequences[0], dict):
        total_tokens = sum(seq.get('num_tokens', 0) for seq in sequences)
    else:
        # Fallback: count tokens directly
        total_tokens = sum(len(seq.get('tokens', [])) if isinstance(seq, dict) else len(seq) 
                          for seq in sequences)
    
    # Build metrics
    metrics = {
        'model': model_name,
        'method': method_name,
        'samples': samples,
        'valid_concepts': valid,
        'invalid_concepts': invalid,
        'invalid_ratio': f"{invalid / (valid + invalid):.4f}" if (valid + invalid) > 0 else "0.0000",
        'threshold_reached': results.get('threshold_reached', False),
        'total_time': f"{elapsed:.2f}",
        'time_to_converge': f"{elapsed:.2f}" if results.get('threshold_reached') else "N/A",
        'tokens_per_sec': f"{total_tokens / elapsed:.2f}" if elapsed > 0 else "0.00",
        'concepts_per_sec': f"{valid / elapsed:.4f}" if elapsed > 0 else "0.0000",
        'total_tokens': total_tokens,
        'avg_tokens_per_seq': f"{total_tokens / samples:.2f}" if samples > 0 else "0.00",
    }
    
    # Add GPU profiling if provided
    if gpu_profiling:
        prof = results.get('profiling', {})
        metrics.update({
            'gpu_load_time': f"{gpu_profiling.get('load_time', 0.0):.2f}",
            'gpu_memory_allocated_gb': f"{gpu_profiling.get('gpu_memory_allocated_gb', 0.0):.2f}",
            'gpu_sampling_allocated_avg_gb': f"{prof.get('gpu_sampling_end_allocated_avg', 0.0):.2f}",
            'gpu_sampling_allocated_max_gb': f"{prof.get('gpu_sampling_end_allocated_max', 0.0):.2f}",
            'time_sampling_avg': f"{prof.get('time_sampling_avg', 0.0):.4f}",
            'time_extraction_avg': f"{prof.get('time_concept_extraction_avg', 0.0):.4f}",
        })
    
    if domain_key:
        metrics['domain'] = domain_key
    if threshold:
        metrics['threshold'] = threshold
    if additional_fields:
        metrics.update(additional_fields)
    
    return metrics


# ============================================================================
# Pretty Printing (Unified)
# ============================================================================

def print_metrics_table(all_metrics: List[Dict], title: str, show_gpu: bool = False):
    """
    Print formatted metrics table.
    
    Adapts to different experiment types automatically.
    """
    if not all_metrics:
        print("⚠ No metrics to display")
        return
    
    print(f"\n{'='*120}")
    print(f"{title}")
    print(f"{'='*120}\n")
    
    # Determine what columns to show based on what's present
    has_domain = any('domain' in m for m in all_metrics)
    has_threshold = any('threshold' in m for m in all_metrics)
    has_quantization = any('quantization' in m for m in all_metrics)
    has_graph_stats = any('nodes_explored' in m for m in all_metrics)
    
    # Build header
    header = f"{'Model':<15} | "
    if has_domain:
        header += f"{'Domain':<10} | "
    if has_threshold:
        header += f"{'Thresh':>6} | "
    if has_quantization:
        header += f"{'Quant':<10} | "
    header += f"{'Method':<25} | {'Samples':>7} | {'Valid':>6} | {'Time':>7} | {'Conv':>5}"
    
    if has_graph_stats:
        header += f" | {'Nodes':>6} | {'Depth':>5}"
    
    if show_gpu:
        header += f" | {'GPU(GB)':>8}"
    
    print(header)
    print("-" * len(header))
    
    # Print rows
    for m in all_metrics:
        converged = "YES" if m.get('threshold_reached', False) else "NO"
        
        row = f"{m.get('model', 'N/A'):<15} | "
        if has_domain:
            row += f"{m.get('domain', 'N/A'):<10} | "
        if has_threshold:
            row += f"{m.get('threshold', 0):>6} | "
        if has_quantization:
            row += f"{m.get('quantization', 'N/A'):<10} | "
        row += f"{m.get('method', 'N/A'):<25} | {m.get('samples', 0):>7} | {m.get('valid_concepts', 0):>6} | "
        row += f"{m.get('total_time', '0.00'):>7}s | {converged:>5}"
        
        if has_graph_stats:
            row += f" | {m.get('nodes_explored', 0):>6} | {m.get('max_depth', 0):>5}"
        
        if show_gpu:
            gpu_mem = m.get('gpu_sampling_allocated_max_gb', m.get('gpu_memory_allocated_gb', '0.00'))
            row += f" | {gpu_mem:>8}"
        
        print(row)
    
    print(f"\n{'='*120}\n")