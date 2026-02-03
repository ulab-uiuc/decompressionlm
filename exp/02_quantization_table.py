#!/usr/bin/env python3
"""
VdC Offset Experiment Analyzer.
Check if all experiment files exist and analyze concept overlap for both models.
"""

import os
import json
import glob
import sys
import struct
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
from itertools import combinations
from collections import defaultdict

# ----------------------------
# Configuration
# ----------------------------
MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]
VARIANTS = ["BF16_BASE", "GPTQ_INT8", "AWQ_4BIT", "GPTQ_INT4", "BNB_4BIT_UNSLOTH"]
VARIANTS_SHORT = {
    "BF16_BASE": "BF16_BASE",
    "GPTQ_INT8": "GPTQ_INT8",
    "AWQ_4BIT": "AWQ_4BIT",
    "GPTQ_INT4": "GPTQ_INT4",
    "BNB_4BIT_UNSLOTH": "BNB_4BIT"
}
SEQ_LENS = [16, 32]
NUM_RUNS = 8

# Output directory
OUTPUT_DIR = "results/vdc_offset_experiment"
VDC_CODES_BINARY = os.path.join(OUTPUT_DIR, "vdc_codes.bin")
VDC_CODES_META = os.path.join(OUTPUT_DIR, "vdc_codes_meta.json")

# ----------------------------
# Helper Functions - BINARY LOADING
# ----------------------------
def load_vdc_codes_binary():
    """Load VdC codes from binary file."""
    if not os.path.exists(VDC_CODES_BINARY):
        print(f"❌ VdC binary file not found: {VDC_CODES_BINARY}")
        return None
    
    try:
        with open(VDC_CODES_BINARY, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic != b'VDC1':
                print(f"❌ Invalid magic number: {magic}")
                return None
            
            version = struct.unpack('I', f.read(4))[0]
            if version != 1:
                print(f"❌ Unsupported version: {version}")
                return None
            
            num_codes = struct.unpack('Q', f.read(8))[0]
            
            # Read dtype info
            dtype_len = struct.unpack('I', f.read(4))[0]
            dtype_str = f.read(dtype_len).decode('utf-8')
            
            # Read binary data
            expected_bytes = num_codes * np.dtype(dtype_str).itemsize
            data_bytes = f.read(expected_bytes)
            
            if len(data_bytes) != expected_bytes:
                print(f"❌ Data size mismatch")
                return None
            
            # Convert to numpy array
            codes_np = np.frombuffer(data_bytes, dtype=dtype_str)
            
            if len(codes_np) != num_codes:
                print(f"❌ Code count mismatch")
                return None
            
        return codes_np
        
    except Exception as e:
        print(f"❌ Error loading binary VdC codes: {e}")
        return None

def extract_vdc_from_filename(filename):
    """Extract VdC code from filename."""
    # Format: MODEL__VARIANT__seqLEN__runX__vdcVALUE.delm.parquet
    parts = filename.split('__')
    if len(parts) < 5:
        return None
    
    vdc_part = parts[-1].replace('.delm.parquet', '')
    if not vdc_part.startswith('vdc'):
        return None
    
    vdc_str = vdc_part[3:].replace('p', '.').replace('m', '-')
    try:
        return float(vdc_str)
    except:
        return None

def check_all_files_exist(vdc_codes):
    """Check if all expected result files exist."""
    missing_files = []
    
    if vdc_codes is None:
        print("❌ Cannot check files without VdC codes")
        return missing_files
    
    for seq_len in SEQ_LENS:
        for model in MODELS:
            for variant in VARIANTS:
                label = f"{model}__{variant}"
                
                for run_idx, vdc_code in enumerate(vdc_codes, 1):
                    # Format filename with VdC code
                    vdc_str = f"{vdc_code:.17f}".replace('.', 'p').replace('-', 'm')
                    filename = f"{label}__seq{seq_len}__run{run_idx}__vdc{vdc_str}.delm.parquet"
                    file_path = os.path.join(OUTPUT_DIR, filename)
                    
                    if not os.path.exists(file_path):
                        missing_files.append({
                            'model': model,
                            'variant': variant,
                            'seq_len': seq_len,
                            'run_idx': run_idx,
                            'vdc_code': vdc_code,
                            'file': filename
                        })
    
    return missing_files

def load_file_metadata(file_path):
    """Load metadata from a .delm.parquet file."""
    try:
        table = pq.read_table(file_path)
        metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
        return metadata
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_concepts_from_metadata(metadata):
    """Extract concepts from metadata using graph_concept_frequencies."""
    if 'graph_concept_frequencies' not in metadata:
        return set()
    
    try:
        import json
        concept_freqs = json.loads(metadata['graph_concept_frequencies'])
        concepts = {concept for concept, _ in concept_freqs}
        return concepts
    except Exception as e:
        print(f"Error parsing concepts: {e}")
        return set()

def load_experiment_data(vdc_codes):
    """Load all experiment data."""
    if vdc_codes is None:
        return None, None
    
    # Load all results
    all_data = {}
    concept_data = {}
    
    for seq_len in SEQ_LENS:
        all_data[seq_len] = {}
        concept_data[seq_len] = {}
        
        for model in MODELS:
            for variant in VARIANTS:
                label = f"{model}__{variant}"
                variant_short = VARIANTS_SHORT.get(variant, variant)
                key = f"{model}__{variant_short}"
                
                all_data[seq_len][key] = {}
                concept_data[seq_len][key] = {}
                
                for run_idx, vdc_code in enumerate(vdc_codes, 1):
                    vdc_str = f"{vdc_code:.17f}".replace('.', 'p').replace('-', 'm')
                    filename = f"{label}__seq{seq_len}__run{run_idx}__vdc{vdc_str}.delm.parquet"
                    file_path = os.path.join(OUTPUT_DIR, filename)
                    
                    if not os.path.exists(file_path):
                        continue
                    
                    metadata = load_file_metadata(file_path)
                    if metadata:
                        # Extract concepts
                        concepts = extract_concepts_from_metadata(metadata)
                        
                        all_data[seq_len][key][run_idx] = {
                            'vdc_code': vdc_code,
                            'concepts': concepts,
                            'concept_count': len(concepts),
                            'metadata': metadata
                        }
                        
                        concept_data[seq_len][key][run_idx] = concepts
    
    return all_data, concept_data

def print_vdc_codes_summary(vdc_codes):
    """Print summary of VdC codes used."""
    if vdc_codes is None:
        print("❌ No VdC codes available")
        return
    
    print(f"\n📊 VDC CODES USED ({len(vdc_codes)} runs, binary format):")
    print()
    print("| Run | VdC Code (full precision) |")
    print("| --- | ------------------------- |")
    for i, code in enumerate(vdc_codes, 1):
        # Use numpy's full precision formatting
        code_str = np.format_float_positional(code, precision=17, unique=True, fractional=True, trim='k')
        print(f"| {i:3d} | {code_str:24} |")
    
    # Load metadata if available
    if os.path.exists(VDC_CODES_META):
        try:
            with open(VDC_CODES_META, 'r') as f:
                metadata = json.load(f)
            print(f"\n📋 BINARY METADATA:")
            print(f"  Seed: {metadata.get('seed', 'unknown')}")
            print(f"  Dtype: {metadata.get('numpy_dtype', 'unknown')}")
            print(f"  Generation method: {metadata.get('generation_method', 'unknown')}")
        except:
            pass

def print_intra_variant_analysis(concept_data, model_variant, seq_len):
    """Analyze concept overlap between different runs of the same variant."""
    model, variant = model_variant.split('__', 1)
    
    print(f"\n{'='*80}")
    print(f"INTRINSIC VdC VARIABILITY: {model} - {variant} - Seq Len {seq_len}")
    print(f"{'='*80}")
    
    if model_variant not in concept_data[seq_len]:
        print("No data available for this variant.")
        return
    
    run_concepts = concept_data[seq_len][model_variant]
    num_runs = len(run_concepts)
    
    if num_runs == 0:
        print("No runs available.")
        return
    
    # Print concept counts per run
    print(f"\n📊 CONCEPT COUNTS PER RUN:")
    print()
    print("| Run | Concepts |")
    print("| --- | -------: |")
    
    concept_counts = []
    for run_idx, concepts in sorted(run_concepts.items()):
        count = len(concepts)
        concept_counts.append(count)
        print(f"| {run_idx:3d} | {count:8,} |")
    
    # Calculate pairwise overlaps
    run_indices = list(sorted(run_concepts.keys()))
    pairs = list(combinations(run_indices, 2))
    
    if len(pairs) > 0:
        print(f"\n🔄 PAIRWISE OVERLAP BETWEEN RUNS ({len(pairs)} pairs):")
        print()
        print("| Pair | Overlap Count | % of A | % of B | Jaccard % |")
        print("| ---- | ------------: | -----: | -----: | --------: |")
        
        pair_stats = []
        for run_a, run_b in pairs:
            concepts_a = run_concepts[run_a]
            concepts_b = run_concepts[run_b]
            
            overlap = concepts_a.intersection(concepts_b)
            overlap_count = len(overlap)
            
            pct_a = (overlap_count / len(concepts_a) * 100) if concepts_a else 0
            pct_b = (overlap_count / len(concepts_b) * 100) if concepts_b else 0
            
            union = concepts_a.union(concepts_b)
            jaccard = (overlap_count / len(union) * 100) if union else 0
            
            pair_name = f"Run{run_a}-Run{run_b}"
            print(f"| {pair_name:10} | {overlap_count:13,} | {pct_a:6.1f}% | {pct_b:6.1f}% | {jaccard:9.1f}% |")
            
            pair_stats.append({
                'pair': pair_name,
                'overlap': overlap_count,
                'pct_a': pct_a,
                'pct_b': pct_b,
                'jaccard': jaccard
            })
        
        # Calculate summary statistics
        if pair_stats:
            avg_overlap = sum(p['overlap'] for p in pair_stats) / len(pair_stats)
            avg_pct_a = sum(p['pct_a'] for p in pair_stats) / len(pair_stats)
            avg_pct_b = sum(p['pct_b'] for p in pair_stats) / len(pair_stats)
            avg_jaccard = sum(p['jaccard'] for p in pair_stats) / len(pair_stats)
            
            print(f"\n📊 AVERAGE PAIRWISE STATS:")
            print(f"  Average overlap: {avg_overlap:,.0f} concepts")
            print(f"  Average % of first run: {avg_pct_a:.1f}%")
            print(f"  Average % of second run: {avg_pct_b:.1f}%")
            print(f"  Average Jaccard: {avg_jaccard:.1f}%")
    
    # Calculate concepts shared by all runs
    all_concept_sets = list(run_concepts.values())
    if len(all_concept_sets) > 1:
        shared_by_all = set.intersection(*all_concept_sets)
        
        print(f"\n🌟 CONCEPTS SHARED BY ALL {num_runs} RUNS:")
        print(f"  Count: {len(shared_by_all):,} concepts")
        
        if shared_by_all:
            # Calculate union of all concepts
            all_concepts = set()
            for concepts in all_concept_sets:
                all_concepts.update(concepts)
            
            shared_pct = (len(shared_by_all) / len(all_concepts) * 100) if all_concepts else 0
            print(f"  Percentage of total unique concepts: {shared_pct:.1f}%")
            
            # Show some shared concepts
            shared_list = list(shared_by_all)
            if len(shared_list) > 0:
                if len(shared_list) > 10:
                    print(f"  Examples (first 10): {', '.join(shared_list[:10])}...")
                else:
                    print(f"  All shared concepts: {', '.join(shared_list)}")
    
    print(f"\n{'='*80}")

def print_inter_variant_analysis(all_data, seq_len, model_filter=None):
    """Analyze concept overlap between different quantization variants."""
    print(f"\n{'='*80}")
    if model_filter:
        print(f"QUANTIZATION VARIANT COMPARISON: {model_filter} - Seq Len {seq_len}")
    else:
        print(f"QUANTIZATION VARIANT COMPARISON: Seq Len {seq_len}")
    print(f"{'='*80}")
    
    if seq_len not in all_data:
        print("No data available.")
        return
    
    # Filter data by model if specified
    variant_data = {}
    for key, run_data in all_data[seq_len].items():
        model, variant = key.split('__', 1)
        if model_filter and model != model_filter:
            continue
        variant_data[key] = run_data
    
    if not variant_data:
        print("No variant data available.")
        return
    
    # Calculate average concepts per variant across all runs
    variant_avg_concepts = {}
    variant_all_concepts = {}  # Union of concepts across all runs for each variant
    variant_concept_counts = defaultdict(list)
    
    for variant_key, run_data in variant_data.items():
        model, variant = variant_key.split('__', 1)
        concept_counts = [data['concept_count'] for data in run_data.values()]
        if concept_counts:
            avg_count = sum(concept_counts) / len(concept_counts)
            
            # Get union of all concepts across runs for this variant
            all_concepts_for_variant = set()
            for data in run_data.values():
                all_concepts_for_variant.update(data['concepts'])
            
            display_key = f"{model}\n{variant}"
            variant_avg_concepts[display_key] = avg_count
            variant_all_concepts[display_key] = all_concepts_for_variant
            variant_concept_counts[display_key] = concept_counts
    
    if not variant_avg_concepts:
        print("No variant data available.")
        return
    
    print(f"\n📊 AVERAGE CONCEPTS PER VARIANT (across {NUM_RUNS} runs):")
    print()
    print("| Model | Variant   | Avg Concepts | Min | Max | Std Dev | Total Unique |")
    print("| ----- | --------- | -----------: | --: | --: | ------: | -----------: |")
    
    for key, avg_count in sorted(variant_avg_concepts.items(), key=lambda x: x[1], reverse=True):
        model, variant = key.split('\n')
        counts = variant_concept_counts[key]
        min_count = min(counts) if counts else 0
        max_count = max(counts) if counts else 0
        
        # Calculate standard deviation
        if len(counts) > 1:
            mean = avg_count
            variance = sum((x - mean) ** 2 for x in counts) / len(counts)
            std_dev = variance ** 0.5
        else:
            std_dev = 0
        
        total_unique = len(variant_all_concepts[key])
        
        print(f"| {model:5} | {variant:10} | {avg_count:11.0f} | {min_count:3} | {max_count:3} | {std_dev:7.1f} | {total_unique:12,} |")
    
    # Calculate overlap between variants (using union of concepts from all runs)
    print(f"\n🔄 VARIANT-TO-VARIANT OVERLAP (using union of all runs):")
    
    # Group by model
    models = {}
    for key in variant_all_concepts.keys():
        model, variant = key.split('\n')
        if model not in models:
            models[model] = []
        models[model].append((variant, variant_all_concepts[key]))
    
    for model, variants in models.items():
        print(f"\n📈 {model}:")
        variants_sorted = sorted(variants, key=lambda x: len(x[1]), reverse=True)
        variant_names = [v[0] for v in variants_sorted]
        
        # Header
        header = "| Variant   | " + " | ".join([f"{v:^12}" for v in variant_names]) + " |"
        separator = "| --------- | " + " | ".join([f"{'':-^12}" for _ in variant_names]) + " |"
        
        print(header)
        print(separator)
        
        # Data rows - showing both overlap count and Jaccard
        for var_a, concepts_a in variants_sorted:
            row_cells = [f"{var_a:10}"]
            
            for var_b, concepts_b in variants_sorted:
                if var_a == var_b:
                    cell = f"{'—':^12}"
                else:
                    overlap = concepts_a.intersection(concepts_b)
                    union = concepts_a.union(concepts_b)
                    overlap_count = len(overlap)
                    jaccard = (overlap_count / len(union) * 100) if union else 0
                    
                    cell = f"{overlap_count:4d}/{jaccard:5.1f}%".center(12)
                
                row_cells.append(cell)
            
            print(" | ".join(row_cells) + " |")
    
    # Calculate concepts shared by all variants of the same model
    for model, variants in models.items():
        if len(variants) > 1:
            all_variant_sets = [concepts for _, concepts in variants]
            shared_by_all_variants = set.intersection(*all_variant_sets)
            
            print(f"\n🌟 {model} - CONCEPTS SHARED BY ALL {len(variants)} VARIANTS:")
            print(f"  Count: {len(shared_by_all_variants):,} concepts")
            
            if shared_by_all_variants:
                # Calculate union of all concepts across all variants
                all_concepts_all_variants = set()
                for concepts in all_variant_sets:
                    all_concepts_all_variants.update(concepts)
                
                shared_pct = (len(shared_by_all_variants) / len(all_concepts_all_variants) * 100) if all_concepts_all_variants else 0
                print(f"  Percentage of total unique concepts: {shared_pct:.1f}%")
                
                # Show some shared concepts
                shared_list = list(shared_by_all_variants)
                if len(shared_list) > 0:
                    if len(shared_list) > 10:
                        print(f"  Examples (first 10): {', '.join(shared_list[:10])}...")
                    else:
                        print(f"  All shared concepts: {', '.join(shared_list)}")
    
    print(f"\n{'='*80}")

def print_comparative_analysis(all_data, seq_len):
    """Print comparative analysis between models."""
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON: Seq Len {seq_len}")
    print(f"{'='*80}")
    
    if seq_len not in all_data:
        print("No data available.")
        return
    
    # Group data by model
    model_data = defaultdict(dict)
    for key, run_data in all_data[seq_len].items():
        model, variant = key.split('__', 1)
        model_data[model][variant] = run_data
    
    if len(model_data) < 2:
        print("Need data from at least 2 models for comparison.")
        return
    
    print(f"\n📊 MODEL-LEVEL STATISTICS:")
    print()
    print("| Model | Variants | Avg Concepts/Variant | Total Unique |")
    print("| ----- | -------: | ------------------: | -----------: |")
    
    for model, variants in sorted(model_data.items()):
        total_concepts = 0
        total_variants = len(variants)
        all_concepts_model = set()
        
        for variant, run_data in variants.items():
            # Get union of all concepts for this variant
            variant_concepts = set()
            for data in run_data.values():
                variant_concepts.update(data['concepts'])
            all_concepts_model.update(variant_concepts)
            
            # Average concept count for this variant
            concept_counts = [data['concept_count'] for data in run_data.values()]
            if concept_counts:
                total_concepts += sum(concept_counts) / len(concept_counts)
        
        avg_concepts_per_variant = total_concepts / total_variants if total_variants > 0 else 0
        total_unique_model = len(all_concepts_model)
        
        print(f"| {model:20} | {total_variants:8d} | {avg_concepts_per_variant:19.0f} | {total_unique_model:12,} |")
    
    # Compare overlap between same variants across models
    print(f"\n🔄 CROSS-MODEL COMPARISON (same quantization method):")
    
    # Find common variants between models
    common_variants = set()
    for model, variants in model_data.items():
        common_variants.update(variants.keys())
    
    # Only compare variants that exist in all models
    common_variants = [v for v in common_variants 
                      if all(v in model_data[m] for m in model_data)]
    
    if common_variants:
        print(f"\nCommon variants: {', '.join(common_variants)}")
        print()
        
        for variant in common_variants:
            print(f"📊 {variant}:")
            variant_concepts = {}
            
            for model in sorted(model_data.keys()):
                if variant in model_data[model]:
                    # Get union of all concepts for this variant in this model
                    all_concepts = set()
                    for data in model_data[model][variant].values():
                        all_concepts.update(data['concepts'])
                    variant_concepts[model] = all_concepts
            
            if len(variant_concepts) >= 2:
                models = list(sorted(variant_concepts.keys()))
                
                # Header
                header = "| Model | " + " | ".join([f"{m:^20}" for m in models]) + " |"
                separator = "| ----- | " + " | ".join([f"{'':-^20}" for _ in models]) + " |"
                
                print(header)
                print(separator)
                
                # Data rows
                for model_a in models:
                    row_cells = [f"{model_a:20}"]
                    
                    for model_b in models:
                        if model_a == model_b:
                            cell = f"{'—':^20}"
                        else:
                            concepts_a = variant_concepts[model_a]
                            concepts_b = variant_concepts[model_b]
                            
                            overlap = concepts_a.intersection(concepts_b)
                            union = concepts_a.union(concepts_b)
                            overlap_count = len(overlap)
                            jaccard = (overlap_count / len(union) * 100) if union else 0
                            
                            cell = f"{overlap_count:5d}/{jaccard:6.1f}%".center(20)
                        
                        row_cells.append(cell)
                    
                    print(" | ".join(row_cells) + " |")
                
                print()
    
    print(f"\n{'='*80}")

# ----------------------------
# Main Function
# ----------------------------
def main():
    print("="*80)
    print("VdC OFFSET EXPERIMENT ANALYZER")
    print("Analyzing concept overlap for Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct")
    print("Binary format VdC codes")
    print("="*80)
    
    # Check if output directory exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"\n❌ Output directory not found: {OUTPUT_DIR}")
        print("Please run the experiment first:")
        print("python vdc_offset_experiment.py")
        sys.exit(1)
    
    # Load VdC codes from binary file
    print("\nLoading VdC codes from binary file...")
    vdc_codes = load_vdc_codes_binary()
    
    if vdc_codes is None:
        print("❌ Failed to load VdC codes from binary file.")
        print("Please run the experiment first.")
        sys.exit(1)
    
    # Print VdC codes summary
    print_vdc_codes_summary(vdc_codes)
    
    # Check if all files exist
    print("\nChecking for required result files...")
    missing_files = check_all_files_exist(vdc_codes)
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} required files:")
        # Group by model and variant for better readability
        missing_by_model = defaultdict(list)
        for missing in missing_files:
            key = f"{missing['model']} - {missing['variant']}"
            missing_by_model[key].append(missing)
        
        for key, files in list(missing_by_model.items())[:10]:  # Show first 10
            print(f"  {key}: {len(files)} missing files")
            if len(files) <= 3:  # Show details for small numbers
                for f in files:
                    print(f"    - Run {f['run_idx']} (VdC: {f['vdc_code']:.6f})")
        
        if len(missing_by_model) > 10:
            print(f"  ... and {len(missing_by_model) - 10} more models/variants")
        
        print(f"\nTotal expected files: {len(MODELS) * len(VARIANTS) * len(SEQ_LENS) * NUM_RUNS}")
        print(f"Missing files: {len(missing_files)}")
        print(f"\nPlease run the experiment first to generate missing files:")
        print("python vdc_offset_experiment.py")
        sys.exit(1)
    
    print("✓ All required files found!")
    
    # Load experiment data
    print("\nLoading experiment data...")
    all_data, concept_data = load_experiment_data(vdc_codes)
    
    if not all_data:
        print("❌ Failed to load experiment data.")
        sys.exit(1)
    
    # Analyze each sequence length
    for seq_len in SEQ_LENS:
        print(f"\n{'='*80}")
        print(f"ANALYSIS FOR SEQUENCE LENGTH: {seq_len}")
        print(f"{'='*80}")
        
        # Analyze intra-variant consistency (VdC effect) for each model
        print(f"\n📊 INTRINSIC VdC VARIABILITY ANALYSIS")
        print(f"(How much do concepts vary with different VdC offsets?)")
        print()
        
        # Get all model-variant combinations
        all_variants = []
        for model in MODELS:
            for variant in VARIANTS_SHORT.values():
                all_variants.append(f"{model}__{variant}")
        
        for model_variant in all_variants:
            print_intra_variant_analysis(concept_data, model_variant, seq_len)
        
        # Analyze inter-variant comparison (quantization effect) for each model
        for model in MODELS:
            print_inter_variant_analysis(all_data, seq_len, model_filter=model)
        
        # Compare between models
        print_comparative_analysis(all_data, seq_len)
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    # Calculate totals
    total_experiments = len(MODELS) * len(VARIANTS) * len(SEQ_LENS) * NUM_RUNS
    total_files = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.delm.parquet')])
    
    print(f"\n📋 DATA SUMMARY:")
    print(f"  Models analyzed: {', '.join(MODELS)}")
    print(f"  Quantization variants per model: {len(VARIANTS)}")
    print(f"  VdC runs per variant: {NUM_RUNS}")
    print(f"  Sequence lengths analyzed: {SEQ_LENS}")
    print(f"  Total experiments analyzed: {total_experiments}")
    print(f"  Total files found: {total_files}")
    print(f"  VdC codes loaded from: {VDC_CODES_BINARY}")
    
    print(f"\n🔍 KEY INSIGHTS AVAILABLE:")
    print("1. Intrinsic VdC Variability: Concept consistency across VdC offsets")
    print("2. Quantization Comparison: Which variants produce similar concepts?")
    print("3. Model Differences: How do Llama and Qwen compare?")
    print("4. Sequence Length Effects: How does seq_len affect concept generation?")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()