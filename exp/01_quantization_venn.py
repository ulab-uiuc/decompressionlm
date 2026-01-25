#!/usr/bin/env python3
"""
Quantization concept overlap analysis.
Analyzes concept overlap between pairs of quantization settings for each model and task.
"""

import os
import json
import glob
import sys
from pathlib import Path
import pyarrow.parquet as pq
from itertools import combinations

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
TASKS = ["us_law", "git_ver_control", "radiology_imaging", "faa"]
SEQ_LENS = [16, 32]

# Task name mapping for display
TASK_DISPLAY_NAMES = {
    "us_law": "US LAW",
    "git_ver_control": "GIT VERSION CONTROL",
    "radiology_imaging": "RADIOLOGY IMAGING",
    "faa": "FAA"
}

# Directory patterns for finding files
DIR_PATTERNS = {
    16: {
        "us_law": "results/quant_ladder__us_law_16_newlines",
        "git_ver_control": "results/quant_ladder__git_ver_control_16_newlines",
        "radiology_imaging": "results/quant_ladder__radiology_imaging_16_newlines",
        "faa": "results/quant_ladder__faa_16_newlines",
    },
    32: {
        "us_law": "results/quant_ladder__us_law_32_newlines",
        "git_ver_control": "results/quant_ladder__git_ver_control_32_newlines",
        "radiology_imaging": "results/quant_ladder__radiology_imaging_32_newlines",
        "faa": "results/quant_ladder__faa_32_newlines",
    }
}

# Expected file naming pattern
FILE_PATTERN = "*__L{}__T1.0.delm.parquet"

# ----------------------------
# Helper Functions
# ----------------------------
def check_all_files_exist():
    """Check if all expected result files exist."""
    missing_files = []
    
    for seq_len in SEQ_LENS:
        for task in TASKS:
            dir_path = DIR_PATTERNS[seq_len][task]
            if not os.path.exists(dir_path):
                missing_files.append(f"Directory missing: {dir_path}")
                continue
                
            for model in MODELS:
                for variant in VARIANTS:
                    # Construct expected filename pattern
                    if model == "Qwen2.5-7B-Instruct":
                        label = f"Qwen2.5-7B-Instruct__{variant}"
                    else:  # Llama-3.1-8B-Instruct
                        label = f"Llama-3.1-8B-Instruct__{variant}"
                    
                    # Determine save_stem based on task
                    if task == "us_law":
                        save_stem = "__us_law_bar_meta"
                    elif task == "git_ver_control":
                        save_stem = "__git_ver_control_bar_meta"
                    elif task == "radiology_imaging":
                        save_stem = "__radiology_imaging_bar_meta"
                    else:  # faa
                        save_stem = "__faa_bar_meta"
                    
                    filename = f"{label}{save_stem}__L{seq_len}__T1.0.delm.parquet"
                    file_path = os.path.join(dir_path, filename)
                    
                    if not os.path.exists(file_path):
                        missing_files.append(file_path)
    
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
        print(f"  Warning: No graph_concept_frequencies in metadata")
        return set()
    
    try:
        concept_freqs = json.loads(metadata['graph_concept_frequencies'])
        # Extract just the concept names, ignore frequencies
        concepts = {concept for concept, _ in concept_freqs}
        return concepts
    except Exception as e:
        print(f"Error parsing concepts from metadata: {e}")
        return set()

def load_concept_data():
    """Load all concept data from parquet files."""
    concept_data = {}
    
    for seq_len in SEQ_LENS:
        concept_data[seq_len] = {}
        
        for task in TASKS:
            concept_data[seq_len][task] = {}
            dir_path = DIR_PATTERNS[seq_len][task]
            
            if not os.path.exists(dir_path):
                print(f"Warning: Directory not found: {dir_path}")
                continue
            
            # Find all parquet files for this task and seq_len
            pattern = os.path.join(dir_path, FILE_PATTERN.format(seq_len))
            files = glob.glob(pattern)
            
            if not files:
                print(f"Warning: No files found for {task} seq_len={seq_len}")
                continue
            
            for file_path in files:
                # Extract model and variant from filename
                filename = os.path.basename(file_path)
                
                # Skip if not a valid model variant
                model_found = None
                variant_found = None
                
                for model in MODELS:
                    for variant in VARIANTS:
                        expected_pattern = f"{model}__{variant}__"
                        if expected_pattern in filename:
                            model_found = model
                            variant_found = variant
                            break
                    if model_found:
                        break
                
                if not model_found or not variant_found:
                    print(f"Warning: Could not parse model/variant from {filename}")
                    continue
                
                # Load metadata
                metadata = load_file_metadata(file_path)
                if metadata:
                    # Extract concepts
                    concepts = extract_concepts_from_metadata(metadata)
                    
                    # Use full label as key
                    label = f"{model_found}__{variant_found}"
                    concept_data[seq_len][task][label] = {
                        'concepts': concepts,
                        'concept_count': len(concepts),
                        'variant_short': variant_found,
                        'variant_display': VARIANTS_SHORT.get(variant_found, variant_found)
                    }
    
    return concept_data

def print_pairwise_overlap_table(concept_info, model, seq_len, task):
    """Print pairwise concept overlap table for all variant pairs."""
    task_title = TASK_DISPLAY_NAMES.get(task, task.upper())
    print(f"\n{'='*80}")
    print(f"PAIRWISE CONCEPT OVERLAP: {model} - {task_title} - Seq Len {seq_len}")
    print(f"{'='*80}")
    
    # Get variant data for this model
    variant_data = {}
    for label, info in concept_info.items():
        if model in label:
            variant_data[info['variant_display']] = info
    
    if not variant_data:
        print("No concept data available.")
        return
    
    # Sort variants by concept count (high to low)
    sorted_variants = sorted(
        variant_data.items(),
        key=lambda x: x[1]['concept_count'],
        reverse=True
    )
    
    # Print concept counts per variant
    print("\n📊 CONCEPT COUNTS PER VARIANT:")
    print()
    print("| Variant   | Concepts |")
    print("| --------- | -------: |")
    for variant_display, info in sorted_variants:
        print(f"| {variant_display:10} | {info['concept_count']:8,} |")
    
    # Generate all unique pairs (order doesn't matter)
    variant_names = [v[0] for v in sorted_variants]
    pairs = list(combinations(variant_names, 2))
    
    # Calculate overlap for each pair
    pair_overlaps = []
    for var1, var2 in pairs:
        concepts1 = variant_data[var1]['concepts']
        concepts2 = variant_data[var2]['concepts']
        
        overlap = concepts1.intersection(concepts2)
        overlap_count = len(overlap)
        
        # Calculate percentages relative to each variant
        pct1 = (overlap_count / len(concepts1) * 100) if concepts1 else 0
        pct2 = (overlap_count / len(concepts2) * 100) if concepts2 else 0
        
        # Calculate Jaccard similarity (intersection over union)
        union = concepts1.union(concepts2)
        jaccard = (overlap_count / len(union) * 100) if union else 0
        
        pair_overlaps.append({
            'var1': var1,
            'var2': var2,
            'overlap_count': overlap_count,
            'pct_of_var1': pct1,
            'pct_of_var2': pct2,
            'jaccard': jaccard
        })
    
    # Sort pairs by overlap count (high to low)
    pair_overlaps.sort(key=lambda x: x['overlap_count'], reverse=True)
    
    # Print pairwise overlap table with BOTH raw overlap and Jaccard
    print(f"\n🔄 PAIRWISE OVERLAP ANALYSIS ({len(pairs)} pairs):")
    print()
    print("| Pair | Overlap Count | % of A | % of B | Jaccard % |")
    print("| ---- | ------------: | -----: | -----: | --------: |")
    
    for i, overlap in enumerate(pair_overlaps, 1):
        var1 = overlap['var1']
        var2 = overlap['var2']
        pair_name = f"{var1}–{var2}"
        
        print(f"| {pair_name:14} | {overlap['overlap_count']:13,} | {overlap['pct_of_var1']:6.1f}% | {overlap['pct_of_var2']:6.1f}% | {overlap['jaccard']:9.1f}% |")
    
    # Calculate and print summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    
    # Concepts shared by all variants
    all_concept_sets = [info['concepts'] for info in variant_data.values()]
    if all_concept_sets:
        shared_by_all = set.intersection(*all_concept_sets)
        print(f"  Concepts shared by all {len(variant_data)} variants: {len(shared_by_all):,}")
        
        if shared_by_all:
            # Calculate union of all concepts
            all_concepts = set()
            for concepts in all_concept_sets:
                all_concepts.update(concepts)
            
            shared_pct = (len(shared_by_all) / len(all_concepts) * 100) if all_concepts else 0
            print(f"  Percentage of total concepts: {shared_pct:.1f}%")
            
            # Show first 10 shared concepts
            shared_list = list(shared_by_all)
            if len(shared_list) > 10:
                print(f"  Examples: {', '.join(shared_list[:10])}...")
            else:
                print(f"  All shared concepts: {', '.join(shared_list)}")
    
    # Calculate average overlap
    if pair_overlaps:
        avg_overlap = sum(o['overlap_count'] for o in pair_overlaps) / len(pair_overlaps)
        avg_pct1 = sum(o['pct_of_var1'] for o in pair_overlaps) / len(pair_overlaps)
        avg_pct2 = sum(o['pct_of_var2'] for o in pair_overlaps) / len(pair_overlaps)
        avg_jaccard = sum(o['jaccard'] for o in pair_overlaps) / len(pair_overlaps)
        
        print(f"\n📊 AVERAGE PAIRWISE STATS:")
        print(f"  Average overlap per pair: {avg_overlap:,.0f} concepts")
        print(f"  Average % of first variant: {avg_pct1:.1f}%")
        print(f"  Average % of second variant: {avg_pct2:.1f}%")
        print(f"  Average Jaccard similarity: {avg_jaccard:.1f}%")
    
    # Find most similar pair (highest Jaccard AND highest overlap)
    if pair_overlaps:
        most_similar_jaccard = max(pair_overlaps, key=lambda x: x['jaccard'])
        most_similar_overlap = max(pair_overlaps, key=lambda x: x['overlap_count'])
        
        print(f"\n🏆 MOST SIMILAR PAIRS:")
        print(f"  By Jaccard: {most_similar_jaccard['var1']}–{most_similar_jaccard['var2']} "
              f"({most_similar_jaccard['jaccard']:.1f}% Jaccard, {most_similar_jaccard['overlap_count']:,} overlap)")
        print(f"  By Overlap: {most_similar_overlap['var1']}–{most_similar_overlap['var2']} "
              f"({most_similar_overlap['overlap_count']:,} overlap, {most_similar_overlap['jaccard']:.1f}% Jaccard)")
    
    # Find least similar pair (lowest Jaccard AND lowest overlap)
    if pair_overlaps:
        least_similar_jaccard = min(pair_overlaps, key=lambda x: x['jaccard'])
        least_similar_overlap = min(pair_overlaps, key=lambda x: x['overlap_count'])
        
        print(f"\n📉 LEAST SIMILAR PAIRS:")
        print(f"  By Jaccard: {least_similar_jaccard['var1']}–{least_similar_jaccard['var2']} "
              f"({least_similar_jaccard['jaccard']:.1f}% Jaccard, {least_similar_jaccard['overlap_count']:,} overlap)")
        print(f"  By Overlap: {least_similar_overlap['var1']}–{least_similar_overlap['var2']} "
              f"({least_similar_overlap['overlap_count']:,} overlap, {least_similar_overlap['jaccard']:.1f}% Jaccard)")
    
    print(f"\n{'='*80}")

def print_comprehensive_overlap_matrix(concept_info, model, seq_len, task):
    """Print a comprehensive overlap matrix showing all pairwise comparisons."""
    task_title = TASK_DISPLAY_NAMES.get(task, task.upper())
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE OVERLAP MATRIX: {model} - {task_title} - Seq Len {seq_len}")
    print(f"{'='*80}")
    
    # Get variant data for this model
    variant_data = {}
    for label, info in concept_info.items():
        if model in label:
            variant_data[info['variant_display']] = info
    
    if not variant_data:
        print("No concept data available.")
        return
    
    # Sort variants by concept count (high to low)
    sorted_variants = sorted(
        variant_data.items(),
        key=lambda x: x[1]['concept_count'],
        reverse=True
    )
    variant_names = [v[0] for v in sorted_variants]
    
    print("\n📊 CONCEPT COUNTS PER VARIANT (sorted by count):")
    print()
    for variant_display, info in sorted_variants:
        print(f"  {variant_display:10}: {info['concept_count']:6,} concepts")
    
    print(f"\n🔄 OVERLAP MATRIX (Count | % of row | % of column | Jaccard %):")
    print()
    
    # Header row
    header = " " * 12 + " | " + " | ".join([f"{v:^40}" for v in variant_names]) + " |"
    separator = "-" * 12 + "-|-" + "-|-".join(["-" * 40 for _ in variant_names]) + "-|"
    
    print(header)
    print(separator)
    
    # Data rows
    for i, var1 in enumerate(variant_names):
        row_cells = [f"{var1:10}"]
        concepts1 = variant_data[var1]['concepts']
        count1 = len(concepts1)
        
        for j, var2 in enumerate(variant_names):
            if i == j:
                # Diagonal: show concept count
                cell = f"{count1:,} concepts".center(40)
            else:
                concepts2 = variant_data[var2]['concepts']
                count2 = len(concepts2)
                
                # Calculate overlap
                overlap = concepts1.intersection(concepts2)
                overlap_count = len(overlap)
                
                # Calculate percentages
                pct_of_var1 = (overlap_count / count1 * 100) if count1 > 0 else 0
                pct_of_var2 = (overlap_count / count2 * 100) if count2 > 0 else 0
                
                # Calculate Jaccard
                union = concepts1.union(concepts2)
                jaccard = (overlap_count / len(union) * 100) if union else 0
                
                # Format cell - show both raw count and percentages
                cell = f"{overlap_count:4d} | {pct_of_var1:5.1f}% | {pct_of_var2:5.1f}% | {jaccard:5.1f}%".center(40)
            
            row_cells.append(cell)
        
        print(" | ".join(row_cells) + " |")
        if i < len(variant_names) - 1:
            print(separator)
    
    print(f"\n{'='*80}")

# ----------------------------
# Main Function
# ----------------------------
def main():
    print("Checking for required result files...")
    
    # Check if all files exist
    missing_files = check_all_files_exist()
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} required files:")
        for file in missing_files[:20]:  # Show first 20 missing files
            print(f"  - {file}")
        if len(missing_files) > 20:
            print(f"  ... and {len(missing_files) - 20} more")
        print("\nPlease run the quantization experiments first.")
        sys.exit(1)
    
    print("✓ All required files found!")
    
    # Load concept data
    print("\nLoading concept data from result files...")
    concept_data = load_concept_data()
    
    # Generate concept overlap analysis
    print("\n" + "="*80)
    print("QUANTIZATION CONCEPT OVERLAP ANALYSIS")
    print("="*80)
    
    for model in MODELS:
        print(f"\n{'='*80}")
        print(f"## {model}")
        print(f"{'='*80}")
        
        for seq_len in SEQ_LENS:
            print(f"\n### Seq len {seq_len}")
            
            for task in TASKS:
                # Get concept info for this combination
                concept_info = concept_data.get(seq_len, {}).get(task, {})
                
                if concept_info:
                    # Print pairwise overlap table
                    print_pairwise_overlap_table(concept_info, model, seq_len, task)
                    
                    # Optional: Print comprehensive matrix (comment out if too detailed)
                    # print_comprehensive_overlap_matrix(concept_info, model, seq_len, task)
                else:
                    task_title = TASK_DISPLAY_NAMES.get(task, task.upper())
                    print(f"\nNo concept data for {model} - {task_title} - Seq Len {seq_len}")
    
    print("\n" + "="*80)
    print("Concept overlap analysis completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()