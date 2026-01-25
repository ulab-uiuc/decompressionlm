#!/usr/bin/env python3
"""
Generate quantization comparison tables from saved .delm.parquet files.
Checks for required files first, then loads data and prints formatted tables.
"""

import os
import json
import glob
import sys
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

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
        
        # Extract graph metrics from metadata
        data = {
            'avg_tokens': float(metadata.get('effective_set_avg_tokens', 0)),
            'nodes': int(metadata.get('graph_num_nodes', 0)),
            'edges': int(metadata.get('graph_num_edges', 0)),
            'density': float(metadata.get('graph_density', 0)),
            'avg_deg': float(metadata.get('graph_avg_degree', 0)),
            'components': int(metadata.get('graph_num_components', 1)),
            'largest_cc': int(metadata.get('graph_largest_component_size', 0)),
        }
        
        # Calculate largest CC percentage
        if data['nodes'] > 0:
            data['largest_cc_pct'] = (data['largest_cc'] / data['nodes']) * 100
        else:
            data['largest_cc_pct'] = 0.0
            
        return data, metadata
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def load_results_data():
    """Load all results data from parquet files."""
    all_data = {}
    
    for seq_len in SEQ_LENS:
        all_data[seq_len] = {}
        
        for task in TASKS:
            all_data[seq_len][task] = {}
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
                
                # Load data
                data, metadata = load_file_metadata(file_path)
                if data:
                    # Use full label as key
                    label = f"{model_found}__{variant_found}"
                    all_data[seq_len][task][label] = data
    
    return all_data

def format_table_rows(data, model, seq_len, task):
    """Format rows for a single table."""
    # Filter data for this specific model, seq_len, and task
    task_data = {}
    for label, metrics in data[seq_len][task].items():
        if model in label:
            task_data[label] = metrics
    
    if not task_data:
        return None
    
    # CORRECTION: Sort by nodes (high to low) as shown in the example tables
    sorted_items = sorted(
        task_data.items(), 
        key=lambda x: x[1]['nodes'], 
        reverse=True
    )
    
    # Create table rows
    table_rows = []
    for rank, (label, metrics) in enumerate(sorted_items, 1):
        # Extract variant short name
        variant_short = label.replace(f"{model}__", "")
        # Map to display name
        variant_display = VARIANTS_SHORT.get(variant_short, variant_short)
        
        row = {
            'rank': rank,
            'variant': variant_display,
            'avg_tokens': metrics['avg_tokens'],
            'nodes': metrics['nodes'],
            'edges': metrics['edges'],
            'density': metrics['density'],
            'avg_deg': metrics['avg_deg'],
            'components': metrics['components'],
            'largest_cc': metrics['largest_cc'],
            'largest_cc_pct': metrics['largest_cc_pct']
        }
        table_rows.append(row)
    
    return table_rows

def print_table(rows, title=""):
    """Print a formatted markdown table."""
    if not rows:
        print(f"No data for {title}")
        return
    
    if title:
        print(f"\n{title}")
    
    print()
    print("| Rank | Variant   | Avg Tokens/Seq | Nodes | Edges | Density | Avg Deg | Components |   Largest CC |")
    print("| ---: | --------- | -------------: | ----: | ----: | ------: | ------: | ---------: | -----------: |")
    
    for row in rows:
        # Format numbers
        avg_tokens = f"{row['avg_tokens']:.1f}"
        nodes = f"{row['nodes']:,}".replace(',', ' ')
        edges = f"{row['edges']:,}".replace(',', ' ')
        
        # Format density - adjust decimal places based on magnitude
        if row['density'] >= 0.01:
            density = f"{row['density']:.4f}"
        elif row['density'] >= 0.001:
            density = f"{row['density']:.5f}"
        else:
            density = f"{row['density']:.6f}"
        
        avg_deg = f"{row['avg_deg']:.2f}"
        components = row['components']
        largest_cc = f"{row['largest_cc']:,} ({row['largest_cc_pct']:.1f}%)".replace(',', ' ')
        
        print(f"| {row['rank']:4d} | {row['variant']:10} | {avg_tokens:>14} | {nodes:>5} | {edges:>5} | {density:>7} | {avg_deg:>7} | {components:10d} | {largest_cc:>12} |")

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
    
    # Load data
    print("\nLoading data from result files...")
    data = load_results_data()
    
    # Generate tables
    print("\n" + "="*80)
    print("QUANTIZATION COMPARISON TABLES")
    print("="*80)
    
    for model in MODELS:
        print(f"\n{'='*80}")
        print(f"## {model}")
        print(f"{'='*80}")
        
        for seq_len in SEQ_LENS:
            print(f"\n### Seq len {seq_len}")
            
            for task in TASKS:
                # Format task title
                task_title = TASK_DISPLAY_NAMES.get(task, task.upper())
                
                print(f"\n{task_title}")
                
                # Generate and print table
                table_rows = format_table_rows(data, model, seq_len, task)
                if table_rows:
                    print_table(table_rows)
                else:
                    print("\nNo data available.\n")
            
            print()  # Add spacing between sequence lengths
    
    print("\n" + "="*80)
    print("All tables generated successfully!")
    print("="*80)

if __name__ == "__main__":
    main()