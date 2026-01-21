#!/usr/bin/env python3
"""
Interactive viewer for .delm.parquet results.

Navigate and explore saved sampling results with decoded sequences.
"""

import pyarrow.parquet as pq
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer
import plotext as plt


def load_results(filepath: str):
    """Load results from .delm.parquet file."""
    table = pq.read_table(filepath)
    metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
    
    # Convert to dict
    data = {
        'codes': table['code'].to_pylist(),
        'sequences': table['sequence'].to_pylist(),
        'lengths': table['sequence_length'].to_pylist(),
        'terminated': table['terminated'].to_pylist(),
        'in_effective_set': table['in_effective_set'].to_pylist(),
    }
    
    # Reconstruct mass history
    mass_history = []
    if 'mass_history_samples' in metadata and 'mass_history_masses' in metadata:
        samples = json.loads(metadata['mass_history_samples'])
        masses = json.loads(metadata['mass_history_masses'])
        mass_history = list(zip(samples, masses))
    
    return data, metadata, mass_history


def display_overview(data, metadata, mass_history):
    """Display overview matching the live sampling output."""
    print(f"\n{'='*80}")
    print("RESULTS OVERVIEW")
    print(f"{'='*80}")
    
    # Plot mass convergence
    if mass_history:
        plt.clf()
        x_vals = [x for x, _ in mass_history]
        y_vals = [y for _, y in mass_history]
        
        prob_threshold = float(metadata.get('prob_threshold', 0.9))
        prefix_len = int(metadata.get('prefix_len', 4))
        
        # Threshold line
        plt.plot(
            x_vals,
            [prob_threshold] * len(x_vals),
            color='red',
            label='threshold'
        )
        
        # Mass curve
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
    
    # Print status
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    
    threshold_reached = metadata.get('threshold_reached', 'Unknown').lower() == 'true'
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
    
    elapsed = metadata.get('elapsed_time', None)
    if elapsed:
        elapsed_val = float(elapsed)
        samples_done = int(metadata.get('total_sequences', 0))
        print(f"\nTime: {elapsed_val:.1f}s ({samples_done/elapsed_val:.1f} samp/s)")
    
    print(f"{'='*80}\n")
    print("📄 Loaded from file (not recomputed)")
    print(f"{'='*80}\n")


def interactive_viewer(filepath: str):
    """
    Interactive viewer for exploring results.
    
    Commands:
        [number]     - View sequence at index
        e[number]    - View nth sequence in effective set
        list [n]     - List first n sequences (default 10)
        elist [n]    - List first n effective set sequences (default 10)
        meta         - Show all metadata
        overview     - Show overview plot and stats again
        q/quit/exit  - Quit
    """
    print(f"Loading results from {filepath}...")
    data, metadata, mass_history = load_results(filepath)
    
    # Get effective set indices
    effective_indices = [i for i, flag in enumerate(data['in_effective_set']) if flag]
    
    # Load tokenizer
    tokenizer = None
    if 'model_name' in metadata:
        try:
            tokenizer_name = metadata.get('tokenizer_name', metadata['model_name'])
            print(f"Loading tokenizer: {tokenizer_name}...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            print("✓ Tokenizer loaded")
        except Exception as e:
            print(f"⚠ Could not load tokenizer: {e}")
            print("Will show raw token IDs instead")
    
    # Display initial overview
    display_overview(data, metadata, mass_history)
    
    # Interactive loop
    print("Interactive viewer ready. Type 'help' for commands.\n")
    
    while True:
        try:
            cmd = input(">>> ").strip()
            
            if not cmd:
                continue
            
            if cmd.lower() in ['q', 'quit', 'exit']:
                print("Goodbye!")
                break
            
            elif cmd.lower() == 'help':
                print("\nCommands:")
                print("  [number]     - View sequence at index")
                print("  e[number]    - View nth sequence in effective set (e.g., 'e5')")
                print("  list [n]     - List first n sequences (default 10)")
                print("  elist [n]    - List first n effective set sequences (default 10)")
                print("  meta         - Show all metadata")
                print("  overview     - Show overview plot and stats again")
                print("  q/quit/exit  - Quit")
                print()
            
            elif cmd.lower() == 'overview':
                display_overview(data, metadata, mass_history)
            
            elif cmd.lower() == 'meta':
                print(f"\n{'='*80}")
                print("METADATA")
                print(f"{'='*80}")
                for key, value in sorted(metadata.items()):
                    if key in ['mass_history_samples', 'mass_history_masses']:
                        print(f"{key:30s}: [array data]")
                    elif key == 'actual_prompt' and len(value) > 100:
                        print(f"{key:30s}: {value[:100]}...")
                    else:
                        print(f"{key:30s}: {value}")
                print(f"{'='*80}\n")
            
            elif cmd.lower().startswith('list'):
                parts = cmd.split()
                n = int(parts[1]) if len(parts) > 1 else 10
                n = min(n, len(data['sequences']))
                
                print(f"\nShowing first {n} sequences:")
                for i in range(n):
                    seq = data['sequences'][i]
                    in_eff = '✓' if data['in_effective_set'][i] else ' '
                    terminated = '⏎' if data['terminated'][i] else '·'
                    
                    if tokenizer:
                        text = tokenizer.decode(seq, skip_special_tokens=True)
                        preview = text[:60] + '...' if len(text) > 60 else text
                    else:
                        preview = str(seq[:10]) + '...'
                    
                    print(f"  [{i:5d}] {in_eff} {terminated} ({len(seq):3d} tok) {preview}")
                print()
            
            elif cmd.lower().startswith('elist'):
                parts = cmd.split()
                n = int(parts[1]) if len(parts) > 1 else 10
                n = min(n, len(effective_indices))
                
                print(f"\nShowing first {n} effective set sequences:")
                for eff_idx in range(n):
                    i = effective_indices[eff_idx]
                    seq = data['sequences'][i]
                    terminated = '⏎' if data['terminated'][i] else '·'
                    
                    if tokenizer:
                        text = tokenizer.decode(seq, skip_special_tokens=True)
                        preview = text[:60] + '...' if len(text) > 60 else text
                    else:
                        preview = str(seq[:10]) + '...'
                    
                    print(f"  e{eff_idx:4d} [idx={i:5d}] {terminated} ({len(seq):3d} tok) {preview}")
                print()
            
            elif cmd.lower().startswith('e'):
                # Effective set index
                try:
                    eff_idx = int(cmd[1:])
                    if eff_idx < 0 or eff_idx >= len(effective_indices):
                        print(f"Error: Effective set index out of range (0-{len(effective_indices)-1})")
                        continue
                    
                    i = effective_indices[eff_idx]
                    seq = data['sequences'][i]
                    
                    print(f"\n{'='*80}")
                    print(f"EFFECTIVE SET SEQUENCE {eff_idx} (absolute index {i})")
                    print(f"{'='*80}")
                    print(f"VdC code    : {data['codes'][i]:.10f}")
                    print(f"Length      : {len(seq)} tokens")
                    print(f"Terminated  : {data['terminated'][i]}")
                    print(f"In eff set  : {data['in_effective_set'][i]}")
                    
                    if tokenizer:
                        text = tokenizer.decode(seq, skip_special_tokens=True)
                        print(f"\nDecoded text:")
                        print("-" * 80)
                        print(text)
                        print("-" * 80)
                    else:
                        print(f"\nToken IDs:")
                        print("-" * 80)
                        print(seq)
                        print("-" * 80)
                    print()
                except ValueError:
                    print("Error: Invalid effective set index format. Use e[number], e.g., 'e5'")
            
            elif cmd.isdigit():
                # Absolute index
                i = int(cmd)
                if i < 0 or i >= len(data['sequences']):
                    print(f"Error: Index out of range (0-{len(data['sequences'])-1})")
                    continue
                
                seq = data['sequences'][i]
                
                print(f"\n{'='*80}")
                print(f"SEQUENCE {i}")
                print(f"{'='*80}")
                print(f"VdC code    : {data['codes'][i]:.10f}")
                print(f"Length      : {len(seq)} tokens")
                print(f"Terminated  : {data['terminated'][i]}")
                print(f"In eff set  : {data['in_effective_set'][i]}")
                
                if tokenizer:
                    text = tokenizer.decode(seq, skip_special_tokens=True)
                    print(f"\nDecoded text:")
                    print("-" * 80)
                    print(text)
                    print("-" * 80)
                else:
                    print(f"\nToken IDs:")
                    print("-" * 80)
                    print(seq)
                    print("-" * 80)
                print()
            
            else:
                print("Unknown command. Type 'help' for commands.")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python view_results.py <path/to/file.delm.parquet>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    interactive_viewer(filepath)
