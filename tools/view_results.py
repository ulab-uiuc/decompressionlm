#!/usr/bin/env python3
"""
Interactive viewer for .delm.parquet results with graph analysis support.

Navigate and explore saved sampling results with decoded sequences and graph metrics.
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

    # Check if graph metrics exist
    has_graph_metrics = 'graph_num_nodes' in metadata

    return data, metadata, mass_history, has_graph_metrics


def is_ascii_concept(text: str) -> bool:
    """Check if concept contains only ASCII characters."""
    return all(ord(c) < 128 for c in text)


def extract_concepts_from_sequences(sequences, tokenizer):
    """
    Extract all concepts from sequences by decoding and splitting by newlines.
    Returns list of (seq_idx, concept_idx, concept_text) tuples.
    """
    all_concepts = []

    for seq_idx, seq in enumerate(sequences):
        # Decode sequence
        text = tokenizer.decode(seq, skip_special_tokens=True)

        # Split into concepts (one per line)
        lines = text.split('\n')
        concepts = [line.strip() for line in lines if line.strip()]

        # Store with indices
        for concept_idx, concept in enumerate(concepts):
            all_concepts.append((seq_idx, concept_idx, concept))

    return all_concepts


def print_concept_flat(metadata, limit=None):
    """
    Print graph concepts from most frequent/connected to least, as:
      c1, c2, c3, ...
    No line breaks (one long line).

    Uses metadata['graph_concept_frequencies'], expected to be JSON like:
      [[concept, degree], [concept, degree], ...]
    """
    if 'graph_concept_frequencies' not in metadata:
        print("\n⚠️  Concept frequencies not saved in this file (old format)")
        print("    Re-run with updated code to save graph_concept_frequencies\n")
        return

    concept_freqs = json.loads(metadata['graph_concept_frequencies'])
    if not concept_freqs:
        print("\n(no concepts)\n")
        return

    if limit is not None:
        try:
            limit = int(limit)
            if limit < 0:
                limit = 0
        except Exception:
            limit = None

    if limit is not None:
        concept_freqs = concept_freqs[:limit]

    concepts = [c for (c, degree) in concept_freqs]
    print(", ".join(concepts))


def display_overview(data, metadata, mass_history, has_graph_metrics):
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

    print(f"{'='*80}")

    # Display graph metrics if available
    if has_graph_metrics:
        print(f"\n{'='*80}")
        print("GRAPH ANALYSIS")
        print(f"{'='*80}")

        print("\nConcept Extraction:")
        print(f"  Total concepts extracted   : {metadata.get('graph_total_concepts', 'N/A')}")
        print(f"  Unique before merging      : {metadata.get('graph_unique_before_merge', 'N/A')}")
        print(f"  Unique after merging       : {metadata.get('graph_unique_after_merge', 'N/A')}")
        print(f"  Total raw edges            : {metadata.get('graph_total_edges_raw', 'N/A')}")

        print("\nGraph Statistics:")
        print(f"  Nodes (concepts)           : {metadata.get('graph_num_nodes', 'N/A')}")
        print(f"  Edges (relations)          : {metadata.get('graph_num_edges', 'N/A')}")
        print(f"  Graph density              : {metadata.get('graph_density', 'N/A')}")
        print(f"  Average degree             : {metadata.get('graph_avg_degree', 'N/A')}")

        print("\nNode Connectivity:")
        orphan_nodes = metadata.get('graph_orphan_nodes', 'N/A')
        orphan_ratio = metadata.get('graph_orphan_ratio', 'N/A')
        if orphan_ratio != 'N/A':
            orphan_ratio = f"{float(orphan_ratio)*100:.1f}%"
        print(f"  Orphan nodes (degree=0)    : {orphan_nodes} ({orphan_ratio})")

        weakly_connected = metadata.get('graph_weakly_connected_nodes', 'N/A')
        weakly_ratio = metadata.get('graph_weakly_connected_ratio', 'N/A')
        if weakly_ratio != 'N/A':
            weakly_ratio = f"{float(weakly_ratio)*100:.1f}%"
        print(f"  Weakly connected (deg≤1)   : {weakly_connected} ({weakly_ratio})")

        # Calculate well connected
        try:
            num_nodes = int(metadata.get('graph_num_nodes', 0))
            weakly_conn = int(metadata.get('graph_weakly_connected_nodes', 0))
            well_connected = num_nodes - weakly_conn
            well_ratio = (well_connected / num_nodes * 100) if num_nodes > 0 else 0
            print(f"  Well connected (deg>1)     : {well_connected} ({well_ratio:.1f}%)")
        except:
            print(f"  Well connected (deg>1)     : N/A")

        print("\nGraph Structure:")
        print(f"  Connected components       : {metadata.get('graph_num_components', 'N/A')}")

        largest_size = metadata.get('graph_largest_component_size', 'N/A')
        largest_ratio = metadata.get('graph_largest_component_ratio', 'N/A')
        if largest_ratio != 'N/A':
            largest_ratio = f"{float(largest_ratio)*100:.1f}%"
        print(f"  Largest component          : {largest_size} nodes ({largest_ratio})")
        print(f"  Largest component density  : {metadata.get('graph_largest_component_density', 'N/A')}")

        print(f"{'='*80}\n")
    else:
        print(f"\n⚠️  Graph analysis metrics not found in file (old format)")
        print(f"    Run estimate_prefix_mass() with enable_graph_analysis=True to add them")
        print(f"{'='*80}\n")

    print("📄 Loaded from file (not recomputed)")
    print(f"{'='*80}\n")


def interactive_viewer(filepath: str):
    """
    Interactive viewer for exploring results.

    Commands:
        [number]            - View sequence at index
        e[number]           - View nth sequence in effective set
        c[number]           - View nth concept
        list [n]            - List first n sequences (default 10)
        elist [n]           - List first n effective set sequences (default 10)
        clist [n]           - List first n concepts (default 20)
        concepts            - Show concept statistics
        concept-flat [n]    - Print concepts: c1, c2, c3, ... (one line, optional top-n)
        meta                - Show all metadata
        graph               - Show graph metrics (if available)
        overview            - Show overview plot and stats again
        q/quit/exit         - Quit
    """
    print(f"Loading results from {filepath}...")
    data, metadata, mass_history, has_graph_metrics = load_results(filepath)

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

    # Extract all concepts if tokenizer available
    all_concepts = []
    if tokenizer:
        print("Extracting concepts from sequences...")
        all_concepts = extract_concepts_from_sequences(data['sequences'], tokenizer)
        print(f"✓ Extracted {len(all_concepts)} concepts from {len(data['sequences'])} sequences")

    # Display initial overview
    display_overview(data, metadata, mass_history, has_graph_metrics)

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
                print("  [number]            - View sequence at index")
                print("  e[number]           - View nth sequence in effective set (e.g., 'e5')")
                print("  c[number]           - View nth concept (e.g., 'c10')")
                print("  list [n]            - List first n sequences (default 10)")
                print("  elist [n]           - List first n effective set sequences (default 10)")
                print("  clist [n]           - List first n concepts (default 20)")
                print("  concepts            - Show concept statistics")
                print("  concept-flat [n]    - Print graph concepts as one comma-separated line (optional top-n)")
                print("  meta                - Show all metadata")
                print("  graph               - Show graph metrics (if available)")
                print("  overview            - Show overview plot and stats again")
                print("  q/quit/exit         - Quit")
                print()

            elif cmd.lower() == 'overview':
                display_overview(data, metadata, mass_history, has_graph_metrics)

            elif cmd.lower().startswith('concept-flat'):
                # concept-flat [n]
                parts = cmd.split()
                n = parts[1] if len(parts) > 1 else None
                print_concept_flat(metadata, n)

            elif cmd.lower() == 'graph':
                if has_graph_metrics:
                    print(f"\n{'='*80}")
                    print("GRAPH ANALYSIS")
                    print(f"{'='*80}")

                    print("\nConcept Extraction:")
                    print(f"  Total concepts extracted   : {metadata.get('graph_total_concepts', 'N/A')}")
                    print(f"  Unique before merging      : {metadata.get('graph_unique_before_merge', 'N/A')}")
                    print(f"  Unique after merging       : {metadata.get('graph_unique_after_merge', 'N/A')}")
                    print(f"  Total raw edges            : {metadata.get('graph_total_edges_raw', 'N/A')}")

                    print("\nGraph Statistics:")
                    print(f"  Nodes (concepts)           : {metadata.get('graph_num_nodes', 'N/A')}")
                    print(f"  Edges (relations)          : {metadata.get('graph_num_edges', 'N/A')}")
                    print(f"  Graph density              : {metadata.get('graph_density', 'N/A')}")
                    print(f"  Average degree             : {metadata.get('graph_avg_degree', 'N/A')}")

                    print("\nNode Connectivity:")
                    orphan_nodes = metadata.get('graph_orphan_nodes', 'N/A')
                    orphan_ratio = metadata.get('graph_orphan_ratio', 'N/A')
                    if orphan_ratio != 'N/A':
                        orphan_ratio = f"{float(orphan_ratio)*100:.1f}%"
                    print(f"  Orphan nodes (degree=0)    : {orphan_nodes} ({orphan_ratio})")

                    weakly_connected = metadata.get('graph_weakly_connected_nodes', 'N/A')
                    weakly_ratio = metadata.get('graph_weakly_connected_ratio', 'N/A')
                    if weakly_ratio != 'N/A':
                        weakly_ratio = f"{float(weakly_ratio)*100:.1f}%"
                    print(f"  Weakly connected (deg≤1)   : {weakly_connected} ({weakly_ratio})")

                    try:
                        num_nodes = int(metadata.get('graph_num_nodes', 0))
                        weakly_conn = int(metadata.get('graph_weakly_connected_nodes', 0))
                        well_connected = num_nodes - weakly_conn
                        well_ratio = (well_connected / num_nodes * 100) if num_nodes > 0 else 0
                        print(f"  Well connected (deg>1)     : {well_connected} ({well_ratio:.1f}%)")
                    except:
                        print(f"  Well connected (deg>1)     : N/A")

                    print("\nGraph Structure:")
                    print(f"  Connected components       : {metadata.get('graph_num_components', 'N/A')}")

                    largest_size = metadata.get('graph_largest_component_size', 'N/A')
                    largest_ratio = metadata.get('graph_largest_component_ratio', 'N/A')
                    if largest_ratio != 'N/A':
                        largest_ratio = f"{float(largest_ratio)*100:.1f}%"
                    print(f"  Largest component          : {largest_size} nodes ({largest_ratio})")
                    print(f"  Largest component density  : {metadata.get('graph_largest_component_density', 'N/A')}")

                    print(f"{'='*80}\n")
                else:
                    print("\n⚠️  No graph metrics in this file (old format)")
                    print("    Load with estimate_prefix_mass(enable_graph_analysis=True) to add them\n")

            elif cmd.lower() == 'concepts':
                if not has_graph_metrics:
                    print("\n⚠️  No graph metrics in this file (old format)")
                    print("    Run estimate_prefix_mass() with enable_graph_analysis=True to add them\n")
                else:
                    # Load concept frequencies from metadata
                    if 'graph_concept_frequencies' in metadata:
                        concept_freqs = json.loads(metadata['graph_concept_frequencies'])

                        print(f"\n{'='*80}")
                        print("FINAL GRAPH CONCEPTS (after filtering & merging)")
                        print(f"{'='*80}")

                        total_extracted = metadata.get('graph_total_concepts', 'N/A')
                        unique_before = metadata.get('graph_unique_before_merge', 'N/A')
                        unique_after = metadata.get('graph_unique_after_merge', 'N/A')

                        print(f"Total concepts extracted   : {total_extracted}")
                        print(f"Unique before merging      : {unique_before}")
                        print(f"Unique after merging       : {unique_after}")

                        if unique_before != 'N/A' and unique_after != 'N/A':
                            reduction = (1 - int(unique_after) / int(unique_before)) * 100
                            print(f"Merge reduction            : {reduction:.1f}%")

                        print(f"\nFinal graph nodes          : {len(concept_freqs)}")
                        print()

                        # Show top 64
                        n = min(64, len(concept_freqs))
                        print(f"Top {n} most connected concepts:")
                        for i, (concept, degree) in enumerate(concept_freqs[:n], 1):
                            preview = concept[:60] + '...' if len(concept) > 60 else concept
                            print(f"  {i:2d}. (degree={degree:3d}) {preview}")

                        print(f"{'='*80}\n")
                    else:
                        # Old file format - no concept list saved
                        print("\n⚠️  Concept list not saved in this file (old format)")
                        print("    Re-run with updated code to save concept frequencies")
                        print("\n📊 Summary stats from metadata:")
                        print(f"    Total extracted: {metadata.get('graph_total_concepts', 'N/A')}")
                        print(f"    Final nodes: {metadata.get('graph_num_nodes', 'N/A')}\n")

            elif cmd.lower().startswith('clist'):
                if not all_concepts:
                    print("\n⚠️  No concepts available (tokenizer not loaded)\n")
                else:
                    parts = cmd.split()
                    n = int(parts[1]) if len(parts) > 1 else 20
                    n = min(n, len(all_concepts))

                    print(f"\nShowing first {n} concepts:")
                    for i in range(n):
                        seq_idx, concept_idx, concept = all_concepts[i]
                        preview = concept[:60] + '...' if len(concept) > 60 else concept
                        print(f"  c{i:5d} [seq {seq_idx:5d}, line {concept_idx}] {preview}")
                    print()

            elif cmd.lower().startswith('c') and cmd[1:].isdigit():
                if not all_concepts:
                    print("\n⚠️  No concepts available (tokenizer not loaded)\n")
                else:
                    concept_idx = int(cmd[1:])
                    if concept_idx < 0 or concept_idx >= len(all_concepts):
                        print(f"Error: Concept index out of range (0-{len(all_concepts)-1})")
                        continue

                    seq_idx, line_idx, concept = all_concepts[concept_idx]

                    print(f"\n{'='*80}")
                    print(f"CONCEPT {concept_idx}")
                    print(f"{'='*80}")
                    print(f"From sequence : {seq_idx}")
                    print(f"Line number   : {line_idx}")
                    print(f"Length        : {len(concept)} chars")
                    print(f"\nConcept text:")
                    print("-" * 80)
                    print(concept)
                    print("-" * 80)

                    # Show context (surrounding concepts from same sequence)
                    same_seq_concepts = [(i, c) for i, (s, l, c) in enumerate(all_concepts) if s == seq_idx]
                    if len(same_seq_concepts) > 1:
                        print(f"\nContext (all concepts from sequence {seq_idx}):")
                        for i, (global_idx, c) in enumerate(same_seq_concepts):
                            marker = ">>> " if global_idx == concept_idx else "    "
                            preview = c[:60] + '...' if len(c) > 60 else c
                            print(f"{marker}[{i}] {preview}")
                    print()

            elif cmd.lower() == 'meta':
                print(f"\n{'='*80}")
                print("METADATA")
                print(f"{'='*80}")

                # Group metadata by category
                sampling_keys = ['prefix_len', 'prob_threshold', 'max_samples', 'max_len',
                                 'batch_size', 'offset', 'display_interval', 'use_chat_template']
                results_keys = ['total_sequences', 'unique_prefixes_discovered', 'final_prefix_mass',
                                'threshold_reached', 'elapsed_time', 'effective_set_size',
                                'effective_set_total_tokens', 'effective_set_min_tokens',
                                'effective_set_max_tokens', 'effective_set_avg_tokens']
                model_keys = ['model_name', 'tokenizer_name', 'transformers_version', 'torch_version']
                graph_keys = [k for k in metadata.keys() if k.startswith('graph_')]

                print("\n--- Sampling Parameters ---")
                for key in sampling_keys:
                    if key in metadata:
                        print(f"  {key:30s}: {metadata[key]}")

                print("\n--- Results ---")
                for key in results_keys:
                    if key in metadata:
                        print(f"  {key:30s}: {metadata[key]}")

                print("\n--- Model Info ---")
                for key in model_keys:
                    if key in metadata:
                        val = metadata[key]
                        if len(val) > 60:
                            val = val[:60] + "..."
                        print(f"  {key:30s}: {val}")

                if graph_keys:
                    print("\n--- Graph Metrics ---")
                    for key in sorted(graph_keys):
                        print(f"  {key:30s}: {metadata[key]}")

                # Show prompt if available
                if 'actual_prompt' in metadata:
                    prompt = metadata['actual_prompt']
                    print("\n--- Prompt ---")
                    if len(prompt) > 200:
                        print(f"  {prompt[:200]}...")
                        print(f"  (truncated, {len(prompt)} chars total)")
                    else:
                        print(f"  {prompt}")

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
