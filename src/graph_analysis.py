"""
Graph-based knowledge analysis for LLM-generated concept sequences.

Extracts concept chains from generated sequences, merges similar concepts,
and computes graph connectivity metrics to measure knowledge structure.
"""

import networkx as nx
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from rapidfuzz.fuzz import ratio
from tqdm import tqdm
import inflect

_INFLECT = inflect.engine()

# Fragment words - 1-token concepts that are noise
FRAGMENT_WORDS = {
    "air", "flight", "minimum", "controlled", "control",
    "terminal", "class", "visual", "instrument", "restricted",
    "area", "zone", "altitude", "level", "route", "approach",
    "departure", "arrival", "pattern", "procedure", "service"
}

# Words NOT to singularize (acronyms, special cases)
SINGULAR_EXCEPTIONS = {
    "class", "airspace", "atc", "faa", "tfr", "ifr", "vfr",
    "ms", "us", "operations", "communications", "services",
    "ilis", "procedures", "systems"
}

# Edge stopwords to remove
EDGE_STOPWORDS = {"the", "a", "an"}


def normalize_concept(text: str) -> str:
    """
    Comprehensive normalization for concept matching.
    
    1) Basic text normalization (Unicode, lowercase, punctuation)
    2) ASCII filtering (reject concepts with non-ASCII characters)
    3) Length filtering (reject 1-character concepts)
    4) Fragment filtering (remove 1-token noise words)
    5) Last-token singularization using inflect (safer than manual)
    6) Edge stopword trimming
    
    Args:
        text: Raw concept text
        
    Returns:
        Normalized concept string (empty if filtered out)
    """
    import unicodedata
    
    # Unicode normalize
    text = unicodedata.normalize('NFKC', text)
    text = text.strip()
    
    # Lowercase
    text = text.lower()
    
    # ASCII filter: reject concepts containing non-ASCII characters
    if not all(ord(c) < 128 for c in text):
        return ''  # Filter out non-ASCII concepts entirely
    
    # Normalize separators (hyphens, slashes to spaces)
    text = text.replace('-', ' ').replace('/', ' ').replace('_', ' ')
    
    # Collapse whitespace
    text = ' '.join(text.split())
    
    # Strip edge punctuation
    text = text.strip('.,;:!?\'"()[]{}')
    
    if not text:
        return ''
    
    # Length filter: reject 1-character concepts
    if len(text) == 1:
        return ''
    
    words = text.split()
    
    # Trim edge stopwords only
    while words and words[0] in EDGE_STOPWORDS:
        words.pop(0)
    while words and words[-1] in EDGE_STOPWORDS:
        words.pop()
    
    if not words:
        return ''
    
    # Fragment killer: drop 1-token noise words
    if len(words) == 1 and words[0] in FRAGMENT_WORDS:
        return ''
    
    # Singularize ONLY the last token (head noun)
    last = words[-1]
    if last.isalpha() and len(last) >= 4 and last not in SINGULAR_EXCEPTIONS:
        sing = _INFLECT.singular_noun(last)  # returns False if already singular
        if sing:
            words[-1] = sing
    
    return ' '.join(words)




def merge_similar_concepts(
    concepts: List[str],
    similarity_threshold: float = 90.0,  # rapidfuzz uses 0-100 scale
    show_progress: bool = True
) -> Dict[str, List[str]]:
    """
    Merge similar concepts using Levenshtein distance with length-based blocking.
    
    Args:
        concepts: List of all concept strings (can have duplicates)
        similarity_threshold: Minimum similarity ratio (0-100) to merge
        show_progress: Whether to show progress bar
        
    Returns:
        Dict mapping canonical_form -> [list of similar variants]
    """
    # Step 1: Normalize and deduplicate
    normalized_map = {}  # normalized -> set of original forms
    for concept in concepts:
        norm = normalize_concept(concept)
        if norm:  # Skip empty strings
            if norm not in normalized_map:
                normalized_map[norm] = set()
            normalized_map[norm].add(concept)
    
    unique_concepts = list(normalized_map.keys())
    
    if len(unique_concepts) == 0:
        return {}
    
    # Step 2: Group by length for efficient blocking
    by_length = defaultdict(list)
    for concept in unique_concepts:
        by_length[len(concept)].append(concept)
    
    # Step 3: Merge similar concepts
    merged = {}  # canonical -> list of variants
    used = set()  # Track which concepts are already merged
    
    lengths = sorted(by_length.keys())
    iterator = tqdm(lengths, desc="Merging concepts") if show_progress else lengths
    
    for length in iterator:
        concepts_at_length = by_length[length]
        
        # Determine nearby lengths to check (±20% or ±3 chars, whichever is larger)
        max_diff = max(3, int(length * 0.2))
        nearby_lengths = [l for l in lengths if abs(l - length) <= max_diff]
        
        for concept in concepts_at_length:
            if concept in used:
                continue
            
            # Try to find existing canonical form that's similar
            matched = False
            for canonical in list(merged.keys()):
                if len(canonical) in nearby_lengths:
                    sim = ratio(concept, canonical)
                    if sim >= similarity_threshold:
                        merged[canonical].append(concept)
                        used.add(concept)
                        matched = True
                        break
            
            if not matched:
                # This becomes a new canonical form
                merged[concept] = [concept]
                used.add(concept)
    
    return merged


def build_concept_graph(
    sequences: List[List[int]],
    tokenizer,
    similarity_threshold: float = 90.0,  # rapidfuzz uses 0-100 scale
    show_progress: bool = True
) -> Tuple[nx.DiGraph, Dict[str, str], Dict]:
    """
    Build a directed graph from concept sequences.
    
    Args:
        sequences: List of token ID sequences
        tokenizer: Tokenizer for decoding
        similarity_threshold: Threshold for merging similar concepts (0-100 scale)
        show_progress: Whether to show progress bars
        
    Returns:
        - graph: NetworkX DiGraph with merged concepts as nodes
        - concept_map: Dict mapping original concept -> canonical form
        - stats: Dict with extraction statistics
    """
    # Step 1: Extract concepts and edges from sequences
    all_concepts = []
    raw_edges = []
    
    iterator = tqdm(sequences, desc="Extracting concepts") if show_progress else sequences
    
    for seq in iterator:
        # Decode sequence
        text = tokenizer.decode(seq, skip_special_tokens=True)
        
        # Split into concepts (one concept per line)
        lines = text.split('\n')
        concepts = [line.strip() for line in lines if line.strip()]
        
        # Check if sequence is incomplete (didn't end with EOS)
        # If last token is not EOS and we have concepts, the last line might be incomplete
        if len(seq) > 0 and len(concepts) > 0:
            # Check if sequence ended with EOS
            has_eos = (seq[-1] == tokenizer.eos_token_id)
            
            # If no EOS, last line might be cut off mid-concept
            if not has_eos and len(concepts) > 1:
                # Keep all but the last concept (which may be incomplete)
                concepts = concepts[:-1]
        
        # Collect all concepts
        all_concepts.extend(concepts)
        
        # Build edges between consecutive concepts (consecutive lines)
        for i in range(len(concepts) - 1):
            raw_edges.append((concepts[i], concepts[i + 1]))
    
    # Step 2: Merge similar concepts
    if show_progress:
        print(f"Found {len(all_concepts)} total concepts, {len(set(all_concepts))} unique")
    
    merged = merge_similar_concepts(
        all_concepts,
        similarity_threshold=similarity_threshold,
        show_progress=show_progress
    )
    
    # Create concept_map: original -> canonical
    concept_map = {}
    for canonical, variants in merged.items():
        for variant in variants:
            # Map normalized form to canonical
            norm = normalize_concept(variant)
            concept_map[norm] = canonical
    
    # Step 3: Build graph with merged concepts
    if show_progress:
        print("Building graph...")
    
    G = nx.DiGraph()
    
    for src, dst in raw_edges:
        # Get canonical forms
        src_norm = normalize_concept(src)
        dst_norm = normalize_concept(dst)
        
        if not src_norm or not dst_norm:
            continue
        
        src_canonical = concept_map.get(src_norm, src_norm)
        dst_canonical = concept_map.get(dst_norm, dst_norm)
        
        # Skip self-loops
        if src_canonical == dst_canonical:
            continue
        
        # Add/update edge with weight
        if G.has_edge(src_canonical, dst_canonical):
            G[src_canonical][dst_canonical]['weight'] += 1
        else:
            G.add_edge(src_canonical, dst_canonical, weight=1)
    
    stats = {
        'total_concepts_extracted': len(all_concepts),
        'unique_concepts_before_merge': len(set(all_concepts)),
        'unique_concepts_after_merge': len(merged),
        'total_raw_edges': len(raw_edges),
    }
    
    return G, concept_map, stats


def compute_graph_metrics(G: nx.DiGraph) -> Dict:
    """
    Compute connectivity and structure metrics for a concept graph.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Dict with graph metrics
    """
    if len(G.nodes()) == 0:
        return {
            'num_nodes': 0,
            'num_edges': 0,
            'density': 0.0,
            'avg_degree': 0.0,
            'orphan_nodes': 0,
            'orphan_ratio': 0.0,
            'weakly_connected_nodes': 0,
            'weakly_connected_ratio': 0.0,
            'num_components': 0,
            'largest_component_size': 0,
            'largest_component_ratio': 0.0,
        }
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Basic metrics
    density = nx.density(G)
    
    # Degree statistics
    degrees = dict(G.degree())  # Treats as undirected for connectivity
    avg_degree = sum(degrees.values()) / num_nodes if num_nodes > 0 else 0
    
    # Node connectivity categories
    orphan_nodes = sum(1 for d in degrees.values() if d == 0)
    weakly_connected = sum(1 for d in degrees.values() if d <= 1)
    
    orphan_ratio = orphan_nodes / num_nodes if num_nodes > 0 else 0
    weakly_connected_ratio = weakly_connected / num_nodes if num_nodes > 0 else 0
    
    # Connected components (using weakly connected for directed graph)
    components = list(nx.weakly_connected_components(G))
    num_components = len(components)
    
    if num_components > 0:
        largest_component = max(components, key=len)
        largest_component_size = len(largest_component)
        largest_component_ratio = largest_component_size / num_nodes
        
        # Density of largest component
        largest_subgraph = G.subgraph(largest_component)
        largest_component_density = nx.density(largest_subgraph)
    else:
        largest_component_size = 0
        largest_component_ratio = 0.0
        largest_component_density = 0.0
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'avg_degree': avg_degree,
        'orphan_nodes': orphan_nodes,
        'orphan_ratio': orphan_ratio,
        'weakly_connected_nodes': weakly_connected,
        'weakly_connected_ratio': weakly_connected_ratio,
        'num_components': num_components,
        'largest_component_size': largest_component_size,
        'largest_component_ratio': largest_component_ratio,
        'largest_component_density': largest_component_density,
    }


def analyze_sequences(
    sequences: List[List[int]],
    tokenizer,
    similarity_threshold: float = 90.0,  # rapidfuzz uses 0-100 scale
    show_progress: bool = True
) -> Dict:
    """
    Full pipeline: build graph and compute metrics.
    
    Args:
        sequences: List of token ID sequences
        tokenizer: Tokenizer for decoding
        similarity_threshold: Threshold for merging similar concepts (0-100 scale)
        show_progress: Whether to show progress bars
        
    Returns:
        Dict containing graph, concept_map, extraction stats, metrics, and concept_frequencies
    """
    # Build graph
    G, concept_map, extraction_stats = build_concept_graph(
        sequences=sequences,
        tokenizer=tokenizer,
        similarity_threshold=similarity_threshold,
        show_progress=show_progress
    )
    
    # Compute metrics
    metrics = compute_graph_metrics(G)
    
    # Compute concept frequencies (by total degree - in + out)
    from collections import Counter
    concept_degrees = {}
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        concept_degrees[node] = in_deg + out_deg
    
    # Sort by degree (most connected first)
    concept_frequencies = sorted(concept_degrees.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'graph': G,
        'concept_map': concept_map,
        'extraction_stats': extraction_stats,
        'metrics': metrics,
        'concept_frequencies': concept_frequencies,  # List of (concept, degree) tuples
    }


def print_graph_analysis(results: Dict):
    """
    Pretty-print graph analysis results.
    
    Args:
        results: Dict from analyze_sequences()
    """
    extraction_stats = results['extraction_stats']
    metrics = results['metrics']
    
    print(f"\n{'='*80}")
    print("GRAPH ANALYSIS")
    print(f"{'='*80}")
    
    print("\nConcept Extraction:")
    print(f"  Total concepts extracted   : {extraction_stats['total_concepts_extracted']}")
    print(f"  Unique before merging      : {extraction_stats['unique_concepts_before_merge']}")
    print(f"  Unique after merging       : {extraction_stats['unique_concepts_after_merge']}")
    print(f"  Total raw edges            : {extraction_stats['total_raw_edges']}")
    
    print("\nGraph Statistics:")
    print(f"  Nodes (concepts)           : {metrics['num_nodes']}")
    print(f"  Edges (relations)          : {metrics['num_edges']}")
    print(f"  Graph density              : {metrics['density']:.4f}")
    print(f"  Average degree             : {metrics['avg_degree']:.2f}")
    
    print("\nNode Connectivity:")
    print(f"  Orphan nodes (degree=0)    : {metrics['orphan_nodes']} ({metrics['orphan_ratio']*100:.1f}%)")
    print(f"  Weakly connected (deg≤1)   : {metrics['weakly_connected_nodes']} ({metrics['weakly_connected_ratio']*100:.1f}%)")
    print(f"  Well connected (deg>1)     : {metrics['num_nodes'] - metrics['weakly_connected_nodes']} ({(1-metrics['weakly_connected_ratio'])*100:.1f}%)")
    
    print("\nGraph Structure:")
    print(f"  Connected components       : {metrics['num_components']}")
    print(f"  Largest component size     : {metrics['largest_component_size']} nodes ({metrics['largest_component_ratio']*100:.1f}%)")
    print(f"  Largest component density  : {metrics['largest_component_density']:.4f}")
    
    print(f"{'='*80}\n")