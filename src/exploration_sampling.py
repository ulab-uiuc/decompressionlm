"""
Graph exploration sampling for hierarchical concept discovery.

Explores concepts as a directed graph where edges represent "related to" relationships.
Discovered concepts are reconnected if found again through different paths.
"""

import torch
from typing import List, Tuple, Dict, Set, Optional, Callable
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.concept_utils import extract_concepts_from_sequence, is_valid_concept, normalize_concept


class ConceptNode:
    """Node in the concept exploration graph."""
    
    def __init__(self, concept: str, depth: int = 0):
        self.concept = concept  # normalized concept
        self.depth = depth  # depth when first discovered
        self.parents = []  # List of ConceptNode that discovered this
        self.children = []  # List of ConceptNode discovered from this
        self.explored = False
        self.sequences = []  # Raw sequences generated from this node
    
    def add_child(self, child: 'ConceptNode'):
        """Add a child node (this node discovered the child)."""
        if child not in self.children:
            self.children.append(child)
        if self not in child.parents:
            child.parents.append(self)
    
    def __repr__(self):
        return f"ConceptNode('{self.concept}', depth={self.depth}, parents={len(self.parents)}, children={len(self.children)})"


def generate_concepts_for_node(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    node: ConceptNode,
    prompt_fn: Callable[[str, str], str],
    domain: str,
    num_sequences: int = 4,
    max_len: int = 32,
    sampling_method: str = "vdc",
) -> Tuple[List[str], List[List[int]]]:
    """
    Generate concepts related to a given concept within a domain.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        node: ConceptNode to explore
        prompt_fn: Function(concept, domain) -> prompt string
        domain: Domain context
        num_sequences: Number of sequences to generate
        max_len: Max sequence length
        sampling_method: "vdc", "random", etc.
        
    Returns:
        (valid_concepts, sequences) - list of normalized concepts and raw sequences
    """
    # Build prompt using provided function
    prompt = prompt_fn(node.concept, domain)
    
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prefix_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
    
    # Generate sequences
    if sampling_method == "vdc":
        from src.vdc import generate_vdc_sequence
        from src.arithmetic import parallel_arithmetic_sample_batch
        
        codes = generate_vdc_sequence(num_sequences)
        batch_results = parallel_arithmetic_sample_batch(
            model, tokenizer, prefix_ids, codes, max_len
        )
    elif sampling_method == "random":
        from src.baseline_sampling import random_sample_batch
        
        batch_results = random_sample_batch(
            model, tokenizer, prefix_ids, num_sequences, max_len, seed=42
        )
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    # Extract valid concepts
    eos_token_id = tokenizer.eos_token_id
    all_valid_concepts = set()
    all_sequences = []
    
    for result in batch_results:
        if len(result) >= 2:
            tokens = result[0]
        else:
            tokens = result
        
        all_sequences.append(tokens)
        
        valid, invalid = extract_concepts_from_sequence(tokens, tokenizer, eos_token_id)
        
        for concept in valid:
            norm = normalize_concept(concept)
            if norm:
                all_valid_concepts.add(norm)
    
    return list(all_valid_concepts), all_sequences


def explore_graph(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    domain: str,
    root_prompt_fn: Callable[[str], str],
    child_prompt_fn: Callable[[str, str], str],
    max_concepts: int = 100,
    max_depth: int = 3,
    sequences_per_node: int = 4,
    max_len: int = 32,
    sampling_method: str = "vdc",
) -> Tuple[ConceptNode, Dict, Dict]:
    """
    Explore concept space as a directed graph using breadth-first traversal.
    
    When a concept is discovered that already exists in the graph, we create
    an edge from the current node to the existing node (reconnection).
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        domain: Domain to explore
        root_prompt_fn: Function(domain) -> prompt for root
        child_prompt_fn: Function(concept, domain) -> prompt for children
        max_concepts: Stop when this many unique concepts discovered
        max_depth: Maximum depth to explore
        sequences_per_node: How many sequences to generate per concept
        max_len: Max sequence length
        sampling_method: Sampling method to use
        
    Returns:
        (root_node, concept_to_node, stats)
    """
    # Initialize root node
    root = ConceptNode(concept=f"ROOT: {domain}", depth=0)
    
    # Tracking
    concept_to_node = {}  # normalized_concept -> ConceptNode
    all_concepts_seen = set()
    queue = deque([root])
    
    stats = {
        "total_sequences_generated": 0,
        "total_nodes_explored": 0,
        "concepts_discovered": 0,
        "max_depth_reached": 0,
        "reconnections": 0,  # How many times we found existing concepts
    }
    
    print(f"\n{'='*80}")
    print(f"GRAPH EXPLORATION: {domain}")
    print(f"{'='*80}")
    print(f"Max concepts: {max_concepts}")
    print(f"Max depth: {max_depth}")
    print(f"Sequences per node: {sequences_per_node}")
    print()
    
    while queue and len(all_concepts_seen) < max_concepts:
        node = queue.popleft()
        
        if node.explored or node.depth >= max_depth:
            continue
        
        print(f"[Depth {node.depth}] Exploring: {node.concept[:60]}...")
        
        # Choose prompt function based on depth
        if node.depth == 0:
            prompt_fn = lambda c, d: root_prompt_fn(d)
        else:
            prompt_fn = child_prompt_fn
        
        # Generate concepts
        concepts, sequences = generate_concepts_for_node(
            model, tokenizer, node, prompt_fn, domain,
            num_sequences=sequences_per_node,
            max_len=max_len,
            sampling_method=sampling_method,
        )
        
        node.explored = True
        node.sequences = sequences
        stats["total_sequences_generated"] += len(sequences)
        stats["total_nodes_explored"] += 1
        stats["max_depth_reached"] = max(stats["max_depth_reached"], node.depth)
        
        new_concepts = 0
        reconnections = 0
        
        for concept in concepts:
            if concept in all_concepts_seen:
                # Reconnection: concept already exists
                existing_node = concept_to_node[concept]
                node.add_child(existing_node)
                reconnections += 1
                stats["reconnections"] += 1
            else:
                # New concept
                all_concepts_seen.add(concept)
                stats["concepts_discovered"] += 1
                new_concepts += 1
                
                # Create new node
                child = ConceptNode(concept, depth=node.depth + 1)
                node.add_child(child)
                concept_to_node[concept] = child
                
                # Add to queue if not too deep
                if child.depth < max_depth:
                    queue.append(child)
        
        print(f"  → Found {len(concepts)} concepts ({new_concepts} new, {reconnections} reconnections)")
        print(f"  → Total concepts: {len(all_concepts_seen)}/{max_concepts}")
        
        if len(all_concepts_seen) >= max_concepts:
            print(f"\n✓ Reached concept limit ({max_concepts})")
            break
    
    print(f"\n{'='*80}")
    print("GRAPH EXPLORATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total concepts: {stats['concepts_discovered']}")
    print(f"Nodes explored: {stats['total_nodes_explored']}")
    print(f"Sequences generated: {stats['total_sequences_generated']}")
    print(f"Reconnections: {stats['reconnections']}")
    print(f"Max depth reached: {stats['max_depth_reached']}")
    print(f"{'='*80}\n")
    
    return root, concept_to_node, stats


def print_graph(node: ConceptNode, visited: Set[str] = None, indent: int = 0, max_depth: int = 3):
    """Print the concept graph (avoiding cycles)."""
    if visited is None:
        visited = set()
    
    if node.concept in visited or node.depth > max_depth:
        return
    
    visited.add(node.concept)
    
    prefix = "  " * indent
    if node.depth == 0:
        print(f"{prefix}[ROOT] {node.concept}")
    else:
        parents_info = f" (parents: {len(node.parents)})" if len(node.parents) > 1 else ""
        print(f"{prefix}├─ {node.concept} ({len(node.children)} children{parents_info})")
    
    for child in node.children:
        print_graph(child, visited, indent + 1, max_depth)


def save_exploration_graph(
    root: ConceptNode,
    stats: Dict,
    save_path: str,
    domain: str,
):
    """Save exploration results to text file."""
    from pathlib import Path
    
    path = Path(save_path)
    if not save_path.endswith(".graph.txt"):
        save_path = save_path + ".graph.txt"
        path = Path(save_path)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"CONCEPT GRAPH EXPLORATION\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"domain: {domain}\n")
        f.write(f"total_concepts: {stats['concepts_discovered']}\n")
        f.write(f"nodes_explored: {stats['total_nodes_explored']}\n")
        f.write(f"sequences_generated: {stats['total_sequences_generated']}\n")
        f.write(f"reconnections: {stats['reconnections']}\n")
        f.write(f"max_depth: {stats['max_depth_reached']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("CONCEPT GRAPH\n")
        f.write("="*80 + "\n\n")
        
        def write_graph(node, visited=None, indent=0):
            if visited is None:
                visited = set()
            
            if node.concept in visited:
                return
            
            visited.add(node.concept)
            
            prefix = "  " * indent
            if node.depth == 0:
                f.write(f"{prefix}[ROOT] {node.concept}\n")
            else:
                parents_info = f" (parents: {len(node.parents)})" if len(node.parents) > 1 else ""
                f.write(f"{prefix}├─ {node.concept} ({len(node.children)} children{parents_info})\n")
            
            for child in node.children:
                write_graph(child, visited, indent + 1)
        
        write_graph(root)
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF GRAPH\n")
        f.write("="*80 + "\n")
    
    print(f"Saved exploration graph to {path}")
