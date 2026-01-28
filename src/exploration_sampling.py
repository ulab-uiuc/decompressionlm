"""
BFS/DFS exploration sampling for hierarchical concept discovery.

Instead of generating from the same prefix repeatedly, we:
1. Start with root prompt: "List concepts about X"
2. For each concept discovered, create new prompt: "List concepts about [concept] in X"
3. Explore breadth-first (BFS) or depth-first (DFS)
4. Build a tree/graph of related concepts
"""

import torch
from typing import List, Tuple, Dict, Set, Optional
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer

from .concept_utils import extract_concepts_from_sequence, is_valid_concept, normalize_concept


class ConceptNode:
    """Node in the concept exploration tree."""
    
    def __init__(self, concept: str, depth: int = 0, parent: Optional['ConceptNode'] = None):
        self.concept = concept  # normalized concept
        self.depth = depth
        self.parent = parent
        self.children = []  # List of ConceptNode
        self.explored = False
        self.sequences = []  # Raw sequences that generated children
    
    def add_child(self, child: 'ConceptNode'):
        """Add a child node."""
        self.children.append(child)
    
    def __repr__(self):
        return f"ConceptNode('{self.concept}', depth={self.depth}, children={len(self.children)})"


def generate_concepts_for_node(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    node: ConceptNode,
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
        domain: Domain context (e.g., "US law and bar exam")
        num_sequences: Number of sequences to generate
        max_len: Max sequence length
        sampling_method: "vdc", "random", etc.
        
    Returns:
        (valid_concepts, sequences) - list of normalized concepts and raw sequences
    """
    # Build prompt for this node
    if node.depth == 0:
        # Root: just list concepts in domain
        prompt = f"List important concepts about {domain}:"
    else:
        # Child: list concepts related to parent concept in domain
        prompt = f"List concepts related to {node.concept} in {domain}:"
    
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prefix_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
    
    # Generate sequences based on sampling method
    if sampling_method == "vdc":
        from .vdc import generate_vdc_sequence
        from .arithmetic import parallel_arithmetic_sample_batch
        
        codes = generate_vdc_sequence(num_sequences)
        batch_results = parallel_arithmetic_sample_batch(
            model, tokenizer, prefix_ids, codes, max_len
        )
    elif sampling_method == "random":
        from .baseline_sampling import random_sample_batch
        
        batch_results = random_sample_batch(
            model, tokenizer, prefix_ids, num_sequences, max_len, seed=42
        )
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    # Extract valid concepts from all sequences
    eos_token_id = tokenizer.eos_token_id
    all_valid_concepts = set()
    all_sequences = []
    
    for result in batch_results:
        # Handle different result formats
        if len(result) >= 2:
            tokens = result[0]
        else:
            tokens = result
        
        all_sequences.append(tokens)
        
        # Extract concepts
        valid, invalid = extract_concepts_from_sequence(tokens, tokenizer, eos_token_id)
        
        # Normalize and deduplicate
        for concept in valid:
            norm = normalize_concept(concept)
            if norm:
                all_valid_concepts.add(norm)
    
    return list(all_valid_concepts), all_sequences


def explore_bfs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    domain: str,
    max_concepts: int = 100,
    max_depth: int = 3,
    sequences_per_node: int = 4,
    max_len: int = 32,
    sampling_method: str = "vdc",
) -> Tuple[ConceptNode, Dict, Dict]:
    """
    Breadth-first exploration of concept space.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        domain: Domain to explore (e.g., "US law and bar exam")
        max_concepts: Stop when this many unique concepts discovered
        max_depth: Maximum depth to explore
        sequences_per_node: How many sequences to generate per concept
        max_len: Max sequence length
        sampling_method: Sampling method to use
        
    Returns:
        (root_node, concept_to_node, stats) - exploration tree, lookup dict, and stats
    """
    # Initialize root node
    root = ConceptNode(concept=f"ROOT: {domain}", depth=0, parent=None)
    
    # Tracking
    concept_to_node = {}  # normalized_concept -> ConceptNode
    all_concepts_seen = set()  # All normalized concepts discovered
    queue = deque([root])  # BFS queue
    
    stats = {
        "total_sequences_generated": 0,
        "total_nodes_explored": 0,
        "concepts_discovered": 0,
        "max_depth_reached": 0,
    }
    
    print(f"\n{'='*80}")
    print(f"BFS EXPLORATION: {domain}")
    print(f"{'='*80}")
    print(f"Max concepts: {max_concepts}")
    print(f"Max depth: {max_depth}")
    print(f"Sequences per node: {sequences_per_node}")
    print()
    
    # BFS exploration
    while queue and len(all_concepts_seen) < max_concepts:
        node = queue.popleft()
        
        # Skip if already explored or too deep
        if node.explored or node.depth >= max_depth:
            continue
        
        print(f"[Depth {node.depth}] Exploring: {node.concept[:60]}...")
        
        # Generate concepts for this node
        concepts, sequences = generate_concepts_for_node(
            model, tokenizer, node, domain,
            num_sequences=sequences_per_node,
            max_len=max_len,
            sampling_method=sampling_method,
        )
        
        node.explored = True
        node.sequences = sequences
        stats["total_sequences_generated"] += len(sequences)
        stats["total_nodes_explored"] += 1
        stats["max_depth_reached"] = max(stats["max_depth_reached"], node.depth)
        
        # Create child nodes for new concepts
        new_concepts = 0
        for concept in concepts:
            if concept not in all_concepts_seen:
                all_concepts_seen.add(concept)
                stats["concepts_discovered"] += 1
                new_concepts += 1
                
                # Create child node
                child = ConceptNode(concept, depth=node.depth + 1, parent=node)
                node.add_child(child)
                concept_to_node[concept] = child
                
                # Add to queue for exploration
                if child.depth < max_depth:
                    queue.append(child)
        
        print(f"  → Found {len(concepts)} concepts ({new_concepts} new)")
        print(f"  → Total concepts: {len(all_concepts_seen)}/{max_concepts}")
        
        # Check if we've hit the concept limit
        if len(all_concepts_seen) >= max_concepts:
            print(f"\n✓ Reached concept limit ({max_concepts})")
            break
    
    print(f"\n{'='*80}")
    print("BFS EXPLORATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total concepts: {stats['concepts_discovered']}")
    print(f"Nodes explored: {stats['total_nodes_explored']}")
    print(f"Sequences generated: {stats['total_sequences_generated']}")
    print(f"Max depth reached: {stats['max_depth_reached']}")
    print(f"{'='*80}\n")
    
    return root, concept_to_node, stats


def explore_dfs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    domain: str,
    max_concepts: int = 100,
    max_depth: int = 3,
    sequences_per_node: int = 4,
    max_len: int = 32,
    sampling_method: str = "vdc",
) -> Tuple[ConceptNode, Dict, Dict]:
    """
    Depth-first exploration of concept space.
    
    Same args as explore_bfs, but uses DFS instead of BFS.
    """
    # Initialize root node
    root = ConceptNode(concept=f"ROOT: {domain}", depth=0, parent=None)
    
    # Tracking
    concept_to_node = {}
    all_concepts_seen = set()
    stack = [root]  # DFS stack
    
    stats = {
        "total_sequences_generated": 0,
        "total_nodes_explored": 0,
        "concepts_discovered": 0,
        "max_depth_reached": 0,
    }
    
    print(f"\n{'='*80}")
    print(f"DFS EXPLORATION: {domain}")
    print(f"{'='*80}")
    print(f"Max concepts: {max_concepts}")
    print(f"Max depth: {max_depth}")
    print(f"Sequences per node: {sequences_per_node}")
    print()
    
    # DFS exploration
    while stack and len(all_concepts_seen) < max_concepts:
        node = stack.pop()
        
        # Skip if already explored or too deep
        if node.explored or node.depth >= max_depth:
            continue
        
        print(f"[Depth {node.depth}] Exploring: {node.concept[:60]}...")
        
        # Generate concepts for this node
        concepts, sequences = generate_concepts_for_node(
            model, tokenizer, node, domain,
            num_sequences=sequences_per_node,
            max_len=max_len,
            sampling_method=sampling_method,
        )
        
        node.explored = True
        node.sequences = sequences
        stats["total_sequences_generated"] += len(sequences)
        stats["total_nodes_explored"] += 1
        stats["max_depth_reached"] = max(stats["max_depth_reached"], node.depth)
        
        # Create child nodes for new concepts (in reverse order for DFS)
        new_concepts = 0
        children_to_add = []
        
        for concept in concepts:
            if concept not in all_concepts_seen:
                all_concepts_seen.add(concept)
                stats["concepts_discovered"] += 1
                new_concepts += 1
                
                # Create child node
                child = ConceptNode(concept, depth=node.depth + 1, parent=node)
                node.add_child(child)
                concept_to_node[concept] = child
                
                # Add to stack for exploration (reversed for DFS)
                if child.depth < max_depth:
                    children_to_add.append(child)
        
        # Add children to stack in reverse order (so first child is explored first)
        stack.extend(reversed(children_to_add))
        
        print(f"  → Found {len(concepts)} concepts ({new_concepts} new)")
        print(f"  → Total concepts: {len(all_concepts_seen)}/{max_concepts}")
        
        # Check if we've hit the concept limit
        if len(all_concepts_seen) >= max_concepts:
            print(f"\n✓ Reached concept limit ({max_concepts})")
            break
    
    print(f"\n{'='*80}")
    print("DFS EXPLORATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total concepts: {stats['concepts_discovered']}")
    print(f"Nodes explored: {stats['total_nodes_explored']}")
    print(f"Sequences generated: {stats['total_sequences_generated']}")
    print(f"Max depth reached: {stats['max_depth_reached']}")
    print(f"{'='*80}\n")
    
    return root, concept_to_node, stats


def print_tree(node: ConceptNode, indent: int = 0, max_depth: int = 3):
    """Print the concept tree."""
    if node.depth > max_depth:
        return
    
    prefix = "  " * indent
    if node.depth == 0:
        print(f"{prefix}[ROOT] {node.concept}")
    else:
        print(f"{prefix}├─ {node.concept} ({len(node.children)} children)")
    
    for child in node.children:
        print_tree(child, indent + 1, max_depth)


def save_exploration_tree(
    root: ConceptNode,
    stats: Dict,
    save_path: str,
    domain: str,
    method: str,  # "bfs" or "dfs"
):
    """Save exploration results to text file."""
    from pathlib import Path
    
    path = Path(save_path)
    if not save_path.endswith(".tree.txt"):
        save_path = save_path + ".tree.txt"
        path = Path(save_path)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*80 + "\n")
        f.write(f"CONCEPT TREE EXPLORATION ({method.upper()})\n")
        f.write("="*80 + "\n\n")
        
        # Metadata
        f.write(f"domain: {domain}\n")
        f.write(f"method: {method}\n")
        f.write(f"total_concepts: {stats['concepts_discovered']}\n")
        f.write(f"nodes_explored: {stats['total_nodes_explored']}\n")
        f.write(f"sequences_generated: {stats['total_sequences_generated']}\n")
        f.write(f"max_depth: {stats['max_depth_reached']}\n\n")
        
        # Tree structure
        f.write("="*80 + "\n")
        f.write("CONCEPT TREE\n")
        f.write("="*80 + "\n\n")
        
        def write_tree(node, indent=0):
            prefix = "  " * indent
            if node.depth == 0:
                f.write(f"{prefix}[ROOT] {node.concept}\n")
            else:
                f.write(f"{prefix}├─ {node.concept} ({len(node.children)} children)\n")
            
            for child in node.children:
                write_tree(child, indent + 1)
        
        write_tree(root)
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF TREE\n")
        f.write("="*80 + "\n")
    
    print(f"Saved exploration tree to {path}")


if __name__ == "__main__":
    print("Exploration sampling module")
    print("\nUsage:")
    print("""
from transformers import AutoModelForCausalLM, AutoTokenizer
from exploration_sampling import explore_bfs, explore_dfs, print_tree

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# BFS exploration
root, concept_map, stats = explore_bfs(
    model=model,
    tokenizer=tokenizer,
    domain="US law and bar exam",
    max_concepts=100,
    max_depth=3,
    sequences_per_node=4,
)

# Print tree
print_tree(root, max_depth=2)

# DFS exploration
root_dfs, concept_map_dfs, stats_dfs = explore_dfs(
    model=model,
    tokenizer=tokenizer,
    domain="machine learning",
    max_concepts=100,
    max_depth=3,
    sequences_per_node=4,
)
    """)
