"""
Graph exploration sampling for hierarchical concept discovery.

Explores concepts as a directed graph where edges represent "related to" relationships.
Discovered concepts are reconnected if found again through different paths.
"""

import torch
from typing import List, Tuple, Dict, Set, Optional, Callable
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.concept_utils import extract_concepts_from_sequence, normalize_concept


class ConceptNode:
    """Node in the concept exploration graph."""

    def __init__(self, concept: str, depth: int = 0):
        self.concept = concept  # normalized concept
        self.depth = depth  # depth when first discovered
        self.parents = []  # List of ConceptNode that discovered this
        self.children = []  # List of ConceptNode discovered from this
        self.explored = False

        # NOTE: do NOT store sequences here (memory heavy)
        # self.sequences = []

    def add_child(self, child: "ConceptNode"):
        """Add a child node (this node discovered the child)."""
        if child not in self.children:
            self.children.append(child)
        if self not in child.parents:
            child.parents.append(self)

    def __repr__(self):
        return (
            f"ConceptNode('{self.concept}', depth={self.depth}, "
            f"parents={len(self.parents)}, children={len(self.children)})"
        )


def _build_prompt_and_prefix_ids(
    tokenizer: AutoTokenizer,
    prompt_text: str,
) -> Dict[str, torch.Tensor]:
    """Tokenize with chat template, return dict suitable for model.generate."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt", padding=False, truncation=False)
    return enc


def _generate_batched_for_nodes(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    sequences_per_node: int,
    max_len: int,
    sampling_method: str,
    sampling_params: Optional[dict] = None,
) -> List[List[List[int]]]:
    """
    Generate sequences for multiple prompts in a single batched call (node-parallel).

    Returns:
        sequences_by_node: List length = len(prompts)
          each element is list of token-id lists length = sequences_per_node
    """
    sampling_params = sampling_params or {}

    # Expand prompts: repeat each prompt sequences_per_node times so each row returns 1 sequence.
    expanded_prompts = []
    for p in prompts:
        expanded_prompts.extend([p] * sequences_per_node)

    # Tokenize + pad to batch
    messages = [[{"role": "user", "content": p}] for p in expanded_prompts]
    formatted_prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages
    ]
    enc = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    # Generation config
    # We generate up to max_len new tokens; we don't need the prompt part.
    gen_kwargs = dict(
        max_new_tokens=max_len,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=False,
    )

    if sampling_method == "greedy":
        gen_kwargs.update(dict(do_sample=False, num_beams=1))
    elif sampling_method == "beam_high":
        # beam sampling (do_sample=True with num_beams>1)
        gen_kwargs.update(dict(do_sample=True, num_beams=4, temperature=1.5, top_p=1.0))
    elif sampling_method == "beam_low":
        gen_kwargs.update(dict(do_sample=True, num_beams=4, temperature=0.5, top_p=1.0))
    elif sampling_method == "random":
        # standard sampling
        seed = int(sampling_params.get("seed", 42))
        g = torch.Generator(device=model.device)
        g.manual_seed(seed)
        gen_kwargs.update(dict(do_sample=True, temperature=1.0, top_p=1.0, generator=g))
    else:
        raise ValueError(f"Node-parallel batching not implemented for sampling_method={sampling_method}")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    # outputs: [batch, prompt+new] token ids (padded)
    # We want only the generated continuation tokens, but extract_concepts_from_sequence
    # can tolerate full sequence; still, better to slice to new tokens.
    # Since prompts are padded, we use attention_mask to find prompt length per row.
    sequences_flat: List[List[int]] = []

    if attention_mask is None:
        # If no attention mask, assume full length prompt (rare)
        for row in outputs:
            sequences_flat.append(row.tolist())
    else:
        prompt_lens = attention_mask.sum(dim=1).tolist()
        for row, plen in zip(outputs, prompt_lens):
            row_list = row.tolist()
            # Keep full row (including prompt) OR only continuation.
            # For concept extraction, continuation-only is usually cleaner:
            sequences_flat.append(row_list[plen:])

    # Re-group per node
    sequences_by_node: List[List[List[int]]] = []
    idx = 0
    for _ in range(len(prompts)):
        sequences_by_node.append(sequences_flat[idx : idx + sequences_per_node])
        idx += sequences_per_node

    return sequences_by_node


def generate_concepts_for_node(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    node: ConceptNode,
    prompt_fn: Callable[[str, str], str],
    domain: str,
    num_sequences: int = 4,
    max_len: int = 32,
    sampling_method: str = "vdc",
    sampling_params: dict = None,
) -> Tuple[List[str], int]:
    """
    Generate concepts related to a given concept within a domain.

    Returns:
        (valid_concepts, sequences_generated_count)
    """
    prompt = prompt_fn(node.concept, domain)

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")

    sampling_params = sampling_params or {}

    # Generate sequences (per-node batching only)
    if sampling_method == "vdc":
        from src.vdc import generate_vdc_sequence
        from src.arithmetic import parallel_arithmetic_sample_batch

        offset = sampling_params.get("offset", 0.0)
        codes = generate_vdc_sequence(num_sequences)
        codes = [(c + offset) % 1.0 for c in codes]

        batch_results = parallel_arithmetic_sample_batch(
            model, tokenizer, prefix_ids, codes, max_len
        )
        sequences = []
        for result in batch_results:
            tokens = result[0] if isinstance(result, (list, tuple)) and len(result) >= 2 else result
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            sequences.append(tokens)

    elif sampling_method in ["random", "beam_high", "beam_low", "greedy"]:
        # Use single-node batched generate (reuse node-parallel helper with 1 prompt)
        sequences_by_node = _generate_batched_for_nodes(
            model=model,
            tokenizer=tokenizer,
            prompts=[prompt],
            sequences_per_node=num_sequences,
            max_len=max_len,
            sampling_method=sampling_method,
            sampling_params=sampling_params,
        )
        sequences = sequences_by_node[0]
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    eos_token_id = tokenizer.eos_token_id
    all_valid_concepts: Set[str] = set()

    for tokens in sequences:
        valid, _invalid = extract_concepts_from_sequence(tokens, tokenizer, eos_token_id)
        for concept in valid:
            norm = normalize_concept(concept)
            if norm:
                all_valid_concepts.add(norm)

    return list(all_valid_concepts), len(sequences)


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
    sampling_params: dict = None,
    node_parallel_batch_size: int = 1,   # <-- NEW: >1 enables node-parallel frontier batching
) -> Tuple[ConceptNode, Dict, Dict]:
    """
    Explore concept space as a directed graph using breadth-first traversal.

    Node-parallel mode:
      For sampling_method in {greedy, beam_high, beam_low, random}, we can explore
      multiple nodes per step by batching their prompts into one model.generate call.

    Returns:
        (root_node, concept_to_node, stats)
    """
    sampling_params = sampling_params or {}

    root = ConceptNode(concept=f"ROOT: {domain}", depth=0)

    concept_to_node: Dict[str, ConceptNode] = {}
    all_concepts_seen: Set[str] = set()
    queue = deque([root])

    stats = {
        "total_sequences_generated": 0,
        "total_nodes_explored": 0,
        "concepts_discovered": 0,
        "max_depth_reached": 0,
        "reconnections": 0,

        # NEW parallelism metadata
        "parallel_sequences_per_node": [],              # list of ints, until completion node (inclusive)
        "average_parallel_sequences_per_node": 0.0,     # mean of the list
    }

    print(f"\n{'='*80}")
    print(f"GRAPH EXPLORATION: {domain}")
    print(f"{'='*80}")
    print(f"Max concepts: {max_concepts}")
    print(f"Max depth: {max_depth}")
    print(f"Sequences per node: {sequences_per_node}")
    print(f"Node-parallel batch size: {node_parallel_batch_size}")
    print(f"Sampling method: {sampling_method}")
    print()

    def node_prompt(node: ConceptNode) -> str:
        if node.depth == 0:
            # root prompt ignores concept
            return root_prompt_fn(domain)
        return child_prompt_fn(node.concept, domain)

    # Node-parallel supported only for these methods
    node_parallel_supported = sampling_method in {"greedy", "beam_high", "beam_low", "random"}

    # Main BFS
    while queue and len(all_concepts_seen) < max_concepts:
        # Collect a frontier chunk
        frontier: List[ConceptNode] = []
        while queue and len(frontier) < max(1, int(node_parallel_batch_size)):
            n = queue.popleft()
            if n.explored or n.depth >= max_depth:
                continue
            frontier.append(n)

        if not frontier:
            # nothing left that is eligible
            break

        # If node-parallel not supported, just process one-by-one
        if not node_parallel_supported or len(frontier) == 1:
            # process sequentially, but still batched inside node
            for node in frontier:
                print(f"[Depth {node.depth}] Exploring: {node.concept[:60]}...")

                prompt = node_prompt(node)

                # Generate concepts
                concepts, seq_count = generate_concepts_for_node(
                    model=model,
                    tokenizer=tokenizer,
                    node=node,
                    prompt_fn=(lambda _c, _d, p=prompt: p),  # wrapper returns prebuilt prompt
                    domain=domain,
                    num_sequences=sequences_per_node,
                    max_len=max_len,
                    sampling_method=sampling_method,
                    sampling_params=sampling_params,
                )

                node.explored = True
                stats["total_sequences_generated"] += seq_count
                stats["total_nodes_explored"] += 1
                stats["max_depth_reached"] = max(stats["max_depth_reached"], node.depth)

                # parallel metadata
                stats["parallel_sequences_per_node"].append(int(seq_count))

                new_concepts = 0
                reconnections = 0

                for concept in concepts:
                    if concept in all_concepts_seen:
                        existing_node = concept_to_node[concept]
                        node.add_child(existing_node)
                        reconnections += 1
                        stats["reconnections"] += 1
                    else:
                        all_concepts_seen.add(concept)
                        stats["concepts_discovered"] += 1
                        new_concepts += 1

                        child = ConceptNode(concept, depth=node.depth + 1)
                        node.add_child(child)
                        concept_to_node[concept] = child

                        if child.depth < max_depth:
                            queue.append(child)

                    if len(all_concepts_seen) >= max_concepts:
                        break

                print(f"  → Found {len(concepts)} concepts ({new_concepts} new, {reconnections} reconnections)")
                print(f"  → Total concepts: {len(all_concepts_seen)}/{max_concepts}")

                if len(all_concepts_seen) >= max_concepts:
                    print(f"\n✓ Reached concept limit ({max_concepts})")
                    break

            if len(all_concepts_seen) >= max_concepts:
                break

            continue

        # Node-parallel path: batch prompts for multiple nodes
        prompts = [node_prompt(n) for n in frontier]

        # One big batched generate call
        sequences_by_node = _generate_batched_for_nodes(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            sequences_per_node=sequences_per_node,
            max_len=max_len,
            sampling_method=sampling_method,
            sampling_params=sampling_params,
        )

        eos_token_id = tokenizer.eos_token_id

        # Process each node's generated sequences
        for node, seqs in zip(frontier, sequences_by_node):
            print(f"[Depth {node.depth}] Exploring: {node.concept[:60]}...")

            # Extract valid concepts from all sequences for this node
            concepts_set: Set[str] = set()
            for tokens in seqs:
                valid, _invalid = extract_concepts_from_sequence(tokens, tokenizer, eos_token_id)
                for c in valid:
                    norm = normalize_concept(c)
                    if norm:
                        concepts_set.add(norm)

            concepts = list(concepts_set)
            seq_count = len(seqs)

            node.explored = True
            stats["total_sequences_generated"] += seq_count
            stats["total_nodes_explored"] += 1
            stats["max_depth_reached"] = max(stats["max_depth_reached"], node.depth)

            # parallel metadata: record per-node count, and truncate correctly if we stop mid-frontier
            stats["parallel_sequences_per_node"].append(int(seq_count))

            new_concepts = 0
            reconnections = 0

            for concept in concepts:
                if concept in all_concepts_seen:
                    existing_node = concept_to_node[concept]
                    node.add_child(existing_node)
                    reconnections += 1
                    stats["reconnections"] += 1
                else:
                    all_concepts_seen.add(concept)
                    stats["concepts_discovered"] += 1
                    new_concepts += 1

                    child = ConceptNode(concept, depth=node.depth + 1)
                    node.add_child(child)
                    concept_to_node[concept] = child

                    if child.depth < max_depth:
                        queue.append(child)

                if len(all_concepts_seen) >= max_concepts:
                    break

            print(f"  → Found {len(concepts)} concepts ({new_concepts} new, {reconnections} reconnections)")
            print(f"  → Total concepts: {len(all_concepts_seen)}/{max_concepts}")

            if len(all_concepts_seen) >= max_concepts:
                print(f"\n✓ Reached concept limit ({max_concepts})")
                break

        if len(all_concepts_seen) >= max_concepts:
            break

    # Finalize averages
    if stats["parallel_sequences_per_node"]:
        stats["average_parallel_sequences_per_node"] = (
            sum(stats["parallel_sequences_per_node"]) / len(stats["parallel_sequences_per_node"])
        )
    else:
        stats["average_parallel_sequences_per_node"] = 0.0

    print(f"\n{'='*80}")
    print("GRAPH EXPLORATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total concepts: {stats['concepts_discovered']}")
    print(f"Nodes explored: {stats['total_nodes_explored']}")
    print(f"Sequences generated: {stats['total_sequences_generated']}")
    print(f"Reconnections: {stats['reconnections']}")
    print(f"Max depth reached: {stats['max_depth_reached']}")
    print(f"Avg parallel sequences/node: {stats['average_parallel_sequences_per_node']:.3f}")
    print(f"{'='*80}\n")

    return root, concept_to_node, stats
