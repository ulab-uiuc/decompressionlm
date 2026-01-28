# decompressionLM - Concept-Based Knowledge Extraction

Extract and explore structured knowledge from LLMs using deterministic and exploratory sampling.

## Project Structure

```
decompressionlm/
├── src/
│   ├── __init__.py              # Package exports
│   ├── arithmetic.py            # Arithmetic sampling (UNCHANGED from original)
│   ├── vdc.py                   # Van der Corput sequences (UNCHANGED)  
│   ├── graph_analysis.py        # Graph analysis (UNCHANGED, optional)
│   ├── concept_utils.py         # Concept validation & filtering
│   ├── baseline_sampling.py     # Random & beam search baselines
│   ├── exploration_sampling.py  # BFS/DFS hierarchical exploration
│   ├── profiling.py            # Time & memory profiling
│   └── concept_sampling.py      # Main flat sampling function
├── scripts/
│   ├── compare_methods.py       # Compare all sampling methods
│   └── view_results.py          # Interactive result viewer
├── examples/
│   └── run_experiments.py       # Usage examples
└── README.md                     # This file
```

## Sampling Methods

### Flat Sampling (concept_sampling.py)
All sequences share the same root prompt. Stops when N unique valid concepts discovered.

- **vdc**: Van der Corput deterministic sampling
- **random**: Random sampling (seed=42 for reproducibility)
- **beam_low**: Beam search, temp=0.5 (focused)
- **beam_high**: Beam search, temp=1.5 (diverse)

### Hierarchical Exploration (exploration_sampling.py)
Recursively explores concept relationships by generating concepts about concepts.

- **bfs**: Breadth-first search (explore all at depth D before D+1)
- **dfs**: Depth-first search (explore one branch fully before others)

Example:
```
ROOT: "US law and bar exam"
├─ constitutional law (from root)
│  ├─ first amendment (from constitutional law)
│  │  ├─ freedom of speech (from first amendment)
│  │  └─ freedom of religion (from first amendment)
│  └─ due process (from constitutional law)
└─ criminal procedure (from root)
```

## Quick Start

### Flat Sampling

```python
from src import sample_concepts
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)

# VdC sampling
results = sample_concepts(
    model=model,
    tokenizer=tokenizer,
    prefix="List machine learning concepts:",
    concept_threshold=100,  # Stop at 100 unique concepts
    sampling_method="vdc",
    save_path="results/ml_vdc",
)
```

### BFS Exploration

```python
from src import explore_bfs, print_tree

root, concept_map, stats = explore_bfs(
    model=model,
    tokenizer=tokenizer,
    domain="US law and bar exam",
    max_concepts=100,
    max_depth=3,
    sequences_per_node=4,
)

# Print tree structure
print_tree(root, max_depth=2)
```

### DFS Exploration

```python
from src import explore_dfs

root_dfs, concept_map_dfs, stats_dfs = explore_dfs(
    model=model,
    tokenizer=tokenizer,
    domain="machine learning",
    max_concepts=100,
    max_depth=3,
)
```

## Compare All Methods

```bash
cd scripts
python compare_methods.py \
    --threshold 100 \
    --max-samples 5000 \
    --prefix "List concepts about neural networks:"
```

Output:
```
Method               | Samples | Valid | Invalid | Threshold | Time
-----------------------------------------------------------------
VdC                  |     523 |   101 |      45 |  REACHED  | 12.3s
Random               |     687 |   102 |      51 |  REACHED  | 16.8s
Beam Low             |     891 |   103 |      38 |  REACHED  | 22.1s
Beam High            |     445 |   100 |      67 |  REACHED  | 11.2s
```

## Concept Validation Rules

A concept is **valid** if:
- Contains only ASCII letters (a-z, A-Z) and spaces
- Length > 1 after stripping whitespace
- Not entirely spaces

Examples:
- ✓ "machine learning"
- ✓ "AI"
- ✗ "ML-based" (hyphen)
- ✗ "AI/ML" (slash)
- ✗ "A" (too short)
- ✗ "123" (numbers)

## Output Formats

### Flat Sampling: `.delm.txt`
```
=== METADATA ===
model: ...
profiling: ...

=== VALID CONCEPTS (by frequency) ===
1 | 42 | machine learning
2 | 38 | neural network
...

=== INVALID CONCEPTS (first 64) ===
1 | 10 | ML
2 | 8  | #AI
...
```

### Exploration: `.tree.txt`
```
=== CONCEPT TREE EXPLORATION (BFS) ===
domain: US law
total_concepts: 103
max_depth: 3

[ROOT] US law and bar exam
  ├─ constitutional law (4 children)
    ├─ first amendment (2 children)
    ├─ due process (3 children)
  ├─ criminal procedure (5 children)
```

## View Results

```bash
cd scripts
python view_results.py ../results/ml_vdc.delm.txt
```

Commands:
- `valid 20` - Show top 20 concepts
- `search neural` - Search for concepts
- `stats` - Show statistics
- `profiling` - Show timing/memory
- `q` - Quit

## Key Features

### Profiling
Every run includes detailed timing and GPU memory tracking:
- Sampling time (min/max/avg per batch)
- Concept extraction time  
- New concept discovery time
- GPU memory (allocated & reserved)

### Stopping Conditions

**Flat sampling**: Stops at exact sequence where concept count ≥ threshold
```
Threshold: 100
Seq 523: 99 concepts
Seq 524: 101 concepts  ← STOP HERE
Seq 525+: DISCARDED
```

**BFS/DFS**: Stops when total unique concepts ≥ threshold across all depths

### Deduplication

Concepts are normalized before deduplication:
1. Strip leading/trailing spaces
2. Collapse internal whitespace
3. Lowercase

"Machine Learning" = "machine  learning" = "MACHINE LEARNING"

## Advanced Usage

### Custom Prompting for Exploration

```python
# Default: "List concepts related to X in Y:"
# Customize by modifying exploration_sampling.py:generate_concepts_for_node()

# For more structured output:
prompt = f"""Generate concepts related to {node.concept} in {domain}.
Output one concept per line.
Be specific and use full terms."""
```

### Combining Methods

```python
# 1. BFS to discover broad structure
root_bfs, _, stats_bfs = explore_bfs(
    model, tokenizer, "neural networks",
    max_concepts=50, max_depth=2
)

# 2. DFS on interesting branches
for child in root_bfs.children:
    if child.concept == "transformers":
        root_dfs, _, _ = explore_dfs(
            model, tokenizer, f"{child.concept} in neural networks",
            max_concepts=100, max_depth=4
        )
```

### Profiling Analysis

```python
results = sample_concepts(..., display_interval=128)

# Check timing breakdown
prof = results['profiling']
print(f"Sampling: {prof['time_sampling_total']:.2f}s")
print(f"Extraction: {prof['time_concept_extraction_total']:.2f}s")

# Check GPU memory
print(f"Peak GPU: {prof['gpu_sampling_end_allocated_max']:.2f} GB")
```

## Backward Compatibility

Old code still works! Your original `bin_entropy.py` is unchanged:

```python
# Old code continues to work
from src.bin_entropy import estimate_prefix_mass

results = estimate_prefix_mass(
    model=model,
    tokenizer=tokenizer,
    prefix_len=8,
    prob_threshold=0.9,
    ...
)
```

## What Changed

1. **No probability tracking** - Just count concepts, no prefix probability mass
2. **Concept-based stopping** - Stop at N concepts, not probability threshold
3. **Multiple sampling methods** - VdC, random, beam (low/high), BFS, DFS
4. **Hierarchical exploration** - NEW: BFS/DFS for concept trees
5. **Better validation** - Strict ASCII alphabetic filtering
6. **Comprehensive profiling** - Time & memory tracking per stage
7. **Human-readable output** - Text files instead of binary parquet

## Citation

```bibtex
@software{decompressionlm2026,
  author = {Luke (Zhaochen Hong)},
  title = {decompressionLM: Concept-Based Knowledge Extraction from LLMs},
  year = {2026},
  url = {https://github.com/...}
}
```
