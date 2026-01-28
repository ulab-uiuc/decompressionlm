# decompressionLM - Refactored

Concept-Based Knowledge Extraction from LLMs using deterministic and exploratory sampling.

## What Changed (Refactoring Summary)

### 1. **Prompt Format Consistency** ✓
- All prompts now use the same template structure
- Prompts follow format: "Generate concepts in [domain] as keywords..."
- Consistent instructions across all methods

### 2. **Removed DFS** ✓
- Removed `explore_dfs` function
- Cleaned up `exploration_sampling.py`
- Only graph exploration (BFS-style with reconnections) remains

### 3. **BFS → Graph Exploration** ✓
- Renamed from "BFS" to "GraphExplore" for accuracy
- Now handles concept reconnections when same concept discovered via different paths
- Tracks parent-child relationships as directed graph edges
- Counts reconnections in stats
- Properly represents concept relationships as graph, not just tree

### 4. **Absolute Imports** ✓
- Changed all imports from relative to absolute (e.g., `from src.concept_utils import ...`)
- `src/__init__.py` is now empty
- Cleaner, more explicit import structure

### 5. **Experiment 4 Parameters Reduced** ✓
- Reduced to **128 concepts** (was 4096/infinity)
- Removed `full_gen` run (was: threshold + infinity runs)
- Only runs single threshold-based experiment for manual verification
- More manageable for fact-checking

### 6. **Consistent Experiment Style** ✓
- Unified formatting across all experiment files
- Consistent function naming and structure
- Standardized metrics extraction
- Common CSV output format

### 7. **No Dates in Folder Names** ✓
- Removed timestamp generation logic
- Output folders: `results/exp1_baselines`, `results/exp2_vdc_offsets`, etc.
- No date suffixes added

### 8. **Decoupled Prompts** ✓
- Created `exp/0_prompt_templates.py`
- All prompts now defined as lambda functions:
  - `flat_concept_list(domain)` - for flat sampling
  - `graph_root_prompt(domain)` - for graph exploration root
  - `graph_child_prompt(concept, domain)` - for graph exploration children
  - `beam_graph_prompt(domain)` - for single-sequence YAML generation
- Experiments import and use these functions
- Easy to tweak prompts in one centralized location

## Project Structure

```
decompressionlm/
├── exp/
│   ├── 0_prompt_templates.py   # NEW: Centralized prompt definitions
│   ├── 1_baseline.py           # Baseline comparison experiments
│   ├── 2_offset.py             # VdC offset comparison
│   ├── 3_quantization.py       # Quantization ladder experiments
│   └── 4_hallucination.py      # Hallucination metrics (REDUCED)
├── src/
│   ├── __init__.py             # Empty (absolute imports)
│   ├── arithmetic.py           # Arithmetic sampling (UNCHANGED)
│   ├── vdc.py                  # Van der Corput sequences (UNCHANGED)
│   ├── concept_utils.py        # Concept validation (UNCHANGED)
│   ├── baseline_sampling.py    # Random & beam search (UNCHANGED)
│   ├── profiling.py            # Time & memory profiling (UNCHANGED)
│   ├── concept_sampling.py     # REFACTORED: Uses prompt_fn parameter
│   └── exploration_sampling.py # REFACTORED: DFS removed, graph reconnections added
└── README.md                   # This file
```

## Key Changes in Detail

### Prompt Templates (`exp/0_prompt_templates.py`)

All prompts now centralized as lambda functions:

```python
# Flat sampling
flat_concept_list = lambda domain: f"""Generate concepts in {domain} as keywords.
Please output ONE concept per line.
Each concept can be multiple words if needed.
Do not include explanations or extra text.
Please begin from any random concept.
Please use English.
"""

# Graph exploration - root
graph_root_prompt = lambda domain: f"""Generate concepts in {domain} as keywords...
"""

# Graph exploration - children
graph_child_prompt = lambda concept, domain: f"""Generate concepts related to {concept} in {domain} as keywords...
"""
```

### Concept Sampling API Change

**Old:**
```python
sample_concepts(
    model=model,
    tokenizer=tokenizer,
    prefix="List concepts about...",  # Raw string
    ...
)
```

**New:**
```python
sample_concepts(
    model=model,
    tokenizer=tokenizer,
    prompt_fn=lambda: flat_concept_list(domain),  # Function returning string
    ...
)
```

### Graph Exploration API Change

**Old:**
```python
explore_bfs(
    model=model,
    tokenizer=tokenizer,
    domain="US law",
    ...
)
```

**New:**
```python
explore_graph(
    model=model,
    tokenizer=tokenizer,
    domain="US law",
    root_prompt_fn=graph_root_prompt,
    child_prompt_fn=graph_child_prompt,
    ...
)
```

Key improvements:
- Renamed from `explore_bfs` to `explore_graph` (more accurate)
- Accepts prompt functions as parameters
- Handles reconnections when concepts rediscovered
- Tracks parent-child relationships as directed edges
- Returns graph stats including reconnection count

## Experiment Changes

### Experiment 1: Baseline Comparison
- ✓ Uses prompt templates
- ✓ "BFS" renamed to "GraphExplore"
- ✓ Consistent formatting
- ✓ No date in output folder

### Experiment 2: VdC Offset Comparison
- ✓ Uses prompt templates
- ✓ Consistent formatting
- ✓ No date in output folder

### Experiment 3: Quantization Ladder
- ✓ Uses prompt templates
- ✓ Consistent formatting
- ✓ No date in output folder

### Experiment 4: Hallucination Metrics
- ✓ Reduced to **128 concepts** (was 4096)
- ✓ Removed `full_gen` run
- ✓ Only threshold-based experiment
- ✓ Designed for manual fact-checking
- ✓ Uses prompt templates
- ✓ Consistent formatting
- ✓ No date in output folder

## Graph Exploration Details

The refactored graph exploration properly handles concept relationships:

1. **Graph Structure**: Concepts form a directed graph where edges represent "related to" relationships
2. **Reconnections**: When a concept is discovered that already exists, we create an edge from the discovering node to the existing node
3. **Multiple Parents**: Nodes can have multiple parents if discovered through different paths
4. **Stats Tracking**: Includes `reconnections` count in stats

Example:
```
ROOT: US law
  ├─ constitutional law
  │   ├─ first amendment
  │   └─ due process (parents: 2)  ← discovered by both constitutional law AND criminal procedure
  └─ criminal procedure
      └─ due process (reconnection!)
```

## Running Experiments

### Experiment 1: Baseline Comparison
```bash
cd exp
python 1_baseline.py
```

### Experiment 4: Hallucination Metrics (Reduced)
```bash
cd exp
python 4_hallucination.py
```

Results saved to `results/exp4_hallucination/` with 128 concepts for manual verification.

## Migration Guide

### If you have existing code using the old API:

**Concept Sampling:**
```python
# Old
from src import sample_concepts
results = sample_concepts(model, tokenizer, prefix="List concepts...")

# New
from exp.prompt_templates import flat_concept_list
from src.concept_sampling import sample_concepts
results = sample_concepts(model, tokenizer, prompt_fn=lambda: flat_concept_list("domain"))
```

**Graph Exploration:**
```python
# Old
from src import explore_bfs
root, map, stats = explore_bfs(model, tokenizer, "domain", ...)

# New
from exp.prompt_templates import graph_root_prompt, graph_child_prompt
from src.exploration_sampling import explore_graph
root, map, stats = explore_graph(
    model, tokenizer, "domain",
    root_prompt_fn=graph_root_prompt,
    child_prompt_fn=graph_child_prompt,
    ...
)
```

## Benefits of Refactoring

1. **Cleaner prompt management**: All prompts in one place, easy to tweak
2. **More accurate naming**: "GraphExplore" instead of "BFS" properly describes the algorithm
3. **Better graph representation**: Handles reconnections and multiple parents correctly
4. **Consistent style**: All experiments follow same patterns
5. **Reduced manual work**: Exp 4 now only 128 concepts for feasible fact-checking
6. **Absolute imports**: More explicit and maintainable
7. **No timestamp clutter**: Clean folder names for persistent results

## Backward Compatibility

The core algorithms (`arithmetic.py`, `vdc.py`, `concept_utils.py`, `baseline_sampling.py`, `profiling.py`) remain **unchanged**. Only the high-level API and experiment organization changed.

## Citation

```bibtex
@software{decompressionlm2026,
  author = {Luke (Zhaochen Hong)},
  title = {decompressionLM: Concept-Based Knowledge Extraction from LLMs},
  year = {2026},
  url = {https://github.com/...}
}
```
