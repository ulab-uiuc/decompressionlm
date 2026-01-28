# decompressionLM Refactored - File Structure

## Complete File Tree

```
decompressionlm_refactored/
├── README.md                      # Comprehensive documentation
├── exp/                           # Experiments
│   ├── 0_prompt_templates.py      # ⭐ NEW: Centralized prompts (lambda functions)
│   ├── 1_baseline.py              # ✨ REFACTORED: Uses templates, graph exploration
│   └── 4_hallucination.py         # ✨ REFACTORED: 128 concepts, single run
└── src/                           # Source code
    ├── __init__.py                # ✨ REFACTORED: Empty (absolute imports)
    ├── arithmetic.py              # ✅ UNCHANGED: Arithmetic sampling
    ├── baseline_sampling.py       # ✅ UNCHANGED: Random & beam search
    ├── concept_sampling.py        # ✨ REFACTORED: Uses prompt_fn parameter
    ├── concept_utils.py           # ✅ UNCHANGED: Concept validation
    ├── exploration_sampling.py    # ✨ REFACTORED: DFS removed, graph with reconnections
    ├── profiling.py               # ✅ UNCHANGED: Time & memory profiling
    └── vdc.py                     # ✅ UNCHANGED: Van der Corput sequences
```

## File Descriptions

### 📁 exp/ (Experiments)

#### ⭐ `0_prompt_templates.py` (NEW)
Centralized prompt definitions as lambda functions.

**Functions:**
- `flat_concept_list(domain)` - For flat sampling
- `graph_root_prompt(domain)` - For graph exploration root
- `graph_child_prompt(concept, domain)` - For graph exploration children
- `beam_graph_prompt(domain)` - For single-sequence YAML generation

**Example:**
```python
from exp.prompt_templates import flat_concept_list
prompt = flat_concept_list("US law and bar exam")
```

#### ✨ `1_baseline.py` (REFACTORED)
Baseline comparison experiment.

**Changes:**
- Imports prompts from `0_prompt_templates.py`
- Uses `explore_graph` instead of `explore_bfs`
- Method renamed: "BFS" → "GraphExplore"
- Consistent style with other experiments
- No date in output folder

**Models:** Llama-3.1-8B, Qwen2.5-7B
**Domain:** US law & Bar Exam
**Methods:** Beam+Graph, GraphExplore, deLM+BeamHigh, deLM+BeamLow, deLM+Random
**Output:** `results/exp1_baselines/`

#### ✨ `4_hallucination.py` (REFACTORED)
Hallucination metrics experiment (reduced for manual verification).

**Changes:**
- Reduced to **128 concepts** (was 4096)
- Removed `full_gen` run entirely
- Only runs threshold-based experiment
- Uses prompt templates
- Consistent style

**Models:** Llama-3.1-8B, Qwen2.5-7B, Mistral-7B, Phi-3-14B
**Domain:** US law & Bar Exam
**Target:** 128 concepts (manual fact-checking)
**Output:** `results/exp4_hallucination/`

### 📁 src/ (Source Code)

#### ✨ `__init__.py` (REFACTORED)
Now empty. Use absolute imports.

**Before:**
```python
from src import sample_concepts
```

**After:**
```python
from src.concept_sampling import sample_concepts
```

#### ✨ `concept_sampling.py` (REFACTORED)
Main concept-based sampling function.

**Key Change:**
- Changed `prefix: str` parameter to `prompt_fn: Callable[[], str]`
- Accepts lambda function that returns prompt string
- Allows dynamic prompt generation

**Usage:**
```python
from exp.prompt_templates import flat_concept_list
from src.concept_sampling import sample_concepts

results = sample_concepts(
    model=model,
    tokenizer=tokenizer,
    prompt_fn=lambda: flat_concept_list("US law"),
    concept_threshold=100,
    ...
)
```

#### ✨ `exploration_sampling.py` (REFACTORED)
Graph exploration (formerly BFS with DFS).

**Major Changes:**
1. **Removed DFS** - `explore_dfs` function deleted
2. **Renamed** - `explore_bfs` → `explore_graph`
3. **Graph reconnections** - Handles rediscovered concepts properly
4. **Multiple parents** - Nodes can have multiple parent edges
5. **Stats tracking** - Added `reconnections` count
6. **Prompt functions** - Accepts `root_prompt_fn` and `child_prompt_fn`

**Key Class:**
- `ConceptNode` - Now tracks both `parents` and `children`

**Usage:**
```python
from exp.prompt_templates import graph_root_prompt, graph_child_prompt
from src.exploration_sampling import explore_graph

root, concept_map, stats = explore_graph(
    model=model,
    tokenizer=tokenizer,
    domain="US law",
    root_prompt_fn=graph_root_prompt,
    child_prompt_fn=graph_child_prompt,
    max_concepts=100,
    max_depth=3,
)
```

**Graph Structure Example:**
```
ROOT: US law
  ├─ constitutional law
  │   ├─ first amendment
  │   └─ due process (parents: 2)  ← multiple parents!
  └─ criminal procedure
      └─ due process (reconnection!)
```

#### ✅ `arithmetic.py` (UNCHANGED)
Arithmetic sampling using deterministic codes.

**Key Functions:**
- `arithmetic_sample_token()` - Single token sampling
- `arithmetic_sample_token_batch()` - Batched token sampling
- `parallel_arithmetic_sample_batch()` - Full sequence generation

#### ✅ `baseline_sampling.py` (UNCHANGED)
Random and beam search sampling methods.

**Key Functions:**
- `random_sample_batch()` - Reproducible random sampling
- `beam_search_batch()` - Beam search with temperature

#### ✅ `concept_utils.py` (UNCHANGED)
Concept validation and filtering.

**Key Functions:**
- `is_valid_concept()` - Check if concept meets validation rules
- `normalize_concept()` - Normalize for deduplication
- `extract_concepts_from_sequence()` - Extract from token sequence
- `track_concepts()` - Update tracking dictionaries

**Validation Rules:**
- Must start with letter
- Can contain letters, digits, spaces, hyphens
- Length > 1
- No punctuation except hyphen

#### ✅ `profiling.py` (UNCHANGED)
Time and GPU memory profiling utilities.

**Key Classes/Functions:**
- `ProfileStats` - Track timing and memory
- `profile_section()` - Context manager for profiling
- `format_time()` - Human-readable time formatting
- `measure_model_memory()` - GPU memory measurement

#### ✅ `vdc.py` (UNCHANGED)
Van der Corput sequence generation.

**Key Function:**
- `generate_vdc_sequence(n, base=2)` - Generate low-discrepancy sequence

### 📄 `README.md`
Comprehensive documentation including:
- What changed (refactoring summary)
- Project structure
- Key changes in detail
- API changes and migration guide
- Running experiments
- Benefits of refactoring

## Files NOT Included (Create from Original)

You'll need to copy these from your original project or create them:

### Optional Experiments (Update Similarly)
- `exp/2_offset.py` - VdC offset comparison
- `exp/3_quantization.py` - Quantization ladder

These weren't refactored but can follow the same pattern:
1. Import prompts from `0_prompt_templates.py`
2. Use absolute imports
3. Pass `prompt_fn` to `sample_concepts`
4. No dates in output folders

## Download Instructions

The complete refactored project is available as `decompressionlm_refactored.tar.gz`.

**Extract:**
```bash
tar -xzf decompressionlm_refactored.tar.gz
cd decompressionlm_refactored
```

**Verify structure:**
```bash
find . -type f | sort
```

**Expected output:**
```
./README.md
./exp/0_prompt_templates.py
./exp/1_baseline.py
./exp/4_hallucination.py
./src/__init__.py
./src/arithmetic.py
./src/baseline_sampling.py
./src/concept_sampling.py
./src/concept_utils.py
./src/exploration_sampling.py
./src/profiling.py
./src/vdc.py
```

## Quick Start

```bash
# Run experiment 1
cd exp
python 1_baseline.py

# Run experiment 4 (reduced for manual verification)
python 4_hallucination.py
```

## Key Improvements Summary

✅ **All prompts centralized** - Easy to tweak in one place
✅ **Graph reconnections** - Proper graph structure, not just tree
✅ **Absolute imports** - Cleaner, more maintainable
✅ **Consistent style** - All experiments follow same patterns
✅ **Reduced manual work** - Exp 4 only 128 concepts
✅ **No timestamp clutter** - Clean persistent folder names
✅ **Better naming** - "GraphExplore" instead of "BFS"
✅ **DFS removed** - Simplified exploration code
