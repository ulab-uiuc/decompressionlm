# DecompressionLM - Prefix Mass Sampling

This project implements a novel approach to understanding what language models have learned vs. memorized by analyzing the **effective support size** of their conditional distributions.

## Algorithm Overview

### Core Idea

Instead of traditional entropy estimation, we directly measure the **probability mass coverage** of prefix patterns and identify the **effective support set** after deduplication.

### Three-Stage Process

#### Stage 1: Mass-Based Sampling

Sample sequences until the cumulative probability mass of discovered prefix patterns exceeds a threshold:

```
For example, if a model can generate 4 sequences with prefix probabilities:
- Prefix A: p=0.2
- Prefix B: p=0.3
- Prefix C: p=0.4
- Prefix D: p=0.1

We sample until we've discovered prefixes with total mass > threshold (e.g., 0.9):
- Sample 1 → Prefix A found → mass = 0.2
- Sample 2 → Prefix B found → mass = 0.5
- Sample 3 → Prefix B again → mass = 0.5 (no change)
- Sample 4 → Prefix C found → mass = 0.9
- STOP (mass >= 0.9)
```

**Key Parameters:**
- `prefix_len`: Length of prefix pattern to track (e.g., 4 tokens)
- `prob_threshold`: Stop when cumulative mass > this (e.g., 0.9 = 90%)

#### Stage 2: Deduplication

Cluster similar sequences using edit distance to find the true effective support:

```python
similarity_threshold = 0.85  # 85% similarity = duplicate
```

This handles:
- Near-duplicate generations
- Minor variations in wording
- Tokenization artifacts

#### Stage 3: Statistics

Report the effective support set characteristics:
- Number of unique sequences
- Total tokens in support set
- Min/max/avg tokens per sequence

## Usage

### Basic Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.bin_entropy import estimate_prefix_mass

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

results = estimate_prefix_mass(
    model=model,
    tokenizer=tokenizer,
    prefix="What is the capital of France?",
    prefix_len=4,              # Track first 4 tokens
    prob_threshold=0.9,        # Stop at 90% mass coverage
    similarity_threshold=0.85, # 85% similarity for dedup
    max_samples=10000,
    max_len=64,
    save_path="results/france.delm.parquet"
)

print(f"Effective support: {results['effective_set_size']} sequences")
print(f"Total mass: {results['final_mass']:.4f}")
```

### Unconditional Sampling (BOS-only)

```python
results = estimate_prefix_mass(
    model=model,
    tokenizer=tokenizer,
    prefix="",  # Empty = BOS-only unconditional
    prefix_len=8,
    prob_threshold=0.95,
    use_chat_template=False,
)
```

## Output Format

Results are saved in `.delm.parquet` format with:

### Data Columns
- `code`: VdC code used for sampling [0, 1)
- `sequence`: Token IDs (list of uint32)
- `sequence_length`: Length in tokens
- `terminated`: Whether sequence hit EOS
- `in_effective_set`: **NEW** - Boolean flag for effective support membership

### Metadata
- Sampling parameters (prefix_len, thresholds, etc.)
- Model/tokenizer info
- Convergence statistics
- **Effective set statistics:**
  - `effective_set_size`: Number of unique sequences
  - `effective_set_min_tokens`: Shortest sequence
  - `effective_set_max_tokens`: Longest sequence
  - `effective_set_avg_tokens`: Average length
  - `effective_set_total_tokens`: Total tokens

## Key Differences from Original

### Original Approach (Bin Entropy)
- Tracked entropy convergence for multiple bin lengths
- Required many samples to estimate H(X)
- Focused on information-theoretic measures

### New Approach (Prefix Mass)
- **Single prefix length** - focus on one granularity
- **Probability mass threshold** - stop when coverage is sufficient
- **Effective support set** - explicit deduplication and counting
- **Actionable metrics** - concrete number of unique sequences

## Why This Matters

This approach reveals:
1. **How many distinct completions** the model produces
2. **How concentrated** the probability mass is
3. **Whether the model has truly learned** or just memorized a few patterns

For example:
- Memorized content: small support (10-50 sequences cover 90% of mass)
- Learned patterns: large support (1000s of sequences needed for 90% mass)

## Installation

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
python test_mass_sampling.py
```

This will test the algorithm on several prompts and save results to `results/`.

## Technical Details

### Arithmetic Sampling
Uses Van der Corput (VdC) quasi-random sequences for unbiased sampling with deterministic reproducibility.

### Prefix Probability Computation
Exact probabilities are computed via:
```
P(prefix) = P(t1) * P(t2|t1) * P(t3|t1,t2) * ...
```

### Similarity Computation
Normalized Levenshtein distance on decoded strings:
```
similarity = 1 - (edit_distance / max_length)
```

## File Structure

```
src/
  __init__.py
  arithmetic.py      # Arithmetic sampling with VdC codes
  bin_entropy.py     # Main mass-based sampling algorithm
  vdc.py            # Van der Corput sequence generation
  plot_utils.py     # Plotting utilities
results/            # Output .delm.parquet files
test_mass_sampling.py  # Example usage
```

## Future Directions

- [ ] Multi-scale analysis (different prefix lengths)
- [ ] Compression-based similarity metrics
- [ ] Support set evolution over training checkpoints
- [ ] Cross-model comparisons
