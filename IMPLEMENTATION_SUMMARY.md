# Implementation Summary: Prefix Mass-Based Sampling

## What Changed

I've completely redesigned the algorithm from bin-based entropy estimation to **prefix mass-based sampling with effective support set computation**.

## New Algorithm (3 Stages)

### Stage 1: Mass-Based Stopping Criterion

**Before:**
- Sample N sequences
- Compute bin entropy H = -Σ p_i log p_i
- Wait for variance convergence

**After:**
- Sample until cumulative **prefix probability mass** > threshold
- Track unique prefix patterns (e.g., first 4 tokens)
- Compute exact probabilities: P(prefix) = P(t1) × P(t2|t1) × P(t3|t1,t2) × ...
- Stop when: Σ P(discovered_prefixes) > threshold (e.g., 0.9)

### Stage 2: Deduplication via Similarity

**New addition:**
- Cluster sequences using normalized edit distance
- Similarity = 1 - (Levenshtein_distance / max_length)
- Keep only representative sequences (similarity_threshold = 0.85)

### Stage 3: Effective Support Set Statistics

**New metrics:**
- Number of unique sequences after deduplication
- Total tokens in support set
- Min/max/avg tokens per sequence
- Distribution of sequence lengths

## Key Parameters

```python
prefix_len = 4              # Length of prefix pattern to track
prob_threshold = 0.9        # Stop at 90% mass coverage
similarity_threshold = 0.85 # 85% similarity = duplicate
```

## File Changes

### Modified Files

**`src/bin_entropy.py`** - Complete rewrite:
- Removed: Multiple bin length tracking, entropy computation
- Added: Mass-based stopping, prefix probability computation
- Added: Sequence clustering/deduplication
- Added: `estimate_prefix_mass()` function (replaces `estimate_bin_entropy()`)

### New Files

**`analyze_results.py`**:
- Load and inspect saved results
- Display effective set statistics
- Compare multiple result files

**`test_mass_sampling.py`**:
- Example usage script
- Tests on multiple prompts

**`README.md`**:
- Complete documentation
- Algorithm explanation
- Usage examples

### Unchanged Files

- `src/arithmetic.py` - Arithmetic sampling (no changes needed)
- `src/vdc.py` - Van der Corput sequence (no changes needed)
- `src/plot_utils.py` - Color utilities (no changes needed)

## Output Format Changes

### New Column in .delm.parquet

**`in_effective_set`**: Boolean flag
- `True` if sequence is in the effective support set after deduplication
- `False` if it's a duplicate/similar sequence

All sequences are still saved, but now you can easily filter to the effective set.

### New Metadata Fields

- `sampling_mode`: "prefix_mass"
- `prefix_len`: Tracked prefix length
- `prob_threshold`: Mass threshold used
- `similarity_threshold`: Deduplication threshold
- `effective_set_size`: Number of unique sequences
- `effective_set_min_tokens`: Shortest sequence
- `effective_set_max_tokens`: Longest sequence
- `effective_set_avg_tokens`: Average length
- `effective_set_total_tokens`: Total tokens
- `final_prefix_mass`: Final cumulative mass
- `unique_prefixes_discovered`: Number of unique prefix patterns

## Usage Example

### Before (Entropy)
```python
results = estimate_bin_entropy(
    model=model,
    tokenizer=tokenizer,
    prefix="What is the capital?",
    bin_prefix_lens=[1, 2, 4, 8],  # Multiple lengths
    variance_threshold=1e-4,
)
```

### After (Mass)
```python
results = estimate_prefix_mass(
    model=model,
    tokenizer=tokenizer,
    prefix="What is the capital?",
    prefix_len=4,              # Single length
    prob_threshold=0.9,        # Stop at 90% mass
    similarity_threshold=0.85, # Deduplicate similar sequences
)

# Access effective support set
print(f"Support size: {results['effective_set_size']}")
print(f"Total tokens: {results['effective_set_stats']['total_tokens']}")
```

## Why This Is Better

### Conceptual Clarity
- **Entropy**: Abstract information measure
- **Mass**: Concrete probability coverage
- **Support size**: Explicit count of unique sequences

### Stopping Criterion
- **Entropy**: Variance-based (indirect)
- **Mass**: Direct probability threshold (clear target)

### Actionable Insights
- **Before**: "Entropy is 8.3 bits"
- **After**: "90% of mass is covered by 45 unique sequences with avg 12 tokens each"

### Deduplication
- **Before**: All sequences counted equally
- **After**: Similar sequences clustered → true diversity measure

## Example Output

```
================================================================================
RESULTS
================================================================================
Status: THRESHOLD REACHED
Total samples: 1,847
Unique prefixes (len=4): 156
Final prefix mass: 0.9023

Effective Support Set:
  Size: 45 sequences
  Total tokens: 532
  Min tokens: 8
  Max tokens: 18
  Avg tokens: 11.8

Time: 12.3s (150.2 samp/s)
================================================================================
```

## Next Steps

You can now:

1. **Run the test script:**
   ```bash
   python test_mass_sampling.py
   ```

2. **Analyze saved results:**
   ```bash
   python analyze_results.py results/your_file.delm.parquet
   ```

3. **Use in your research:**
   - Compare support sizes across different prompts
   - Measure memorization vs. learning
   - Track support evolution across checkpoints

## Dependencies

Added one new dependency:
- `python-Levenshtein>=0.21.0` for edit distance computation

All others remain the same.
