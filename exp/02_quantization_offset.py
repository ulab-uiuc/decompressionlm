#!/usr/bin/env python3
"""
VdC Offset Experiment Runner.
For Llama 3.1-8B-Instruct and Qwen2.5-7B-Instruct, run the same question 8 times with different VdC offsets.
Save results for later analysis.
"""

import os
import gc
import json
import random
import torch
import numpy as np
import sys
import struct
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from src.bin_entropy import estimate_prefix_mass
import pyarrow as pa
import pyarrow.parquet as pq

# ----------------------------
# Configuration
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# All quantization variants to test (both models)
VARIANTS = [
    # ==========================================
    # 1. QWEN-2.5-7B-INSTRUCT
    # ==========================================
    {
        "label": "Qwen2.5-7B-Instruct__BF16_BASE",
        "repo": "Qwen/Qwen2.5-7B-Instruct",
        "dtype": torch.bfloat16,
    },
    {
        "label": "Qwen2.5-7B-Instruct__GPTQ_INT8",
        "repo": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
        "dtype": "auto",
    },
    {
        "label": "Qwen2.5-7B-Instruct__AWQ_4BIT",
        "repo": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "dtype": "auto",
    },
    {
        "label": "Qwen2.5-7B-Instruct__GPTQ_INT4",
        "repo": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        "dtype": "auto",
    },
    {
        "label": "Qwen2.5-7B-Instruct__BNB_4BIT_UNSLOTH",
        "repo": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "dtype": "auto",
    },
    # ==========================================
    # 2. LLAMA-3.1-8B-INSTRUCT
    # ==========================================
    {
        "label": "Llama-3.1-8B-Instruct__BF16_BASE",
        "repo": "meta-llama/Llama-3.1-8B-Instruct",
        "dtype": torch.bfloat16,
    },
    {
        "label": "Llama-3.1-8B-Instruct__GPTQ_INT8",
        "repo": "abdo-Mansour/Meta-Llama-3.1-8B-Instruct-GPTQ-8bit",
        "dtype": "auto",
    },
    {
        "label": "Llama-3.1-8B-Instruct__AWQ_4BIT",
        "repo": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "dtype": "auto",
    },
    {
        "label": "Llama-3.1-8B-Instruct__GPTQ_INT4",
        "repo": "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
        "dtype": "auto",
    },
    {
        "label": "Llama-3.1-8B-Instruct__BNB_4BIT_UNSLOTH",
        "repo": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "dtype": "auto",
    },
]

# Experiment parameters
SEQ_LENS = [16, 32]
NUM_RUNS = 8  # Number of different VdC offsets to try
PROMPT = """Generate United States bar exam legal concepts as keywords.

Please output ONE concept per line.
Each concept can be multiple words if needed.
Do not include explanations or extra text.
Please begin from any random concept.
Please use English.
"""

# Output directories
OUTPUT_DIR = "results/vdc_offset_experiment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File to store generated VdC codes in BINARY format
VDC_CODES_BINARY = os.path.join(OUTPUT_DIR, "vdc_codes.bin")
VDC_CODES_META = os.path.join(OUTPUT_DIR, "vdc_codes_meta.json")

# ----------------------------
# Helper Functions - BINARY FORMAT
# ----------------------------
def generate_vdc_codes_binary(num_codes, seed=42):
    """Generate random VdC codes in [0, 1) and save in binary format."""
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate codes as numpy float64
    codes_np = np.zeros(num_codes, dtype=np.float64)
    for i in range(num_codes):
        base = i / num_codes
        jitter = np.random.uniform(-0.01, 0.01)
        codes_np[i] = (base + jitter) % 1.0
    
    print(f"\nGenerated {num_codes} VdC codes (seed={seed}):")
    print(f"  NumPy dtype: {codes_np.dtype}")
    print(f"  Memory layout: {codes_np.flags}")
    
    # Print with full precision
    for i, code in enumerate(codes_np):
        # Use numpy's string representation for full precision
        code_str = np.format_float_positional(code, precision=17, unique=True, fractional=True, trim='k')
        print(f"  Run {i+1}: {code_str}")
    
    # Save binary data (raw bytes)
    with open(VDC_CODES_BINARY, 'wb') as f:
        # Write header: magic number, version, count, dtype info
        f.write(b'VDC1')  # Magic number
        f.write(struct.pack('I', 1))  # Version
        f.write(struct.pack('Q', num_codes))  # Number of codes
        
        # Write dtype information
        dtype_str = str(codes_np.dtype).encode('utf-8')
        f.write(struct.pack('I', len(dtype_str)))
        f.write(dtype_str)
        
        # Write the actual data
        f.write(codes_np.tobytes())
    
    print(f"\n✓ Saved binary data to: {VDC_CODES_BINARY}")
    print(f"  File size: {os.path.getsize(VDC_CODES_BINARY)} bytes")
    print(f"  Data size: {codes_np.nbytes} bytes for {num_codes} codes")
    
    # Save metadata as JSON
    metadata = {
        "seed": seed,
        "num_codes": num_codes,
        "generated_with_seed": seed,
        "description": "VdC codes generated for offset experiments",
        "generation_method": "evenly spaced with jitter [-0.01, 0.01]",
        "numpy_dtype": str(codes_np.dtype),
        "binary_file": VDC_CODES_BINARY,
        "created_at": np.datetime64('now').astype(str),
        "note": "Codes stored in binary format for exact reproducibility"
    }
    
    with open(VDC_CODES_META, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved metadata to: {VDC_CODES_META}")
    
    return codes_np

def load_vdc_codes_binary():
    """Load VdC codes from binary file with verification."""
    if not os.path.exists(VDC_CODES_BINARY):
        print(f"❌ VdC binary file not found: {VDC_CODES_BINARY}")
        return None
    
    try:
        with open(VDC_CODES_BINARY, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic != b'VDC1':
                print(f"❌ Invalid magic number in {VDC_CODES_BINARY}: {magic}")
                return None
            
            version = struct.unpack('I', f.read(4))[0]
            if version != 1:
                print(f"❌ Unsupported version {version} in {VDC_CODES_BINARY}")
                return None
            
            num_codes = struct.unpack('Q', f.read(8))[0]
            
            # Read dtype info
            dtype_len = struct.unpack('I', f.read(4))[0]
            dtype_str = f.read(dtype_len).decode('utf-8')
            
            # Read binary data
            expected_bytes = num_codes * np.dtype(dtype_str).itemsize
            data_bytes = f.read(expected_bytes)
            
            if len(data_bytes) != expected_bytes:
                print(f"❌ Data size mismatch: expected {expected_bytes}, got {len(data_bytes)}")
                return None
            
            # Convert to numpy array
            codes_np = np.frombuffer(data_bytes, dtype=dtype_str)
            
            # Verify we got the right number of codes
            if len(codes_np) != num_codes:
                print(f"❌ Code count mismatch: expected {num_codes}, got {len(codes_np)}")
                return None
            
        print(f"\n✓ Loaded {len(codes_np)} VdC codes from binary file:")
        print(f"  NumPy dtype: {codes_np.dtype}")
        print(f"  Shape: {codes_np.shape}")
        
        # Print with full precision
        for i, code in enumerate(codes_np):
            code_str = np.format_float_positional(code, precision=17, unique=True, fractional=True, trim='k')
            print(f"  Run {i+1}: {code_str}")
        
        # Load metadata if available
        if os.path.exists(VDC_CODES_META):
            with open(VDC_CODES_META, 'r') as f:
                metadata = json.load(f)
            print(f"\n📋 METADATA:")
            print(f"  Seed: {metadata.get('seed', 'unknown')}")
            print(f"  Generation method: {metadata.get('generation_method', 'unknown')}")
        
        return codes_np
        
    except Exception as e:
        print(f"❌ Error loading binary VdC codes: {e}")
        import traceback
        traceback.print_exc()
        return None

def verify_vdc_codes(codes_np):
    """Verify VdC codes meet requirements."""
    if codes_np is None:
        return False
    
    print(f"\n🔍 VDC CODE VERIFICATION:")
    
    # Check dtype
    print(f"  Data type: {codes_np.dtype}")
    if codes_np.dtype != np.float64:
        print(f"  ⚠ Warning: Expected float64, got {codes_np.dtype}")
    
    # Check range
    in_range = np.all((codes_np >= 0) & (codes_np < 1))
    print(f"  All in range [0,1): {in_range}")
    if not in_range:
        min_val = np.min(codes_np)
        max_val = np.max(codes_np)
        print(f"    Min: {min_val}")
        print(f"    Max: {max_val}")
    
    # Check uniqueness (allowing for floating point epsilon)
    unique_codes = np.unique(codes_np)
    print(f"  Unique codes: {len(unique_codes)} out of {len(codes_np)}")
    if len(unique_codes) != len(codes_np):
        print(f"    ⚠ Warning: {len(codes_np) - len(unique_codes)} duplicate(s)")
    
    # Check statistics
    print(f"  Statistics:")
    print(f"    Min: {np.min(codes_np):.17f}")
    print(f"    Max: {np.max(codes_np):.17f}")
    print(f"    Mean: {np.mean(codes_np):.17f}")
    print(f"    Std: {np.std(codes_np):.17f}")
    
    # Check byte representation
    print(f"  Memory:")
    print(f"    Total bytes: {codes_np.nbytes}")
    print(f"    Bytes per code: {codes_np.dtype.itemsize}")
    
    return in_range and (len(unique_codes) == len(codes_np))

# ----------------------------
# Experiment Functions
# ----------------------------
def run_vdc_experiment(variant, vdc_codes, seq_len, max_samples=2048, batch_size=128):
    """Run experiment with multiple VdC codes for a single variant."""
    label = variant["label"]
    repo = variant["repo"]
    dtype = variant["dtype"]
    
    print(f"\n{'='*80}")
    print(f"Running VdC experiment for: {label}")
    print(f"Sequence length: {seq_len}")
    print(f"Number of runs: {len(vdc_codes)}")
    print(f"Seed: {SEED} (for reproducibility)")
    print(f"{'='*80}")
    
    # Load model and tokenizer
    print(f"\nLoading model and tokenizer for {label}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Special handling for specific models
        if repo == "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4":
            config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
            qc = getattr(config, "quantization_config", None)
            if qc:
                qc["desc_act"] = False
                config.quantization_config = qc
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                config=config,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        
        print(f"✓ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model {label}: {e}")
        return
    
    # Run experiments for each VdC code
    for run_idx, vdc_code in enumerate(vdc_codes):
        print(f"\n--- Run {run_idx+1}/{len(vdc_codes)} (VdC: {vdc_code:.17f}) ---")
        
        # Create output filename with full precision VdC code
        vdc_str = f"{vdc_code:.17f}".replace('.', 'p').replace('-', 'm')
        output_file = os.path.join(
            OUTPUT_DIR,
            f"{label}__seq{seq_len}__run{run_idx+1}__vdc{vdc_str}.delm.parquet"
        )
        
        # Check if result already exists
        if os.path.exists(output_file):
            print(f"✓ Result already exists: {os.path.basename(output_file)}")
            print(f"  Skipping... (delete file to regenerate)")
            continue
        
        # Run new experiment
        try:
            print(f"  Running experiment with seed={SEED}...")
            
            # Set deterministic behavior for torch
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            
            results = estimate_prefix_mass(
                model=model,
                tokenizer=tokenizer,
                prefix=PROMPT,
                prefix_len=seq_len,
                prob_threshold=1.0,  # Disable threshold
                max_samples=max_samples,
                max_len=seq_len,
                use_chat_template=True,
                batch_size=batch_size,
                display_interval=batch_size,
                save_path=output_file,
                model_name=label,
                enable_graph_analysis=True,
                offset=float(vdc_code),  # Convert numpy float64 to Python float
            )
            
            # Save VdC code information in metadata with BINARY representation
            try:
                table = pq.read_table(output_file)
                metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
                metadata['experiment_seed'] = str(SEED)
                metadata['vdc_code'] = str(vdc_code)  # Full precision string
                metadata['vdc_code_float'] = f"{vdc_code:.17f}"  # Formatted for readability
                metadata['vdc_code_bytes'] = vdc_code.tobytes().hex()  # Binary representation
                metadata['vdc_code_dtype'] = str(vdc_code.dtype)
                metadata['run_index'] = str(run_idx + 1)
                metadata['total_runs'] = str(len(vdc_codes))
                metadata['sequence_length'] = str(seq_len)
                
                # Convert metadata back to bytes
                new_metadata = {k.encode(): v.encode() for k, v in metadata.items()}
                
                # Create new table with updated metadata
                new_table = table.replace_schema_metadata(new_metadata)
                
                # Write back to file
                pq.write_table(new_table, output_file)
            except Exception as e:
                print(f"  Warning: Could not update metadata: {e}")
            
            print(f"  ✓ Saved to: {os.path.basename(output_file)}")
            print(f"  Concepts generated: {results.get('effective_set_size', 'N/A')}")
            
        except Exception as e:
            print(f"  ❌ Error in run {run_idx+1}: {e}")
            import traceback
            traceback.print_exc()
    
    # Clean up
    print(f"\nCleaning up memory for {label}...")
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("✓ Memory cleaned")

# ----------------------------
# Main Function
# ----------------------------
def main():
    print("="*80)
    print("VdC OFFSET EXPERIMENT RUNNER - BINARY PRECISION")
    print(f"Seed: {SEED} (for reproducibility)")
    print("Testing Llama 3.1-8B-Instruct and Qwen2.5-7B-Instruct")
    print("="*80)
    
    print(f"\n📝 CONFIGURATION:")
    print(f"  Random seed: {SEED}")
    print(f"  Total quantization variants: {len(VARIANTS)}")
    print(f"  Sequence lengths: {SEQ_LENS}")
    print(f"  VdC runs per variant: {NUM_RUNS}")
    total_experiments = len(VARIANTS) * len(SEQ_LENS) * NUM_RUNS
    print(f"  Total expected experiments: {total_experiments}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Binary data file: {VDC_CODES_BINARY}")
    print(f"  Metadata file: {VDC_CODES_META}")
    
    # Generate or load VdC codes in BINARY format
    vdc_codes_np = load_vdc_codes_binary()
    if vdc_codes_np is None:
        print(f"\nGenerating new VdC codes with seed={SEED} (binary format)...")
        vdc_codes_np = generate_vdc_codes_binary(NUM_RUNS, seed=SEED)
    else:
        if len(vdc_codes_np) != NUM_RUNS:
            print(f"\nWarning: Loaded {len(vdc_codes_np)} codes, but expected {NUM_RUNS}")
            print(f"Generating new codes with seed={SEED}...")
            vdc_codes_np = generate_vdc_codes_binary(NUM_RUNS, seed=SEED)
        else:
            print(f"\n✓ Using existing binary VdC codes")
    
    # Verify codes
    if not verify_vdc_codes(vdc_codes_np):
        print(f"\n❌ VdC codes failed verification!")
        return
    
    # Run experiments for each sequence length
    for seq_len in SEQ_LENS:
        print(f"\n{'='*80}")
        print(f"STARTING EXPERIMENT: Seq Len = {seq_len}")
        print(f"{'='*80}")
        
        # Run each quantization variant
        for variant_idx, variant in enumerate(VARIANTS):
            variant_label = variant["label"]
            print(f"\n{'='*80}")
            print(f"PROCESSING [{variant_idx+1}/{len(VARIANTS)}]: {variant_label}")
            print(f"{'='*80}")
            
            # Run all VdC codes for this variant
            run_vdc_experiment(
                variant=variant,
                vdc_codes=vdc_codes_np,
                seq_len=seq_len,
                max_samples=2048,  # Reduced for speed
                batch_size=128
            )
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    
    # Summary
    print(f"\n📋 EXPERIMENT SUMMARY:")
    print(f"  Models tested: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct")
    print(f"  Random seed used: {SEED}")
    print(f"  Quantization variants: {len(VARIANTS)} (5 per model)")
    print(f"  VdC codes used: {len(vdc_codes_np)}")
    print(f"  VdC data type: {vdc_codes_np.dtype}")
    print(f"  Sequence lengths: {SEQ_LENS}")
    print(f"  Output directory: {OUTPUT_DIR}")
    
    # Count generated files
    expected_files = total_experiments
    actual_files = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.delm.parquet')])
    
    print(f"\n📊 FILES GENERATED:")
    print(f"  Expected: {expected_files} files")
    print(f"  Actual: {actual_files} files")
    
    if actual_files < expected_files:
        missing = expected_files - actual_files
        print(f"  Missing: {missing} files ({missing/expected_files*100:.1f}%)")
        print(f"  Run script again to generate missing files:")
        print(f"    python {sys.argv[0]}")
    else:
        print(f"  ✓ All files generated successfully!")
        print(f"  ✓ Results are reproducible with seed={SEED}")
    
    # Binary file info
    print(f"\n💾 BINARY FILES:")
    print(f"  VdC codes (binary): {VDC_CODES_BINARY}")
    print(f"    Size: {os.path.getsize(VDC_CODES_BINARY)} bytes")
    print(f"  Metadata (JSON): {VDC_CODES_META}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()