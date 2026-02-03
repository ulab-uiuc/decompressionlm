#!/usr/bin/env python3
"""
Complete hallucination experiment on MMLU-Pro Law ranked models.

Part 1: Extract concept graphs from models (if not already done)
Part 2: Verify concepts against CourtListener API
Part 3: Analyze correlation with MMLU-Pro Law performance

This tests: Do better MMLU performers have lower hallucination rates?
"""

import os
import gc
import json
import time
import torch
import hashlib
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import plotext as plt
from tqdm import tqdm
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional

# ----------------------------
# Configuration
# ----------------------------

# Models ranked by MMLU-Pro Law performance (top to bottom)
MODELS = [
    # "google/gemma-2-27b-it", # at most 16B, this is too large
    "google/gemma-2-9b-it",
    # "mistralai/Mistral-Small-Instruct-2409", # at most 16B, this is too large
    "mistralai/Mistral-Nemo-Instruct-2407",
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen2-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "microsoft/Phi-3-mini-128k-instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "abacusai/Llama-3-Smaug-8B",
    "ibm-granite/granite-3.1-8b-instruct",
    "mistralai/Ministral-8B-Instruct-2410",
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    "ibm-granite/granite-3.1-2b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "m-a-p/neo_7b_instruct_v0.1",
    "ibm-granite/granite-3.1-3b-a800m-instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct",
    "deepseek-ai/deepseek-math-7b-instruct",
    "ibm-granite/granite-3.1-1b-a400m-instruct",
]

# Concept extraction parameters
EXTRACTION_PROMPT = """Generate United States bar exam legal concepts as keywords.

Please output ONE concept per line.
Each concept can be multiple words if needed.
Do not include explanations or extra text.
Please begin from any random concept.
Please use English.
"""

EXTRACTION_DIR = "results/mmlu_pro_law_concept_extraction_16"
MAX_LEN = 16
PREFIX_LEN = 16
PROB_THRESHOLD = 1.0  # Disabled
MAX_SAMPLES = 8192
BATCH_SIZE = 64
DISPLAY_INTERVAL = BATCH_SIZE

# Verification parameters
SAMPLE_SIZE = 200  # Number of concepts to sample per model
VERIFICATION_THRESHOLD = 1  # Minimum hits to consider verified
VERIFICATION_PAGE_SIZE = 1
RATE_LIMIT_DELAY = 0.05  # seconds between API calls
CACHE_DIR = "./temp/courtlistener_cache"
CACHE_TTL_HOURS = 24 * 7  # 1 week

# Output directory
OUTPUT_DIR = "results/hallucination_analysis_mmlu_law"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXTRACTION_DIR, exist_ok=True)

# ----------------------------
# CourtListener API Client
# ----------------------------

class CourtListenerClient:
    """Minimal CourtListener search client with local cache."""

    def __init__(
        self,
        api_token: Optional[str] = None,
        base_url: str = "https://www.courtlistener.com",
        cache_dir: str = "./temp/courtlistener_cache",
        cache_ttl_hours: float = 24 * 7,
        timeout: float = 15.0,
        user_agent: str = "decompressionLM-hallucination-checker/1.0",
    ):
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_s = max(0.0, cache_ttl_hours * 3600.0)
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        if api_token:
            self.session.headers.update({"Authorization": f"Token {api_token}"})

        self.search_endpoints = [
            f"{self.base_url}/api/rest/v4/search/",
            f"{self.base_url}/api/rest/v3/search/",
        ]

    @staticmethod
    def _stable_params(params: Dict) -> List[Tuple[str, str]]:
        items: List[Tuple[str, str]] = []
        for k, v in params.items():
            if v is None:
                continue
            if isinstance(v, bool):
                v = "true" if v else "false"
            items.append((str(k), str(v)))
        return sorted(items, key=lambda x: (x[0], x[1]))

    def _cache_key(self, url: str, params: Dict) -> str:
        stable = self._stable_params(params)
        blob = json.dumps(
            {"url": url, "params": stable},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_read(self, key: str) -> Optional[Dict]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            fetched_at = float(payload.get("_fetched_at", 0.0))
            if self.cache_ttl_s > 0 and (time.time() - fetched_at) > self.cache_ttl_s:
                return None
            return payload.get("data")
        except Exception:
            return None

    def _cache_write(self, key: str, data: Dict) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"_fetched_at": time.time(), "data": data}
        tmp = p.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp.replace(p)

    @staticmethod
    def _extract_count(resp_json: Dict) -> Optional[int]:
        """Extract result count from CourtListener response."""
        if not isinstance(resp_json, dict):
            return None

        if "count" in resp_json and isinstance(resp_json["count"], int):
            return resp_json["count"]

        meta = resp_json.get("meta")
        if isinstance(meta, dict):
            for k in ("total_count", "count", "total"):
                if k in meta and isinstance(meta[k], int):
                    return meta[k]

        for k in ("total", "total_results", "totalCount"):
            if k in resp_json and isinstance(resp_json[k], int):
                return resp_json[k]

        return None

    def _request_json_with_backoff(
        self,
        url: str,
        params: Dict,
        max_retries: int = 6,
        base_backoff_s: float = 0.5,
        max_backoff_s: float = 30.0,
    ) -> Dict:
        """Robust GET with cache and exponential backoff."""
        key = self._cache_key(url, params)
        cached = self._cache_read(key)
        if cached is not None:
            return cached

        last_err: Optional[str] = None

        for attempt in range(max_retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)

                if r.status_code in (429, 500, 502, 503, 504):
                    retry_after = r.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            sleep_s = float(retry_after)
                        except Exception:
                            sleep_s = base_backoff_s * (2 ** attempt)
                    else:
                        sleep_s = base_backoff_s * (2 ** attempt)

                    sleep_s = min(max_backoff_s, max(0.1, sleep_s))
                    time.sleep(sleep_s)
                    last_err = f"HTTP {r.status_code}"
                    continue

                r.raise_for_status()
                data = r.json()
                self._cache_write(key, data)
                return data

            except requests.exceptions.RequestException as e:
                last_err = str(e)
                sleep_s = min(max_backoff_s, base_backoff_s * (2 ** attempt))
                time.sleep(max(0.1, sleep_s))
            except ValueError as e:
                last_err = f"JSON decode error: {e}"
                break

        return {"_error": last_err or "unknown error"}

    def search(
        self,
        query: str,
        page_size: int = 1,
        extra_params: Optional[Dict] = None,
    ) -> Tuple[Dict, str]:
        """Perform a search query on CourtListener."""
        params = {"q": query, "page_size": page_size}
        if extra_params:
            params.update(extra_params)

        last = None
        for endpoint in self.search_endpoints:
            data = self._request_json_with_backoff(endpoint, params)
            last = (data, endpoint)

            if isinstance(data, dict) and data.get("_error"):
                continue

            cnt = self._extract_count(data) if isinstance(data, dict) else None
            if cnt is not None:
                return data, endpoint

        if last is not None:
            return last[0], last[1]
        return {"_error": "no endpoints attempted"}, "none"

    def verify_concept(
        self,
        concept: str,
        threshold: int = 1,
        page_size: int = 1,
        extra_params: Optional[Dict] = None,
    ) -> Tuple[bool, int, str, Optional[str]]:
        """
        Verify if a concept appears in CourtListener.
        
        Returns:
            (is_verified, total_count, endpoint_used, error)
        """
        data, endpoint_used = self.search(
            query=f"\"{concept}\"" if concept and " " in concept else concept,
            page_size=page_size,
            extra_params=extra_params,
        )

        if not isinstance(data, dict):
            return False, 0, endpoint_used, "non-dict response"

        if data.get("_error"):
            return False, 0, endpoint_used, data.get("_error")

        total = self._extract_count(data) or 0
        is_verified = total >= threshold

        return is_verified, total, endpoint_used, None


# ----------------------------
# Part 1: Concept Extraction (reuse from original)
# ----------------------------

def extract_concepts_for_model(model_name: str) -> bool:
    """Extract concept graph for one model."""
    from src.bin_entropy import estimate_prefix_mass
    
    print(f"\n{'='*80}")
    print(f"CONCEPT EXTRACTION: {model_name}")
    print(f"{'='*80}")
    
    safe_name = model_name.replace("/", "_")
    save_name = f"{safe_name}__us_law_concepts__L{PREFIX_LEN}__T{PROB_THRESHOLD}.delm.parquet"
    save_path = os.path.join(EXTRACTION_DIR, save_name)
    
    if os.path.exists(save_path):
        print(f"✓ Concepts already extracted: {save_path}")
        return True
    
    print("Loading model and tokenizer...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        print("Extracting concepts...")
        if "gemma" in model_name.lower():
            model.config.use_cache = False

        results = estimate_prefix_mass(
            model=model,
            tokenizer=tokenizer,
            prefix=EXTRACTION_PROMPT,
            prefix_len=PREFIX_LEN,
            prob_threshold=PROB_THRESHOLD,
            max_samples=MAX_SAMPLES,
            max_len=MAX_LEN,
            use_chat_template=True,
            batch_size=BATCH_SIZE,
            display_interval=DISPLAY_INTERVAL,
            save_path=save_path,
            model_name=model_name,
            enable_graph_analysis=True
        )
        
        print(f"\nExtraction complete:")
        print(f"  Unique prefixes: {results['unique_prefixes']}")
        print(f"  Samples done: {results['samples_done']}")
        print(f"  Effective set size: {results['effective_set_size']}")
        print(f"  Saved to: {save_path}")
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"⚠️  Error extracting concepts: {e}")
        import traceback
        traceback.print_exc()
        return False


# ----------------------------
# Part 2: Load and Sample Concepts from Parquet
# ----------------------------

def load_concepts_from_parquet(file_path: str) -> List[Tuple[str, int]]:
    """Load concepts and their frequencies from .delm.parquet file."""
    try:
        table = pq.read_table(file_path)
        metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
        
        if 'graph_concept_frequencies' not in metadata:
            print(f"  Warning: No graph_concept_frequencies in {file_path}")
            return []
        
        concept_freqs = json.loads(metadata['graph_concept_frequencies'])
        return concept_freqs  # List of [concept, frequency/degree] pairs
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return []


def sample_concepts_stratified(
    concept_freqs: List[Tuple[str, int]], 
    n_samples: int
) -> List[Tuple[str, int]]:
    """
    Sample concepts with stratified sampling across frequency ranges.
    
    Distribution:
    - Top 25%: high frequency (50 concepts)
    - Middle 50%: medium frequency (100 concepts)
    - Bottom 25%: low frequency (50 concepts)
    """
    if len(concept_freqs) <= n_samples:
        return concept_freqs
    
    # Sort by frequency (descending)
    sorted_concepts = sorted(concept_freqs, key=lambda x: x[1], reverse=True)
    
    # Calculate sample sizes
    n_high = int(n_samples * 0.25)
    n_mid = int(n_samples * 0.50)
    n_low = n_samples - n_high - n_mid
    
    # Define stratum boundaries
    total = len(sorted_concepts)
    high_idx = int(total * 0.25)
    mid_idx = int(total * 0.75)
    
    sampled = []
    
    # High frequency: top 50
    high_stratum = sorted_concepts[:high_idx]
    sampled.extend(high_stratum[:n_high])
    
    # Medium frequency: evenly sample 100
    mid_stratum = sorted_concepts[high_idx:mid_idx]
    if len(mid_stratum) > 0:
        step = max(1, len(mid_stratum) // n_mid)
        sampled.extend(mid_stratum[::step][:n_mid])
    
    # Low frequency: evenly sample 50
    low_stratum = sorted_concepts[mid_idx:]
    if len(low_stratum) > 0:
        step = max(1, len(low_stratum) // n_low)
        sampled.extend(low_stratum[::step][:n_low])
    
    return sampled[:n_samples]


# ----------------------------
# Part 3: Hallucination Verification
# ----------------------------

def verify_concepts_for_model(model_name: str, client: CourtListenerClient) -> pd.DataFrame:
    """Verify concepts for one model against CourtListener."""
    print(f"\n{'='*80}")
    print(f"HALLUCINATION VERIFICATION: {model_name}")
    print(f"{'='*80}")
    
    # Construct file path
    safe_name = model_name.replace("/", "_")
    concept_file = f"{safe_name}__us_law_concepts__L{PREFIX_LEN}__T{PROB_THRESHOLD}.delm.parquet"
    concept_path = os.path.join(EXTRACTION_DIR, concept_file)
    
    if not os.path.exists(concept_path):
        print(f"  ⚠️  Concept file not found: {concept_path}")
        return None
    
    # Load concepts
    print(f"  Loading concepts from {concept_file}...")
    concept_freqs = load_concepts_from_parquet(concept_path)
    
    if not concept_freqs:
        print(f"  ⚠️  No concepts found")
        return None
    
    print(f"  Total concepts: {len(concept_freqs)}")
    
    # Sample concepts
    sampled_concepts = sample_concepts_stratified(concept_freqs, SAMPLE_SIZE)
    print(f"  Sampled: {len(sampled_concepts)} concepts")
    
    # Verify each concept
    results = []
    
    print(f"  Verifying against CourtListener...")
    for concept, frequency in tqdm(sampled_concepts, desc="  Concepts"):
        try:
            is_verified, hit_count, endpoint, error = client.verify_concept(
                concept=concept,
                threshold=VERIFICATION_THRESHOLD,
                page_size=VERIFICATION_PAGE_SIZE,
            )
            
            results.append({
                'model': model_name,
                'concept': concept,
                'frequency': frequency,
                'verified': is_verified,
                'hit_count': hit_count,
                'endpoint': endpoint,
                'error': error,
                'is_hallucination': not is_verified and error is None
            })
            
            # Rate limiting
            if RATE_LIMIT_DELAY > 0:
                time.sleep(RATE_LIMIT_DELAY)
                
        except Exception as e:
            print(f"    ⚠️  Error verifying '{concept}': {e}")
            results.append({
                'model': model_name,
                'concept': concept,
                'frequency': frequency,
                'verified': False,
                'hit_count': 0,
                'endpoint': 'error',
                'error': str(e),
                'is_hallucination': False  # Don't count errors as hallucinations
            })
            continue
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        # Calculate statistics
        total = len(df)
        errors = df['error'].notna().sum()
        valid = total - errors
        
        if valid > 0:
            verified = df['verified'].sum()
            hallucinations = df['is_hallucination'].sum()
            hallucination_rate = hallucinations / valid
            
            print(f"\n  Results:")
            print(f"    Total concepts: {total}")
            print(f"    Errors: {errors}")
            print(f"    Valid checks: {valid}")
            print(f"    Verified: {verified} ({verified/valid*100:.1f}%)")
            print(f"    Hallucinations: {hallucinations} ({hallucination_rate*100:.1f}%)")
            
            # Frequency correlation
            valid_df = df[df['error'].isna()]
            if len(valid_df) > 1:
                corr = valid_df[['frequency', 'verified']].corr().iloc[0, 1]
                print(f"    Freq-Verified Correlation: {corr:+.3f}")
    
    return df


# ----------------------------
# Part 4: Analysis & Visualization
# ----------------------------

def plot_model_comparison_terminal(df: pd.DataFrame):
    """Create terminal-based plots comparing models."""
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON - HALLUCINATION RATE vs MMLU-Pro Law Rank")
    print(f"{'='*80}\n")
    
    # Calculate stats per model
    model_stats = []
    for i, model_name in enumerate(MODELS):
        model_df = df[df['model'] == model_name]
        
        if len(model_df) == 0:
            continue
        
        # Exclude errors from hallucination calculation
        valid_df = model_df[model_df['error'].isna()]
        
        if len(valid_df) == 0:
            continue
        
        verified = valid_df['verified'].sum()
        hallucinations = valid_df['is_hallucination'].sum()
        total_valid = len(valid_df)
        hallucination_rate = hallucinations / total_valid if total_valid > 0 else 0
        
        model_stats.append({
            'model': model_name.split('/')[-1][:20],
            'rank': i + 1,
            'n_concepts': len(model_df),
            'n_valid': total_valid,
            'verified': verified,
            'hallucinations': hallucinations,
            'hallucination_rate': hallucination_rate,
            'error_rate': (len(model_df) - total_valid) / len(model_df)
        })
    
    if not model_stats:
        print("No data to plot")
        return
    
    stats_df = pd.DataFrame(model_stats)
    
    # Plot: Hallucination rate vs MMLU rank
    print("\n1. Hallucination Rate vs MMLU-Pro Law Rank")
    print("-" * 80)
    
    plt.clf()
    plt.scatter(
        stats_df['rank'].values,
        stats_df['hallucination_rate'].values * 100,
        marker='dot'
    )
    
    plt.xlabel("MMLU-Pro Law Rank (1=best)")
    plt.ylabel("Hallucination Rate (%)")
    plt.title("Do better MMLU models hallucinate less?")
    plt.plotsize(100, 25)
    plt.show()
    
    # Calculate correlation
    rank_corr = np.corrcoef(stats_df['rank'], stats_df['hallucination_rate'])[0, 1]
    print(f"\nRank-Hallucination Correlation: {rank_corr:+.3f}")
    print("(Positive = better MMLU → more hallucinations [unexpected])")
    print("(Negative = better MMLU → fewer hallucinations [expected])")
    
    # Print detailed stats
    print("\n\n2. Detailed Model Statistics")
    print("-" * 80)
    print(f"{'Rank':<5} {'Model':<22} {'Valid':>6} {'Verified':>9} {'Halluc':>8} {'Hall%':>7} {'Err%':>6}")
    print("-" * 80)
    
    for _, row in stats_df.iterrows():
        print(f"{row['rank']:>4} {row['model']:<22} "
              f"{row['n_valid']:>6.0f} "
              f"{row['verified']:>9.0f} "
              f"{row['hallucinations']:>8.0f} "
              f"{row['hallucination_rate']*100:>6.1f}% "
              f"{row['error_rate']*100:>5.1f}%")


def plot_top_vs_bottom_terminal(df: pd.DataFrame):
    """Compare top 5 vs bottom 5 models."""
    print(f"\n{'='*80}")
    print(f"TOP 5 vs BOTTOM 5 MODELS")
    print(f"{'='*80}\n")
    
    top_5_models = MODELS[:5]
    bottom_5_models = MODELS[-5:]
    
    top_df = df[df['model'].isin(top_5_models) & df['error'].isna()]
    bottom_df = df[df['model'].isin(bottom_5_models) & df['error'].isna()]
    
    if len(top_df) == 0 or len(bottom_df) == 0:
        print("Insufficient data")
        return
    
    print("Top 5 models (best MMLU-Pro Law):")
    for model in top_5_models:
        print(f"  - {model}")
    
    print("\nBottom 5 models (worst MMLU-Pro Law):")
    for model in bottom_5_models:
        print(f"  - {model}")
    
    # Calculate hallucination rates
    top_hall_rate = top_df['is_hallucination'].sum() / len(top_df) * 100
    bottom_hall_rate = bottom_df['is_hallucination'].sum() / len(bottom_df) * 100
    
    print("\nHallucination Rates:")
    print(f"  Top 5:    {top_hall_rate:.1f}%")
    print(f"  Bottom 5: {bottom_hall_rate:.1f}%")
    print(f"  Difference: {bottom_hall_rate - top_hall_rate:+.1f}%")
    
    # Statistical test
    from scipy.stats import chi2_contingency
    
    contingency = np.array([
        [top_df['is_hallucination'].sum(), (~top_df['is_hallucination']).sum()],
        [bottom_df['is_hallucination'].sum(), (~bottom_df['is_hallucination']).sum()]
    ])
    
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    print(f"\n  Chi-square test: χ²={chi2:.3f}, p={p_value:.4f}")
    if p_value < 0.05:
        print(f"  ✓ Significant difference (p < 0.05)")
    else:
        print(f"  ✗ No significant difference (p >= 0.05)")


def generate_summary_stats(df: pd.DataFrame, output_path: str):
    """Generate summary statistics table."""
    summary_data = []
    
    for i, model_name in enumerate(MODELS):
        model_df = df[df['model'] == model_name]
        
        if len(model_df) == 0:
            continue
        
        valid_df = model_df[model_df['error'].isna()]
        
        if len(valid_df) == 0:
            continue
        
        verified = valid_df['verified'].sum()
        hallucinations = valid_df['is_hallucination'].sum()
        total_valid = len(valid_df)
        
        summary_data.append({
            'MMLU_Rank': i + 1,
            'Model': model_name,
            'Total_Concepts': len(model_df),
            'Valid_Checks': total_valid,
            'Verified': verified,
            'Hallucinations': hallucinations,
            'Hallucination_Rate': hallucinations / total_valid if total_valid > 0 else 0,
            'Verification_Rate': verified / total_valid if total_valid > 0 else 0,
            'Error_Count': len(model_df) - total_valid,
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n✓ Saved summary: {output_path}")
    
    return summary_df


# ----------------------------
# Main Pipeline
# ----------------------------

def main():
    print("="*80)
    print("COMPLETE HALLUCINATION EXPERIMENT - MMLU-Pro Law Models")
    print("="*80)
    print(f"\nPhase 1: Concept Extraction")
    print(f"Phase 2: CourtListener Verification")
    print(f"Phase 3: Analysis & Visualization")
    print(f"\nModels: {len(MODELS)} models ranked by MMLU-Pro Law")
    print(f"Sample size: {SAMPLE_SIZE} concepts per model")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Get API token
    api_token = os.environ.get("COURTLISTENER_TOKEN")
    if api_token:
        print(f"✓ CourtListener API token found")
    else:
        print(f"⚠️  No CourtListener API token (set COURTLISTENER_TOKEN env var)")
    
    # Phase 1: Extract concepts
    print("\n" + "="*80)
    print("PHASE 1: CONCEPT EXTRACTION")
    print("="*80)
    
    for model_name in MODELS:
        extract_concepts_for_model(model_name)
    
    # Phase 2: Verify concepts
    print("\n" + "="*80)
    print("PHASE 2: HALLUCINATION VERIFICATION")
    print("="*80)
    
    client = CourtListenerClient(
        api_token=api_token,
        cache_dir=CACHE_DIR,
        cache_ttl_hours=CACHE_TTL_HOURS,
    )
    
    all_results = []
    
    for model_name in MODELS:
        # Check if already verified
        safe_name = model_name.replace("/", "_")
        result_file = os.path.join(OUTPUT_DIR, f"{safe_name}__hallucination.csv")
        
        if os.path.exists(result_file):
            print(f"\n✓ Already verified: {model_name}")
            df = pd.read_csv(result_file)
            all_results.append(df)
            continue
        
        # Verify concepts
        df = verify_concepts_for_model(model_name, client)
        
        if df is not None and len(df) > 0:
            # Save individual result
            df.to_csv(result_file, index=False)
            print(f"  ✓ Saved: {result_file}")
            all_results.append(df)
    
    # Phase 3: Analysis
    print("\n" + "="*80)
    print("PHASE 3: ANALYSIS & VISUALIZATION")
    print("="*80)
    
    if not all_results:
        print("\n⚠️  No results to analyze")
        return
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_file = os.path.join(OUTPUT_DIR, "all_hallucination_results.csv")
    combined_df.to_csv(combined_file, index=False)
    print(f"\n✓ Saved combined results: {combined_file}")
    
    # Generate visualizations
    plot_model_comparison_terminal(combined_df)
    plot_top_vs_bottom_terminal(combined_df)
    
    # Generate summary statistics
    summary_path = os.path.join(OUTPUT_DIR, "hallucination_summary_stats.csv")
    summary_df = generate_summary_stats(combined_df, summary_path)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - Combined data: all_hallucination_results.csv")
    print(f"  - Summary stats: hallucination_summary_stats.csv")
    print(f"  - Individual CSVs: *__hallucination.csv")
    print(f"\nConcept extractions saved to: {EXTRACTION_DIR}")
    print(f"CourtListener cache saved to: {CACHE_DIR}")
    
    print("\n" + "="*80)
    print("KEY HYPOTHESIS TEST")
    print("="*80)
    print("\nDo models with better MMLU-Pro Law scores have:")
    print("  1. Lower hallucination rates?")
    print("  2. Higher verification rates?")
    print("\nCheck the analysis above for answers!")


if __name__ == "__main__":
    main()