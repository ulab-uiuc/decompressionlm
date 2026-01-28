"""Profiling utilities for tracking time and GPU memory usage."""

import time
import torch
from contextlib import contextmanager
from typing import Dict, Optional, List
from collections import defaultdict


class ProfileStats:
    """Track timing and memory statistics."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.gpu_memory = defaultdict(list)
        self.counters = defaultdict(int)
        
    def record_time(self, stage: str, duration: float):
        """Record a timing measurement."""
        self.timings[stage].append(duration)
        self.counters[stage] += 1
    
    def record_gpu_memory(self, stage: str):
        """Record current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            self.gpu_memory[stage].append((allocated, reserved))
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {}
        
        # Timing stats
        for stage, durations in self.timings.items():
            summary[f"time_{stage}_total"] = sum(durations)
            summary[f"time_{stage}_count"] = len(durations)
            summary[f"time_{stage}_avg"] = sum(durations) / len(durations) if durations else 0
            summary[f"time_{stage}_min"] = min(durations) if durations else 0
            summary[f"time_{stage}_max"] = max(durations) if durations else 0
        
        # GPU memory stats
        for stage, measurements in self.gpu_memory.items():
            if measurements:
                allocated = [m[0] for m in measurements]
                reserved = [m[1] for m in measurements]
                
                summary[f"gpu_{stage}_allocated_avg"] = sum(allocated) / len(allocated)
                summary[f"gpu_{stage}_allocated_min"] = min(allocated)
                summary[f"gpu_{stage}_allocated_max"] = max(allocated)
                
                summary[f"gpu_{stage}_reserved_avg"] = sum(reserved) / len(reserved)
                summary[f"gpu_{stage}_reserved_min"] = min(reserved)
                summary[f"gpu_{stage}_reserved_max"] = max(reserved)
        
        return summary
    
    def print_summary(self):
        """Print formatted summary."""
        print(f"\n{'='*80}")
        print("PROFILING SUMMARY")
        print(f"{'='*80}")
        
        # Timing
        print("\nTiming (seconds):")
        for stage in sorted(self.timings.keys()):
            durations = self.timings[stage]
            total = sum(durations)
            count = len(durations)
            avg = total / count if count > 0 else 0
            min_t = min(durations) if durations else 0
            max_t = max(durations) if durations else 0
            
            print(f"  {stage:30s}: total={total:8.2f}s  avg={avg:8.4f}s  min={min_t:8.4f}s  max={max_t:8.4f}s  (n={count})")
        
        # GPU Memory
        if self.gpu_memory:
            print("\nGPU Memory (GB):")
            for stage in sorted(self.gpu_memory.keys()):
                measurements = self.gpu_memory[stage]
                allocated = [m[0] for m in measurements]
                reserved = [m[1] for m in measurements]
                
                print(f"  {stage:30s}:")
                print(f"    Allocated: avg={sum(allocated)/len(allocated):6.2f}  min={min(allocated):6.2f}  max={max(allocated):6.2f}")
                print(f"    Reserved:  avg={sum(reserved)/len(reserved):6.2f}  min={min(reserved):6.2f}  max={max(reserved):6.2f}")
        
        print(f"{'='*80}\n")


@contextmanager
def profile_section(stats: ProfileStats, stage: str, record_memory: bool = True):
    """
    Context manager for profiling a code section.
    
    Usage:
        stats = ProfileStats()
        with profile_section(stats, "sampling"):
            # ... code to profile ...
    """
    start_time = time.time()
    
    if record_memory:
        stats.record_gpu_memory(f"{stage}_start")
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        stats.record_time(stage, duration)
        
        if record_memory:
            stats.record_gpu_memory(f"{stage}_end")


def format_time(seconds: float) -> str:
    """Format time duration as human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m{secs:.1f}s"


def measure_model_memory():
    """Measure static model memory usage."""
    if not torch.cuda.is_available():
        return {}
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    
    return {
        "model_memory_allocated_gb": allocated,
        "model_memory_reserved_gb": reserved,
    }
