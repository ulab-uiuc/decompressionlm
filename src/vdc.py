"""Utilities for generating 1D low-discrepancy sequences based on the Van der Corput (VdC) construction"""

def generate_vdc_sequence(n: int, base: int = 2) -> list[float]:
    """
    Generate the first n points of the Van der Corput sequence.
    
    The Van der Corput sequence in base b is constructed by:
    1. Take integer i (1-indexed)
    2. Write i in base b: i = (d_k...d_1d_0)_b
    3. Reflect: VdC(i) = (0.d_0d_1...d_k)_b
    
    Args:
        n: Number of points to generate
        base: Base for sequence (default 2 for binary)
        
    Returns:
        List of n floats in [0, 1)
        
    Example:
        >>> generate_vdc_sequence(8, base=2)
        [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625]
    """
    sequence = []
    
    for i in range(1, n + 1):
        vdc_value = 0.0
        denominator = base
        num = i
        
        while num > 0:
            remainder = num % base
            vdc_value += remainder / denominator
            denominator *= base
            num //= base
            
        sequence.append(vdc_value)
    
    return sequence