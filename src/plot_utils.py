"""Plotting utilities for bin entropy visualization."""

from typing import List, Tuple


def get_bin_colors(n_bins: int) -> List[Tuple[int, int, int]]:
    """
    Generate visually distinct RGB colors for bin lengths.
    Avoids white and very light colors for visibility.
    
    Args:
        n_bins: Number of different bin lengths
        
    Returns:
        List of RGB tuples (r, g, b) where each value is 0-255
    """
    if n_bins <= 0:
        return []
    
    # Predefined palette for common cases (up to 12 bin lengths)
    # These are carefully chosen to be distinct and visible on dark backgrounds
    palette = [
        (255, 85, 85),    # Red
        (85, 170, 255),   # Blue
        (85, 255, 85),    # Green
        (255, 170, 85),   # Orange
        (255, 85, 255),   # Magenta
        (85, 255, 255),   # Cyan
        (255, 255, 85),   # Yellow
        (170, 85, 255),   # Purple
        (255, 170, 170),  # Light Red
        (170, 255, 170),  # Light Green
        (170, 170, 255),  # Light Blue
        (255, 170, 255),  # Light Magenta
    ]
    
    if n_bins <= len(palette):
        return palette[:n_bins]
    
    # For more than 12 bins, generate colors using HSV-like distribution
    colors = []
    for i in range(n_bins):
        # Distribute hues evenly around the color wheel
        hue = i / n_bins
        
        # Convert HSV to RGB (with S=0.8, V=0.9 for good visibility)
        # Avoiding very light colors
        h = hue * 6.0
        x = 1.0 - abs((h % 2.0) - 1.0)
        
        if h < 1:
            r, g, b = 1.0, x, 0.0
        elif h < 2:
            r, g, b = x, 1.0, 0.0
        elif h < 3:
            r, g, b = 0.0, 1.0, x
        elif h < 4:
            r, g, b = 0.0, x, 1.0
        elif h < 5:
            r, g, b = x, 0.0, 1.0
        else:
            r, g, b = 1.0, 0.0, x
        
        # Apply saturation and value
        s, v = 0.8, 0.9
        r = int(((r - 1.0) * s + 1.0) * v * 255)
        g = int(((g - 1.0) * s + 1.0) * v * 255)
        b = int(((b - 1.0) * s + 1.0) * v * 255)
        
        colors.append((r, g, b))
    
    return colors