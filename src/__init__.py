"""
decompressionLM - Concept-based knowledge extraction from LLMs.

Main sampling methods:
- VdC (Van der Corput deterministic sampling)
- Random (reproducible random sampling)
- Beam search (low/high temperature)
- BFS (breadth-first concept exploration)
- DFS (depth-first concept exploration)
"""

__version__ = "2.0.0"

# Core sampling
from .concept_sampling import sample_concepts

# Exploration methods
from .exploration_sampling import explore_bfs, explore_dfs, ConceptNode

# Utilities
from .concept_utils import is_valid_concept, normalize_concept
from .profiling import ProfileStats, profile_section

__all__ = [
    "sample_concepts",
    "explore_bfs",
    "explore_dfs",
    "ConceptNode",
    "is_valid_concept",
    "normalize_concept",
    "ProfileStats",
    "profile_section",
]
