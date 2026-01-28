"""Concept filtering and validation utilities for decompressionLM."""

import re
from typing import Set, List, Tuple


# Pattern rules:
# - Must start with a letter
# - Tokens are alphanumeric (letters/digits)
# - Tokens may be separated by a single space or a single hyphen
# - No leading/trailing separators, no double separators
# - No punctuation or symbols besides hyphen
_VALID_CONCEPT_RE = re.compile(r'^[A-Za-z][A-Za-z0-9]*(?:[ -][A-Za-z0-9]+)*$')


def is_valid_concept(text: str) -> bool:
    """
    Check if a concept is valid according to filtering rules.

    Allowed:
    - Letters a-zA-Z
    - Digits 0-9 (but concept must start with a letter)
    - Spaces between tokens
    - Hyphen "-" between tokens (single hyphen only)

    Disallowed:
    - Leading/trailing hyphen or space (after stripping)
    - Multiple consecutive separators ("--", "  ", "- ", " -")
    - Any punctuation/symbols other than "-"
    - Non-ASCII letters (e.g., résumé)
    - Concepts that are empty or length <= 1 after stripping
    - Standalone numbers like "123" (must start with a letter)
    """
    if not text:
        return False

    stripped = text.strip()

    # Length must be > 1
    if len(stripped) <= 1:
        return False

    # Must match allowed pattern
    return _VALID_CONCEPT_RE.match(stripped) is not None


def normalize_concept(text: str) -> str:
    """
    Normalize a valid concept for deduplication.

    - Strip leading/trailing spaces
    - Collapse internal whitespace
    - Lowercase for case-insensitive matching

    Note: This does NOT rewrite hyphens.
          So "long-term" and "long term" remain distinct concepts.
          If you want them merged, replace '-' with ' ' before collapsing whitespace.
    """
    if not is_valid_concept(text):
        return ''

    text = text.strip()
    text = ' '.join(text.split())
    text = text.lower()
    return text


def extract_concepts_from_sequence(
    tokens: List[int],
    tokenizer,
    eos_token_id: int
) -> Tuple[List[str], List[str]]:
    """
    Extract valid and invalid concepts from a token sequence.
    """
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    lines = text.split('\n')

    has_eos = (len(tokens) > 0 and tokens[-1] == eos_token_id)

    # If no EOS and we have lines, last line might be incomplete - drop it
    if not has_eos and len(lines) > 1:
        lines = lines[:-1]

    valid = []
    invalid = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if is_valid_concept(stripped):
            valid.append(stripped)
        else:
            invalid.append(stripped)

    return valid, invalid


def track_concepts(
    valid_concepts: List[str],
    invalid_concepts: List[str],
    valid_set: Set[str],
    invalid_set: Set[str],
    valid_freq: dict,
    invalid_freq: dict
) -> Tuple[int, int]:
    """
    Update concept tracking with new concepts from a sequence.
    """
    new_valid = 0
    new_invalid = 0

    for concept in valid_concepts:
        norm = normalize_concept(concept)
        if norm:
            if norm not in valid_set:
                valid_set.add(norm)
                new_valid += 1
            valid_freq[norm] = valid_freq.get(norm, 0) + 1

    for concept in invalid_concepts:
        if concept not in invalid_set:
            invalid_set.add(concept)
            new_invalid += 1
        invalid_freq[concept] = invalid_freq.get(concept, 0) + 1

    return new_valid, new_invalid


if __name__ == "__main__":
    # Test filtering rules
    test_cases = [
        ("hello world", True),
        ("A", False),  # Too short
        ("ab", True),
        ("Hello World", True),

        # Numbers now allowed (with constraints)
        ("L2", True),
        ("Section 230", True),
        ("ResNet 50", True),
        ("Rule 10b", True),
        ("B2B", True),

        # Hyphens allowed between tokens
        ("hello-world", True),
        ("long-term memory", True),

        # Still disallowed
        ("123", False),  # Must start with a letter
        ("hello123", True),  # alphanumeric token ok
        ("hello world 123", True),  # token "123" is allowed here because it's after a separator AND token pattern allows it
                                  # If you want to disallow pure-digit tokens, tell me and I’ll tighten the regex.
        ("hello--world", False),  # double hyphen
        ("-hello", False),  # leading hyphen
        ("hello-", False),  # trailing hyphen
        ("hello-world!", False),  # punctuation
        ("hello_world", False),  # underscore
        ("  hello  ", True),
        ("   ", False),
        ("", False),
        ("h", False),
        ("résumé", False),  # Non-ASCII
    ]

    print("Testing concept validation:")
    for text, expected in test_cases:
        result = is_valid_concept(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text}' -> {result} (expected {expected})")

    print("\nTesting normalization:")
    for text, valid in test_cases:
        if valid:
            norm = normalize_concept(text)
            print(f"  '{text}' -> '{norm}'")
