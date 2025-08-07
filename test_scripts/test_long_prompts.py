#!/usr/bin/env python3
"""
Test script to verify long prompts feature
"""

import logging
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from task_evaluators import MathTaskEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_long_prompts():
    """Test long vs short prompts"""
    print("Testing Long Prompts Feature:")
    print("=" * 60)

    # Test sample
    sample = {
        "latex": "x^2 + 3x + 1",
        "question": "Determine if this function is even, odd, or neither.",
    }

    # Test short prompts
    print("\n--- SHORT PROMPTS ---")
    evaluator_short = MathTaskEvaluator("math_parity", use_long_prompts=False)
    short_text = evaluator_short._create_short_text_prompt(sample)
    short_image = evaluator_short._create_short_image_prompt(sample)

    print("Short text prompt:")
    print(short_text)
    print("\nShort image prompt:")
    print(short_image)

    # Test long prompts
    print("\n--- LONG PROMPTS ---")
    evaluator_long = MathTaskEvaluator("math_parity", use_long_prompts=True)
    long_text = evaluator_long._create_long_text_prompt(sample)
    long_image = evaluator_long._create_long_image_prompt(sample)

    print("Long text prompt:")
    print(long_text)
    print("\nLong image prompt:")
    print(long_image)

    # Test using the main create methods
    print("\n--- INTEGRATION TEST ---")
    main_short = evaluator_short.create_text_prompt(sample)
    main_long = evaluator_long.create_text_prompt(sample)

    print("Main method (short):", len(main_short), "characters")
    print("Main method (long):", len(main_long), "characters")

    assert len(main_long) > len(
        main_short
    ), "Long prompt should be longer than short prompt"
    print("✓ Long prompts are working correctly!")


if __name__ == "__main__":
    test_long_prompts()
