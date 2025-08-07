#!/usr/bin/env python3
"""
Test script to verify the new choice parsing system works
"""

import logging
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_choice_parsing():
    """Test the new choice parsing system"""
    print("Testing choice parsing with mock responses:")
    print("=" * 50)

    try:
        from models import OpenAIModel

        # Create a mock model (won't actually call API)
        model = OpenAIModel("gpt-4o-mini")

        # Test cases with different response formats
        test_cases = [
            {
                "response": "I think the answer is yes",
                "choices": ["yes", "no"],
                "expected": "yes",
            },
            {
                "response": "The answer is A",
                "choices": ["even", "odd"],
                "expected": "even",
            },
            {
                "response": "My choice is 0",
                "choices": ["true", "false"],
                "expected": "true",
            },
            {
                "response": "definitely false",
                "choices": ["true", "false"],
                "expected": "false",
            },
        ]

        for i, test_case in enumerate(test_cases):
            try:
                result = model._fallback_parse_choice(
                    test_case["response"], test_case["choices"]
                )
                expected = test_case["expected"]
                status = "✓ PASS" if result == expected else "✗ FAIL"
                print(
                    f"Test {i+1}: '{test_case['response']}' -> {result} (expected {expected}) {status}"
                )
            except Exception as e:
                print(f"Test {i+1}: ERROR - {e}")

    except ImportError as e:
        print(f"Could not import models: {e}")
        print("This is expected if OpenAI package is not installed")

    print("\nDone!")


if __name__ == "__main__":
    test_choice_parsing()
