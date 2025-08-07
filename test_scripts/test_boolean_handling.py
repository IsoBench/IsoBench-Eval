#!/usr/bin/env python3
"""
Test script to verify boolean ground truth handling
"""

import logging
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.task_evaluators import AlgorithmTaskEvaluator

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_boolean_conversion():
    """Test boolean to choice index conversion"""
    evaluator = AlgorithmTaskEvaluator("graph_connectivity")

    # Test cases
    test_cases = [
        {"label": True, "choices": ["yes", "no"], "expected": 0},
        {"label": False, "choices": ["yes", "no"], "expected": 1},
        {"label": "true", "choices": ["yes", "no"], "expected": 0},
        {"label": "false", "choices": ["yes", "no"], "expected": 1},
        {"label": "TRUE", "choices": ["yes", "no"], "expected": 0},
        {"label": "FALSE", "choices": ["yes", "no"], "expected": 1},
    ]

    print("Testing boolean ground truth conversion:")
    print("=" * 50)

    for i, test_case in enumerate(test_cases):
        sample = {"label": test_case["label"]}
        choices = test_case["choices"]
        expected = test_case["expected"]

        result = evaluator.process_ground_truth(sample, choices)

        status = "✓ PASS" if result == expected else "✗ FAIL"
        print(
            f"Test {i+1}: {test_case['label']} -> {result} (expected {expected}) {status}"
        )

    print("\nDone!")


if __name__ == "__main__":
    test_boolean_conversion()
