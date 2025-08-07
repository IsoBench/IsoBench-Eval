#!/usr/bin/env python3
"""
Test script to verify choice extraction from dataset and value-based parsing
"""

import logging
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from task_evaluators import AlgorithmTaskEvaluator

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_choice_extraction():
    """Test choice extraction from dataset"""
    print("Testing choice extraction from dataset:")
    print("=" * 50)

    # Create a mock evaluator with sample data
    evaluator = AlgorithmTaskEvaluator("graph_connectivity")

    # Mock dataset with boolean values
    mock_samples = [
        {"label": True},
        {"label": False},
        {"label": "true"},
        {"label": "false"},
    ]

    evaluator.dataset = mock_samples

    # Extract choices
    choices = evaluator.extract_choices_from_dataset()
    print(f"Extracted choices: {choices}")

    # Test ground truth processing
    print("\nTesting ground truth processing:")
    for i, sample in enumerate(mock_samples):
        gt = evaluator.process_ground_truth(sample, choices)
        print(f"Sample {i+1}: {sample['label']} -> {gt}")

    print("\nDone!")


if __name__ == "__main__":
    test_choice_extraction()
