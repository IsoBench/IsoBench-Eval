#!/usr/bin/env python3
"""
Test script for the resume functionality
"""

import logging
import tempfile
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_resume_functionality():
    """Test the resume functionality"""

    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        logger.info(f"Testing resume functionality with output directory: {output_dir}")

        try:
            from models import OpenAIModel
            from evaluator import IsoBenchEvaluator

            # Create model (using gpt-3.5-turbo for faster testing)
            model = OpenAIModel("gpt-3.5-turbo")

            # Create evaluator
            evaluator = IsoBenchEvaluator(output_dir=output_dir)

            # Run a very limited evaluation (2 samples, text only, one task)
            logger.info("=== First run (should evaluate samples) ===")
            evaluator.evaluate_model(
                model,
                tasks=["math_parity"],
                modalities=["text"],
                max_samples_per_task=2,
                resume=True,
            )

            # Check if cache files were created
            model_dir = output_dir / "gpt-3.5-turbo"
            cache_file = model_dir / "math_parity.json"

            logger.info(f"Cache file exists: {cache_file.exists()}")
            if cache_file.exists():
                import json

                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
                logger.info(f"Cached entries: {len(cached_data)}")

            # Clear results and run again (should use cache)
            evaluator.results = []
            logger.info("\n=== Second run (should use cache) ===")
            evaluator.evaluate_model(
                model,
                tasks=["math_parity"],
                modalities=["text"],
                max_samples_per_task=2,
                resume=True,
            )

            # Clear results and run with fresh start
            evaluator.results = []
            logger.info("\n=== Third run with fresh start (should re-evaluate) ===")
            evaluator.evaluate_model(
                model,
                tasks=["math_parity"],
                modalities=["text"],
                max_samples_per_task=2,
                resume=False,  # Fresh start
            )

            logger.info("\n✅ Resume functionality test completed!")

        except Exception as e:
            logger.error(f"❌ Resume functionality test failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    test_resume_functionality()
