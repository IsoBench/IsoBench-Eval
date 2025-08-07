#!/usr/bin/env python3
"""
Test script to verify the comprehensive logging functionality.
"""

import logging
import tempfile
from pathlib import Path

from models import create_model
from evaluator import IsoBenchEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_logging():
    """Test the logging functionality with a small evaluation"""

    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        logger.info(f"Testing logging with output directory: {output_dir}")

        try:
            # Create model (using OpenAI as default test model)
            model_name = "gpt-3.5-turbo"
            logger.info(f"Testing with model: {model_name}")

            # Create evaluator with logging enabled
            evaluator = IsoBenchEvaluator(
                output_dir=output_dir,
                use_long_prompts=False,  # Use short prompts for faster testing
            )

            # Run a limited evaluation
            # We'll limit to just 1-2 samples per task to test quickly
            logger.info("Starting limited evaluation for logging test...")

            # Override max_samples for testing
            original_evaluate = evaluator.evaluate_model

            def limited_evaluate(model_name, use_long_prompts=False):
                # Just evaluate one task with few samples
                from task_evaluators import IsoBenchTaskEvaluator

                # Test with math_geometry task and only 2 samples
                task = "math_geometry"
                logger.info(f"Testing logging with task: {task}")

                task_evaluator = evaluator.get_task_evaluator(
                    task, use_long_prompts=use_long_prompts
                )

                # Evaluate just text modality with 2 samples
                result = task_evaluator.evaluate_modality(
                    model_name=model_name,
                    modality="text",
                    max_samples=2,  # Just 2 samples for testing
                    output_dir=output_dir,
                )

                evaluator.results.append(result)
                logger.info(f"Test evaluation completed for {task}")

            # Run limited evaluation
            limited_evaluate(model_name)

            # Create evaluation summary
            evaluator.create_evaluation_summary(model_name)

            # Check if log files were created
            model_dir = output_dir / model_name.replace("/", "_")
            log_files = list(model_dir.glob("*.jsonl"))
            summary_file = model_dir / "evaluation_summary.json"

            logger.info("=== Logging Test Results ===")
            logger.info(f"Model directory created: {model_dir.exists()}")
            logger.info(f"Log files found: {len(log_files)}")
            for log_file in log_files:
                logger.info(f"  - {log_file.name}")
            logger.info(f"Summary file created: {summary_file.exists()}")

            # Read and show sample log content
            if log_files:
                sample_log = log_files[0]
                logger.info(f"\n=== Sample content from {sample_log.name} ===")
                with open(sample_log, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    logger.info(f"Total log entries: {len(lines)}")
                    if lines:
                        import json

                        sample_entry = json.loads(lines[0])
                        logger.info("Sample log entry structure:")
                        for key in sample_entry.keys():
                            logger.info(f"  - {key}")

            # Show summary content
            if summary_file.exists():
                logger.info(f"\n=== Summary content ===")
                import json

                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                    logger.info(f"Model: {summary['model_name']}")
                    logger.info(f"Tasks summarized: {len(summary['task_summaries'])}")
                    for task, stats in summary["task_summaries"].items():
                        logger.info(
                            f"  {task}: {stats['total_samples']} samples, {stats['overall_accuracy']:.1%} accuracy"
                        )

            logger.info("\n✅ Logging test completed successfully!")

        except Exception as e:
            logger.error(f"❌ Logging test failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    test_logging()
