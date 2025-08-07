#!/usr/bin/env python3
"""
Aggregate Multiple Models Results into Table 1
==============================================

This script aggregates evaluation results from multiple models that were evaluated
separately and creates a comprehensive Table 1 report similar to the original paper.

Usage:
    python aggregate_results.py --output-dir isobench_results
    python aggregate_results.py --output-dir isobench_results --models gpt-4 gpt-3.5-turbo
    python aggregate_results.py --output-dir isobench_results --include-task-breakdown

Author: Deqing Fu
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ResultsAggregator:
    """Aggregates results from multiple model evaluations"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            raise ValueError(f"Output directory does not exist: {output_dir}")

    def discover_models(self) -> List[str]:
        """Discover all models that have evaluation results"""
        models = []
        for model_dir in self.output_dir.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith("."):
                # Check if this directory has evaluation results
                if (
                    any(model_dir.glob("*.json"))
                    or (model_dir / "evaluation_summary.json").exists()
                ):
                    models.append(model_dir.name)

        logger.info(f"Discovered {len(models)} models with results: {models}")
        return sorted(models)

    def load_model_summary(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load evaluation summary for a specific model"""
        model_dir = self.output_dir / model_name
        summary_file = model_dir / "evaluation_summary.json"

        if not summary_file.exists():
            logger.warning(f"No evaluation summary found for model {model_name}")
            logger.info(
                f"Try running: python eval.py --model {model_name} to generate summary"
            )
            return None

        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading summary for {model_name}: {e}")
            return None

    def calculate_model_accuracy(self, summary: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall text and image accuracy for a model"""
        text_stats = {"total": 0, "correct": 0}
        image_stats = {"total": 0, "correct": 0}

        for task_name, task_summary in summary.get("task_summaries", {}).items():
            modality_breakdown = task_summary.get("modality_breakdown", {})

            for modality, stats in modality_breakdown.items():
                if modality == "text":
                    text_stats["total"] += stats.get("total", 0)
                    text_stats["correct"] += stats.get("correct", 0)
                elif modality == "image":
                    image_stats["total"] += stats.get("total", 0)
                    image_stats["correct"] += stats.get("correct", 0)

        text_accuracy = (
            text_stats["correct"] / text_stats["total"]
            if text_stats["total"] > 0
            else 0.0
        )
        image_accuracy = (
            image_stats["correct"] / image_stats["total"]
            if image_stats["total"] > 0
            else 0.0
        )

        return {
            "text_accuracy": text_accuracy,
            "image_accuracy": image_accuracy,
            "text_total": text_stats["total"],
            "text_correct": text_stats["correct"],
            "image_total": image_stats["total"],
            "image_correct": image_stats["correct"],
        }

    def generate_table1_report(
        self, models: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate comprehensive Table 1 report for multiple models"""
        if models is None:
            models = self.discover_models()

        if not models:
            raise ValueError("No models found with evaluation results")

        report_data = []

        for model_name in models:
            logger.info(f"Processing results for model: {model_name}")
            summary = self.load_model_summary(model_name)

            if summary is None:
                logger.warning(f"Skipping model {model_name} due to missing summary")
                continue

            accuracies = self.calculate_model_accuracy(summary)

            # Calculate performance gap
            gap = accuracies["text_accuracy"] - accuracies["image_accuracy"]

            report_data.append(
                {
                    "Model": model_name,
                    "Text Accuracy": f"{accuracies['text_accuracy']:.1%}",
                    "Image Accuracy": f"{accuracies['image_accuracy']:.1%}",
                    "Gap (Text - Image)": f"{gap:.1%}",
                    "Gap (Points)": f"{gap*100:.1f}",
                    "Text Samples": accuracies["text_total"],
                    "Text Correct": accuracies["text_correct"],
                    "Image Samples": accuracies["image_total"],
                    "Image Correct": accuracies["image_correct"],
                }
            )

        df = pd.DataFrame(report_data)

        # Save the comprehensive report
        output_file = self.output_dir / "table1_comprehensive_report.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Comprehensive Table 1 report saved to: {output_file}")

        # Also save a simplified version (just the main columns)
        simple_df = df[
            [
                "Model",
                "Text Accuracy",
                "Image Accuracy",
                "Gap (Text - Image)",
                "Gap (Points)",
            ]
        ]
        simple_output = self.output_dir / "table1_report.csv"
        simple_df.to_csv(simple_output, index=False)
        logger.info(f"Simplified Table 1 report saved to: {simple_output}")

        return df

    def generate_task_breakdown(
        self, models: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate detailed task-by-task breakdown report"""
        if models is None:
            models = self.discover_models()

        # Collect all unique tasks
        all_tasks = set()
        model_summaries = {}

        for model_name in models:
            summary = self.load_model_summary(model_name)
            if summary:
                model_summaries[model_name] = summary
                all_tasks.update(summary.get("task_summaries", {}).keys())

        all_tasks = sorted(all_tasks)

        # Create breakdown data
        breakdown_data = []

        for model_name in models:
            if model_name not in model_summaries:
                continue

            summary = model_summaries[model_name]
            task_summaries = summary.get("task_summaries", {})

            for task in all_tasks:
                if task in task_summaries:
                    task_data = task_summaries[task]
                    modality_breakdown = task_data.get("modality_breakdown", {})

                    text_acc = modality_breakdown.get("text", {}).get("accuracy", 0.0)
                    image_acc = modality_breakdown.get("image", {}).get("accuracy", 0.0)

                    breakdown_data.append(
                        {
                            "Model": model_name,
                            "Task": task,
                            "Text Accuracy": f"{text_acc:.1%}",
                            "Image Accuracy": f"{image_acc:.1%}",
                            "Gap": f"{(text_acc - image_acc):.1%}",
                            "Total Samples": task_data.get("total_samples", 0),
                            "Correct Samples": task_data.get("correct_samples", 0),
                            "Overall Accuracy": f"{task_data.get('overall_accuracy', 0.0):.1%}",
                        }
                    )
                else:
                    # Task not found for this model
                    breakdown_data.append(
                        {
                            "Model": model_name,
                            "Task": task,
                            "Text Accuracy": "N/A",
                            "Image Accuracy": "N/A",
                            "Gap": "N/A",
                            "Total Samples": 0,
                            "Correct Samples": 0,
                            "Overall Accuracy": "N/A",
                        }
                    )

        df = pd.DataFrame(breakdown_data)

        # Save task breakdown report
        output_file = self.output_dir / "task_breakdown_report.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Task breakdown report saved to: {output_file}")

        return df


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Aggregate IsoBench evaluation results from multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate all models in the default output directory
  python aggregate_results.py
  
  # Aggregate specific models
  python aggregate_results.py --models gpt-4 gpt-3.5-turbo gemini-1.5-pro
  
  # Use custom output directory and include task breakdown
  python aggregate_results.py --output-dir my_results --include-task-breakdown
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="isobench_results",
        help="Output directory containing model results (default: isobench_results)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to include (default: all discovered models)",
    )

    parser.add_argument(
        "--include-task-breakdown",
        action="store_true",
        help="Generate detailed task-by-task breakdown report",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def main():
    """Main aggregation function"""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=== IsoBench Results Aggregator ===")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Models: {args.models or 'auto-discover'}")
    logger.info(f"Include task breakdown: {args.include_task_breakdown}")

    try:
        aggregator = ResultsAggregator(args.output_dir)

        # Generate main Table 1 report
        logger.info("Generating Table 1 comprehensive report...")
        df = aggregator.generate_table1_report(args.models)

        print("\n=== Table 1 Report ===")
        print(
            df[
                [
                    "Model",
                    "Text Accuracy",
                    "Image Accuracy",
                    "Gap (Text - Image)",
                    "Gap (Points)",
                ]
            ].to_string(index=False)
        )

        # Generate task breakdown if requested
        if args.include_task_breakdown:
            logger.info("Generating task breakdown report...")
            breakdown_df = aggregator.generate_task_breakdown(args.models)

            print(f"\n=== Task Breakdown Summary ===")
            print(
                f"Generated breakdown for {len(breakdown_df)} model-task combinations"
            )

        logger.info("Aggregation completed successfully!")

    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
