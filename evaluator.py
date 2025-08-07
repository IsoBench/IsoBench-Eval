"""
Main evaluator class for IsoBench dataset.

This module contains the main IsoBenchEvaluator class that coordinates
evaluation across all tasks and models, and handles result aggregation.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from data_structures import EvaluationResult, AggregatedResult
from models import BaseModel
from task_evaluators import (
    MathTaskEvaluator,
    ScienceTaskEvaluator,
    AlgorithmTaskEvaluator,
    GameTaskEvaluator,
    IsoBenchTaskEvaluator,
)

logger = logging.getLogger(__name__)


class IsoBenchEvaluator:
    """Main evaluator for IsoBench dataset"""

    # Task definitions
    MATH_TASKS = ["math_parity", "math_convexity", "math_breakpoint"]
    SCIENCE_TASKS = ["chemistry", "physics"]
    ALGORITHM_TASKS = ["graph_connectivity", "graph_maxflow", "graph_isomorphism"]
    GAME_TASKS = ["winner_id", "puzzle"]

    ALL_TASKS = MATH_TASKS + SCIENCE_TASKS + ALGORITHM_TASKS + GAME_TASKS

    def __init__(
        self, output_dir: str = "isobench_results", use_long_prompts: bool = False
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[EvaluationResult] = []
        self.use_long_prompts = use_long_prompts

        if use_long_prompts:
            logger.info("Using long prompts (paper appendix style)")
        else:
            logger.info("Using short prompts (default style)")

    def get_task_evaluator(self, task_name: str) -> IsoBenchTaskEvaluator:
        """Get appropriate evaluator for task"""
        if task_name in self.MATH_TASKS:
            return MathTaskEvaluator(task_name, use_long_prompts=self.use_long_prompts)
        elif task_name in self.SCIENCE_TASKS:
            return ScienceTaskEvaluator(
                task_name, use_long_prompts=self.use_long_prompts
            )
        elif task_name in self.ALGORITHM_TASKS:
            return AlgorithmTaskEvaluator(
                task_name, use_long_prompts=self.use_long_prompts
            )
        elif task_name in self.GAME_TASKS:
            return GameTaskEvaluator(task_name, use_long_prompts=self.use_long_prompts)
        else:
            raise ValueError(f"Unknown task: {task_name}")

    def evaluate_model(
        self,
        model: BaseModel,
        tasks: List[str] = None,
        modalities: List[str] = None,
        max_samples_per_task: int = None,
        use_long_prompts: bool = False,
    ) -> List[EvaluationResult]:
        """Evaluate model on specified tasks and modalities"""
        if tasks is None:
            tasks = self.ALL_TASKS

        if modalities is None:
            modalities = ["text", "image"]

        model_results = []

        for task in tasks:
            logger.info(f"Starting evaluation of {model.model_name} on task: {task}")

            try:
                evaluator = self.get_task_evaluator(task)

                for modality in modalities:
                    result = evaluator.evaluate_modality(
                        model,
                        modality,
                        max_samples_per_task,
                        output_dir=self.output_dir,
                    )
                    model_results.append(result)
                    self.results.append(result)

                    logger.info(
                        f"Task: {task}, Modality: {modality}, Accuracy: {result.accuracy:.3f}"
                    )

            except Exception as e:
                logger.error(f"Error evaluating {task}: {e}")

        return model_results

    def aggregate_results(self, model_name: str) -> AggregatedResult:
        """Aggregate results for a specific model"""
        model_results = [r for r in self.results if r.model_name == model_name]

        if not model_results:
            logger.warning(f"No results found for model: {model_name}")
            return AggregatedResult(model_name, 0.0, 0.0)

        # Separate image and text results
        image_results = [r for r in model_results if r.modality == "image"]
        text_results = [r for r in model_results if r.modality == "text"]

        # Calculate macro averages (average of task accuracies)
        image_accuracy = (
            np.mean([r.accuracy for r in image_results]) if image_results else 0.0
        )
        text_accuracy = (
            np.mean([r.accuracy for r in text_results]) if text_results else 0.0
        )

        # Create task-level breakdown
        task_results = {}
        for result in model_results:
            if result.task not in task_results:
                task_results[result.task] = {}
            task_results[result.task][result.modality] = result.accuracy

        return AggregatedResult(
            model_name=model_name,
            image_accuracy=image_accuracy,
            text_accuracy=text_accuracy,
            task_results=task_results,
        )

    def save_detailed_results(self, filename: str = "detailed_results.json"):
        """Save detailed results to JSON"""
        output_file = self.output_dir / filename

        results_data = []
        for result in self.results:
            results_data.append(
                {
                    "task": result.task,
                    "modality": result.modality,
                    "model_name": result.model_name,
                    "accuracy": result.accuracy,
                    "total_samples": result.total_samples,
                    "correct_samples": result.correct_samples,
                    "predictions": result.predictions,
                    "ground_truth": result.ground_truth,
                    "details": result.details,
                }
            )

        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Detailed results saved to: {output_file}")

    def generate_table1_report(self, models: List[str] = None) -> pd.DataFrame:
        """Generate Table 1 style report similar to the paper"""
        if models is None:
            models = list(set([r.model_name for r in self.results]))

        report_data = []

        for model in models:
            agg_result = self.aggregate_results(model)

            # Calculate image-text performance gap
            gap = agg_result.text_accuracy - agg_result.image_accuracy

            report_data.append(
                {
                    "Model": model,
                    "Text Accuracy": f"{agg_result.text_accuracy:.1%}",
                    "Image Accuracy": f"{agg_result.image_accuracy:.1%}",
                    "Gap (Text - Image)": f"{gap:.1%}",
                    "Gap (Points)": f"{gap*100:.1f}",
                }
            )

        df = pd.DataFrame(report_data)

        # Save report
        report_file = self.output_dir / "table1_report.csv"
        df.to_csv(report_file, index=False)
        logger.info(f"Table 1 report saved to: {report_file}")

        return df

    def create_evaluation_summary(self, model_name: str):
        """Create a summary of the detailed evaluation logs"""
        model_dir = self.output_dir / model_name.replace("/", "_")
        summary_file = model_dir / "evaluation_summary.json"

        if not model_dir.exists():
            logger.warning(f"No detailed logs found for model {model_name}")
            return

        # Collect summary statistics
        summary = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "task_summaries": {},
        }

        # Process each task log file
        for log_file in model_dir.glob("*.json"):
            task_name = log_file.stem

            # Read and analyze the log file
            total_samples = 0
            correct_samples = 0
            modality_stats = {}

            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    log_data = json.load(f)

                    # Process each entry in the JSON array
                    for entry in log_data:
                        total_samples += 1

                        if entry.get("evaluation", {}).get("is_correct", False):
                            correct_samples += 1

                        # Track modality stats
                        modality = entry.get("modality", "unknown")
                        if modality not in modality_stats:
                            modality_stats[modality] = {"total": 0, "correct": 0}
                        modality_stats[modality]["total"] += 1
                        if entry.get("evaluation", {}).get("is_correct", False):
                            modality_stats[modality]["correct"] += 1

                # Calculate accuracies
                overall_accuracy = (
                    correct_samples / total_samples if total_samples > 0 else 0
                )

                for modality in modality_stats:
                    total = modality_stats[modality]["total"]
                    correct = modality_stats[modality]["correct"]
                    modality_stats[modality]["accuracy"] = (
                        correct / total if total > 0 else 0
                    )

                summary["task_summaries"][task_name] = {
                    "total_samples": total_samples,
                    "correct_samples": correct_samples,
                    "overall_accuracy": overall_accuracy,
                    "modality_breakdown": modality_stats,
                    "log_file": str(log_file.name),
                }

            except Exception as e:
                logger.error(f"Error processing log file {log_file}: {e}")

        # Save summary
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Created evaluation summary: {summary_file}")
        return summary
