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

from .data_structures import EvaluationResult, AggregatedResult
from .models import BaseModel
from .task_evaluators import (
    MathTaskEvaluator,
    ScienceTaskEvaluator,
    AlgorithmTaskEvaluator,
    GameTaskEvaluator,
    IsoBenchTaskEvaluator,
)
from .radar_plots import RadarPlotGenerator

logger = logging.getLogger(__name__)


class IsoBenchEvaluator:
    """Main evaluator for IsoBench dataset"""

    # Task definitions
    MATH_TASKS = ["math_parity", "math_convexity", "math_breakpoint"]
    SCIENCE_TASKS = ["chemistry", "physics"]
    ALGORITHM_TASKS = ["graph_connectivity", "graph_maxflow", "graph_isomorphism"]
    # GAME_TASKS = ["winner_id", "puzzle"]
    GAME_TASKS = ["winner_id"]  # Temporarily disable "puzzle" task due to issues
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
        resume: bool = True,
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
                        resume=resume,
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
        """Generate enhanced Table 1 style report with macro-task breakdown"""
        if models is None:
            models = list(set([r.model_name for r in self.results]))

        report_data = []
        macro_task_names = ["Math", "Science", "Algorithm", "Game"]

        for model in models:
            # Load evaluation summary for more detailed data
            model_dir = self.output_dir / model.replace("/", "_")
            summary_file = model_dir / "evaluation_summary.json"

            if not summary_file.exists():
                # Fallback to aggregated results for overall row only
                agg_result = self.aggregate_results(model)
                gap = agg_result.text_accuracy - agg_result.image_accuracy

                report_data.append(
                    {
                        "Model": model,
                        "Task": "All",
                        "Text Accuracy": f"{agg_result.text_accuracy:.1%}",
                        "Image Accuracy": f"{agg_result.image_accuracy:.1%}",
                        "Gap (Text - Image)": f"{gap:.1%}",
                        "Gap (Points)": f"{gap*100:.1f}",
                        "Text Samples": 0,
                        "Text Correct": 0,
                        "Image Samples": 0,
                        "Image Correct": 0,
                    }
                )
                continue

            with open(summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)

            # Add macro-task rows
            macro_summaries = summary.get("macro_task_summaries", {})
            for macro_task in macro_task_names:
                if macro_task in macro_summaries:
                    data = macro_summaries[macro_task]
                    text_accuracy = data.get("text_accuracy", 0)
                    image_accuracy = data.get("image_accuracy", 0)
                    gap = text_accuracy - image_accuracy

                    report_data.append(
                        {
                            "Model": model,
                            "Task": macro_task,
                            "Text Accuracy": f"{text_accuracy:.1%}",
                            "Image Accuracy": f"{image_accuracy:.1%}",
                            "Gap (Text - Image)": f"{gap:.1%}",
                            "Gap (Points)": f"{gap*100:.1f}",
                            "Text Samples": data.get("total_text_samples", 0),
                            "Text Correct": data.get("correct_text_samples", 0),
                            "Image Samples": data.get("total_image_samples", 0),
                            "Image Correct": data.get("correct_image_samples", 0),
                        }
                    )

            # Add overall "All" row
            overall_summary = summary.get("overall_summary", {})
            text_accuracy = overall_summary.get("text_accuracy", 0)
            image_accuracy = overall_summary.get("image_accuracy", 0)
            gap = text_accuracy - image_accuracy

            report_data.append(
                {
                    "Model": model,
                    "Task": "All",
                    "Text Accuracy": f"{text_accuracy:.1%}",
                    "Image Accuracy": f"{image_accuracy:.1%}",
                    "Gap (Text - Image)": f"{gap:.1%}",
                    "Gap (Points)": f"{gap*100:.1f}",
                    "Text Samples": overall_summary.get("total_text_samples", 0),
                    "Text Correct": overall_summary.get("correct_text_samples", 0),
                    "Image Samples": overall_summary.get("total_image_samples", 0),
                    "Image Correct": overall_summary.get("correct_image_samples", 0),
                }
            )

        df = pd.DataFrame(report_data)

        # Reorder columns for better readability
        column_order = [
            "Model",
            "Task",
            "Text Accuracy",
            "Image Accuracy",
            "Gap (Text - Image)",
            "Gap (Points)",
            "Text Samples",
            "Text Correct",
            "Image Samples",
            "Image Correct",
        ]
        df = df[column_order]

        # Save report to model-specific directory for single model evaluations
        if len(models) == 1:
            model_dir = self.output_dir / models[0].replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)
            report_file = model_dir / "individual_report.csv"
        else:
            # Save to main output directory for multi-model reports
            report_file = self.output_dir / "table1_comprehensive_report.csv"

        df.to_csv(report_file, index=False)
        logger.info(f"Enhanced Table 1 report saved to: {report_file}")

        return df

    def generate_radar_plots(self, model_names: List[str] = None):
        """Generate radar plots for model evaluation results"""
        if model_names is None:
            model_names = list(set([r.model_name for r in self.results]))

        logger.info("Generating radar plots for evaluation results...")

        try:
            radar_generator = RadarPlotGenerator(str(self.output_dir))
            radar_generator.generate_all_plots(model_names)
            logger.info("Radar plots generated successfully")
        except Exception as e:
            logger.error(f"Error generating radar plots: {e}")
            logger.info("Continuing without radar plots...")

    def create_evaluation_summary(self, model_name: str):
        """Create a comprehensive summary of the detailed evaluation logs"""
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
            "macro_task_summaries": {},
            "overall_summary": {},
        }

        # Process each task log file
        for log_file in model_dir.glob("*.json"):
            task_name = log_file.stem
            if task_name == "evaluation_summary":
                continue

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

                # Calculate gap between text and image
                text_acc = modality_stats.get("text", {}).get("accuracy", 0)
                image_acc = modality_stats.get("image", {}).get("accuracy", 0)
                gap = text_acc - image_acc

                summary["task_summaries"][task_name] = {
                    "total_samples": total_samples,
                    "correct_samples": correct_samples,
                    "overall_accuracy": overall_accuracy,
                    "modality_breakdown": modality_stats,
                    "gap": gap,
                    "log_file": str(log_file.name),
                }

            except Exception as e:
                logger.error(f"Error processing log file {log_file}: {e}")

        # Calculate macro-task summaries
        macro_task_groups = {
            "Math": self.MATH_TASKS,
            "Science": self.SCIENCE_TASKS,
            "Algorithm": self.ALGORITHM_TASKS,
            "Game": self.GAME_TASKS,
        }

        for macro_task, tasks in macro_task_groups.items():
            macro_stats = {
                "text": {"total": 0, "correct": 0},
                "image": {"total": 0, "correct": 0},
            }

            for task in tasks:
                if task in summary["task_summaries"]:
                    task_data = summary["task_summaries"][task]
                    for modality in ["text", "image"]:
                        if modality in task_data["modality_breakdown"]:
                            macro_stats[modality]["total"] += task_data[
                                "modality_breakdown"
                            ][modality]["total"]
                            macro_stats[modality]["correct"] += task_data[
                                "modality_breakdown"
                            ][modality]["correct"]

            # Calculate macro accuracies
            text_accuracy = (
                macro_stats["text"]["correct"] / macro_stats["text"]["total"]
                if macro_stats["text"]["total"] > 0
                else 0
            )
            image_accuracy = (
                macro_stats["image"]["correct"] / macro_stats["image"]["total"]
                if macro_stats["image"]["total"] > 0
                else 0
            )
            gap = text_accuracy - image_accuracy

            summary["macro_task_summaries"][macro_task] = {
                "text_accuracy": text_accuracy,
                "image_accuracy": image_accuracy,
                "gap": gap,
                "total_text_samples": macro_stats["text"]["total"],
                "total_image_samples": macro_stats["image"]["total"],
                "correct_text_samples": macro_stats["text"]["correct"],
                "correct_image_samples": macro_stats["image"]["correct"],
            }

        # Calculate overall summary
        total_text_stats = {"total": 0, "correct": 0}
        total_image_stats = {"total": 0, "correct": 0}

        for task_data in summary["task_summaries"].values():
            for modality in ["text", "image"]:
                if modality in task_data["modality_breakdown"]:
                    if modality == "text":
                        total_text_stats["total"] += task_data["modality_breakdown"][
                            modality
                        ]["total"]
                        total_text_stats["correct"] += task_data["modality_breakdown"][
                            modality
                        ]["correct"]
                    else:
                        total_image_stats["total"] += task_data["modality_breakdown"][
                            modality
                        ]["total"]
                        total_image_stats["correct"] += task_data["modality_breakdown"][
                            modality
                        ]["correct"]

        overall_text_accuracy = (
            total_text_stats["correct"] / total_text_stats["total"]
            if total_text_stats["total"] > 0
            else 0
        )
        overall_image_accuracy = (
            total_image_stats["correct"] / total_image_stats["total"]
            if total_image_stats["total"] > 0
            else 0
        )
        overall_gap = overall_text_accuracy - overall_image_accuracy

        summary["overall_summary"] = {
            "text_accuracy": overall_text_accuracy,
            "image_accuracy": overall_image_accuracy,
            "gap": overall_gap,
            "total_text_samples": total_text_stats["total"],
            "total_image_samples": total_image_stats["total"],
            "correct_text_samples": total_text_stats["correct"],
            "correct_image_samples": total_image_stats["correct"],
        }

        # Save summary
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Created enhanced evaluation summary: {summary_file}")
        return summary
