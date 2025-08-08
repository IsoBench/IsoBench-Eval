"""
Radar plot visualization for IsoBench evaluation results.

This module creates radar plots showing model performance across different
tasks and modalities.
"""

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.patches as patches

plt.rcParams["font.family"] = "serif"
logger = logging.getLogger(__name__)


class RadarPlotGenerator:
    """Generates radar plots for IsoBench evaluation results"""

    def __init__(self, output_dir: str = "isobench_results"):
        self.output_dir = Path(output_dir)

        # Task categories for macro-level plots
        self.MACRO_TASKS = {
            "Math": ["math_parity", "math_convexity", "math_breakpoint"],
            "Science": ["chemistry", "physics"],
            "Algorithm": ["graph_connectivity", "graph_maxflow", "graph_isomorphism"],
            # "Game": ["winner_id", "puzzle"],
            "Game": ["winner_id"],  # Temporarily disable "puzzle" task due to issues
        }

        # All tasks for detailed plots
        self.ALL_TASKS = [
            "math_parity",
            "math_convexity",
            "math_breakpoint",
            "chemistry",
            "physics",
            "graph_connectivity",
            "graph_maxflow",
            "graph_isomorphism",
            "winner_id",
            # "puzzle",  # Temporarily disabled
        ]

    def load_model_summary(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load evaluation summary for a specific model"""
        model_dir = self.output_dir / model_name.replace("/", "_")
        summary_file = model_dir / "evaluation_summary.json"

        if not summary_file.exists():
            logger.warning(f"No evaluation summary found for model {model_name}")
            return None

        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading summary for {model_name}: {e}")
            return None

    def _format_model_name(self, model_name: str) -> str:
        """Format model name with proper capitalization (e.g., GPT-4, GPT-5)"""
        # Replace 'gpt' with 'GPT' (case insensitive)
        import re

        formatted_name = re.sub(r"\bgpt\b", "GPT", model_name, flags=re.IGNORECASE)
        return formatted_name

    def create_radar_plot(self, model_name: str, plot_type: str = "detailed"):
        """
        Create radar plots for a model

        Args:
            model_name: Name of the model
            plot_type: "detailed" for individual tasks, "macro" for macro-task groups
        """
        summary = self.load_model_summary(model_name)
        if not summary:
            return

        if plot_type == "detailed":
            self._create_detailed_radar_plot(model_name, summary)
        elif plot_type == "macro":
            self._create_macro_radar_plot(model_name, summary)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")

    def _create_detailed_radar_plot(self, model_name: str, summary: Dict[str, Any]):
        """Create radar plot with individual task accuracies"""
        task_summaries = summary.get("task_summaries", {})

        # Extract data for all tasks
        text_accuracies = []
        image_accuracies = []
        task_labels = []

        for task in self.ALL_TASKS:
            if task in task_summaries:
                modality_breakdown = task_summaries[task].get("modality_breakdown", {})
                text_acc = modality_breakdown.get("text", {}).get("accuracy", 0) * 100
                image_acc = modality_breakdown.get("image", {}).get("accuracy", 0) * 100
            else:
                text_acc = 0
                image_acc = 0

            text_accuracies.append(text_acc)
            image_accuracies.append(image_acc)
            task_labels.append(task.replace("_", " ").title().replace(" ", "\n"))

        # Create the radar plot
        self._plot_radar(
            task_labels,
            text_accuracies,
            image_accuracies,
            title=f"{self._format_model_name(model_name)} - IsoBench Performance Detailed",
            filename=f"{model_name.replace('/', '_')}_detailed_radar.png",
        )

    def _create_macro_radar_plot(self, model_name: str, summary: Dict[str, Any]):
        """Create radar plot with macro-task group accuracies"""
        macro_summaries = summary.get("macro_task_summaries", {})

        # Extract data for macro tasks
        text_accuracies = []
        image_accuracies = []
        macro_labels = []

        for macro_task in ["Math", "Science", "Algorithm", "Game"]:
            if macro_task in macro_summaries:
                text_acc = macro_summaries[macro_task].get("text_accuracy", 0) * 100
                image_acc = macro_summaries[macro_task].get("image_accuracy", 0) * 100
            else:
                text_acc = 0
                image_acc = 0

            text_accuracies.append(text_acc)
            image_accuracies.append(image_acc)
            macro_labels.append(macro_task)

        # Create the radar plot
        self._plot_radar(
            macro_labels,
            text_accuracies,
            image_accuracies,
            title=f"{self._format_model_name(model_name)} - IsoBench Performance",
            filename=f"{model_name.replace('/', '_')}_macro_radar.png",
        )

    def _plot_radar(
        self,
        labels: List[str],
        text_data: List[float],
        image_data: List[float],
        title: str,
        filename: str,
    ):
        """Create and save a radar plot"""
        # Number of variables
        num_vars = len(labels)

        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Complete the circle

        # Add first value to end to close the radar chart
        text_data += text_data[:1]
        image_data += image_data[:1]

        # Create the figure and polar subplot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        # Plot text modality (red)
        ax.plot(
            angles,
            text_data,
            "o-",
            linewidth=2,
            label="Text Modality",
            color="red",
            alpha=0.8,
        )
        ax.fill(angles, text_data, alpha=0.1, color="red")

        # Plot image modality (blue)
        ax.plot(
            angles,
            image_data,
            "o-",
            linewidth=2,
            label="Image Modality",
            color="blue",
            alpha=0.8,
        )
        ax.fill(angles, image_data, alpha=0.1, color="blue")

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=18, fontweight="bold")

        # Move labels further from the circle to avoid overlap
        ax.tick_params(axis="x", pad=30)

        # Set y-axis limits (0-100 for percentages)
        ax.set_ylim(0, 100)
        # ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=15)
        ax.grid(True, alpha=0.3)

        # Add title and legend
        plt.title(title, size=18, fontweight="bold", pad=20)
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=15)

        # Save the plot
        output_file = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.01)
        plt.close()

        logger.info(f"Radar plot saved to: {output_file}")

    def create_comparison_radar_plot(
        self, model_names: List[str], plot_type: str = "macro"
    ):
        """
        Create a comparison radar plot showing multiple models

        Args:
            model_names: List of model names to compare
            plot_type: "detailed" or "macro"
        """
        if len(model_names) > 4:
            logger.warning("Too many models for comparison plot. Using first 4 models.")
            model_names = model_names[:4]

        # Colors for different models
        colors = ["red", "blue", "green", "orange"]

        # Load summaries for all models
        summaries = {}
        for model_name in model_names:
            summary = self.load_model_summary(model_name)
            if summary:
                summaries[model_name] = summary

        if not summaries:
            logger.error("No valid model summaries found for comparison")
            return

        if plot_type == "macro":
            labels = ["Math", "Science", "Algorithm", "Game"]
            data_key = "macro_task_summaries"
            filename = "models_macro_comparison_radar.png"
            title = "Model Comparison - IsoBench Performance (Text Modality)"
        else:
            labels = [
                task.replace("_", " ").title().replace(" ", "\n")
                for task in self.ALL_TASKS
            ]
            data_key = "task_summaries"
            filename = "models_detailed_comparison_radar.png"
            title = "Model Comparison - IsoBench Performance Detailed (Text Modality)"

        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection="polar"))

        # Set font family to Sans Serif
        plt.rcParams["font.family"] = "serif"

        # Number of variables
        num_vars = len(labels)
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]

        # Plot each model
        for i, (model_name, summary) in enumerate(summaries.items()):
            color = colors[i % len(colors)]

            if plot_type == "macro":
                macro_summaries = summary.get(data_key, {})
                accuracies = []
                for macro_task in labels:
                    if macro_task in macro_summaries:
                        acc = macro_summaries[macro_task].get("text_accuracy", 0) * 100
                    else:
                        acc = 0
                    accuracies.append(acc)
            else:
                task_summaries = summary.get(data_key, {})
                accuracies = []
                for task in self.ALL_TASKS:
                    if task in task_summaries:
                        modality_breakdown = task_summaries[task].get(
                            "modality_breakdown", {}
                        )
                        acc = (
                            modality_breakdown.get("text", {}).get("accuracy", 0) * 100
                        )
                    else:
                        acc = 0
                    accuracies.append(acc)

            # Close the plot
            accuracies += accuracies[:1]

            # Plot the line
            ax.plot(
                angles,
                accuracies,
                "o-",
                linewidth=2,
                label=self._format_model_name(model_name),
                color=color,
                alpha=0.8,
            )
            ax.fill(angles, accuracies, alpha=0.1, color=color)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=18, fontweight="bold")

        # Move labels further from the circle to avoid overlap
        # ax.tick_params(axis="both", pad=0)

        # Set y-axis
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=15)
        ax.grid(True, alpha=0.3)

        # Add title and legend
        plt.title(title, size=18, fontweight="bold", pad=20)
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=15)

        # Save the plot
        output_file = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.01)
        plt.close()

        logger.info(f"Comparison radar plot saved to: {output_file}")

    def generate_all_plots(self, model_names: List[str]):
        """Generate all radar plots for given models"""
        for model_name in model_names:
            logger.info(f"Generating radar plots for {model_name}")
            self.create_radar_plot(model_name, "detailed")
            self.create_radar_plot(model_name, "macro")

        # Create comparison plots if multiple models
        if len(model_names) > 1:
            logger.info("Generating comparison radar plots")
            self.create_comparison_radar_plot(model_names, "detailed")
            self.create_comparison_radar_plot(model_names, "macro")
