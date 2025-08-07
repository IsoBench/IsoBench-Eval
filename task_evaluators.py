"""
Task evaluators for IsoBench dataset.

This module contains specialized evaluators for different categories of tasks
including mathematics, science, algorithms, and games.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datasets import load_dataset
from PIL import Image

from data_structures import EvaluationResult
from models import BaseModel

logger = logging.getLogger(__name__)


class IsoBenchTaskEvaluator:
    """Base evaluator for IsoBench tasks"""

    def __init__(self, task_name: str, use_long_prompts: bool = False):
        self.task_name = task_name
        self.dataset = None
        self.use_long_prompts = use_long_prompts
        self._load_dataset()

    def _load_dataset(self):
        """Load the specific task dataset"""
        try:
            self.dataset = load_dataset(
                "isobench/IsoBench", self.task_name, split="validation"
            )
            logger.info(
                f"Loaded {len(self.dataset)} samples for task: {self.task_name}"
            )

            # Extract choices from dataset
            self.extract_choices_from_dataset()
            if hasattr(self, "_dataset_choices"):
                logger.info(f"Extracted choices from dataset: {self._dataset_choices}")

        except Exception as e:
            logger.error(f"Failed to load dataset for task {self.task_name}: {e}")

    def create_text_prompt(self, sample: Dict[str, Any]) -> str:
        """Create text prompt based on task type"""
        if self.use_long_prompts:
            return self._create_long_text_prompt(sample)
        else:
            return self._create_short_text_prompt(sample)

    def create_image_prompt(self, sample: Dict[str, Any]) -> str:
        """Create prompt for image-based evaluation"""
        if self.use_long_prompts:
            return self._create_long_image_prompt(sample)
        else:
            return self._create_short_image_prompt(sample)

    def _create_short_text_prompt(self, sample: Dict[str, Any]) -> str:
        """Create short text prompt - original implementation"""
        if "question" in sample:
            return sample["question"]
        elif "latex" in sample:
            return (
                f"Given the function: {sample['latex']}, answer the following question."
            )
        elif "code" in sample:
            return (
                f"Given the function: {sample['code']}, answer the following question."
            )
        else:
            return "Please analyze the given input and provide your answer."

    def _create_short_image_prompt(self, sample: Dict[str, Any]) -> str:
        """Create short image prompt - original implementation"""
        if "question" in sample:
            return sample["question"]
        else:
            return "Please analyze the image and provide your answer."

    def _create_long_text_prompt(self, sample: Dict[str, Any]) -> str:
        """Create long text prompt - paper appendix style"""
        # Base implementation - override in specific task evaluators
        if "question" in sample:
            return sample["question"]
        return "Please carefully analyze the given input and provide your answer."

    def _create_long_image_prompt(self, sample: Dict[str, Any]) -> str:
        """Create long image prompt - paper appendix style"""
        # Base implementation - override in specific task evaluators
        if "question" in sample:
            return sample["question"]
        return "Please carefully analyze the image and provide your answer."

    def _save_detailed_log(
        self,
        model_name: str,
        sample_idx: int,
        sample: Dict[str, Any],
        modality: str,
        prompt: str,
        choices: Optional[List[str]],
        model_response: str,
        parsed_prediction: Union[str, int],
        ground_truth: Union[str, int],
        is_correct: bool,
        output_dir: Path,
    ):
        """Save detailed evaluation log for a single sample"""

        # Create directory structure: output_dir/model_name/
        model_dir = output_dir / model_name.replace(
            "/", "_"
        )  # Handle model names with slashes
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create detailed log entry
        log_entry = {
            "sample_index": sample_idx,
            "task_name": self.task_name,
            "modality": modality,
            "timestamp": datetime.now().isoformat(),
            "dataset_sample": {
                # Include relevant fields from the dataset sample
                "label": sample.get("label"),
                "answer": sample.get("answer"),
                "question": sample.get("question"),
                "latex": sample.get("latex"),
                "code": sample.get("code"),
                "domain": sample.get("domain"),
                "description": sample.get("description"),
                "image_available": "image" in sample,
                # Add other relevant fields as needed
                "raw_sample_keys": list(sample.keys()),
            },
            "evaluation": {
                "choices": choices,
                "input_prompt": prompt,
                "model_response": model_response,
                "parsed_prediction": parsed_prediction,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "prompt_type": "long" if self.use_long_prompts else "short",
            },
        }

        # Save to task-specific file
        log_file = model_dir / f"{self.task_name}.json"

        # Load existing data or create new list
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    log_data = []
        else:
            log_data = []

        # Append new entry
        log_data.append(log_entry)

        # Save back to JSON file with indent=2
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        logger.debug(
            f"Saved detailed log for {self.task_name} sample {sample_idx} to {log_file}"
        )

    def get_choices(self, sample: Dict[str, Any]) -> Optional[List[str]]:
        """Extract choices from sample if available"""
        # First try to get choices from explicit 'choices' field
        if "choices" in sample:
            if isinstance(sample["choices"], str):
                # Parse string representation of choices
                import ast

                try:
                    return ast.literal_eval(sample["choices"])
                except:
                    return None
            elif isinstance(sample["choices"], list):
                return sample["choices"]

        # If no explicit choices, try to extract unique labels from the dataset
        # This will be populated by the evaluator when loading the dataset
        if hasattr(self, "_dataset_choices") and self._dataset_choices:
            return self._dataset_choices

        return None

    def extract_choices_from_dataset(self):
        """Extract unique choices from the dataset's label column"""
        if not self.dataset:
            return None

        unique_labels = set()
        for sample in self.dataset:
            label = sample.get("label", sample.get("answer"))
            if label is not None:
                # Convert boolean to string representation
                if isinstance(label, bool):
                    unique_labels.add("yes" if label else "no")
                elif isinstance(label, str):
                    # Clean up string labels
                    clean_label = label.lower().strip()
                    if clean_label in ["true", "false"]:
                        unique_labels.add("yes" if clean_label == "true" else "no")
                    else:
                        unique_labels.add(clean_label)
                else:
                    unique_labels.add(str(label))

        if unique_labels:
            # Sort for consistency
            self._dataset_choices = sorted(list(unique_labels))
            return self._dataset_choices
        return None

    def process_ground_truth(
        self, sample: Dict[str, Any], choices: Optional[List[str]] = None
    ) -> Union[int, str, bool]:
        """Process ground truth value, can be overridden by subclasses for special handling"""
        gt = sample.get("label", sample.get("answer"))

        # If we have choices, return the actual choice value, not index
        if choices and gt is not None:
            # Handle boolean values
            if isinstance(gt, bool):
                return "yes" if gt else "no"
            elif isinstance(gt, str):
                gt_lower = gt.lower().strip()
                if gt_lower in ["true", "false"]:
                    return "yes" if gt_lower == "true" else "no"
                return gt_lower

        # Handle non-choice cases (like numeric answers)
        if isinstance(gt, str):
            try:
                return int(gt)
            except ValueError:
                return gt

        return gt

    def evaluate_modality(
        self,
        model: BaseModel,
        modality: str,
        max_samples: int = None,
        output_dir: Path = None,
    ) -> EvaluationResult:
        """Evaluate model on specific modality"""
        if not self.dataset:
            raise ValueError(f"Dataset not loaded for task {self.task_name}")

        predictions = []
        ground_truth = []
        correct_count = 0

        samples = list(self.dataset)
        if max_samples:
            samples = samples[:max_samples]

        logger.info(
            f"Evaluating {model.model_name} on {self.task_name} - {modality} modality ({len(samples)} samples)"
        )

        for i, sample in enumerate(samples):
            try:
                choices = self.get_choices(sample)
                logger.debug(
                    f"Sample {i}: choices = {choices}, label = {sample.get('label')}"
                )

                # Store the original model response for logging
                model_response = ""
                prompt = ""

                if modality == "text":
                    # Use text-based representations
                    prompt = self.create_text_prompt(sample)
                    prediction = model.predict_text(prompt, choices)
                    # Get the raw response (we'll need to modify the model to return this)
                    model_response = getattr(model, "_last_response", str(prediction))

                elif modality == "image":
                    # Use image representation
                    if "image" not in sample:
                        logger.warning(
                            f"No image found in sample {i} for task {self.task_name}"
                        )
                        prediction = -1
                        model_response = "ERROR: No image available"
                        prompt = "N/A - No image available"
                    else:
                        prompt = self.create_image_prompt(sample)
                        prediction = model.predict_image_text(
                            sample["image"], prompt, choices
                        )
                        model_response = getattr(
                            model, "_last_response", str(prediction)
                        )

                else:
                    raise ValueError(f"Unknown modality: {modality}")

                # Get ground truth using the specialized method
                gt = self.process_ground_truth(sample, choices)

                predictions.append(prediction)
                ground_truth.append(gt)

                # Compare prediction and ground truth appropriately
                if isinstance(prediction, int) and isinstance(gt, int):
                    # Both are indices
                    is_correct = prediction == gt
                elif isinstance(prediction, str) and isinstance(gt, str):
                    # Both are strings, compare case-insensitively
                    is_correct = prediction.lower().strip() == gt.lower().strip()
                else:
                    # Mixed types, not correct
                    is_correct = False

                if is_correct:
                    correct_count += 1

                # Save detailed log for this sample
                if output_dir:
                    self._save_detailed_log(
                        model_name=model.model_name,
                        sample_idx=i,
                        sample=sample,
                        modality=modality,
                        prompt=prompt,
                        choices=choices,
                        model_response=model_response,
                        parsed_prediction=prediction,
                        ground_truth=gt,
                        is_correct=is_correct,
                        output_dir=output_dir,
                    )

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(samples)} samples")

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                predictions.append(-1)
                ground_truth.append(
                    sample.get("label", sample.get("answer", "unknown"))
                )

                # Save error log
                if output_dir:
                    self._save_detailed_log(
                        model_name=model.model_name,
                        sample_idx=i,
                        sample=sample,
                        modality=modality,
                        prompt="ERROR",
                        choices=None,
                        model_response=f"ERROR: {str(e)}",
                        parsed_prediction=-1,
                        ground_truth=sample.get(
                            "label", sample.get("answer", "unknown")
                        ),
                        is_correct=False,
                        output_dir=output_dir,
                    )

        accuracy = correct_count / len(samples) if samples else 0.0

        return EvaluationResult(
            task=self.task_name,
            modality=modality,
            model_name=model.model_name,
            accuracy=accuracy,
            total_samples=len(samples),
            correct_samples=correct_count,
            predictions=predictions,
            ground_truth=ground_truth,
        )


class MathTaskEvaluator(IsoBenchTaskEvaluator):
    """Evaluator for mathematics tasks (parity, convexity, breakpoint)"""

    def _create_short_text_prompt(self, sample: Dict[str, Any]) -> str:
        """Create math-specific text prompt"""
        base_prompt = ""

        if self.task_name == "math_parity":
            if "latex" in sample:
                base_prompt = f"Determine if the function f(x) = {sample['latex']} is even, odd, or neither."
            elif "code" in sample:
                base_prompt = f"Given the function defined as: {sample['code']}, determine if it is even, odd, or neither."

        elif self.task_name == "math_convexity":
            domain_info = ""
            if "domain" in sample and sample["domain"]:
                domain_info = f" for {sample['domain']}"

            if "latex" in sample:
                base_prompt = f"Determine if the function f(x) = {sample['latex']} is convex or concave{domain_info}."
            elif "code" in sample:
                base_prompt = f"Given the function defined as: {sample['code']}, determine if it is convex or concave{domain_info}."

        elif self.task_name == "math_breakpoint":
            if "latex" in sample:
                base_prompt = f"Count the number of breakpoints in the piecewise linear function: {sample['latex']}."
            elif "code" in sample:
                base_prompt = f"Given the piecewise function: {sample['code']}, count the number of breakpoints."

        return base_prompt

    def _create_long_text_prompt(self, sample: Dict[str, Any]) -> str:
        """Create long math-specific text prompt - based on paper appendix style"""
        base_prompt = ""

        if self.task_name == "math_parity":
            if "latex" in sample:
                base_prompt = f"""You are given a mathematical function f(x) = {sample['latex']}.

Your task is to determine whether this function has even symmetry, odd symmetry, or neither.

Recall the definitions:
- A function f(x) is EVEN if f(-x) = f(x) for all x in the domain. Even functions are symmetric about the y-axis.
- A function f(x) is ODD if f(-x) = -f(x) for all x in the domain. Odd functions have rotational symmetry about the origin.
- A function is NEITHER if it satisfies neither of these conditions.

Analyze the given function mathematically by substituting -x for x and comparing the result to the original function and its negative.

Please read the problem carefully and provide your answer. Think step by step about the mathematical properties involved."""
            elif "code" in sample:
                base_prompt = f"""You are given a mathematical function defined programmatically as: {sample['code']}.

Your task is to determine whether this function has even symmetry, odd symmetry, or neither.

Recall the definitions:
- A function f(x) is EVEN if f(-x) = f(x) for all x in the domain. Even functions are symmetric about the y-axis.
- A function f(x) is ODD if f(-x) = -f(x) for all x in the domain. Odd functions have rotational symmetry about the origin.
- A function is NEITHER if it satisfies neither of these conditions.

Analyze the given function by considering how it behaves when the input is negated.

Please read the problem carefully and provide your answer. Think step by step about the mathematical properties involved."""

        elif self.task_name == "math_convexity":
            domain_info = ""
            if "domain" in sample and sample["domain"]:
                domain_info = f" within the domain {sample['domain']}"

            if "latex" in sample:
                base_prompt = f"""You are given a mathematical function f(x) = {sample['latex']}.

Your task is to determine whether this function is convex or concave{domain_info}.

Recall the definitions:
- A function is CONVEX if its second derivative f''(x) ≥ 0 throughout the domain, or equivalently, if the line segment between any two points on the graph lies above the graph.
- A function is CONCAVE if its second derivative f''(x) ≤ 0 throughout the domain, or equivalently, if the line segment between any two points on the graph lies below the graph.

Analyze the given function by computing its second derivative and determining its sign{domain_info}.

Please read the problem carefully and provide your answer. Think step by step about the mathematical properties involved."""
            elif "code" in sample:
                base_prompt = f"""You are given a mathematical function defined as: {sample['code']}.

Your task is to determine whether this function is convex or concave{domain_info}.

Recall the definitions:
- A function is CONVEX if its second derivative f''(x) ≥ 0 throughout the domain, or equivalently, if the line segment between any two points on the graph lies above the graph.
- A function is CONCAVE if its second derivative f''(x) ≤ 0 throughout the domain, or equivalently, if the line segment between any two points on the graph lies below the graph.

Analyze the given function by considering its curvature properties{domain_info}.

Please read the problem carefully and provide your answer. Think step by step about the mathematical properties involved."""

        elif self.task_name == "math_breakpoint":
            if "latex" in sample:
                base_prompt = f"""You are given a piecewise linear function: {sample['latex']}.

Your task is to count the number of breakpoints (also called corner points or vertices) in this function.

A breakpoint occurs where:
1. The function changes its slope (the derivative is not continuous)
2. Two linear pieces meet at a point
3. The function has a "corner" or "vertex"

Carefully examine the piecewise definition and identify all points where the slope changes. Count each such point as one breakpoint.

Please read the problem carefully and provide your answer. Think step by step about the mathematical properties involved."""
            elif "code" in sample:
                base_prompt = f"""You are given a piecewise linear function defined as: {sample['code']}.

Your task is to count the number of breakpoints (also called corner points or vertices) in this function.

A breakpoint occurs where:
1. The function changes its slope (the derivative is not continuous)
2. Two linear pieces meet at a point
3. The function has a "corner" or "vertex"

Analyze the code definition and identify all points where linear pieces meet and the slope changes. Count each such point as one breakpoint.

Please read the problem carefully and provide your answer. Think step by step about the mathematical properties involved."""

        return base_prompt

    def _create_short_image_prompt(self, sample: Dict[str, Any]) -> str:
        """Create math-specific image prompt"""
        if self.task_name == "math_parity":
            return "Looking at the function plot, determine if the function is even, odd, or neither."
        elif self.task_name == "math_convexity":
            return "Looking at the function plot, determine if the function is convex or concave."
        elif self.task_name == "math_breakpoint":
            return "Looking at the piecewise linear function plot, count the number of breakpoints (corners/vertices)."

        return "Please analyze the mathematical function shown in the image."

    def _create_long_image_prompt(self, sample: Dict[str, Any]) -> str:
        """Create long math-specific image prompt"""
        if self.task_name == "math_parity":
            return """Looking at the function plot, determine if the function is even, odd, or neither.

Recall the definitions:
- A function f(x) is EVEN if f(-x) = f(x) for all x in the domain. Even functions are symmetric about the y-axis.
- A function f(x) is ODD if f(-x) = -f(x) for all x in the domain. Odd functions have rotational symmetry about the origin.
- A function is NEITHER if it satisfies neither of these conditions.

Carefully examine the visual symmetries in the plot to make your determination."""
        elif self.task_name == "math_convexity":
            return """Looking at the function plot, determine if the function is convex or concave.

Recall the definitions:
- A function is CONVEX if the line segment between any two points on the graph lies above the graph. The function curves upward.
- A function is CONCAVE if the line segment between any two points on the graph lies below the graph. The function curves downward.

Examine the curvature of the function in the plot to make your determination."""
        elif self.task_name == "math_breakpoint":
            return """Looking at the piecewise linear function plot, count the number of breakpoints (corners/vertices).

A breakpoint occurs where:
1. The function changes its slope (the derivative is not continuous)
2. Two linear pieces meet at a point
3. The function has a "corner" or "vertex"

Carefully examine the plot and count each point where the function changes direction or slope."""

        return "Please carefully analyze the mathematical function shown in the image and provide your answer."


class ScienceTaskEvaluator(IsoBenchTaskEvaluator):
    """Evaluator for science tasks (chemistry, medical)"""

    def create_text_prompt(self, sample: Dict[str, Any]) -> str:
        """Create science-specific text prompt"""
        if "question" in sample:
            prompt = sample["question"]
            if "description" in sample:
                prompt += f"\n\nContext: {sample['description']}"
            return prompt
        return "Please analyze the given scientific problem and provide your answer."

    def create_image_prompt(self, sample: Dict[str, Any]) -> str:
        """Create science-specific image prompt"""
        if "question" in sample:
            return sample["question"]
        return "Please analyze the scientific diagram/image and answer the question."


class AlgorithmTaskEvaluator(IsoBenchTaskEvaluator):
    """Evaluator for algorithm tasks (path, search, sort, connectivity)"""

    def create_text_prompt(self, sample: Dict[str, Any]) -> str:
        """Create algorithm-specific text prompt"""
        base_prompt = ""

        if self.task_name == "graph_connectivity":
            adj_matrix = sample.get("adjacency_matrix", "")
            node1 = sample.get("query_node_1", "")
            node2 = sample.get("query_node_2", "")
            base_prompt = f"Given the adjacency matrix: {adj_matrix}, determine if nodes {node1} and {node2} are connected."

        elif self.task_name == "graph_maxflow":
            adj_matrix = sample.get("adjacency_matrix", "")
            source = sample.get("source_node", "")
            sink = sample.get("sink_node", "")
            base_prompt = f"Given the capacity matrix: {adj_matrix}, find the maximum flow from source node {source} to sink node {sink}."

        elif self.task_name == "graph_isomorphism":
            adj_g = sample.get("adjacency_matrix_G", "")
            adj_h = sample.get("adjacency_matrix_H", "")
            base_prompt = f"Given two adjacency matrices G: {adj_g} and H: {adj_h}, determine if the graphs are isomorphic."

        return base_prompt

    def create_image_prompt(self, sample: Dict[str, Any]) -> str:
        """Create algorithm-specific image prompt"""
        if self.task_name == "graph_connectivity":
            node1 = sample.get("query_node_1", "")
            node2 = sample.get("query_node_2", "")
            return f"Looking at the graph, determine if nodes {node1} and {node2} are connected."

        elif self.task_name == "graph_maxflow":
            return "Looking at the flow network, determine the maximum flow from source to sink."

        elif self.task_name == "graph_isomorphism":
            return "Looking at the two graphs, determine if they are isomorphic."

        return "Please analyze the graph(s) shown in the image."


class GameTaskEvaluator(IsoBenchTaskEvaluator):
    """Evaluator for game tasks"""

    def create_text_prompt(self, sample: Dict[str, Any]) -> str:
        """Create game-specific text prompt"""
        # Implementation depends on specific game tasks
        if "question" in sample:
            return sample["question"]
        return "Please analyze the game situation and provide your answer."

    def create_image_prompt(self, sample: Dict[str, Any]) -> str:
        """Create game-specific image prompt"""
        if "question" in sample:
            return sample["question"]
        return "Please analyze the game board/situation shown in the image."
