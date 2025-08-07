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

from .data_structures import EvaluationResult
from .models import BaseModel

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

    def _load_cached_results(
        self, model_name: str, output_dir: Path
    ) -> Dict[int, Dict]:
        """Load cached results for a specific model and task"""
        model_dir = output_dir / model_name.replace("/", "_")
        log_file = model_dir / f"{self.task_name}.json"

        if not log_file.exists():
            return {}

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)

            # Create a mapping from (sample_index, modality) to cached result
            cached = {}
            for entry in log_data:
                key = (entry["sample_index"], entry["modality"])
                cached[key] = entry

            logger.info(f"Loaded {len(log_data)} cached results for {self.task_name}")
            return cached

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error loading cached results from {log_file}: {e}")
            return {}

    def _clear_cache_if_fresh_start(
        self, model_name: str, output_dir: Path, resume: bool
    ):
        """Clear cache file if not resuming"""
        if not resume:
            model_dir = output_dir / model_name.replace("/", "_")
            log_file = model_dir / f"{self.task_name}.json"
            if log_file.exists():
                log_file.unlink()
                logger.info(f"Cleared cached results for {self.task_name}")

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
        resume: bool = True,
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

        # Handle caching
        cached_results = {}
        if resume and output_dir:
            cached_results = self._load_cached_results(model.model_name, output_dir)
        elif output_dir:
            self._clear_cache_if_fresh_start(model.model_name, output_dir, resume)

        logger.info(
            f"Evaluating {model.model_name} on {self.task_name} - {modality} modality ({len(samples)} samples)"
        )
        if cached_results:
            cached_count = sum(
                1 for i in range(len(samples)) if (i, modality) in cached_results
            )
            logger.info(
                f"Found {cached_count} cached results, will evaluate {len(samples) - cached_count} new samples"
            )

        for i, sample in enumerate(samples):
            # Check if this sample is already cached
            cache_key = (i, modality)
            if cache_key in cached_results:
                # Load from cache
                cached_entry = cached_results[cache_key]
                cached_eval = cached_entry["evaluation"]

                prediction = cached_eval["parsed_prediction"]
                gt = cached_eval["ground_truth"]
                is_correct = cached_eval["is_correct"]

                predictions.append(prediction)
                ground_truth.append(gt)
                if is_correct:
                    correct_count += 1

                logger.debug(
                    f"Sample {i}: Loaded from cache - prediction: {prediction}, ground_truth: {gt}, correct: {is_correct}"
                )
                continue

            # Evaluate new sample
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
    """Evaluator for science tasks (chemistry, physics)"""

    def _create_short_text_prompt(self, sample: Dict[str, Any]) -> str:
        """Create science-specific text prompt"""
        if "question" in sample:
            prompt = sample["question"]
            if "description" in sample:
                prompt += f"\n\nContext: {sample['description']}"
            return prompt
        return "Please analyze the given scientific problem and provide your answer."

    def _create_short_image_prompt(self, sample: Dict[str, Any]) -> str:
        """Create science-specific image prompt"""
        if "question" in sample:
            return sample["question"]
        return "Please analyze the scientific diagram/image and answer the question."

    def _create_long_text_prompt(self, sample: Dict[str, Any]) -> str:
        """Create long science-specific text prompt - based on paper appendix style"""
        base_prompt = ""

        if self.task_name == "chemistry":
            base_prompt = """You are presented with a chemistry problem that requires careful analysis.

Please read the problem statement and any provided context carefully. Use your knowledge of chemical principles, molecular structure, chemical reactions, and laboratory procedures to solve the problem.

Consider the following when analyzing:
- Molecular structures and bonding
- Chemical properties and reactions
- Stoichiometry and chemical equations
- Laboratory techniques and measurements
- Physical and chemical changes

Problem: """

        elif self.task_name == "physics":
            base_prompt = """You are presented with a physics problem that requires systematic analysis.

Please read the problem statement and any provided context carefully. Use your knowledge of physical principles, mathematical relationships, and scientific reasoning to solve the problem.

Consider the following when analyzing:
- Physical laws and principles
- Mathematical relationships and equations
- Units and dimensional analysis
- Graphical representations
- Experimental setup and measurements

Problem: """

        if "question" in sample:
            base_prompt += sample["question"]
            if "description" in sample:
                base_prompt += f"\n\nAdditional Context: {sample['description']}"

        base_prompt += "\n\nPlease think through the problem step by step and provide your answer with clear reasoning."

        return base_prompt

    def _create_long_image_prompt(self, sample: Dict[str, Any]) -> str:
        """Create long science-specific image prompt - based on paper appendix style"""
        base_prompt = ""

        if self.task_name == "chemistry":
            base_prompt = """You are shown a chemistry diagram, molecular structure, experimental setup, or chemical representation.

Your task is to carefully analyze the visual information and answer the associated question.

When examining the image, consider:
- Molecular structures, bonds, and geometry
- Chemical symbols, formulas, and notation
- Laboratory equipment and experimental setups
- Reaction mechanisms and pathways
- Physical and chemical properties
- Measurements, graphs, and data representations

Look carefully at all visual elements including:
- Colors, shapes, and symbols
- Labels, numbers, and text
- Spatial relationships and arrangements
- Any legends, keys, or reference information

"""

        elif self.task_name == "physics":
            base_prompt = """You are shown a physics diagram, experimental setup, graph, or physical representation.

Your task is to carefully analyze the visual information and answer the associated question.

When examining the image, consider:
- Physical systems and their components
- Forces, motion, and energy relationships
- Graphs, charts, and data representations
- Experimental apparatus and measurements
- Geometric relationships and spatial arrangements
- Physical quantities and their units

Look carefully at all visual elements including:
- Shapes, directions, and orientations
- Numbers, scales, and measurements
- Labels, symbols, and notation
- Any legends, axes, or reference information

"""

        if "question" in sample:
            base_prompt += f"Question: {sample['question']}"

        base_prompt += "\n\nAnalyze the image systematically and provide your answer with clear reasoning based on what you observe."

        return base_prompt


class AlgorithmTaskEvaluator(IsoBenchTaskEvaluator):
    """Evaluator for algorithm tasks (graph connectivity, maxflow, isomorphism)"""

    def _create_short_text_prompt(self, sample: Dict[str, Any]) -> str:
        """Create short algorithm-specific text prompt"""
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

    def _create_short_image_prompt(self, sample: Dict[str, Any]) -> str:
        """Create short algorithm-specific image prompt"""
        if self.task_name == "graph_connectivity":
            node1 = sample.get("query_node_1", "")
            node2 = sample.get("query_node_2", "")
            return f"Looking at the graph, determine if nodes {node1} and {node2} are connected."

        elif self.task_name == "graph_maxflow":
            return "Looking at the flow network, determine the maximum flow from source to sink."

        elif self.task_name == "graph_isomorphism":
            return "Looking at the two graphs, determine if they are isomorphic."

        return "Please analyze the graph(s) shown in the image."

    def _create_long_text_prompt(self, sample: Dict[str, Any]) -> str:
        """Create long algorithm-specific text prompt - based on paper appendix style"""
        base_prompt = ""

        if self.task_name == "graph_connectivity":
            adj_matrix = sample.get("adjacency_matrix", "")
            node1 = sample.get("query_node_1", "")
            node2 = sample.get("query_node_2", "")

            base_prompt = f"""You are given a graph connectivity problem.

You need to determine whether two specific nodes in a graph are connected (i.e., there exists a path between them).

Graph representation: The graph is represented as an adjacency matrix:
{adj_matrix}

In this matrix:
- Entry (i,j) = 1 means there is an edge from node i to node j
- Entry (i,j) = 0 means there is no direct edge from node i to node j
- The graph may be directed or undirected

Query: Determine if node {node1} and node {node2} are connected.

Two nodes are connected if there exists a path (sequence of edges) that connects them, either directly or through intermediate nodes.

Analyze the adjacency matrix systematically to trace possible paths between the query nodes."""

        elif self.task_name == "graph_maxflow":
            adj_matrix = sample.get("adjacency_matrix", "")
            source = sample.get("source_node", "")
            sink = sample.get("sink_node", "")

            base_prompt = f"""You are given a maximum flow problem in a flow network.

You need to find the maximum amount of flow that can be sent from a source node to a sink node.

Network representation: The flow network is represented as a capacity matrix:
{adj_matrix}

In this matrix:
- Entry (i,j) represents the maximum capacity of the edge from node i to node j
- Entry (i,j) = 0 means there is no edge from node i to node j

Source node: {source}
Sink node: {sink}

Your task is to find the maximum flow from the source to the sink.

Consider using algorithms like Ford-Fulkerson, and think about:
- Finding augmenting paths from source to sink
- The bottleneck capacity along each path
- The total maximum flow possible"""

        elif self.task_name == "graph_isomorphism":
            adj_g = sample.get("adjacency_matrix_G", "")
            adj_h = sample.get("adjacency_matrix_H", "")

            base_prompt = f"""You are given a graph isomorphism problem.

You need to determine whether two graphs are isomorphic (structurally identical).

Graph G adjacency matrix:
{adj_g}

Graph H adjacency matrix:
{adj_h}

Two graphs are isomorphic if:
- They have the same number of vertices and edges
- There exists a bijection (one-to-one mapping) between their vertices that preserves adjacency relationships
- In other words, you can relabel the vertices of one graph to make it identical to the other

When checking for isomorphism, consider:
- Degree sequences (must be identical)
- Number of vertices and edges
- Structural properties like cycles, paths, connectivity
- Try to find a vertex mapping that preserves all edge relationships"""

        base_prompt += "\n\nAnalyze the problem systematically and provide your answer with clear reasoning."
        return base_prompt

    def _create_long_image_prompt(self, sample: Dict[str, Any]) -> str:
        """Create long algorithm-specific image prompt - based on paper appendix style"""
        base_prompt = ""

        if self.task_name == "graph_connectivity":
            node1 = sample.get("query_node_1", "")
            node2 = sample.get("query_node_2", "")

            base_prompt = f"""You are shown a graph visualization.

Your task is to determine whether two specific nodes are connected by analyzing the visual representation.

Query nodes: {node1} and {node2}

When examining the graph:
- Look for direct edges between the query nodes
- Trace possible paths through intermediate nodes
- Pay attention to node labels and colors
- Consider that connections might be through multiple hops

Two nodes are connected if there exists any path (sequence of edges) between them, either directly or through intermediate vertices.

Carefully trace the graph structure to determine connectivity."""

        elif self.task_name == "graph_maxflow":
            base_prompt = """You are shown a flow network diagram.

Your task is to determine the maximum flow from the source to the sink by analyzing the visual representation.

When examining the network:
- Identify the source node (usually specially marked or colored)
- Identify the sink node (usually specially marked or colored)  
- Look at edge capacities (numbers on edges)
- Consider multiple paths from source to sink
- The maximum flow is limited by bottleneck edges

Find all possible paths from source to sink and determine the maximum total flow that can be achieved. Remember that the flow through any edge cannot exceed its capacity.

Analyze the network systematically to find the maximum flow value."""

        elif self.task_name == "graph_isomorphism":
            base_prompt = """You are shown two graph visualizations.

Your task is to determine whether these two graphs are isomorphic (structurally identical) by analyzing their visual representations.

Two graphs are isomorphic if:
- They have the same number of vertices and edges
- You can relabel the vertices of one graph to make it look identical to the other
- They have the same structural properties

When comparing the graphs:
- Count vertices and edges in both graphs
- Compare degree sequences (how many edges each vertex has)
- Look for similar structural patterns
- Check if vertices with the same degree have the same connectivity patterns
- Try to mentally map vertices from one graph to the other

Examine both graphs carefully and determine if they represent the same underlying structure."""

        base_prompt += "\n\nLook carefully at the visual representation and provide your answer based on systematic analysis."
        return base_prompt


class GameTaskEvaluator(IsoBenchTaskEvaluator):
    """Evaluator for game tasks"""

    def _create_short_text_prompt(self, sample: Dict[str, Any]) -> str:
        """Create short game-specific text prompt"""
        if "question" in sample:
            return sample["question"]
        return "Please analyze the game situation and provide your answer."

    def _create_short_image_prompt(self, sample: Dict[str, Any]) -> str:
        """Create short game-specific image prompt"""
        if "question" in sample:
            return sample["question"]
        return "Please analyze the game board/situation shown in the image."

    def _create_long_text_prompt(self, sample: Dict[str, Any]) -> str:
        """Create long game-specific text prompt - based on paper appendix style"""
        base_prompt = ""

        if self.task_name == "winner_id":
            base_prompt = """You are presented with a game scenario that requires strategic analysis.

Your task is to analyze the game state and determine the winner or optimal outcome.

When analyzing the game:
- Consider the rules and objectives
- Evaluate the current positions and advantages
- Think about possible moves and strategies
- Determine who has the winning position or advantage

Game scenario: """

        elif self.task_name == "puzzle":
            base_prompt = """You are presented with a puzzle that requires logical reasoning and problem-solving.

Your task is to analyze the puzzle systematically and find the correct solution.

When solving the puzzle:
- Identify the constraints and rules
- Look for patterns and relationships
- Consider different approaches and strategies
- Apply logical reasoning step by step

Puzzle: """

        if "question" in sample:
            base_prompt += sample["question"]

        base_prompt += "\n\nAnalyze the problem carefully and provide your answer with clear reasoning."
        return base_prompt

    def _create_long_image_prompt(self, sample: Dict[str, Any]) -> str:
        """Create long game-specific image prompt - based on paper appendix style"""
        base_prompt = ""

        if self.task_name == "winner_id":
            base_prompt = """You are shown a game board or game situation.

Your task is to analyze the visual game state and determine the winner or optimal outcome.

When examining the game image:
- Identify the game type and rules
- Analyze piece positions and player advantages
- Consider possible moves and strategies
- Look for winning conditions or decisive positions
- Pay attention to any highlighted or special elements

"""

        elif self.task_name == "puzzle":
            base_prompt = """You are shown a visual puzzle that requires logical analysis.

Your task is to examine the puzzle image carefully and find the correct solution.

When analyzing the puzzle image:
- Identify all visual elements and their relationships
- Look for patterns, sequences, or logical connections  
- Consider spatial arrangements and orientations
- Apply systematic reasoning to find the solution
- Pay attention to colors, shapes, numbers, or symbols

"""

        if "question" in sample:
            base_prompt += f"Question: {sample['question']}"

        base_prompt += "\n\nExamine the image systematically and provide your solution with clear reasoning."
        return base_prompt
