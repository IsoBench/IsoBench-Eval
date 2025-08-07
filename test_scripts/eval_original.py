"""
IsoBench Evaluation Framework
============================

A comprehensive evaluation framework for IsoBench dataset supporting multiple
foundation models (OpenAI GPT, Google Gemini, Anthropic Claude) across all
tasks and modalities (image and text representations).

Features:
- Modular design with separate components for each model and task
- Support for all IsoBench tasks: math, science, algorithms, games
- Image and text modality evaluation
- Detailed per-task and aggregate reporting
- Results aggregation similar to Table 1 in the paper

Author: Deqing Fu
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from datasets import load_dataset
import time
import logging
from pathlib import Path
import base64
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results"""

    task: str
    modality: str
    model_name: str
    accuracy: float
    total_samples: int
    correct_samples: int
    predictions: List[Any] = field(default_factory=list)
    ground_truth: List[Any] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedResult:
    """Container for aggregated results across tasks"""

    model_name: str
    image_accuracy: float
    text_accuracy: float
    task_results: Dict[str, Dict[str, float]] = field(default_factory=dict)


class BaseModel(ABC):
    """Abstract base class for all models"""

    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.rate_limit_delay = 1.0  # seconds between API calls

    @abstractmethod
    def predict_text(self, prompt: str, choices: List[str] = None) -> Union[str, int]:
        """Make prediction with text input"""
        pass

    @abstractmethod
    def predict_image_text(
        self, image: Image.Image, prompt: str, choices: List[str] = None
    ) -> Union[str, int]:
        """Make prediction with image + text input"""
        pass

    def _apply_rate_limit(self):
        """Apply rate limiting between API calls"""
        time.sleep(self.rate_limit_delay)


class OpenAIModel(BaseModel):
    """OpenAI GPT model implementation"""

    def __init__(self, model_name: str = "gpt-4-turbo", api_key: str = None):
        super().__init__(model_name, api_key)
        try:
            import openai

            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")

    def predict_text(self, prompt: str, choices: List[str] = None) -> Union[str, int]:
        """Make prediction with text input"""
        try:
            if choices:
                prompt += f"\n\nChoices: {choices}\nPlease respond with only the letter (A, B, C, D) or number (0, 1, 2, 3) of your choice."

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
            )

            result = response.choices[0].message.content.strip()
            self._apply_rate_limit()

            # Parse result to get choice index
            if choices:
                return self._parse_choice(result, choices)
            return result

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return -1  # Error indicator

    def predict_image_text(
        self, image: Image.Image, prompt: str, choices: List[str] = None
    ) -> Union[str, int]:
        """Make prediction with image + text input"""
        try:
            # Convert PIL image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            if choices:
                prompt += f"\n\nChoices: {choices}\nPlease respond with only the letter (A, B, C, D) or number (0, 1, 2, 3) of your choice."

            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",  # Use vision model for image inputs
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_str}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=100,
                temperature=0.0,
            )

            result = response.choices[0].message.content.strip()
            self._apply_rate_limit()

            if choices:
                return self._parse_choice(result, choices)
            return result

        except Exception as e:
            logger.error(f"OpenAI Vision API error: {e}")
            return -1

    def _parse_choice(self, response: str, choices: List[str]) -> int:
        """Parse model response to extract choice index"""
        response = response.upper().strip()

        # Check for letter choices (A, B, C, D)
        letters = ["A", "B", "C", "D"]
        for i, letter in enumerate(letters[: len(choices)]):
            if letter in response:
                return i

        # Check for number choices (0, 1, 2, 3)
        for i in range(len(choices)):
            if str(i) in response:
                return i

        # Default to first choice if parsing fails
        logger.warning(f"Could not parse choice from response: {response}")
        return 0


class GeminiModel(BaseModel):
    """Google Gemini model implementation"""

    def __init__(self, model_name: str = "gemini-pro", api_key: str = None):
        super().__init__(model_name, api_key)
        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
            self.text_model = genai.GenerativeModel("gemini-pro")
            self.vision_model = genai.GenerativeModel("gemini-pro-vision")
        except ImportError:
            raise ImportError(
                "Please install google-generativeai package: pip install google-generativeai"
            )

    def predict_text(self, prompt: str, choices: List[str] = None) -> Union[str, int]:
        """Make prediction with text input"""
        try:
            if choices:
                prompt += f"\n\nChoices: {choices}\nPlease respond with only the letter (A, B, C, D) or number (0, 1, 2, 3) of your choice."

            response = self.text_model.generate_content(prompt)
            result = response.text.strip()
            self._apply_rate_limit()

            if choices:
                return self._parse_choice(result, choices)
            return result

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return -1

    def predict_image_text(
        self, image: Image.Image, prompt: str, choices: List[str] = None
    ) -> Union[str, int]:
        """Make prediction with image + text input"""
        try:
            if choices:
                prompt += f"\n\nChoices: {choices}\nPlease respond with only the letter (A, B, C, D) or number (0, 1, 2, 3) of your choice."

            response = self.vision_model.generate_content([prompt, image])
            result = response.text.strip()
            self._apply_rate_limit()

            if choices:
                return self._parse_choice(result, choices)
            return result

        except Exception as e:
            logger.error(f"Gemini Vision API error: {e}")
            return -1

    def _parse_choice(self, response: str, choices: List[str]) -> int:
        """Parse model response to extract choice index"""
        response = response.upper().strip()

        # Check for letter choices (A, B, C, D)
        letters = ["A", "B", "C", "D"]
        for i, letter in enumerate(letters[: len(choices)]):
            if letter in response:
                return i

        # Check for number choices (0, 1, 2, 3)
        for i in range(len(choices)):
            if str(i) in response:
                return i

        # Default to first choice if parsing fails
        logger.warning(f"Could not parse choice from response: {response}")
        return 0


class ClaudeModel(BaseModel):
    """Anthropic Claude model implementation"""

    def __init__(self, model_name: str = "claude-3-opus-20240229", api_key: str = None):
        super().__init__(model_name, api_key)
        try:
            import anthropic

            self.client = anthropic.Anthropic(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
            )
        except ImportError:
            raise ImportError("Please install anthropic package: pip install anthropic")

    def predict_text(self, prompt: str, choices: List[str] = None) -> Union[str, int]:
        """Make prediction with text input"""
        try:
            if choices:
                prompt += f"\n\nChoices: {choices}\nPlease respond with only the letter (A, B, C, D) or number (0, 1, 2, 3) of your choice."

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=100,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

            result = response.content[0].text.strip()
            self._apply_rate_limit()

            if choices:
                return self._parse_choice(result, choices)
            return result

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return -1

    def predict_image_text(
        self, image: Image.Image, prompt: str, choices: List[str] = None
    ) -> Union[str, int]:
        """Make prediction with image + text input"""
        try:
            # Convert PIL image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            if choices:
                prompt += f"\n\nChoices: {choices}\nPlease respond with only the letter (A, B, C, D) or number (0, 1, 2, 3) of your choice."

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=100,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_str,
                                },
                            },
                        ],
                    }
                ],
            )

            result = response.content[0].text.strip()
            self._apply_rate_limit()

            if choices:
                return self._parse_choice(result, choices)
            return result

        except Exception as e:
            logger.error(f"Claude Vision API error: {e}")
            return -1

    def _parse_choice(self, response: str, choices: List[str]) -> int:
        """Parse model response to extract choice index"""
        response = response.upper().strip()

        # Check for letter choices (A, B, C, D)
        letters = ["A", "B", "C", "D"]
        for i, letter in enumerate(letters[: len(choices)]):
            if letter in response:
                return i

        # Check for number choices (0, 1, 2, 3)
        for i in range(len(choices)):
            if str(i) in response:
                return i

        # Default to first choice if parsing fails
        logger.warning(f"Could not parse choice from response: {response}")
        return 0


class IsoBenchTaskEvaluator:
    """Base evaluator for IsoBench tasks"""

    def __init__(self, task_name: str, use_long_prompts: bool = False):
        self.task_name = task_name
        self.use_long_prompts = use_long_prompts
        self.dataset = None
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

    def get_choices(self, sample: Dict[str, Any]) -> Optional[List[str]]:
        """Extract choices from sample if available"""
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
        return None

    def evaluate_modality(
        self, model: BaseModel, modality: str, max_samples: int = None
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

                if modality == "text":
                    # Use text-based representations
                    prompt = self.create_text_prompt(sample)
                    prediction = model.predict_text(prompt, choices)

                elif modality == "image":
                    # Use image representation
                    if "image" not in sample:
                        logger.warning(
                            f"No image found in sample {i} for task {self.task_name}"
                        )
                        prediction = -1
                    else:
                        prompt = self.create_image_prompt(sample)
                        prediction = model.predict_image_text(
                            sample["image"], prompt, choices
                        )

                else:
                    raise ValueError(f"Unknown modality: {modality}")

                # Get ground truth
                if isinstance(sample["label"], str):
                    gt = int(sample["label"])
                else:
                    gt = sample["label"]

                predictions.append(prediction)
                ground_truth.append(gt)

                if prediction == gt:
                    correct_count += 1

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(samples)} samples")

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                predictions.append(-1)
                ground_truth.append(sample["label"])

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
        """Create short math-specific text prompt"""
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

    def _create_short_image_prompt(self, sample: Dict[str, Any]) -> str:
        """Create short math-specific image prompt"""
        if self.task_name == "math_parity":
            return "Looking at the function plot, determine if the function is even, odd, or neither."
        elif self.task_name == "math_convexity":
            return "Looking at the function plot, determine if the function is convex or concave."
        elif self.task_name == "math_breakpoint":
            return "Looking at the piecewise linear function plot, count the number of breakpoints (corners/vertices)."

        return "Please analyze the mathematical function shown in the image."

    def _create_long_text_prompt(self, sample: Dict[str, Any]) -> str:
        """Create long math-specific text prompt - based on paper appendix style"""
        base_prompt = ""
        instruction_suffix = "\n\nPlease read the problem carefully and provide your answer. Think step by step about the mathematical properties involved."

        if self.task_name == "math_parity":
            if "latex" in sample:
                base_prompt = f"""You are given a mathematical function f(x) = {sample['latex']}.

Your task is to determine whether this function has even symmetry, odd symmetry, or neither.

Recall the definitions:
- A function f(x) is EVEN if f(-x) = f(x) for all x in the domain. Even functions are symmetric about the y-axis.
- A function f(x) is ODD if f(-x) = -f(x) for all x in the domain. Odd functions have rotational symmetry about the origin.
- A function is NEITHER if it satisfies neither of these conditions.

Analyze the given function mathematically by substituting -x for x and comparing the result to the original function and its negative."""
            elif "code" in sample:
                base_prompt = f"""You are given a mathematical function defined programmatically as: {sample['code']}.

Your task is to determine whether this function has even symmetry, odd symmetry, or neither.

Recall the definitions:
- A function f(x) is EVEN if f(-x) = f(x) for all x in the domain. Even functions are symmetric about the y-axis.
- A function f(x) is ODD if f(-x) = -f(x) for all x in the domain. Odd functions have rotational symmetry about the origin.
- A function is NEITHER if it satisfies neither of these conditions.

Analyze the given function by considering how it behaves when the input is negated."""

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

Analyze the given function by computing its second derivative and determining its sign{domain_info}."""
            elif "code" in sample:
                base_prompt = f"""You are given a mathematical function defined as: {sample['code']}.

Your task is to determine whether this function is convex or concave{domain_info}.

Recall the definitions:
- A function is CONVEX if its second derivative f''(x) ≥ 0 throughout the domain, or equivalently, if the line segment between any two points on the graph lies above the graph.
- A function is CONCAVE if its second derivative f''(x) ≤ 0 throughout the domain, or equivalently, if the line segment between any two points on the graph lies below the graph.

Analyze the given function by considering its curvature properties{domain_info}."""

        elif self.task_name == "math_breakpoint":
            if "latex" in sample:
                base_prompt = f"""You are given a piecewise linear function: {sample['latex']}.

Your task is to count the number of breakpoints (also called corner points or vertices) in this function.

A breakpoint occurs where:
1. The function changes its slope (the derivative is not continuous)
2. Two linear pieces meet at a point
3. The function has a "corner" or "vertex"

Carefully examine the piecewise definition and identify all points where the slope changes. Count each such point as one breakpoint."""
            elif "code" in sample:
                base_prompt = f"""You are given a piecewise linear function defined as: {sample['code']}.

Your task is to count the number of breakpoints (also called corner points or vertices) in this function.

A breakpoint occurs where:
1. The function changes its slope (the derivative is not continuous)
2. Two linear pieces meet at a point
3. The function has a "corner" or "vertex"

Carefully examine the piecewise definition and identify all points where the slope changes. Count each such point as one breakpoint."""

        return base_prompt + instruction_suffix

    def _create_long_image_prompt(self, sample: Dict[str, Any]) -> str:
        """Create long math-specific image prompt - based on paper appendix style"""
        base_prompt = ""

        if self.task_name == "math_parity":
            base_prompt = """You are shown a graph of a mathematical function.

Your task is to determine whether this function has even symmetry, odd symmetry, or neither by analyzing its visual properties.

Recall the definitions:
- A function is EVEN if it is symmetric about the y-axis. If you fold the graph along the y-axis, the two halves should match perfectly.
- A function is ODD if it has rotational symmetry about the origin (point symmetry). If you rotate the graph 180 degrees about the origin, it should look the same.
- A function is NEITHER if it exhibits neither of these symmetries.

Carefully examine the graph and look for these symmetry properties. Pay attention to how the function behaves for positive and negative x-values."""

        elif self.task_name == "math_convexity":
            base_prompt = """You are shown a graph of a mathematical function.

Your task is to determine whether this function is convex or concave by analyzing its curvature.

Recall the definitions:
- A function is CONVEX if it curves upward like a cup (∪ shape). Any line segment connecting two points on the graph lies above the graph.
- A function is CONCAVE if it curves downward like a cap (∩ shape). Any line segment connecting two points on the graph lies below the graph.

Look at the overall shape and curvature of the function. Pay attention to whether the function opens upward or downward."""

        elif self.task_name == "math_breakpoint":
            base_prompt = """You are shown a graph of a piecewise linear function.

Your task is to count the number of breakpoints (corner points or vertices) in this function.

A breakpoint is a point where:
1. The slope of the function changes abruptly
2. Two linear segments meet at a sharp corner
3. The function has a "vertex" or "corner"

Carefully trace the function from left to right and count every point where you see a sharp change in direction or slope. Do not count smooth curves - only sharp corners where linear pieces meet."""

        return (
            base_prompt
            + "\n\nLook carefully at the graph and provide your answer based on visual analysis."
        )


class ScienceTaskEvaluator(IsoBenchTaskEvaluator):
    """Evaluator for science tasks (chemistry, physics)"""

    def _create_short_text_prompt(self, sample: Dict[str, Any]) -> str:
        """Create short science-specific text prompt"""
        if "question" in sample:
            prompt = sample["question"]
            if "description" in sample:
                prompt += f"\n\nContext: {sample['description']}"
            return prompt
        return "Please analyze the given scientific problem and provide your answer."

    def _create_short_image_prompt(self, sample: Dict[str, Any]) -> str:
        """Create short science-specific image prompt"""
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
        if base_prompt.task_name == "graph_maxflow":
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
        self.use_long_prompts = use_long_prompts
        self.results: List[EvaluationResult] = []

        if use_long_prompts:
            logger.info("Using long prompts (paper appendix style)")
        else:
            logger.info("Using short prompts")

    def get_task_evaluator(self, task_name: str) -> IsoBenchTaskEvaluator:
        """Get appropriate evaluator for task"""
        if task_name in self.MATH_TASKS:
            return MathTaskEvaluator(task_name, self.use_long_prompts)
        elif task_name in self.SCIENCE_TASKS:
            return ScienceTaskEvaluator(task_name, self.use_long_prompts)
        elif task_name in self.ALGORITHM_TASKS:
            return AlgorithmTaskEvaluator(task_name, self.use_long_prompts)
        elif task_name in self.GAME_TASKS:
            return GameTaskEvaluator(task_name, self.use_long_prompts)
        else:
            raise ValueError(f"Unknown task: {task_name}")


def main():
    """Example usage of the IsoBench evaluation framework"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate models on IsoBench")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["gpt4", "claude", "gemini"],
        default=["gpt4"],
        help="Models to evaluate",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=IsoBenchEvaluator.ALL_TASKS,
        default=None,
        help="Tasks to evaluate (default: all)",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["text", "image"],
        default=["text", "image"],
        help="Modalities to evaluate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per task (for testing)",
    )
    parser.add_argument(
        "--long-prompts",
        action="store_true",
        help="Use long prompts from paper appendix",
    )
    parser.add_argument(
        "--output-dir", default="isobench_results", help="Output directory for results"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = IsoBenchEvaluator(
        output_dir=args.output_dir, use_long_prompts=args.long_prompts
    )

    # Initialize models
    models = []
    if "gpt4" in args.models:
        models.append(OpenAIModel("gpt-4-turbo"))
    if "claude" in args.models:
        models.append(ClaudeModel("claude-3-opus-20240229"))
    if "gemini" in args.models:
        models.append(GeminiModel("gemini-pro"))

    # Run evaluation
    for model in models:
        logger.info(f"Starting evaluation for {model.model_name}")
        evaluator.evaluate_model(
            model=model,
            tasks=args.tasks,
            modalities=args.modalities,
            max_samples_per_task=args.max_samples,
        )

    # Save results and generate reports
    evaluator.save_detailed_results()

    # Generate Table 1 style report
    report_df = evaluator.generate_table1_report()
    print("\n" + "=" * 60)
    print("ISOBENCH EVALUATION RESULTS (Table 1 Style)")
    print("=" * 60)
    print(report_df.to_string(index=False))
    print("=" * 60)

    # Save task-level breakdown
    task_breakdown = {}
    for model_name in [m.model_name for m in models]:
        agg_result = evaluator.aggregate_results(model_name)
        task_breakdown[model_name] = agg_result.task_results

    with open(evaluator.output_dir / "task_breakdown.json", "w") as f:
        json.dump(task_breakdown, f, indent=2)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()

    def evaluate_model(
        self,
        model: BaseModel,
        tasks: List[str] = None,
        modalities: List[str] = None,
        max_samples_per_task: int = None,
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
                        model, modality, max_samples_per_task
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
