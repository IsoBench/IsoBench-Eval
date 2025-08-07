"""
Data structures for IsoBench evaluation results.

This module contains dataclasses and result containers used throughout
the evaluation framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


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
