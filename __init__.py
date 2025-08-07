"""
IsoBench Evaluation Framework
============================

A comprehensive evaluation framework for IsoBench dataset supporting multiple
foundation models across all tasks and modalities.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .data_structures import EvaluationResult, AggregatedResult
from .models import BaseModel, OpenAIModel, GeminiModel, ClaudeModel
from .evaluator import IsoBenchEvaluator
from .task_evaluators import (
    IsoBenchTaskEvaluator,
    MathTaskEvaluator,
    ScienceTaskEvaluator,
    AlgorithmTaskEvaluator,
    GameTaskEvaluator,
)

__all__ = [
    "EvaluationResult",
    "AggregatedResult",
    "BaseModel",
    "OpenAIModel",
    "GeminiModel",
    "ClaudeModel",
    "IsoBenchEvaluator",
    "IsoBenchTaskEvaluator",
    "MathTaskEvaluator",
    "ScienceTaskEvaluator",
    "AlgorithmTaskEvaluator",
    "GameTaskEvaluator",
]
