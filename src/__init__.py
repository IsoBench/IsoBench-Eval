"""
IsoBench Evaluation Framework - Source Package
=============================================

This package contains the core components of the IsoBench evaluation framework
for benchmarking foundation models across multiple modalities and tasks.

Modules:
- models: Foundation model implementations (OpenAI, Gemini, Claude)
- data_structures: Core data structures for evaluation results
- task_evaluators: Task-specific evaluation logic
- evaluator: Main evaluation orchestrator
"""

from .models import BaseModel, OpenAIModel, GeminiModel, ClaudeModel
from .data_structures import EvaluationResult, AggregatedResult
from .task_evaluators import (
    IsoBenchTaskEvaluator,
    MathTaskEvaluator,
    ScienceTaskEvaluator,
    AlgorithmTaskEvaluator,
    GameTaskEvaluator,
)
from .evaluator import IsoBenchEvaluator

__all__ = [
    "BaseModel",
    "OpenAIModel",
    "GeminiModel",
    "ClaudeModel",
    "EvaluationResult",
    "AggregatedResult",
    "IsoBenchTaskEvaluator",
    "MathTaskEvaluator",
    "ScienceTaskEvaluator",
    "AlgorithmTaskEvaluator",
    "GameTaskEvaluator",
    "IsoBenchEvaluator",
]
