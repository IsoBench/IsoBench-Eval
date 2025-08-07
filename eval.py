#!/usr/bin/env python3
"""
IsoBench Evaluation Framework - Main Evaluation Script
====================================================

Main evaluation script for running IsoBench evaluations with foundation models.
Uses GPT-5 as the default model with support for OpenAI GPT, Google Gemini,
and Anthropic Claude models across all tasks and modalities.

Usage:
    python eval.py --model gpt-5 --tasks math_parity math_convexity --max-samples 100
    python eval.py --model gemini-pro --modalities text --output-dir results/
    python eval.py --model claude-3-opus --help

Author: AI Assistant
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from models import OpenAIModel, GeminiModel, ClaudeModel
from evaluator import IsoBenchEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("isobench_evaluation.log"),
    ],
)
logger = logging.getLogger(__name__)


def create_model(model_name: str, api_key: Optional[str] = None):
    """Create and return the specified model instance"""
    model_name_lower = model_name.lower()

    if any(name in model_name_lower for name in ["gpt", "openai"]):
        return OpenAIModel(model_name, api_key)

    elif any(name in model_name_lower for name in ["gemini", "google"]):
        return GeminiModel(model_name, api_key)

    elif any(name in model_name_lower for name in ["claude", "anthropic"]):
        return ClaudeModel(model_name, api_key)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Supported models include OpenAI GPT models, Google Gemini models, and Anthropic Claude models"
        )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="IsoBench Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full evaluation with GPT-5 (default)
  python eval.py

  # Evaluate specific tasks with custom model
  python eval.py --model gpt-4 --tasks math_parity math_convexity

  # Run with limited samples for testing
  python eval.py --model gemini-2.0-flash-exp --max-samples 50

  # Text modality only
  python eval.py --modalities text --output-dir text_only_results

Available tasks:
  Math: math_parity, math_convexity, math_breakpoint
  Science: chemistry, physics  
  Algorithms: graph_connectivity, graph_maxflow, graph_isomorphism
  Games: winner_id, puzzle
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="Model to evaluate (default: gpt-5). Options: gpt-5, gpt-4, gemini-2.0-flash-exp, gemini-1.5-pro, claude-3-opus",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Specific tasks to evaluate (default: all tasks)",
    )

    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["text", "image"],
        default=["text", "image"],
        help="Modalities to evaluate (default: text image)",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per task (default: all samples)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="isobench_results",
        help="Output directory for results (default: isobench_results)",
    )

    parser.add_argument(
        "--long-prompts",
        action="store_true",
        help="Use long prompts from paper appendix (default: short prompts)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the model (can also use environment variables)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=== IsoBench Evaluation Framework ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Tasks: {args.tasks or 'all'}")
    logger.info(f"Modalities: {args.modalities}")
    logger.info(f"Max samples per task: {args.max_samples or 'all'}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Long prompts: {args.long_prompts}")

    try:
        # Create model instance
        logger.info("Initializing model...")
        model = create_model(args.model, args.api_key)
        logger.info(f"Using model: {model.model_name}")

        # Create evaluator
        logger.info("Initializing evaluator...")
        evaluator = IsoBenchEvaluator(
            args.output_dir, use_long_prompts=args.long_prompts
        )

        # Run evaluation
        logger.info("Starting evaluation...")
        evaluator.evaluate_model(args.model, use_long_prompts=args.long_prompts)

        # Create evaluation summary for the model
        evaluator.create_evaluation_summary(args.model)

        # Generate report
        if args.save_detailed_results:
            evaluator.save_detailed_results()

        # Generate Table 1 report
        report = evaluator.generate_table1_report([args.model])
        print("\nEvaluation Results:")
        print(report.to_string(index=False))

        # Print individual task results
        agg_result = evaluator.aggregate_results(model.model_name)
        if agg_result.task_results:
            print(f"\n=== Task-level Results for {model.model_name} ===")
            for task, modality_results in agg_result.task_results.items():
                print(f"\n{task}:")
                for modality, accuracy in modality_results.items():
                    print(f"  {modality.capitalize()}: {accuracy:.3f}")

        logger.info(
            f"Evaluation completed successfully. Results saved to: {args.output_dir}"
        )

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
