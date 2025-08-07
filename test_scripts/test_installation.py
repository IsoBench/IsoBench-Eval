#!/usr/bin/env python3
"""
Test script to verify IsoBench Evaluation Framework installation
"""

import sys
import importlib


def test_imports():
    """Test if all required modules can be imported"""
    required_modules = ["pandas", "numpy", "PIL", "datasets"]

    optional_modules = ["openai", "google.genai", "anthropic"]

    print("=== Testing Required Dependencies ===")
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            return False

    print("\n=== Testing Optional Dependencies (Model APIs) ===")
    for module in optional_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"⚠️  {module}: {e} (install if you plan to use this model)")

    return True


def test_framework():
    """Test if the framework modules can be imported"""
    print("\n=== Testing Framework Modules ===")

    try:
        from data_structures import EvaluationResult, AggregatedResult

        print("✅ data_structures")
    except ImportError as e:
        print(f"❌ data_structures: {e}")
        return False

    try:
        from src.models import BaseModel, OpenAIModel, GeminiModel, ClaudeModel

        print("✅ models")
    except ImportError as e:
        print(f"❌ models: {e}")
        return False

    try:
        from src.task_evaluators import IsoBenchTaskEvaluator, MathTaskEvaluator

        print("✅ task_evaluators")
    except ImportError as e:
        print(f"❌ task_evaluators: {e}")
        return False

    try:
        from src.evaluator import IsoBenchEvaluator

        print("✅ evaluator")
    except ImportError as e:
        print(f"❌ evaluator: {e}")
        return False

    return True


def main():
    """Main test function"""
    print("IsoBench Evaluation Framework - Installation Test\n")

    deps_ok = test_imports()
    framework_ok = test_framework()

    print("\n=== Test Summary ===")
    if deps_ok and framework_ok:
        print("✅ All tests passed! Framework is ready to use.")
        print("\nNext steps:")
        print("1. Set your API keys as environment variables")
        print("2. Run: python eval.py --help")
        print("3. Start with: python eval.py --max-samples 10")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("Install missing dependencies with: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
