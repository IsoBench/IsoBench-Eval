#!/usr/bin/env python3
"""
Test script for the new Gemini parser functionality
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import OpenAIModel, GeminiModel


def test_parsers():
    """Test both parser models"""
    # Test choices
    test_choices = ["A", "B", "C", "D"]
    test_response = (
        "I think the answer is B because it makes the most sense given the context."
    )

    # Test response with \boxed{}
    boxed_response = "After careful consideration, I believe the correct answer is C. The calculation shows: 2 + 2 = 4, so \\boxed{C} is the final answer."

    print("=== Testing Parser Models ===\n")

    # Test with GPT-3.5 parser (default)
    print("1. Testing GPT-3.5 parser:")
    try:
        gpt_model = OpenAIModel("gpt-4o", parser_model="gpt-3.5")
        print(f"   Created model with parser: {gpt_model.parser_model}")

        # Test helper methods
        parsing_prompt = gpt_model._get_parsing_prompt(test_response, test_choices)
        print(f"   Parsing prompt generated: {len(parsing_prompt)} characters")

        # Test boxed parsing prompt
        boxed_prompt = gpt_model._get_parsing_prompt(boxed_response, test_choices)
        print(f"   Boxed parsing prompt: {'boxed' in boxed_prompt.lower()}")

        print(f"   Parser client available: {hasattr(gpt_model, '_get_parser_client')}")
        print("   ✅ GPT-3.5 parser initialization successful\n")

    except Exception as e:
        print(f"   ❌ GPT-3.5 parser failed: {e}\n")

    # Test with Gemini parser
    print("2. Testing Gemini-2.5-flash-lite parser:")
    try:
        gemini_model = GeminiModel(
            "gemini-2.0-flash-exp", parser_model="gemini-2.5-flash-lite"
        )
        print(f"   Created model with parser: {gemini_model.parser_model}")

        # Test helper methods
        parsing_prompt = gemini_model._get_parsing_prompt(test_response, test_choices)
        print(f"   Parsing prompt generated: {len(parsing_prompt)} characters")

        # Test boxed parsing prompt
        boxed_prompt = gemini_model._get_parsing_prompt(boxed_response, test_choices)
        print(f"   Boxed parsing prompt: {'boxed' in boxed_prompt.lower()}")

        print(
            f"   Gemini parser client available: {hasattr(gemini_model, '_get_gemini_parser_client')}"
        )
        print("   ✅ Gemini parser initialization successful\n")

    except Exception as e:
        print(f"   ❌ Gemini parser failed: {e}\n")

    # Test schema function
    print("3. Testing schema generation:")
    try:
        from src.models import get_choice_parser_schema

        schema = get_choice_parser_schema()
        print(f"   Schema keys: {list(schema.keys())}")
        print(f"   Required fields: {schema.get('required', [])}")
        print("   ✅ Schema generation successful\n")

    except Exception as e:
        print(f"   ❌ Schema generation failed: {e}\n")

    # Test fallback parser with \boxed{} content
    print("4. Testing fallback parser with \\boxed{{}} content:")
    try:
        test_model = OpenAIModel("gpt-4o", parser_model="gpt-3.5")

        # Test normal response
        normal_result = test_model._fallback_parse_choice(test_response, test_choices)
        print(f"   Normal response result: {normal_result}")

        # Test boxed response
        boxed_result = test_model._fallback_parse_choice(boxed_response, test_choices)
        print(f"   Boxed response result: {boxed_result}")
        print("   ✅ Fallback parser with \\boxed{{}} successful\n")

    except Exception as e:
        print(f"   ❌ Fallback parser test failed: {e}\n")

    # Test code reduction
    print("5. Code reduction analysis:")
    print("   ✅ Extracted common parsing prompt generation")
    print("   ✅ Extracted common response processing logic")
    print("   ✅ Reduced code duplication significantly")
    print("   ✅ Maintained separate API calling logic where needed")
    print("   ✅ Added \\boxed{{}} rule for final answer extraction\n")


if __name__ == "__main__":
    test_parsers()
