#!/usr/bin/env python3
"""
Quick test for the \boxed{} functionality
"""

import sys
import os
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_boxed_parsing():
    """Test the boxed pattern extraction"""

    test_cases = [
        {
            "response": "After careful analysis, I believe the answer is \\boxed{B} because it satisfies all conditions.",
            "choices": ["A", "B", "C", "D"],
            "expected": "B",
        },
        {
            "response": "Let me solve this step by step: 2x + 3 = 7, so x = 2. Therefore \\boxed{even}",
            "choices": ["even", "odd", "neither"],
            "expected": "even",
        },
        {
            "response": "First I calculate the derivative... The final answer is \\boxed{convex}.",
            "choices": ["convex", "concave", "neither"],
            "expected": "convex",
        },
        {
            "response": "No boxed content here, just regular text with answer D",
            "choices": ["A", "B", "C", "D"],
            "expected": None,  # Should fall back to regular parsing
        },
    ]

    print("=== Testing \\boxed{} Pattern Extraction ===\n")

    for i, case in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"  Response: {case['response']}")
        print(f"  Choices: {case['choices']}")

        # Extract boxed content
        boxed_pattern = r"\\boxed\{([^}]*)\}"
        boxed_matches = re.findall(boxed_pattern, case["response"], re.IGNORECASE)

        if boxed_matches:
            boxed_content = boxed_matches[-1].strip()
            print(f"  Boxed content found: '{boxed_content}'")

            # Try to match to choices
            result = None
            for choice in case["choices"]:
                if choice.lower() == boxed_content.lower():
                    result = choice
                    break

            if not result:
                for choice in case["choices"]:
                    if choice.lower() in boxed_content.lower():
                        result = choice
                        break

            print(f"  Matched choice: {result}")
            print(f"  Expected: {case['expected']}")

            if result == case["expected"]:
                print("  ✅ PASS\n")
            else:
                print("  ❌ FAIL\n")
        else:
            print("  No \\boxed{} content found")
            print(f"  Expected: {case['expected']}")
            if case["expected"] is None:
                print("  ✅ PASS (no boxed content expected)\n")
            else:
                print("  ❌ FAIL (boxed content was expected)\n")


if __name__ == "__main__":
    test_boxed_parsing()
