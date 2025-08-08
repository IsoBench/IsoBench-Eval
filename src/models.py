"""
Model implementations for IsoBench evaluation.

This module contains all foundation model implementations including OpenAI GPT,
Google Gemini, and Anthropic Claude models.
"""

import os
import base64
import io
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Union
from PIL import Image

logger = logging.getLogger(__name__)


# Define structured output schema for Gemini parser
def get_choice_parser_schema():
    """Get the JSON schema for choice parsing"""
    return {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The selected choice from the available options",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of how the choice was determined",
            },
        },
        "required": ["answer", "reasoning"],
        "propertyOrdering": ["answer", "reasoning"],
    }


class BaseModel(ABC):
    """Abstract base class for all models"""

    def __init__(
        self, model_name: str, api_key: str = None, parser_model: str = "gpt-3.5"
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.rate_limit_delay = 0.1  # seconds between API calls
        self.parser_model = parser_model  # "gpt-3.5" or "gemini-2.5-flash-lite"
        self._parser_client = None  # Will be initialized when needed
        self._gemini_parser_client = None  # For Gemini parser
        self._last_response = ""  # Store last raw response for logging

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

    def _get_parser_client(self):
        """Get or create GPT-3.5-turbo client for parsing"""
        if self._parser_client is None:
            try:
                import openai

                self._parser_client = openai.OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY") or self.api_key
                )
            except ImportError:
                raise ImportError(
                    "OpenAI package required for choice parsing: pip install openai"
                )
        return self._parser_client

    def _get_gemini_parser_client(self):
        """Get or create Gemini-2.5-flash-lite client for parsing"""
        if self._gemini_parser_client is None:
            try:
                from google import genai
                from google.genai.types import GenerateContentConfig

                api_key = (
                    os.getenv("GEMINI_API_KEY")
                    or os.getenv("GOOGLE_API_KEY")
                    or self.api_key
                )
                if not api_key:
                    raise ValueError("Gemini API key required for gemini parser")

                self._gemini_parser_client = genai.Client(api_key=api_key)

            except ImportError:
                raise ImportError(
                    "Google GenerativeAI package required for gemini parsing: pip install google-generativeai"
                )
        return self._gemini_parser_client

    def _parse_choice(self, response: str, choices: List[str]) -> str:
        """Parse model response to extract choice value using configured parser"""
        if self.parser_model == "gemini-2.5-flash-lite":
            return self._parse_choice_gemini(response, choices)
        else:
            return self._parse_choice_gpt(response, choices)

    def _get_parsing_prompt(self, response: str, choices: List[str]) -> str:
        """Generate parsing prompt based on task type"""
        # Check if this is a chess puzzle task (binary choice with chess moves)
        is_chess_puzzle = (
            len(choices) == 2
            and "no-move" in choices
            and any(len(choice) >= 4 and choice != "no-move" for choice in choices)
        )

        if is_chess_puzzle:
            return f"""You are a chess move parser. Given a model's response to a chess puzzle, extract the chess move.

Model Response: "{response}"

Available Choices: {choices}

Your task:
1. First, check if there is a \\boxed{{}} expression in the response. If present, use ONLY what's inside the \\boxed{{}} as the model's final answer
2. Look for chess moves in Algebraic Coordinate Notation (e.g., "d2d1", "e5a1", "c4f4") in the response (or in the \\boxed{{}} content if present)
3. If you find a chess move, select it
4. If no valid chess move is found or mentioned, select "no-move"

The answer must be one of these exact values: {choices}"""
        else:
            return f"""You are a response parser. Given a model's response to a multiple choice question, extract the answer and reasoning.

Model Response: "{response}"

Available Choices: {choices}

Your task:
1. First, check if there is a \\boxed{{}} expression in the response. If present, use ONLY what's inside the \\boxed{{}} as the model's final answer and compare it to the available choices
2. If no \\boxed{{}} is present, determine which choice the model selected from the available options using the full response
3. Parse the response to extract the model's reasoning behind the choice. You can copy the reasoning directly from the response.

The answer must be one of these exact values: {choices}"""

    def _process_parsed_response(
        self, result: str, response: str, choices: List[str], parser_type: str
    ) -> str:
        """Process parsed response and validate choice"""
        import json

        try:
            parsed = json.loads(result)
            choice_value = parsed.get("answer", choices[0])
            reasoning = parsed.get("reasoning", "No reasoning provided")

            logger.info(
                f"{parser_type} parsed choice: {choice_value}, Reasoning: {reasoning}"
            )

            # Validate choice value
            if choice_value in choices:
                return choice_value
            else:
                logger.warning(
                    f"Invalid choice value '{choice_value}', defaulting to first choice '{choices[0]}'"
                )
                return choices[0]

        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse JSON from {parser_type} parser response: {result}"
            )
            # Fallback to simple parsing
            return self._fallback_parse_choice(response, choices)

    def _parse_choice_gemini(self, response: str, choices: List[str]) -> str:
        """Parse model response to extract choice value using Gemini-2.5-flash-lite with structured output"""
        try:
            from google.genai.types import GenerateContentConfig

            client = self._get_gemini_parser_client()
            parsing_prompt = self._get_parsing_prompt(response, choices)

            # Configure structured output
            config = GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=get_choice_parser_schema(),
            )

            # Make the request with structured output
            parsing_response = client.models.generate_content(
                model="gemini-2.5-flash-lite", contents=parsing_prompt, config=config
            )

            result = parsing_response.text.strip()
            return self._process_parsed_response(result, response, choices, "Gemini")

        except Exception as e:
            logger.error(f"Error in Gemini-2.5-flash-lite choice parsing: {e}")
            # Fallback to simple parsing
            return self._fallback_parse_choice(response, choices)

    def _parse_choice_gpt(self, response: str, choices: List[str]) -> str:
        """Parse model response to extract choice value using GPT-3.5-turbo"""
        try:
            client = self._get_parser_client()
            parsing_prompt = self._get_parsing_prompt(response, choices)

            # Add JSON format instruction for GPT
            parsing_prompt += """

Respond with valid JSON in this exact format:
{
    "answer": "<exact_choice_value>",
    "reasoning": "<brief explanation of how you identified the choice>"
}"""

            parsing_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": parsing_prompt}],
            )

            result = parsing_response.choices[0].message.content.strip()
            return self._process_parsed_response(result, response, choices, "GPT")

        except Exception as e:
            logger.error(f"Error in GPT-3.5 choice parsing: {e}")
            # Fallback to simple parsing
            return self._fallback_parse_choice(response, choices)

    def _fallback_parse_choice(self, response: str, choices: List[str]) -> str:
        """Fallback simple choice parsing when structured parser fails"""
        response_upper = response.upper().strip()
        response_lower = response.lower().strip()

        # First, check for \boxed{} content - prioritize this as the final answer
        import re

        boxed_pattern = r"\\boxed\{([^}]*)\}"
        boxed_matches = re.findall(boxed_pattern, response, re.IGNORECASE)

        if boxed_matches:
            # Use the last \boxed{} content as the final answer
            boxed_content = boxed_matches[-1].strip()
            logger.info(f"Found \\boxed{{}} content: {boxed_content}")

            # Try to match boxed content to choices
            for choice in choices:
                if choice.lower() == boxed_content.lower():
                    return choice

            # If exact match fails, check if boxed content contains any choice
            for choice in choices:
                if choice.lower() in boxed_content.lower():
                    return choice

        # Special handling for chess puzzle tasks
        is_chess_puzzle = (
            len(choices) == 2
            and "no-move" in choices
            and any(len(choice) >= 4 and choice != "no-move" for choice in choices)
        )

        if is_chess_puzzle:
            # Find the chess move in choices (not "no-move")
            chess_move = next(
                (choice for choice in choices if choice != "no-move"), None
            )
            if chess_move:
                # Look for chess move pattern in response (4+ characters like "e6e7")
                chess_pattern = r"[a-h][1-8][a-h][1-8]"
                matches = re.findall(chess_pattern, response_lower)

                for match in matches:
                    if match == chess_move.lower():
                        return chess_move

                # Also check for exact match
                if chess_move.lower() in response_lower:
                    return chess_move

            # If no chess move found, default to "no-move"
            return "no-move"

        # First, try to find exact matches (case insensitive)
        for choice in choices:
            if choice.lower() in response_lower:
                return choice

        # Check for letter choices (A, B, C, D) mapped to indices
        letters = ["A", "B", "C", "D"]
        for i, letter in enumerate(letters[: len(choices)]):
            if letter in response_upper:
                return choices[i]

        # Check for number choices (0, 1, 2, 3) mapped to indices
        for i in range(len(choices)):
            if str(i) in response:
                return choices[i]

        # Default to first choice if parsing fails
        logger.warning(f"Could not parse choice from response: {response}")
        return choices[0]


class OpenAIModel(BaseModel):
    """OpenAI GPT model implementation"""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str = None,
        parser_model: str = "gpt-3.5",
    ):
        super().__init__(model_name, api_key, parser_model)
        try:
            import openai

            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")

    def predict_text(self, prompt: str, choices: List[str] = None) -> Union[str, int]:
        """Make prediction with text input"""
        try:

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )

            result = response.choices[0].message.content.strip()
            self._last_response = result  # Store raw response for logging
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

            response = self.client.chat.completions.create(
                model=self.model_name,
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
            )

            result = response.choices[0].message.content.strip()
            self._last_response = result  # Store raw response for logging
            self._apply_rate_limit()

            if choices:
                return self._parse_choice(result, choices)
            return result

        except Exception as e:
            logger.error(f"OpenAI Vision API error: {e}")
            return -1


class GeminiModel(BaseModel):
    """Google Gemini model implementation"""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-exp",
        api_key: str = None,
        parser_model: str = "gpt-3.5",
    ):
        super().__init__(model_name, api_key, parser_model)
        try:
            from google import genai
            from google.genai.types import GenerateContentConfig
        except ImportError:
            raise ImportError(
                "Google GenerativeAI package not installed. Run: pip install google-generativeai"
            )

        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "Google API key not provided. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable or pass api_key parameter."
            )

        # Configuration for generation
        config_params = {}
        self.config = GenerateContentConfig(**config_params)
        self.gemini_client = genai.Client(api_key=api_key)

    def predict_text(self, prompt: str, choices: List[str] = None) -> Union[str, int]:
        """Make prediction with text input"""
        try:

            response = self.gemini_client.models.generate_content(
                model=self.model_name, contents=prompt, config=self.config
            )
            result = response.text.strip()
            self._last_response = result  # Store raw response for logging
            self._apply_rate_limit()

            if choices:
                return self._parse_choice(result, choices)
            return result

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            # Add retry logic for rate limiting
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                logger.info("Rate limited, waiting and retrying...")
                time.sleep(2)
                try:
                    response = self.gemini_client.models.generate_content(
                        model=self.model_name, contents=prompt, config=self.config
                    )
                    result = response.text.strip()
                    self._last_response = result  # Store raw response for logging
                    return self._parse_choice(result, choices) if choices else result
                except Exception as e2:
                    logger.error(f"Retry failed: {e2}")
            return -1

    def predict_image_text(
        self, image: Image.Image, prompt: str, choices: List[str] = None
    ) -> Union[str, int]:
        """Make prediction with image + text input"""
        try:
            # Convert PIL image to format Gemini expects
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Create multimodal content
            contents = [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": img_str}},
            ]

            response = self.gemini_client.models.generate_content(
                model=self.model_name, contents=contents, config=self.config
            )
            result = response.text.strip()
            self._last_response = result  # Store raw response for logging
            self._apply_rate_limit()

            if choices:
                return self._parse_choice(result, choices)
            return result

        except Exception as e:
            logger.error(f"Gemini Vision API error: {e}")
            # Add retry logic for rate limiting
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                logger.info("Rate limited, waiting and retrying...")
                time.sleep(2)
                try:
                    response = self.gemini_client.models.generate_content(
                        model=self.model_name, contents=contents, config=self.config
                    )
                    result = response.text.strip()
                    self._last_response = result  # Store raw response for logging
                    return self._parse_choice(result, choices) if choices else result
                except Exception as e2:
                    logger.error(f"Retry failed: {e2}")
            return -1


class ClaudeModel(BaseModel):
    """Anthropic Claude model implementation"""

    def __init__(
        self,
        model_name: str = "claude-3-opus-20240229",
        api_key: str = None,
        parser_model: str = "gpt-3.5",
    ):
        super().__init__(model_name, api_key, parser_model)
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

            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )

            result = response.content[0].text.strip()
            self._last_response = result  # Store raw response for logging
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

            response = self.client.messages.create(
                model=self.model_name,
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
            self._last_response = result  # Store raw response for logging
            self._apply_rate_limit()

            if choices:
                return self._parse_choice(result, choices)
            return result

        except Exception as e:
            logger.error(f"Claude Vision API error: {e}")
            return -1
