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


class BaseModel(ABC):
    """Abstract base class for all models"""

    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.rate_limit_delay = 1.0  # seconds between API calls
        self._parser_client = None  # Will be initialized when needed
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

    def _parse_choice(self, response: str, choices: List[str]) -> str:
        """Parse model response to extract choice value using GPT-3.5-turbo"""
        try:
            client = self._get_parser_client()

            parsing_prompt = f"""You are a response parser. Given a model's response to a multiple choice question, extract the answer and reasoning.

Model Response: "{response}"

Available Choices: {choices}

Your task:
1. Determine which choice the model selected from the available options
2. Parse the response to extract the model's reasoning behind the choice. You can copy the reasoning directly from the response.

Respond with valid JSON in this exact format:
{{
    "answer": "<exact_choice_value>",
    "reasoning": "<brief explanation of how the model determined the choice>"
}}

The answer must be one of these exact values: {choices}"""

            parsing_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": parsing_prompt}],
            )

            result = parsing_response.choices[0].message.content.strip()

            # Parse the JSON response
            import json

            try:
                parsed = json.loads(result)
                choice_value = parsed.get("answer", choices[0])
                reasoning = parsed.get("reasoning", "No reasoning provided")

                logger.info(f"Parsed choice: {choice_value}, Reasoning: {reasoning}")

                # Validate choice value
                if choice_value in choices:
                    return choice_value
                else:
                    logger.warning(
                        f"Invalid choice value '{choice_value}', defaulting to first choice '{choices[0]}'"
                    )
                    return choices[0]

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from parser response: {result}")
                # Fallback to simple parsing
                return self._fallback_parse_choice(response, choices)

        except Exception as e:
            logger.error(f"Error in GPT-3.5 choice parsing: {e}")
            # Fallback to simple parsing
            return self._fallback_parse_choice(response, choices)

    def _fallback_parse_choice(self, response: str, choices: List[str]) -> str:
        """Fallback simple choice parsing when GPT-3.5 parser fails"""
        response_upper = response.upper().strip()
        response_lower = response.lower().strip()

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

    def __init__(self, model_name: str = "gpt-4o", api_key: str = None):
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
                prompt += f"\n\nChoices: {choices}\nPlease provide your reasoning and then clearly state your final answer as one of the given choices."

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

            if choices:
                prompt += f"\n\nChoices: {choices}\nPlease provide your reasoning and then clearly state your final answer as one of the given choices."

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

    def __init__(self, model_name: str = "gemini-2.0-flash-exp", api_key: str = None):
        super().__init__(model_name, api_key)
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
            if choices:
                prompt += f"\n\nChoices: {choices}\nPlease provide your reasoning and then clearly state your final answer as one of the given choices."

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
            if choices:
                prompt += f"\n\nChoices: {choices}\nPlease provide your reasoning and then clearly state your final answer as one of the given choices."

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
                prompt += f"\n\nChoices: {choices}\nPlease provide your reasoning and then clearly state your final answer as one of the given choices."

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

            if choices:
                prompt += f"\n\nChoices: {choices}\nPlease provide your reasoning and then clearly state your final answer as one of the given choices."

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
