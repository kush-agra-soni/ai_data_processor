# ai/llm_client.py

import os
import time
from typing import Dict, Any, Optional

import google.generativeai as genai


# ==========================================================
# Gemini LLM Client
# ==========================================================
class GeminiClient:
    """
    Thin, safe wrapper around Gemini API.
    - Stateless
    - Retry-aware
    - Timeout-guarded
    - JSON-only output expected
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        timeout_seconds: int = 30
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "Gemini API key not found. "
                "Set GEMINI_API_KEY env variable."
            )

        genai.configure(api_key=self.api_key)

        self.model = genai.GenerativeModel(model_name)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.timeout_seconds = timeout_seconds

    # ----------------------------
    # Public call
    # ----------------------------
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2
    ) -> str:
        """
        Generate a response from Gemini.

        Returns:
            raw text response (expected to be JSON)
        """
        prompt = self._build_prompt(system_prompt, user_prompt)

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": 2048,
                    }
                )

                text = response.text
                if not text:
                    raise RuntimeError("Empty response from Gemini")

                return text.strip()

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff ** attempt)
                else:
                    break

        raise RuntimeError(
            f"Gemini request failed after {self.max_retries} attempts: {last_error}"
        )

    # ----------------------------
    # Prompt assembly
    # ----------------------------
    @staticmethod
    def _build_prompt(system_prompt: str, user_prompt: str) -> str:
        """
        Combine system + user prompt into a single instruction.
        """
        return f"""
SYSTEM INSTRUCTIONS:
{system_prompt}

USER INPUT:
{user_prompt}

IMPORTANT:
- Respond with ONLY valid JSON
- Do NOT include markdown
- Do NOT include explanations outside JSON
""".strip()


# ==========================================================
# Factory
# ==========================================================
def get_llm_client(**kwargs) -> GeminiClient:
    """
    Factory method for GeminiClient.
    """
    return GeminiClient(**kwargs)
