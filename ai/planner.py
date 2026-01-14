# ai/planner.py

import json
from typing import Dict, Any

from ai.llm_client import GeminiClient, get_llm_client
from ai.decision_schema import validate_decision
from ai.profiler import DataProfiler, get_profiler


# ==========================================================
# System prompt (hard constraints)
# ==========================================================
SYSTEM_PROMPT = """
You are an automated data-engineering planner.

Your task:
- Analyze the provided dataset profile
- Propose an ordered list of pipeline steps
- Use ONLY the allowed step names
- Provide parameters only when necessary
- Assign a risk level to each step
- Output ONLY valid JSON that conforms to the provided schema

Rules:
- Do NOT invent steps
- Do NOT reference code or libraries
- Do NOT output explanations outside JSON
- Prefer safe, minimal pipelines
- Avoid destructive steps unless clearly justified
- Always end with 'validate'
"""


# ==========================================================
# Planner
# ==========================================================
class AIPlanner:
    """
    Orchestrates:
    DataProfiler → Gemini → Decision Schema validation
    """

    def __init__(
        self,
        llm_client: GeminiClient | None = None,
        profiler: DataProfiler | None = None
    ):
        self.llm = llm_client or get_llm_client()
        self.profiler = profiler or get_profiler()

    # ----------------------------
    # Public entry
    # ----------------------------
    def plan(self, df) -> Dict[str, Any]:
        """
        Generate a validated execution plan for the given DataFrame.
        """
        profile = self.profiler.profile(df)
        user_prompt = self._build_user_prompt(profile)

        raw = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2
        )

        decision = self._parse_json(raw)
        validate_decision(decision)

        return decision

    # ======================================================
    # Internals
    # ======================================================
    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """
        Strict JSON parsing with hard failure on errors.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM returned invalid JSON: {e}\nRaw output:\n{text}"
            )

    @staticmethod
    def _build_user_prompt(profile: Dict[str, Any]) -> str:
        """
        Convert profile to a compact JSON prompt.
        """
        return json.dumps(
            {
                "dataset_profile": profile,
                "instructions": {
                    "objective": "prepare clean, model-ready dataset",
                    "constraints": {
                        "fully_automated": True,
                        "prefer_non_destructive": True,
                        "avoid_data_loss": True
                    }
                }
            },
            indent=2
        )


# ==========================================================
# Factory
# ==========================================================
def get_planner(**kwargs) -> AIPlanner:
    """
    Factory method for AIPlanner.
    """
    return AIPlanner(**kwargs)
