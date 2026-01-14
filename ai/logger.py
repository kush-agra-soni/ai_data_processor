# ai/logger.py

import sys
import time
from typing import Optional, List, Dict, Any


# ==========================================================
# Log levels
# ==========================================================
INFO = "INFO"
WARN = "WARN"
ERROR = "ERROR"
DEBUG = "DEBUG"


# ==========================================================
# Base AI Logger
# ==========================================================
class AILogger:
    """
    Central AI logger abstraction.

    - Works for CLI, Streamlit, CI, cron
    - Real-time streaming
    - Structured + human-readable
    """

    def __init__(
        self,
        stream=sys.stdout,
        level: str = INFO,
        persist: bool = False,
        log_file: Optional[str] = None
    ):
        self.stream = stream
        self.level = level
        self.persist = persist
        self.log_file = log_file
        self._buffer: List[str] = []

    # ----------------------------
    # Core logging
    # ----------------------------
    def _emit(self, level: str, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{level}] {message}"

        self.stream.write(line + "\n")
        self.stream.flush()

        self._buffer.append(line)

        if self.persist and self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ----------------------------
    # Public helpers
    # ----------------------------
    def info(self, message: str):
        self._emit(INFO, message)

    def warn(self, message: str):
        self._emit(WARN, message)

    def error(self, message: str):
        self._emit(ERROR, message)

    def debug(self, message: str):
        if self.level == DEBUG:
            self._emit(DEBUG, message)

    # ======================================================
    # AI-specific helpers
    # ======================================================
    def ai_plan(self, decision: Dict[str, Any]):
        self.info(
            f"AI plan generated | confidence={decision.get('confidence')}"
        )

    def ai_step_start(self, step: Dict[str, Any]):
        self.info(
            f"STEP START → {step['name']} | "
            f"risk={step.get('risk', 'n/a')} | "
            f"reason={step.get('reason', 'n/a')}"
        )

    def ai_step_end(self, step_name: str):
        self.info(f"STEP END → {step_name}")

    def ai_blocked(self, step_name: str, reason: str):
        self.warn(f"STEP BLOCKED → {step_name} | {reason}")

    def ai_failure(self, step_name: str, error: Exception):
        self.error(f"STEP FAILED → {step_name} | {error}")

    def ai_validation(self, report: Dict[str, Any]):
        status = "PASSED" if report.get("is_valid") else "FAILED"
        self.info(f"VALIDATION {status}")

    # ----------------------------
    # Access logs
    # ----------------------------
    def get_logs(self) -> List[str]:
        return self._buffer.copy()


# ==========================================================
# Streamlit-compatible logger
# ==========================================================
class StreamlitAILogger(AILogger):
    """
    Logger adapter for Streamlit UI.
    """

    def __init__(self, placeholder, level: str = INFO):
        super().__init__(stream=sys.stdout, level=level)
        self.placeholder = placeholder
        self._ui_buffer: List[str] = []

    def _emit(self, level: str, message: str):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] [{level}] {message}"
        self._ui_buffer.append(line)
        self.placeholder.code("\n".join(self._ui_buffer))


# ==========================================================
# Factory
# ==========================================================
def get_logger(**kwargs) -> AILogger:
    """
    Factory for AILogger.
    """
    return AILogger(**kwargs)
