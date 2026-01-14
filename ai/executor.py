# ai/executor.py

from typing import Dict, Any, Callable
import sys
import time

import pandas as pd

from core.pipeline import get_pipeline
from ai.decision_schema import is_high_risk


# ==========================================================
# Terminal logger (real-time, human readable)
# ==========================================================
class ExecutorLogger:
    def __init__(self, stream=sys.stdout):
        self.stream = stream

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.stream.write(f"[{timestamp}] {message}\n")
        self.stream.flush()

    def step_start(self, step: Dict[str, Any]):
        self.log(
            f"▶ STEP: {step['name']} | "
            f"risk={step.get('risk', 'unknown')} | "
            f"reason={step.get('reason', 'n/a')}"
        )

    def step_end(self, step_name: str):
        self.log(f"✔ COMPLETED: {step_name}")

    def warn(self, message: str):
        self.log(f"⚠ WARNING: {message}")

    def error(self, message: str):
        self.log(f"✖ ERROR: {message}")


# ==========================================================
# Executor
# ==========================================================
class AIExecutor:
    """
    Executes an AI decision plan against the pipeline.
    """

    def __init__(
        self,
        pipeline=None,
        logger: ExecutorLogger | None = None,
        strict_mode: bool = True
    ):
        self.pipeline = pipeline or get_pipeline()
        self.logger = logger or ExecutorLogger()
        self.strict_mode = strict_mode

        # step registry (hard-coded safety)
        self._registry: Dict[str, Callable] = {
            "detect": self._detect,
            "clean": self._clean,
            "standardize": self._standardize,
            "handle_outliers": self._handle_outliers,
            "impute": self._impute,
            "extract_dates": self._extract_dates,
            "enforce_types": self._enforce_types,
            "auto_convert_types": self._auto_convert_types,
            "encode": self._encode,
            "select_features": self._select_features,
            "feature_importance": self._feature_importance,
            "scale": self._scale,
            "resample": self._resample,
            "validate": self._validate,
        }

    # ----------------------------
    # Public entry
    # ----------------------------
    def execute(
        self,
        df: pd.DataFrame,
        decision: Dict[str, Any],
        target_column: str | None = None
    ) -> Dict[str, Any]:
        """
        Execute a validated AI decision.
        """
        self.logger.log("🚀 Starting AI-driven pipeline execution")

        if self.strict_mode and is_high_risk(decision):
            self.logger.warn(
                "High-risk steps detected. "
                "Execution continues because strict_mode=True only blocks INVALID steps."
            )

        context: Dict[str, Any] = {
            "df": df,
            "X": None,
            "y": None,
            "artifacts": {},
            "reports": [],
        }

        if target_column and target_column in df.columns:
            context["y"] = df[target_column]
            context["X"] = df.drop(columns=[target_column])

        for step in decision["steps"]:
            name = step["name"]
            params = step.get("params", {})

            if name not in self._registry:
                raise RuntimeError(f"Step '{name}' not registered")

            self.logger.step_start(step)

            try:
                self._registry[name](context, params)
                self.logger.step_end(name)
            except Exception as e:
                self.logger.error(f"{name} failed: {e}")
                if self.strict_mode:
                    raise
                else:
                    self.logger.warn("Continuing due to strict_mode=False")

        self.logger.log("🏁 Pipeline execution finished")

        return {
            "data": context.get("df"),
            "X": context.get("X"),
            "y": context.get("y"),
            "artifacts": context["artifacts"],
            "reports": context["reports"],
        }

    # ======================================================
    # Step handlers
    # ======================================================
    def _detect(self, ctx, params):
        df, meta = self.pipeline.detect(ctx["df"])
        ctx["df"] = df
        ctx["artifacts"]["detection"] = meta

    def _clean(self, ctx, params):
        ctx["df"] = self.pipeline.clean(ctx["df"])

    def _standardize(self, ctx, params):
        ctx["df"] = self.pipeline.standardize(ctx["df"])

    def _handle_outliers(self, ctx, params):
        df, meta = self.pipeline.handle_outliers(ctx["df"], **params)
        ctx["df"] = df
        ctx["artifacts"]["outliers"] = meta

    def _impute(self, ctx, params):
        df, meta = self.pipeline.impute(ctx["df"], **params)
        ctx["df"] = df
        ctx["artifacts"]["imputation"] = meta

    def _extract_dates(self, ctx, params):
        df, meta = self.pipeline.extract_dates(ctx["df"])
        ctx["df"] = df
        ctx["artifacts"]["date_features"] = meta

    def _enforce_types(self, ctx, params):
        df, meta = self.pipeline.enforce_types(ctx["df"], params.get("type_map", {}))
        ctx["df"] = df
        ctx["artifacts"]["type_enforcement"] = meta

    def _auto_convert_types(self, ctx, params):
        df, meta = self.pipeline.auto_convert_types(ctx["df"])
        ctx["df"] = df
        ctx["artifacts"]["auto_type_conversion"] = meta

    def _encode(self, ctx, params):
        df, meta = self.pipeline.encode(ctx["df"], **params)
        ctx["df"] = df
        ctx["artifacts"]["encoding"] = meta

        # update X if target exists
        if ctx.get("y") is not None:
            ctx["X"] = df.drop(columns=[ctx["y"].name], errors="ignore")

    def _select_features(self, ctx, params):
        if ctx["X"] is None or ctx["y"] is None:
            raise RuntimeError("Feature selection requires X and y")

        X, meta = self.pipeline.select_features(ctx["X"], ctx["y"], **params)
        ctx["X"] = X
        ctx["artifacts"]["feature_selection"] = meta

    def _feature_importance(self, ctx, params):
        if ctx["X"] is None or ctx["y"] is None:
            raise RuntimeError("Feature importance requires X and y")

        df, meta = self.pipeline.feature_importance(ctx["X"], ctx["y"], **params)
        ctx["artifacts"]["feature_importance"] = {
            "ranking": df.to_dict(orient="records"),
            "meta": meta,
        }

    def _scale(self, ctx, params):
        if ctx["X"] is None:
            raise RuntimeError("Scaling requires X")

        X_scaled, meta = self.pipeline.scale(ctx["X"], **params)
        ctx["X"] = X_scaled
        ctx["artifacts"]["scaling"] = meta

    def _resample(self, ctx, params):
        if ctx["X"] is None or ctx["y"] is None:
            raise RuntimeError("Resampling requires X and y")

        Xr, yr, meta = self.pipeline.resample(ctx["X"], ctx["y"], **params)
        ctx["X"], ctx["y"] = Xr, yr
        ctx["artifacts"]["resampling"] = meta

    def _validate(self, ctx, params):
        report = self.pipeline.validate(ctx["df"])
        ctx["reports"].append(report)

        if not report["is_valid"]:
            self.logger.warn("Validation failed — dataset still has issues")


# ==========================================================
# Factory
# ==========================================================
def get_executor(**kwargs) -> AIExecutor:
    """
    Factory for AIExecutor.
    """
    return AIExecutor(**kwargs)
