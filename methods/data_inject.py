# methods/data_inject.py

from typing import Optional, Dict, Any
import os

import pandas as pd

from core._1_loader import load_file, load_from_database
from ai.planner import get_planner
from ai.executor import get_executor
from ai.logger import AILogger


# ==========================================================
# Manual file-based injection
# ==========================================================
def inject_from_file(
    file_path: str,
    use_ai: bool = True,
    target_column: Optional[str] = None,
    strict_mode: bool = True,
    logger: Optional[AILogger] = None
) -> Dict[str, Any]:
    """
    Inject data manually from CSV / JSON / XLSX.

    Returns:
        execution result dict
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    logger = logger or AILogger()
    logger.info(f"Injecting data from file: {file_path}")

    df = load_file(file_path)
    logger.info(f"Loaded dataset with shape {df.shape}")

    if not use_ai:
        logger.warn("AI disabled — returning raw dataset only")
        return {"data": df}

    planner = get_planner()
    executor = get_executor(logger=logger, strict_mode=strict_mode)

    logger.info("Generating AI execution plan")
    decision = planner.plan(df)
    logger.ai_plan(decision)

    result = executor.execute(
        df=df,
        decision=decision,
        target_column=target_column
    )

    logger.info("Data injection pipeline completed")
    return result


# ==========================================================
# Database-based injection (scheduled / cron)
# ==========================================================
def inject_from_database(
    query: str,
    connection,
    use_ai: bool = True,
    target_column: Optional[str] = None,
    strict_mode: bool = True,
    logger: Optional[AILogger] = None
) -> Dict[str, Any]:
    """
    Inject data from a database query.

    Intended for scheduled / automated runs.
    """
    logger = logger or AILogger()
    logger.info("Injecting data from database")

    df = load_from_database(query, connection)
    logger.info(f"Loaded dataset with shape {df.shape}")

    if not use_ai:
        logger.warn("AI disabled — returning raw dataset only")
        return {"data": df}

    planner = get_planner()
    executor = get_executor(logger=logger, strict_mode=strict_mode)

    logger.info("Generating AI execution plan")
    decision = planner.plan(df)
    logger.ai_plan(decision)

    result = executor.execute(
        df=df,
        decision=decision,
        target_column=target_column
    )

    logger.info("Database injection pipeline completed")
    return result
