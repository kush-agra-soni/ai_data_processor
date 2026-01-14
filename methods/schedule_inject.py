# methods/schedule_inject.py

"""
Scheduled injection runner.

Designed for:
- cron jobs
- GitHub Actions
- headless servers

Responsibilities:
- load config
- connect to DB
- call data injection
- exit with proper status codes
"""

import os
import sys
import yaml
import argparse
from typing import Dict, Any, Optional

import psycopg2

from methods.data_inject import inject_from_database
from ai.logger import AILogger


# ==========================================================
# Config loaders
# ==========================================================
def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ==========================================================
# Database connection
# ==========================================================
def get_postgres_connection(db_cfg: Dict[str, Any]):
    return psycopg2.connect(
        host=db_cfg["host"],
        port=db_cfg.get("port", 5432),
        database=db_cfg["database"],
        user=db_cfg["user"],
        password=db_cfg["password"],
        connect_timeout=db_cfg.get("connect_timeout", 10),
    )


# ==========================================================
# Main scheduled runner
# ==========================================================
def run_scheduled_job(
    db_config_path: str,
    pipeline_config_path: Optional[str] = None,
    use_ai: bool = True,
    strict_mode: bool = True
):
    logger = AILogger()
    logger.info("Starting scheduled data injection job")

    # ----------------------------
    # Load DB config
    # ----------------------------
    db_cfg = load_yaml(db_config_path)
    query = db_cfg["query"]
    target_column = db_cfg.get("target_column")

    # ----------------------------
    # Connect to DB
    # ----------------------------
    logger.info("Connecting to database")
    conn = get_postgres_connection(db_cfg["connection"])

    try:
        result = inject_from_database(
            query=query,
            connection=conn,
            use_ai=use_ai,
            target_column=target_column,
            strict_mode=strict_mode,
            logger=logger
        )
        logger.info("Scheduled injection completed successfully")
        return result

    except Exception as e:
        logger.error(f"Scheduled injection failed: {e}")
        raise

    finally:
        conn.close()
        logger.info("Database connection closed")


# ==========================================================
# CLI entrypoint (cron / GH Actions)
# ==========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Scheduled Data Injection Runner"
    )

    parser.add_argument(
        "--db-config",
        required=True,
        help="Path to database YAML config"
    )

    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI planner/executor"
    )

    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Continue execution on step failures"
    )

    args = parser.parse_args()

    try:
        run_scheduled_job(
            db_config_path=args.db_config,
            use_ai=not args.no_ai,
            strict_mode=not args.non_strict
        )
        sys.exit(0)

    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
