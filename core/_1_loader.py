import os
import pandas as pd
from typing import Union

SUPPORTED_FORMATS = {
    ".csv": "csv",
    ".json": "json",
    ".xlsx": "excel"
}


def load_file(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from a supported file into a pandas DataFrame.

    Supported formats:
    - CSV
    - JSON
    - XLSX

    Args:
        file_path (str): Path to the data file
        **kwargs: Optional pandas loader arguments

    Returns:
        pd.DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported formats: {list(SUPPORTED_FORMATS.keys())}"
        )

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path, **kwargs)
        elif ext == ".json":
            df = pd.read_json(file_path, **kwargs)
        elif ext == ".xlsx":
            df = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError("Unreachable loader state")

    except Exception as e:
        raise RuntimeError(f"Failed to load file '{file_path}': {e}")

    if df.empty:
        raise ValueError("Loaded DataFrame is empty")

    return df


def load_from_database(
    query: str,
    connection: Union[str, object],
    **kwargs
) -> pd.DataFrame:
    """
    Execute SQL query and load result into a pandas DataFrame.

    Args:
        query (str): SQL query
        connection (str or DB connection): SQLAlchemy engine or DBAPI connection
        **kwargs: Optional pandas read_sql_query arguments

    Returns:
        pd.DataFrame
    """
    try:
        df = pd.read_sql_query(query, connection, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Database query failed: {e}")

    if df.empty:
        raise ValueError("Database query returned no rows")

    return df
