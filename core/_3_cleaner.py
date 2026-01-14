import pandas as pd
from typing import List


class Cleaner:
    """
    Deterministic structural and column-level cleaning.
    No logging, no prints, no side effects.
    """

    # ----------------------------
    # Cell-level normalization
    # ----------------------------
    def standardize_empty_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize all missing-like values to pandas NA.
        """
        df = df.copy()
        return df.applymap(lambda x: pd.NA if pd.isna(x) else x)

    # ----------------------------
    # Column name cleaning
    # ----------------------------
    def remove_special_characters(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", " ", regex=True)
        return df

    def convert_to_lowercase(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.lower()
        return df

    def remove_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip()
        return df

    def replace_space_with_underscore(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.replace(" ", "_", regex=False)
        return df

    # ----------------------------
    # Structural cleanup
    # ----------------------------
    def remove_empty_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        return df.dropna(axis=1, how="all")

    def remove_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        return df.loc[~df.isna().all(axis=1)]

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        return df.drop_duplicates().reset_index(drop=True)

    # ----------------------------
    # Identifier detection
    # ----------------------------
    def identifier_column_remover(
        self,
        df: pd.DataFrame,
        max_unique_ratio: float = 0.98,
        monotonic_only: bool = True
    ) -> pd.DataFrame:
        """
        Remove likely identifier columns (IDs, surrogate keys).

        Rules:
        - Numeric
        - High uniqueness
        - Monotonic increasing (optional)
        """
        df = df.copy()
        cols_to_drop: List[str] = []

        row_count = len(df)

        for col in df.columns:
            series = df[col]

            if not pd.api.types.is_integer_dtype(series):
                continue

            if series.isna().any():
                continue

            unique_ratio = series.nunique() / max(row_count, 1)

            if unique_ratio >= max_unique_ratio:
                if monotonic_only and not series.is_monotonic_increasing:
                    continue
                cols_to_drop.append(col)

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        return df

    # ----------------------------
    # Canonical execution entry
    # ----------------------------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Canonical cleaner entry point.
        Execution order is fixed and deterministic.
        """
        df = self.standardize_empty_cells(df)

        df = self.remove_special_characters(df)
        df = self.convert_to_lowercase(df)
        df = self.remove_whitespace(df)
        df = self.replace_space_with_underscore(df)

        df = self.identifier_column_remover(df)
        df = self.remove_empty_columns(df)
        df = self.remove_empty_rows(df)
        df = self.remove_duplicates(df)

        return df
