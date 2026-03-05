from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatasetAnalysisResult:
    raw_shape: tuple[int, int]
    clean_shape: tuple[int, int]
    dropped_duplicates: int
    missing_values_filled: int


def load_dataset(file_path: str) -> pd.DataFrame:
    extension = Path(file_path).suffix.lower()

    if extension == ".csv":
        return pd.read_csv(file_path)

    if extension == ".xlsx":
        try:
            return pd.read_excel(file_path, engine="openpyxl")
        except ImportError as exc:
            raise ImportError(
                "Reading .xlsx files requires 'openpyxl'. Add 'openpyxl>=3.1.0' to dependencies."
            ) from exc

    if extension == ".xls":
        try:
            return pd.read_excel(file_path, engine="xlrd")
        except ImportError as exc:
            raise ImportError(
                "Reading .xls files requires 'xlrd'. Add 'xlrd>=2.0.1' to dependencies."
            ) from exc

    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, DatasetAnalysisResult]:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    before_dupes = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    dropped_duplicates = before_dupes - len(cleaned)

    missing_before = int(cleaned.isna().sum().sum())

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    categorical_cols = cleaned.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    for col in categorical_cols:
        mode = cleaned[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        cleaned[col] = cleaned[col].fillna(fill_value)

    missing_after = int(cleaned.isna().sum().sum())

    result = DatasetAnalysisResult(
        raw_shape=df.shape,
        clean_shape=cleaned.shape,
        dropped_duplicates=dropped_duplicates,
        missing_values_filled=missing_before - missing_after,
    )
    return cleaned, result


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    summary: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_by_column": df.isna().sum().to_dict(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    if numeric_cols:
        summary["numeric_describe"] = df[numeric_cols].describe().to_dict()

    return summary


def detect_patterns(df: pd.DataFrame) -> dict[str, Any]:
    numeric_df = df.select_dtypes(include=[np.number])
    patterns: dict[str, Any] = {
        "strong_correlations": [],
        "high_skew_columns": [],
    }

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True)
        for i, col1 in enumerate(corr.columns):
            for col2 in corr.columns[i + 1 :]:
                value = float(corr.loc[col1, col2])
                if abs(value) >= 0.6:
                    patterns["strong_correlations"].append(
                        {"feature_1": col1, "feature_2": col2, "correlation": value}
                    )

    if not numeric_df.empty:
        skew_series = numeric_df.skew(numeric_only=True)
        patterns["high_skew_columns"] = [
            {"column": col, "skew": float(value)}
            for col, value in skew_series.items()
            if abs(value) > 1.0
        ]

    return patterns
