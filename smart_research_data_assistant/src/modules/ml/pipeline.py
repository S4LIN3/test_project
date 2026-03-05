from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class MLResult:
    task_type: str
    leaderboard: pd.DataFrame
    best_model_name: str
    best_model_pipeline: Pipeline
    feature_columns: list[str]


def _detect_task_type(target_series: pd.Series) -> str:
    if target_series.dtype == "object" or target_series.nunique(dropna=True) <= 20:
        return "classification"
    return "regression"


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def _classification_models() -> dict[str, Any]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
    }


def _regression_models() -> dict[str, Any]:
    return {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    }


def train_models(df: pd.DataFrame, target_column: str, test_size: float = 0.2) -> MLResult:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not present in dataframe.")

    data = df.dropna(subset=[target_column]).copy()
    X = data.drop(columns=[target_column])
    y = data[target_column]

    task_type = _detect_task_type(y)
    preprocessor = _build_preprocessor(X)

    if task_type == "classification":
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        models = _classification_models()
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        models = _regression_models()

    rows: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_model_name = ""
    best_pipeline: Pipeline | None = None

    for model_name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        if task_type == "classification":
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")
            score = float(f1)
            rows.append({"model": model_name, "accuracy": acc, "f1_weighted": f1})
        else:
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            r2 = r2_score(y_test, preds)
            score = float(r2)
            rows.append({"model": model_name, "rmse": rmse, "r2": r2})

        if score > best_score:
            best_score = score
            best_model_name = model_name
            best_pipeline = pipeline

    leaderboard = pd.DataFrame(rows)
    if task_type == "classification":
        leaderboard = leaderboard.sort_values("f1_weighted", ascending=False)
    else:
        leaderboard = leaderboard.sort_values("r2", ascending=False)

    if best_pipeline is None:
        raise RuntimeError("Training failed to produce a model.")

    return MLResult(
        task_type=task_type,
        leaderboard=leaderboard,
        best_model_name=best_model_name,
        best_model_pipeline=best_pipeline,
        feature_columns=X.columns.tolist(),
    )


def predict(best_model: Pipeline, feature_df: pd.DataFrame) -> np.ndarray:
    return best_model.predict(feature_df)


def train_with_pycaret(df: pd.DataFrame, target_column: str) -> tuple[str, Any] | None:
    try:
        if _detect_task_type(df[target_column]) == "classification":
            from pycaret.classification import compare_models, setup
        else:
            from pycaret.regression import compare_models, setup
    except Exception:
        return None

    setup(
        data=df,
        target=target_column,
        session_id=42,
        verbose=False,
        html=False,
    )
    model = compare_models()
    return ("pycaret_best_model", model)

