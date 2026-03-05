from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_visualizations(df: pd.DataFrame) -> dict[str, go.Figure]:
    figures: dict[str, go.Figure] = {}

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    if numeric_cols:
        first_num = numeric_cols[0]
        figures["histogram"] = px.histogram(
            df,
            x=first_num,
            nbins=30,
            title=f"Distribution of {first_num}",
            template="plotly_white",
        )

    if len(numeric_cols) >= 2:
        figures["scatter"] = px.scatter(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
            template="plotly_white",
        )

    if categorical_cols and numeric_cols:
        grouped = (
            df.groupby(categorical_cols[0], dropna=False)[numeric_cols[0]]
            .mean()
            .reset_index()
            .sort_values(numeric_cols[0], ascending=False)
            .head(15)
        )
        figures["bar"] = px.bar(
            grouped,
            x=categorical_cols[0],
            y=numeric_cols[0],
            title=f"Average {numeric_cols[0]} by {categorical_cols[0]}",
            template="plotly_white",
        )

    if numeric_cols:
        idx_col = df.index.name if df.index.name else "index"
        line_df = df.reset_index().rename(columns={"index": idx_col})
        figures["line"] = px.line(
            line_df,
            x=idx_col,
            y=numeric_cols[0],
            title=f"Trend of {numeric_cols[0]}",
            template="plotly_white",
        )

    return figures
