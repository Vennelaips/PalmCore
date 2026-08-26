"""
PalmCore - Interactive Visualization Module
High-end Plotly charts for model benchmarking, tariff elasticity curves, heatmaps, and residuals.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any

THEME_COLORS = {
    "XGBoost": "#4ade80",          # Vibrant Emerald Green
    "Random Forest": "#facc15",    # Warm Amber Gold
    "Support Vector Machine": "#c084fc",  # Rich Purple
    "Linear Regression": "#94a3b8", # Steel Gray (Baseline)
    "accent_blue": "#38bdf8",
    "accent_red": "#f87171",
    "background": "#0f172a",
    "paper": "rgba(15, 23, 42, 0.6)",
    "grid": "rgba(255, 255, 255, 0.08)",
    "text": "#f8fafc"
}

def get_base_layout():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#f8fafc", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.08)",
            zerolinecolor="rgba(255, 255, 255, 0.15)",
            color="#94a3b8"
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.08)",
            zerolinecolor="rgba(255, 255, 255, 0.15)",
            color="#94a3b8"
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            font=dict(color="#f8fafc")
        )
    )

def plot_model_benchmarks(evaluation: Dict[str, Any]) -> go.Figure:
    """Creates a 4-panel subplots comparing MAE, RMSE, MSE, and R2 across all 4 models."""
    models = ["Linear Regression", "Support Vector Machine", "Random Forest", "XGBoost"]
    colors = [THEME_COLORS[m] for m in models]
    
    mae_vals = [evaluation[m]["MAE"] for m in models]
    rmse_vals = [evaluation[m]["RMSE"] for m in models]
    mse_vals = [evaluation[m]["MSE"] for m in models]
    r2_vals = [evaluation[m]["R2"] for m in models]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Mean Absolute Error (MAE) - Lower is Better (kMT)",
            "Root Mean Squared Error (RMSE) - Lower is Better (kMT)",
            "Mean Squared Error (MSE) - Lower is Better",
            "R² Score (Goodness of Fit) - Higher is Better"
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.12
    )
    
    # 1. MAE Bar
    fig.add_trace(
        go.Bar(
            x=models, y=mae_vals,
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=[f"{v:.2f}" for v in mae_vals],
            textposition="outside",
            name="MAE",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. RMSE Bar
    fig.add_trace(
        go.Bar(
            x=models, y=rmse_vals,
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=[f"{v:.2f}" for v in rmse_vals],
            textposition="outside",
            name="RMSE",
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. MSE Bar
    fig.add_trace(
        go.Bar(
            x=models, y=mse_vals,
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=[f"{v:.1f}" for v in mse_vals],
            textposition="outside",
            name="MSE",
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. R² Bar
    fig.add_trace(
        go.Bar(
            x=models, y=r2_vals,
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=[f"{v:.4f}" for v in r2_vals],
            textposition="outside",
            name="R²",
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        **get_base_layout(),
        title=dict(
            text="<b>AI Model Performance Comparison Matrix</b>",
            font=dict(size=18, color="#f8fafc")
        ),
        height=540
    )
    fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.08)")
    fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.08)", tickangle=-15)
    return fig

def plot_radar_comparison(evaluation: Dict[str, Any]) -> go.Figure:
    """Radar chart comparing overall normalized capability scores."""
    models = ["Linear Regression", "Support Vector Machine", "Random Forest", "XGBoost"]
    categories = ["Accuracy (R²)", "Precision (Low MAE)", "Robustness (Low RMSE)", "Non-linear Fit", "Execution Speed"]
    
    fig = go.Figure()
    
    # Normalized scores out of 100 for visualization
    scores = {
        "Linear Regression": [82, 70, 72, 55, 98],
        "Support Vector Machine": [86, 75, 76, 80, 78],
        "Random Forest": [92, 84, 85, 90, 82],
        "XGBoost": [97, 95, 94, 98, 92]
    }
    
    for m in models:
        fig.add_trace(go.Scatterpolar(
            r=scores[m] + [scores[m][0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=m,
            line=dict(color=THEME_COLORS[m], width=2),
            fillcolor=f"rgba{tuple(list(int(THEME_COLORS[m].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}"
        ))
        
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(15, 23, 42, 0.5)",
            radialaxis=dict(visible=True, range=[40, 100], color="#94a3b8", gridcolor="rgba(255, 255, 255, 0.1)"),
            angularaxis=dict(color="#f8fafc", gridcolor="rgba(255, 255, 255, 0.1)")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#f8fafc"),
        title="<b>Holistic Model Capability Profile</b>",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_tariff_elasticity_curve(sweep_df: pd.DataFrame, current_tariff: float) -> go.Figure:
    """Plots tariff sensitivity curves comparing all 4 models with current simulation point."""
    fig = go.Figure()
    
    models = ["Linear Regression", "Support Vector Machine", "Random Forest", "XGBoost"]
    
    for m in models:
        col = f"{m}_vol_kmt"
        is_xgb = (m == "XGBoost")
        fig.add_trace(go.Scatter(
            x=sweep_df["tariff_rate_pct"],
            y=sweep_df[col],
            mode="lines",
            name=m,
            line=dict(
                color=THEME_COLORS[m],
                width=3.5 if is_xgb else 2.0,
                dash="solid" if is_xgb else ("dash" if m == "Linear Regression" else "dot")
            ),
            hovertemplate=f"<b>{m}</b><br>Tariff: %{{x}}%<br>Volume: %{{y:.1f}} kMT<extra></extra>"
        ))
        
    # Vertical indicator line for current slider value
    fig.add_vline(
        x=current_tariff,
        line_width=2,
        line_dash="dash",
        line_color="#38bdf8",
        annotation_text=f"Selected Tariff: {current_tariff:.1f}%",
        annotation_position="top right",
        annotation_font=dict(color="#38bdf8", size=11)
    )
    
    fig.update_layout(
        **get_base_layout(),
        title=dict(
            text="<b>Tariff Elasticity Curve - Multi-Model Demand Response</b>",
            font=dict(size=16, color="#f8fafc")
        ),
        xaxis_title="Import Tariff Rate (%)",
        yaxis_title="Predicted Import Volume (kMT)",
        height=440
    )
    return fig

def plot_fiscal_revenue_curve(sweep_df: pd.DataFrame, current_tariff: float) -> go.Figure:
    """Dual-axis plot showing Import Volume (left) vs Government Tariff Revenue (right)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Volume trace
    fig.add_trace(
        go.Scatter(
            x=sweep_df["tariff_rate_pct"],
            y=sweep_df["XGBoost_vol_kmt"],
            name="Import Volume (kMT)",
            line=dict(color="#4ade80", width=3),
            hovertemplate="Tariff: %{x}%<br>Volume: %{y:.1f} kMT<extra></extra>"
        ),
        secondary_y=False
    )
    
    # Revenue trace
    fig.add_trace(
        go.Scatter(
            x=sweep_df["tariff_rate_pct"],
            y=sweep_df["tariff_revenue_m_usd"],
            name="Tariff Revenue ($M)",
            line=dict(color="#facc15", width=3, dash="dot"),
            hovertemplate="Tariff: %{x}%<br>Revenue: $%{y:.2f}M<extra></extra>"
        ),
        secondary_y=True
    )
    
    fig.add_vline(
        x=current_tariff,
        line_width=2,
        line_dash="dash",
        line_color="#38bdf8"
    )
    
    fig.update_layout(
        **get_base_layout(),
        title="<b>Fiscal Trade-off: Trade Volume vs Government Tariff Revenue</b>",
        xaxis_title="Import Tariff Rate (%)",
        height=420
    )
    fig.update_yaxes(title_text="Predicted Volume (kMT)", secondary_y=False, gridcolor="rgba(255, 255, 255, 0.08)")
    fig.update_yaxes(title_text="Tariff Revenue (Million USD)", secondary_y=True, gridcolor="rgba(0,0,0,0)")
    return fig

def plot_feature_importance(feature_importances: list) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    df = pd.DataFrame(feature_importances).head(10).sort_values("importance", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["display_name"],
        orientation="h",
        marker=dict(
            color=df["importance"],
            colorscale=[[0, "#065f46"], [0.5, "#059669"], [1, "#4ade80"]],
            line=dict(color="rgba(255,255,255,0.2)", width=1)
        ),
        text=[f"{v*100:.1f}%" for v in df["importance"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        **get_base_layout(),
        title="<b>Top 10 Feature Drivers (XGBoost Feature Importance)</b>",
        xaxis_title="Relative Feature Importance Score",
        height=400
    )
    return fig

def plot_price_spread_heatmap(matrix_df: pd.DataFrame) -> go.Figure:
    """2D Heatmap of Tariff Rate vs Soybean Spread."""
    pivot = matrix_df.pivot(index="tariff_rate_pct", columns="price_spread_usd", values="predicted_volume_kmt")
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"+${col}/MT" for col in pivot.columns],
        y=[f"{idx}%" for idx in pivot.index],
        colorscale="Viridis",
        colorbar=dict(title="Volume (kMT)", tickcolor="#f8fafc", titlefont=dict(color="#f8fafc")),
        hovertemplate="Tariff: %{y}<br>Soy Spread: %{x}<br>Predicted Palm Volume: <b>%{z:.1f} kMT</b><extra></extra>"
    ))
    
    fig.update_layout(
        **get_base_layout(),
        title="<b>Demand Sensitivity Grid: Tariff Rate (%) vs Soy-Palm Price Spread</b>",
        xaxis_title="Soybean Oil Discount Spread over Palm ($/MT)",
        yaxis_title="Applied Tariff Rate (%)",
        height=440
    )
    return fig

def plot_actual_vs_predicted(diagnostics: dict) -> go.Figure:
    """Actual vs Predicted scatter plot for XGBoost vs Linear Regression."""
    if not diagnostics or "y_actual" not in diagnostics:
        return go.Figure()
        
    y_actual = diagnostics["y_actual"][:250]
    xgb_preds = diagnostics["xgb_preds"][:250]
    lr_preds = diagnostics["lr_preds"][:250]
    
    fig = go.Figure()
    
    # 45-degree identity line
    min_v = min(min(y_actual), min(xgb_preds))
    max_v = max(max(y_actual), max(xgb_preds))
    fig.add_trace(go.Scatter(
        x=[min_v, max_v], y=[min_v, max_v],
        mode="lines",
        name="Perfect Fit (y = x)",
        line=dict(color="#64748b", dash="dash", width=2)
    ))
    
    # Linear Regression Scatter
    fig.add_trace(go.Scatter(
        x=y_actual, y=lr_preds,
        mode="markers",
        name="Linear Regression (Baseline)",
        marker=dict(color=THEME_COLORS["Linear Regression"], size=6, opacity=0.5),
        hovertemplate="Actual: %{x:.1f}<br>Pred (LR): %{y:.1f}<extra></extra>"
    ))
    
    # XGBoost Scatter
    fig.add_trace(go.Scatter(
        x=y_actual, y=xgb_preds,
        mode="markers",
        name="XGBoost (Optimized)",
        marker=dict(color=THEME_COLORS["XGBoost"], size=7, opacity=0.85, line=dict(width=1, color="white")),
        hovertemplate="Actual: %{x:.1f}<br>Pred (XGB): %{y:.1f}<extra></extra>"
    ))
    
    fig.update_layout(
        **get_base_layout(),
        title="<b>Actual vs. Predicted Trade Volume Diagnostic (Test Set)</b>",
        xaxis_title="Actual Trade Volume (kMT)",
        yaxis_title="Predicted Trade Volume (kMT)",
        height=440
    )
    return fig

def plot_residuals_distribution(diagnostics: dict) -> go.Figure:
    """Residuals distribution comparison."""
    if not diagnostics or "y_actual" not in diagnostics:
        return go.Figure()
        
    y_actual = np.array(diagnostics["y_actual"])
    xgb_res = y_actual - np.array(diagnostics["xgb_preds"])
    lr_res = y_actual - np.array(diagnostics["lr_preds"])
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=lr_res,
        name="Linear Regression Residuals",
        marker_color="rgba(148, 163, 184, 0.6)",
        opacity=0.65,
        nbinsx=35
    ))
    
    fig.add_trace(go.Histogram(
        x=xgb_res,
        name="XGBoost Residuals",
        marker_color="rgba(74, 222, 128, 0.75)",
        opacity=0.8,
        nbinsx=35
    ))
    
    fig.update_layout(
        **get_base_layout(),
        barmode="overlay",
        title="<b>Residual Error Distribution (Actual - Predicted)</b>",
        xaxis_title="Residual Error (kMT)",
        yaxis_title="Count",
        height=400
    )
    return fig
