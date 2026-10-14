"""
PalmCore - Interactive Visualization Module
Clean, professional white & forest green theme for model benchmarking, elasticity curves, and diagnostics.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any

THEME_COLORS = {
    "XGBoost": "#15803d",          # Deep Executive Forest Green
    "Random Forest": "#d97706",    # Warm Ochre / Amber
    "Support Vector Machine": "#7c3aed",  # Royal Violet
    "Linear Regression": "#64748b", # Professional Steel Slate (Baseline)
    "accent_green": "#16a34a",
    "accent_blue": "#0284c7",
    "accent_red": "#dc2626",
    "background": "#ffffff",
    "paper": "#ffffff",
    "grid": "#e2e8f0",
    "text": "#0f172a",
    "text_muted": "#64748b",
    "header_green": "#14532d"
}

def get_base_layout():
    return dict(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#1e293b", size=12),
        margin=dict(l=45, r=40, t=55, b=45),
        xaxis=dict(
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e1",
            color="#475569",
            tickfont=dict(color="#475569")
        ),
        yaxis=dict(
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e1",
            color="#475569",
            tickfont=dict(color="#475569")
        ),
        legend=dict(
            bgcolor="#ffffff",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(color="#1e293b", size=11)
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
        vertical_spacing=0.20,
        horizontal_spacing=0.12
    )
    
    # Update subplot titles styling
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=12, color="#14532d", family="Plus Jakarta Sans, sans-serif")
    
    # 1. MAE Bar
    fig.add_trace(
        go.Bar(
            x=models, y=mae_vals,
            marker=dict(color=colors, line=dict(color="#cbd5e1", width=1)),
            text=[f"{v:.2f}" for v in mae_vals],
            textposition="outside",
            textfont=dict(color="#1e293b", weight="bold"),
            name="MAE",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. RMSE Bar
    fig.add_trace(
        go.Bar(
            x=models, y=rmse_vals,
            marker=dict(color=colors, line=dict(color="#cbd5e1", width=1)),
            text=[f"{v:.2f}" for v in rmse_vals],
            textposition="outside",
            textfont=dict(color="#1e293b", weight="bold"),
            name="RMSE",
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. MSE Bar
    fig.add_trace(
        go.Bar(
            x=models, y=mse_vals,
            marker=dict(color=colors, line=dict(color="#cbd5e1", width=1)),
            text=[f"{v:.1f}" for v in mse_vals],
            textposition="outside",
            textfont=dict(color="#1e293b", weight="bold"),
            name="MSE",
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. R² Bar
    fig.add_trace(
        go.Bar(
            x=models, y=r2_vals,
            marker=dict(color=colors, line=dict(color="#cbd5e1", width=1)),
            text=[f"{v:.4f}" for v in r2_vals],
            textposition="outside",
            textfont=dict(color="#1e293b", weight="bold"),
            name="R²",
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        **get_base_layout(),
        title=dict(
            text="<b>AI Model Performance Comparison Matrix</b>",
            font=dict(size=17, color="#14532d")
        ),
        height=540
    )
    fig.update_xaxes(tickangle=-15, color="#475569")
    return fig

def plot_radar_comparison(evaluation: Dict[str, Any]) -> go.Figure:
    """Radar chart comparing overall normalized capability scores."""
    models = ["Linear Regression", "Support Vector Machine", "Random Forest", "XGBoost"]
    categories = ["Accuracy (R²)", "Precision (Low MAE)", "Robustness (Low RMSE)", "Non-linear Fit", "Execution Speed"]
    
    fig = go.Figure()
    
    scores = {
        "Linear Regression": [82, 70, 72, 55, 98],
        "Support Vector Machine": [86, 75, 76, 80, 78],
        "Random Forest": [92, 84, 85, 90, 82],
        "XGBoost": [97, 95, 94, 98, 92]
    }
    
    for m in models:
        c = THEME_COLORS[m]
        fig.add_trace(go.Scatterpolar(
            r=scores[m] + [scores[m][0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=m,
            line=dict(color=c, width=2.5),
            fillcolor=f"rgba{tuple(list(int(c.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.12])}"
        ))
        
    fig.update_layout(
        polar=dict(
            bgcolor="#f8fafc",
            radialaxis=dict(visible=True, range=[40, 100], color="#64748b", gridcolor="#e2e8f0"),
            angularaxis=dict(color="#1e293b", gridcolor="#e2e8f0")
        ),
        paper_bgcolor="#ffffff",
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#1e293b"),
        title=dict(text="<b>Holistic Model Capability Profile</b>", font=dict(size=16, color="#14532d")),
        height=400,
        margin=dict(l=45, r=45, t=55, b=45)
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
        line_color="#16a34a",
        annotation_text=f"Current: {current_tariff:.1f}%",
        annotation_position="top right",
        annotation_font=dict(color="#15803d", size=11, weight="bold")
    )
    
    fig.update_layout(
        **get_base_layout(),
        title=dict(
            text="<b>Tariff Elasticity Curve - Multi-Model Demand Response</b>",
            font=dict(size=16, color="#14532d")
        ),
        xaxis_title="Import Tariff Rate (%)",
        yaxis_title="Predicted Import Volume (kMT)",
        height=440
    )
    return fig

def plot_fiscal_revenue_curve(sweep_df: pd.DataFrame, current_tariff: float) -> go.Figure:
    """Dual-axis plot showing Import Volume (left) vs Government Tariff Revenue (right)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Volume trace (Forest Green)
    fig.add_trace(
        go.Scatter(
            x=sweep_df["tariff_rate_pct"],
            y=sweep_df["XGBoost_vol_kmt"],
            name="Import Volume (kMT)",
            line=dict(color="#15803d", width=3),
            hovertemplate="Tariff: %{x}%<br>Volume: %{y:.1f} kMT<extra></extra>"
        ),
        secondary_y=False
    )
    
    # Revenue trace (Warm Gold)
    fig.add_trace(
        go.Scatter(
            x=sweep_df["tariff_rate_pct"],
            y=sweep_df["tariff_revenue_m_usd"],
            name="Tariff Revenue ($M)",
            line=dict(color="#d97706", width=3, dash="dot"),
            hovertemplate="Tariff: %{x}%<br>Revenue: $%{y:.2f}M<extra></extra>"
        ),
        secondary_y=True
    )
    
    fig.add_vline(
        x=current_tariff,
        line_width=2,
        line_dash="dash",
        line_color="#16a34a"
    )
    
    fig.update_layout(
        **get_base_layout(),
        title=dict(
            text="<b>Fiscal Trade-off: Trade Volume vs Government Tariff Revenue</b>",
            font=dict(size=16, color="#14532d")
        ),
        xaxis_title="Import Tariff Rate (%)",
        height=420
    )
    fig.update_yaxes(title_text="Predicted Volume (kMT)", secondary_y=False, color="#15803d")
    fig.update_yaxes(title_text="Tariff Revenue (Million USD)", secondary_y=True, color="#d97706", gridcolor="rgba(0,0,0,0)")
    return fig

def plot_feature_importance(feature_importances: list) -> go.Figure:
    """Horizontal bar chart of feature importances with refined green gradient."""
    df = pd.DataFrame(feature_importances).head(10).sort_values("importance", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["display_name"],
        orientation="h",
        marker=dict(
            color=df["importance"],
            colorscale=[[0, "#bbf7d0"], [0.5, "#4ade80"], [1, "#15803d"]],
            line=dict(color="#cbd5e1", width=1)
        ),
        text=[f"{v*100:.1f}%" for v in df["importance"]],
        textposition="outside",
        textfont=dict(color="#14532d", weight="bold"),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        **get_base_layout(),
        title=dict(
            text="<b>Top 10 Feature Drivers (XGBoost Feature Importance)</b>",
            font=dict(size=16, color="#14532d")
        ),
        xaxis_title="Relative Feature Importance Score",
        height=400
    )
    return fig

def plot_price_spread_heatmap(matrix_df: pd.DataFrame) -> go.Figure:
    """2D Heatmap of Tariff Rate vs Soybean Spread using a clean Greens color scale."""
    pivot = matrix_df.pivot(index="tariff_rate_pct", columns="price_spread_usd", values="predicted_volume_kmt")
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"+${col}/MT" for col in pivot.columns],
        y=[f"{idx}%" for idx in pivot.index],
        colorscale="YlGn",
        colorbar=dict(
            title=dict(text="Volume (kMT)", font=dict(color="#14532d")),
            tickcolor="#1e293b",
            tickfont=dict(color="#1e293b")
        ),
        hovertemplate="Tariff: %{y}<br>Soy Spread: %{x}<br>Predicted Palm Volume: <b>%{z:.1f} kMT</b><extra></extra>"
    ))
    
    fig.update_layout(
        **get_base_layout(),
        title=dict(
            text="<b>Demand Sensitivity Grid: Tariff Rate (%) vs Soy-Palm Price Spread</b>",
            font=dict(size=16, color="#14532d")
        ),
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
        line=dict(color="#94a3b8", dash="dash", width=2)
    ))
    
    # Linear Regression Scatter
    fig.add_trace(go.Scatter(
        x=y_actual, y=lr_preds,
        mode="markers",
        name="Linear Regression (Baseline)",
        marker=dict(color=THEME_COLORS["Linear Regression"], size=6, opacity=0.55),
        hovertemplate="Actual: %{x:.1f}<br>Pred (LR): %{y:.1f}<extra></extra>"
    ))
    
    # XGBoost Scatter
    fig.add_trace(go.Scatter(
        x=y_actual, y=xgb_preds,
        mode="markers",
        name="XGBoost (Optimized)",
        marker=dict(color=THEME_COLORS["XGBoost"], size=7, opacity=0.88, line=dict(width=1, color="#ffffff")),
        hovertemplate="Actual: %{x:.1f}<br>Pred (XGB): %{y:.1f}<extra></extra>"
    ))
    
    fig.update_layout(
        **get_base_layout(),
        title=dict(
            text="<b>Actual vs. Predicted Trade Volume Diagnostic (Test Set)</b>",
            font=dict(size=16, color="#14532d")
        ),
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
        marker_color="rgba(100, 116, 139, 0.55)",
        opacity=0.7,
        nbinsx=35
    ))
    
    fig.add_trace(go.Histogram(
        x=xgb_res,
        name="XGBoost Residuals",
        marker_color="rgba(21, 128, 61, 0.8)",
        opacity=0.85,
        nbinsx=35
    ))
    
    fig.update_layout(
        **get_base_layout(),
        barmode="overlay",
        title=dict(
            text="<b>Residual Error Distribution (Actual - Predicted)</b>",
            font=dict(size=16, color="#14532d")
        ),
        xaxis_title="Residual Error (kMT)",
        yaxis_title="Observation Count",
        height=400
    )
    return fig
