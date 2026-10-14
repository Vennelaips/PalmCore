"""
PalmCore - Utility functions and clean professional light-green CSS theme styling
"""

import streamlit as st
import pandas as pd
import numpy as np

def apply_custom_css():
    """Injects clean, professional light theme styling with white background and forest green accents."""
    custom_css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
    }

    /* Overall App Background - Pure Clean White */
    .stApp {
        background-color: #ffffff;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3.5rem;
        max-width: 95% !important;
    }

    /* Sidebar Styling - Crisp Light Pearl Background */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
        color: #14532d;
    }

    /* Brand Header Banner */
    .palm-hero-banner {
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 50%, #ecfdf5 100%);
        border: 1px solid #bbf7d0;
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(22, 101, 52, 0.06);
    }
    
    .palm-hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: #14532d;
        background: linear-gradient(90deg, #14532d 0%, #15803d 60%, #047857 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    
    .palm-hero-subtitle {
        font-size: 1.02rem;
        color: #475569;
        font-weight: 400;
        line-height: 1.6;
    }

    .palm-hero-subtitle b {
        color: #15803d;
        font-weight: 600;
    }

    /* Metric Cards - Crisp White with Forest Green Highlights */
    .palm-metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 20px;
        transition: all 0.25s ease-in-out;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        position: relative;
        overflow: hidden;
    }
    
    .palm-metric-card:hover {
        transform: translateY(-2px);
        border-color: #86efac;
        box-shadow: 0 8px 22px rgba(22, 101, 52, 0.09);
    }

    .palm-metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3.5px;
        background: linear-gradient(90deg, #15803d, #22c55e);
    }

    .palm-metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #15803d;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .palm-metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        font-family: 'JetBrains Mono', 'Plus Jakarta Sans', monospace;
    }

    .palm-metric-delta {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 6px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 20px;
    }

    .delta-positive {
        color: #15803d;
        background: #dcfce7;
        border: 1px solid #bbf7d0;
    }

    .delta-negative {
        color: #b91c1c;
        background: #fee2e2;
        border: 1px solid #fecaca;
    }

    .delta-neutral {
        color: #1e293b;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
    }

    /* Model Benchmark Highlight Card */
    .benchmark-highlight-card {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 4px 16px rgba(22, 101, 52, 0.05);
    }

    .benchmark-highlight-card h4 {
        color: #14532d;
        font-weight: 700;
    }

    /* Section Headers */
    .palm-section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #14532d;
        margin-top: 1.6rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
        border-left: 4px solid #16a34a;
        padding-left: 10px;
    }

    /* Badges */
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-xgb {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }

    .badge-rf {
        background: #fef9c3;
        color: #854d0e;
        border: 1px solid #fde047;
    }

    .badge-svm {
        background: #f3e8ff;
        color: #6b21a8;
        border: 1px solid #d8b4fe;
    }

    .badge-lr {
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
    }

    /* Interactive Buttons */
    .stButton > button {
        background-color: #15803d;
        color: #ffffff;
        font-weight: 600;
        border-radius: 8px;
        border: 1px solid #14532d;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #166534;
        color: #ffffff;
        border-color: #14532d;
        box-shadow: 0 4px 12px rgba(22, 101, 52, 0.2);
    }

    /* Sliders & Inputs */
    .stSlider label {
        color: #1e293b;
        font-weight: 600;
    }

    /* Dataframe Header */
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        background-color: #ffffff;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def render_kpi_card(title: str, value: str, delta: str = None, delta_type: str = "positive"):
    """Renders a styled KPI card with badge."""
    delta_class = f"delta-{delta_type}"
    delta_html = f'<div class="palm-metric-delta {delta_class}">{delta}</div>' if delta else ''
    
    html = f"""
    <div class="palm-metric-card">
        <div class="palm-metric-label">{title}</div>
        <div class="palm-metric-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
