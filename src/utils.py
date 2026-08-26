"""
PalmCore - Utility functions and custom CSS theme styling
"""

import streamlit as st
import pandas as pd
import numpy as np

def apply_custom_css():
    """Injects high-end dark glassmorphic styling and typography for PalmCore."""
    custom_css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 95% !important;
    }

    /* Brand Header Banner */
    .palm-hero-banner {
        background: linear-gradient(135deg, rgba(16, 44, 30, 0.95) 0%, rgba(26, 77, 46, 0.9) 50%, rgba(13, 27, 42, 0.95) 100%);
        border: 1px solid rgba(74, 222, 128, 0.2);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
    }
    
    .palm-hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4ade80 0%, #facc15 50%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    
    .palm-hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Metric Cards */
    .palm-metric-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        transition: all 0.25s ease-in-out;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .palm-metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(74, 222, 128, 0.4);
        box-shadow: 0 8px 24px rgba(74, 222, 128, 0.15);
    }

    .palm-metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #4ade80, #facc15);
    }

    .palm-metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .palm-metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }

    .palm-metric-delta {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 4px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 8px;
        border-radius: 6px;
    }

    .delta-positive {
        color: #4ade80;
        background: rgba(74, 222, 128, 0.12);
    }

    .delta-negative {
        color: #f87171;
        background: rgba(248, 113, 113, 0.12);
    }

    .delta-neutral {
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.12);
    }

    /* Model Benchmark Highlight Card */
    .benchmark-highlight-card {
        background: linear-gradient(145deg, rgba(6, 78, 59, 0.4) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(52, 211, 153, 0.35);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }

    /* Section Headers */
    .palm-section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

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
        background: rgba(74, 222, 128, 0.15);
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.3);
    }

    .badge-rf {
        background: rgba(250, 204, 21, 0.15);
        color: #facc15;
        border: 1px solid rgba(250, 204, 21, 0.3);
    }

    .badge-svm {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }

    .badge-lr {
        background: rgba(148, 163, 184, 0.15);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.3);
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
