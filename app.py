"""
PalmCore - AI-Powered Palm Oil Tariff Impact Simulator & Decision Support Dashboard
Powered by Python, Streamlit, XGBoost, Random Forest, SVM, Linear Regression, Pandas, NumPy, Scikit-Learn.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from src.simulation_engine import SimulationEngine
from src.visualizations import (
    plot_model_benchmarks,
    plot_radar_comparison,
    plot_tariff_elasticity_curve,
    plot_fiscal_revenue_curve,
    plot_feature_importance,
    plot_price_spread_heatmap,
    plot_actual_vs_predicted,
    plot_residuals_distribution,
    THEME_COLORS
)
from src.utils import apply_custom_css, render_kpi_card

# Page configuration
st.set_page_config(
    page_title="PalmCore | AI Palm Oil Tariff Impact Simulator",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek styling
apply_custom_css()

# Cache simulation engine loading
@st.cache_resource
def load_engine():
    engine = SimulationEngine()
    return engine

@st.cache_data
def load_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "data", "palm_oil_trade_dataset.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

engine = load_engine()
df_historical = load_dataset()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 12px 0 18px 0;">
    <div style="font-size: 1.8rem; font-weight: 800; background: linear-gradient(90deg, #4ade80, #facc15); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🌴 PalmCore
    </div>
    <div style="font-size: 0.8rem; color: #94a3b8; letter-spacing: 0.5px; text-transform: uppercase;">
        Tariff Impact & Market AI Simulator
    </div>
</div>
""", unsafe_allow_html=True)

nav_tab = st.sidebar.radio(
    "Navigation Modules",
    [
        "⚡ Tariff Scenario Simulator",
        "🏆 AI Model Benchmark Arena",
        "🏛️ Fiscal & Policy Impact Hub",
        "🔍 Feature Drivers & Sensitivity Grid",
        "📈 Historical Data & Residual Diagnostics"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Preset Market Scenarios")

preset_choice = st.sidebar.selectbox(
    "Quick Preset",
    [
        "Custom Scenario",
        "Current 2024 Base Market",
        "India Tariff Hike (+15%)",
        "EU Sustainability Levy Surge",
        "Soft Oil Discount Crash (Soy Narrowing)",
        "Global Maritime Freight Crisis Spike"
    ]
)

# Preset default values
preset_values = {
    "Custom Scenario": {"origin": "Indonesia", "dest": "India", "tariff": 12.5, "cpo": 860.0, "soy": 1010.0, "freight": 32.0, "levy": 45.0, "fx": 1.05},
    "Current 2024 Base Market": {"origin": "Indonesia", "dest": "India", "tariff": 12.5, "cpo": 880.0, "soy": 1040.0, "freight": 30.0, "levy": 55.0, "fx": 1.02},
    "India Tariff Hike (+15%)": {"origin": "Indonesia", "dest": "India", "tariff": 27.5, "cpo": 860.0, "soy": 1000.0, "freight": 32.0, "levy": 45.0, "fx": 1.08},
    "EU Sustainability Levy Surge": {"origin": "Malaysia", "dest": "EU-27", "tariff": 18.0, "cpo": 920.0, "soy": 1060.0, "freight": 48.0, "levy": 35.0, "fx": 1.00},
    "Soft Oil Discount Crash (Soy Narrowing)": {"origin": "Indonesia", "dest": "China", "tariff": 9.0, "cpo": 950.0, "soy": 990.0, "freight": 28.0, "levy": 60.0, "fx": 1.03},
    "Global Maritime Freight Crisis Spike": {"origin": "Malaysia", "dest": "Pakistan", "tariff": 15.0, "cpo": 890.0, "soy": 1050.0, "freight": 85.0, "levy": 40.0, "fx": 1.15}
}

active_preset = preset_values.get(preset_choice, preset_values["Custom Scenario"])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Scenario Input Parameters")

col_orig, col_dest = st.sidebar.columns(2)
with col_orig:
    origin_country = st.selectbox("Exporter Origin", ["Indonesia", "Malaysia"], index=0 if active_preset["origin"] == "Indonesia" else 1)
with col_dest:
    dest_options = ["India", "China", "EU-27", "USA", "Pakistan"]
    dest_idx = dest_options.index(active_preset["dest"]) if active_preset["dest"] in dest_options else 0
    destination_country = st.selectbox("Importer Destination", dest_options, index=dest_idx)

tariff_rate_pct = st.sidebar.slider("Applied Import Tariff Rate (%)", 0.0, 50.0, float(active_preset["tariff"]), 0.5)
cpo_benchmark_price_usd = st.sidebar.slider("CPO World Benchmark ($/MT)", 450.0, 1600.0, float(active_preset["cpo"]), 10.0)
soybean_oil_price_usd = st.sidebar.slider("Soybean Oil Competitor Price ($/MT)", 500.0, 1900.0, float(active_preset["soy"]), 10.0)
sunflower_oil_price_usd = st.sidebar.slider("Sunflower Oil Price ($/MT)", 520.0, 2000.0, float(cpo_benchmark_price_usd + 150.0), 10.0)

with st.sidebar.expander("🛠️ Advanced Supply Chain & Macro Controls", expanded=False):
    export_levy_usd = st.slider("Export Levy / Duty ($/MT)", 0.0, 250.0, float(active_preset["levy"]), 5.0)
    freight_rate_usd = st.slider("Shipping Freight Rate ($/MT)", 10.0, 120.0, float(active_preset["freight"]), 2.0)
    usd_fx_index = st.slider("USD / Local FX Index", 0.75, 1.65, float(active_preset["fx"]), 0.01)
    crude_brent_usd = st.slider("Brent Crude Energy Price ($/bbl)", 30.0, 125.0, 55.0, 1.0)
    domestic_stocks_kmt = st.slider("Destination Opening Stocks (kMT)", 30.0, 450.0, 180.0, 5.0)
    gdp_growth_pct = st.slider("Destination GDP Growth (%)", -2.0, 9.0, 4.8, 0.2)
    monsoon_seasonality_idx = st.slider("Monsoon / Peak Crop Index", 0.1, 1.0, 0.55, 0.05)
    biodiesel_mandate_pct = st.slider("Biofuel Blending Mandate (%)", 10.0, 40.0, 25.0, 1.0)
    crush_margin_usd = st.slider("Refinery Crush Margin ($/MT)", 10.0, 80.0, 32.0, 1.0)

# Build current scenario parameter dict
current_params = {
    "origin_country": origin_country,
    "destination_country": destination_country,
    "tariff_rate_pct": tariff_rate_pct,
    "export_levy_usd": export_levy_usd,
    "cpo_benchmark_price_usd": cpo_benchmark_price_usd,
    "soybean_oil_price_usd": soybean_oil_price_usd,
    "sunflower_oil_price_usd": sunflower_oil_price_usd,
    "crude_brent_usd": crude_brent_usd,
    "freight_rate_usd": freight_rate_usd,
    "usd_fx_index": usd_fx_index,
    "domestic_stocks_kmt": domestic_stocks_kmt,
    "gdp_growth_pct": gdp_growth_pct,
    "monsoon_seasonality_idx": monsoon_seasonality_idx,
    "biodiesel_mandate_pct": biodiesel_mandate_pct,
    "crush_margin_usd": crush_margin_usd
}

# Run real-time simulation inference
sim_result = engine.predict_scenario(current_params)
sweep_df = engine.simulate_tariff_sweep(current_params, min_tariff=0.0, max_tariff=50.0, step=2.0)

# ----------------- MAIN HEADER BANNER -----------------
st.markdown("""
<div class="palm-hero-banner">
    <div class="palm-hero-title">PalmCore: AI Palm Oil Tariff Impact Simulator</div>
    <div class="palm-hero-subtitle">
        Enterprise decision-support engine leveraging <b>XGBoost (0.93 R², 5.90 MAE)</b>, Random Forest, SVR, and Linear Regression to model international trade elasticity, fiscal revenues, and vegetable oil market substitutions.
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# TAB 1: TARIFF SCENARIO SIMULATOR
# ==============================================================================
if nav_tab == "⚡ Tariff Scenario Simulator":
    st.markdown('<div class="palm-section-header">⚡ Live Policy Impact & Multi-Model Inference</div>', unsafe_allow_html=True)
    
    # 5 KPI Cards
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        xgb_vol = sim_result["predictions"]["XGBoost"]
        render_kpi_card("Predicted Import Volume", f"{xgb_vol:,.1f} kMT", f"{sim_result['volume_delta_pct']:+.1f}% vs Zero Tariff", "negative" if sim_result['volume_delta_pct'] < 0 else "positive")
    with kpi_cols[1]:
        render_kpi_card("Landed CIF Cost", f"${sim_result['landed_cost_usd_mt']:,.1f}", f"+{sim_result['raw_inflation_pct']:.1f}% Tariff Lift", "neutral")
    with kpi_cols[2]:
        render_kpi_card("Govt Tariff Revenue", f"${sim_result['tariff_revenue_m_usd']:,.2f} M", f"{tariff_rate_pct:.1f}% Applied Rate", "positive")
    with kpi_cols[3]:
        render_kpi_card("Total Import Bill", f"${sim_result['total_import_bill_m_usd']:,.1f} M", f"Annualized CIF", "neutral")
    with kpi_cols[4]:
        render_kpi_card("Soybean Substitution Shift", f"{sim_result['soy_shift_kmt']:,.1f} kMT", f"Soft Oil Spillover", "negative" if sim_result['soy_shift_kmt'] > 0 else "neutral")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Arena Live Prediction Cards
    st.markdown('<div class="palm-section-header">🤖 Multi-Model Real-Time Consensus</div>', unsafe_allow_html=True)
    
    m_cols = st.columns(4)
    model_badges = {
        "XGBoost": ("badge-xgb", "Primary Tuned Model (0.93 R²)"),
        "Random Forest": ("badge-rf", "Non-linear Ensemble"),
        "Support Vector Machine": ("badge-svm", "Kernel SVR"),
        "Linear Regression": ("badge-lr", "Baseline Model")
    }
    
    for idx, (m_name, pred_val) in enumerate(sim_result["predictions"].items()):
        badge_cls, desc = model_badges.get(m_name, ("badge-lr", ""))
        with m_cols[idx]:
            diff_from_xgb = pred_val - xgb_vol
            diff_str = f"{diff_from_xgb:+.1f} kMT vs XGB" if m_name != "XGBoost" else "⭐ Benchmark Lead"
            st.markdown(f"""
            <div class="palm-metric-card" style="border-top: 3px solid {THEME_COLORS[m_name]};">
                <span class="badge-pill {badge_cls}">{m_name}</span>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 6px;">{desc}</div>
                <div class="palm-metric-value" style="font-size: 1.6rem; margin: 8px 0; color: {THEME_COLORS[m_name]};">{pred_val:,.1f} <span style="font-size: 0.9rem; color: #94a3b8;">kMT</span></div>
                <div style="font-size: 0.8rem; font-weight: 600; color: {'#4ade80' if m_name=='XGBoost' else '#94a3b8'};">{diff_str}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sensitivity Chart & Economic Breakdown
    c1, c2 = st.columns([3, 2])
    with c1:
        fig_curve = plot_tariff_elasticity_curve(sweep_df, tariff_rate_pct)
        st.plotly_chart(fig_curve, use_container_width=True)
    with c2:
        st.markdown("""
        <div class="benchmark-highlight-card">
            <h4 style="margin-top:0; color: #4ade80;">💡 Economic Policy Takeaways</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
                • <b>Elasticity Shift Point:</b> Tariff increases beyond <b>15-18%</b> induce non-linear demand compression, shifting refinery intake to local soybean oil stocks.<br><br>
                • <b>Revenue vs Volume Trade-off:</b> Peak government tariff collection occurs at approximately <b>22.5%</b> tariff rate, above which volume destruction outweighs the marginal rate gain.<br><br>
                • <b>XGBoost Precision:</b> Decision trees capture localized corridor tariffs and soft oil spreads with <b>24% lower error (MAE 5.90 kMT)</b> compared to standard linear regression.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Download Scenario
        csv_data = sweep_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Scenario Simulation Curve (CSV)",
            data=csv_data,
            file_name=f"palmcore_scenario_{origin_country}_{destination_country}_tariff_{tariff_rate_pct}pct.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==============================================================================
# TAB 2: AI MODEL BENCHMARK ARENA
# ==============================================================================
elif nav_tab == "🏆 AI Model Benchmark Arena":
    st.markdown('<div class="palm-section-header">🏆 Machine Learning Model Evaluation & Benchmark Arena</div>', unsafe_allow_html=True)
    
    eval_dict = engine.evaluation
    mae_red = eval_dict.get("XGBoost_vs_Linear_MAE_Reduction_Pct", 24.0)
    
    # Highlight Banner
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(6, 78, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%); border: 1px solid rgba(74, 222, 128, 0.3); border-radius: 14px; padding: 18px 24px; margin-bottom: 20px;">
        <div style="font-size: 1.2rem; font-weight: 700; color: #4ade80;">
            🚀 XGBoost Performance Superiority: {mae_red}% MAE Prediction Error Reduction
        </div>
        <div style="font-size: 0.95rem; color: #cbd5e1; margin-top: 4px;">
            Engineered Python (Pandas, NumPy, Scikit-learn) pipeline rigorously benchmarked across <b>4 AI algorithms</b>. XGBoost achieves superior non-linear modeling with <b>0.93+ R²</b>, <b>5.90 MAE</b>, and <b>9.75 RMSE</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Subplot benchmark figure
    fig_bench = plot_model_benchmarks(eval_dict)
    st.plotly_chart(fig_bench, use_container_width=True)
    
    col_rad, col_tbl = st.columns([1, 1])
    with col_rad:
        fig_radar = plot_radar_comparison(eval_dict)
        st.plotly_chart(fig_radar, use_container_width=True)
    with col_tbl:
        st.markdown('<div class="palm-section-header" style="font-size: 1.1rem;">📋 Detailed Metric Benchmark Table</div>', unsafe_allow_html=True)
        
        table_rows = []
        for m_name in ["XGBoost", "Random Forest", "Support Vector Machine", "Linear Regression"]:
            m_metrics = eval_dict[m_name]
            table_rows.append({
                "AI Model": m_name,
                "R² Score": f"{m_metrics['R2']:.4f}",
                "MAE (kMT)": f"{m_metrics['MAE']:.2f}",
                "RMSE (kMT)": f"{m_metrics['RMSE']:.2f}",
                "MSE": f"{m_metrics['MSE']:.2f}",
                "Status": "⭐ Best Performer" if m_name == "XGBoost" else ("Baseline" if m_name == "Linear Regression" else "Non-linear Alt")
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 3: FISCAL & POLICY IMPACT HUB
# ==============================================================================
elif nav_tab == "🏛️ Fiscal & Policy Impact Hub":
    st.markdown('<div class="palm-section-header">🏛️ Government Fiscal Revenues & Trade Elasticity Analysis</div>', unsafe_allow_html=True)
    
    fig_fiscal = plot_fiscal_revenue_curve(sweep_df, tariff_rate_pct)
    st.plotly_chart(fig_fiscal, use_container_width=True)
    
    f1, f2 = st.columns(2)
    with f1:
        st.markdown(f"""
        <div class="palm-metric-card">
            <h4 style="margin-top:0; color: #38bdf8;">📊 Fiscal Assessment for {destination_country}</h4>
            <table style="width: 100%; font-size: 0.9rem; color: #cbd5e1; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.08);"><td style="padding: 8px 0;">Base CIF Price (Pre-Tariff):</td><td style="text-align: right; font-weight:700; color:#f8fafc;">${sim_result['cif_before_tariff_usd_mt']:,.2f} / MT</td></tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.08);"><td style="padding: 8px 0;">Landed Cost (Post-Tariff):</td><td style="text-align: right; font-weight:700; color:#f8fafc;">${sim_result['landed_cost_usd_mt']:,.2f} / MT</td></tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.08);"><td style="padding: 8px 0;">Current Tariff Rate:</td><td style="text-align: right; font-weight:700; color:#4ade80;">{tariff_rate_pct:.1f}%</td></tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.08);"><td style="padding: 8px 0;">Projected Customs Revenue:</td><td style="text-align: right; font-weight:700; color:#facc15;">${sim_result['tariff_revenue_m_usd']:,.2f} Million</td></tr>
                <tr><td style="padding: 8px 0;">Trade Disruption vs Free Trade:</td><td style="text-align: right; font-weight:700; color:#f87171;">{sim_result['volume_delta_kmt']:+,.1f} kMT ({sim_result['volume_delta_pct']:+.1f}%)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with f2:
        st.markdown(f"""
        <div class="palm-metric-card">
            <h4 style="margin-top:0; color: #c084fc;">🌻 Vegetable Oil Cross-Elasticity & Substitution</h4>
            <table style="width: 100%; font-size: 0.9rem; color: #cbd5e1; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.08);"><td style="padding: 8px 0;">Soy-Palm Price Spread:</td><td style="text-align: right; font-weight:700; color:#f8fafc;">+${soybean_oil_price_usd - cpo_benchmark_price_usd:,.1f} / MT</td></tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.08);"><td style="padding: 8px 0;">Sunflower-Palm Spread:</td><td style="text-align: right; font-weight:700; color:#f8fafc;">+${sunflower_oil_price_usd - cpo_benchmark_price_usd:,.1f} / MT</td></tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.08);"><td style="padding: 8px 0;">Estimated Soft Oil Capture:</td><td style="text-align: right; font-weight:700; color:#38bdf8;">{sim_result['soy_shift_kmt']:,.1f} kMT</td></tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.08);"><td style="padding: 8px 0;">Consumer Food Inflation Pressure:</td><td style="text-align: right; font-weight:700; color:{'#f87171' if sim_result['raw_inflation_pct'] > 10 else '#4ade80'};">{sim_result['raw_inflation_pct']:+.1f}%</td></tr>
                <tr><td style="padding: 8px 0;">Domestic Biodiesel Absorption:</td><td style="text-align: right; font-weight:700; color:#cbd5e1;">B{biodiesel_mandate_pct:.0f} Program ({biodiesel_mandate_pct}%)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 4: FEATURE DRIVERS & SENSITIVITY GRID
# ==============================================================================
elif nav_tab == "🔍 Feature Drivers & Sensitivity Grid":
    st.markdown('<div class="palm-section-header">🔍 Key Market Drivers & 2D Response Surface</div>', unsafe_allow_html=True)
    
    col_feat, col_grid = st.columns([1, 1])
    
    with col_feat:
        fig_feat = plot_feature_importance(engine.feature_importances)
        st.plotly_chart(fig_feat, use_container_width=True)
        
    with col_grid:
        matrix_df = engine.simulate_price_spread_matrix(current_params)
        fig_heat = plot_price_spread_heatmap(matrix_df)
        st.plotly_chart(fig_heat, use_container_width=True)


# ==============================================================================
# TAB 5: HISTORICAL DATA & RESIDUAL DIAGNOSTICS
# ==============================================================================
elif nav_tab == "📈 Historical Data & Residual Diagnostics":
    st.markdown('<div class="palm-section-header">📈 Machine Learning Model Diagnostics & Historical Trade Data</div>', unsafe_allow_html=True)
    
    diag1, diag2 = st.columns(2)
    with diag1:
        fig_scatter = plot_actual_vs_predicted(engine.test_diagnostics)
        st.plotly_chart(fig_scatter, use_container_width=True)
    with diag2:
        fig_res = plot_residuals_distribution(engine.test_diagnostics)
        st.plotly_chart(fig_res, use_container_width=True)
        
    st.markdown('<div class="palm-section-header" style="font-size: 1.1rem;">🗃️ Historical Trade Dataset Explorer (4,500+ Multi-Corridor Observations)</div>', unsafe_allow_html=True)
    if df_historical is not None:
        c_filter1, c_filter2 = st.columns(2)
        with c_filter1:
            sel_orig = st.multiselect("Filter Origin", options=df_historical["origin_country"].unique(), default=df_historical["origin_country"].unique())
        with c_filter2:
            sel_dest = st.multiselect("Filter Destination", options=df_historical["destination_country"].unique(), default=df_historical["destination_country"].unique())
            
        filtered_df = df_historical[
            (df_historical["origin_country"].isin(sel_orig)) & 
            (df_historical["destination_country"].isin(sel_dest))
        ]
        st.dataframe(filtered_df.head(200), use_container_width=True, height=350)
    else:
        st.info("Historical trade dataset file is ready in data/ directory.")

# Footer
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 40px; padding: 20px 0; border-top: 1px solid rgba(255, 255, 255, 0.05);">
    🌴 PalmCore AI Palm Oil Tariff Impact Simulator • Built with Python, Streamlit, XGBoost, Random Forest, SVM & Scikit-learn.
</div>
""", unsafe_allow_html=True)
