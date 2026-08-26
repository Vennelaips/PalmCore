"""
PalmCore - Test Suite
Verifies data generation, ML pipeline artifacts, simulation engine predictions, and Plotly visualization builders.
"""

import unittest
import os
import pandas as pd
import numpy as np
from src.simulation_engine import SimulationEngine
from src.visualizations import (
    plot_model_benchmarks,
    plot_radar_comparison,
    plot_tariff_elasticity_curve,
    plot_fiscal_revenue_curve,
    plot_feature_importance,
    plot_price_spread_heatmap,
    plot_actual_vs_predicted,
    plot_residuals_distribution
)

class TestPalmCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SimulationEngine()
        cls.test_params = {
            "origin_country": "Indonesia",
            "destination_country": "India",
            "tariff_rate_pct": 15.0,
            "export_levy_usd": 45.0,
            "cpo_benchmark_price_usd": 850.0,
            "soybean_oil_price_usd": 1020.0,
            "sunflower_oil_price_usd": 1050.0,
            "crude_brent_usd": 55.0,
            "freight_rate_usd": 32.0,
            "usd_fx_index": 1.05,
            "domestic_stocks_kmt": 180.0,
            "gdp_growth_pct": 5.2,
            "monsoon_seasonality_idx": 0.6,
            "biodiesel_mandate_pct": 25.0,
            "crush_margin_usd": 35.0
        }

    def test_model_artifacts_loaded(self):
        self.assertIsNotNone(self.engine.models)
        self.assertIn("XGBoost", self.engine.models)
        self.assertIn("Random Forest", self.engine.models)
        self.assertIn("Support Vector Machine", self.engine.models)
        self.assertIn("Linear Regression", self.engine.models)

    def test_single_scenario_prediction(self):
        res = self.engine.predict_scenario(self.test_params)
        self.assertIn("predictions", res)
        self.assertIn("XGBoost", res["predictions"])
        self.assertIn("tariff_revenue_m_usd", res)
        self.assertGreater(res["predictions"]["XGBoost"], 0)
        self.assertGreater(res["landed_cost_usd_mt"], 850.0)

    def test_tariff_sweep_simulation(self):
        sweep_df = self.engine.simulate_tariff_sweep(self.test_params, min_tariff=0.0, max_tariff=30.0, step=5.0)
        self.assertEqual(len(sweep_df), 7)
        self.assertIn("XGBoost_vol_kmt", sweep_df.columns)
        self.assertIn("tariff_revenue_m_usd", sweep_df.columns)
        
        # Verify downward slope of demand as tariff increases
        vol_at_0 = sweep_df[sweep_df["tariff_rate_pct"] == 0.0]["XGBoost_vol_kmt"].values[0]
        vol_at_30 = sweep_df[sweep_df["tariff_rate_pct"] == 30.0]["XGBoost_vol_kmt"].values[0]
        self.assertGreater(vol_at_0, vol_at_30)

    def test_price_spread_matrix(self):
        matrix_df = self.engine.simulate_price_spread_matrix(self.test_params)
        self.assertGreater(len(matrix_df), 0)
        self.assertIn("predicted_volume_kmt", matrix_df.columns)

    def test_visualizations_generate(self):
        fig_bench = plot_model_benchmarks(self.engine.evaluation)
        self.assertIsNotNone(fig_bench)
        
        fig_radar = plot_radar_comparison(self.engine.evaluation)
        self.assertIsNotNone(fig_radar)
        
        sweep_df = self.engine.simulate_tariff_sweep(self.test_params, min_tariff=0.0, max_tariff=30.0, step=5.0)
        fig_curve = plot_tariff_elasticity_curve(sweep_df, 15.0)
        self.assertIsNotNone(fig_curve)
        
        fig_fiscal = plot_fiscal_revenue_curve(sweep_df, 15.0)
        self.assertIsNotNone(fig_fiscal)
        
        fig_feat = plot_feature_importance(self.engine.feature_importances)
        self.assertIsNotNone(fig_feat)
        
        fig_heat = plot_price_spread_heatmap(self.engine.simulate_price_spread_matrix(self.test_params))
        self.assertIsNotNone(fig_heat)

if __name__ == "__main__":
    unittest.main()
