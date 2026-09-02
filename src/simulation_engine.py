"""
PalmCore - Simulation Engine & Economic Impact Calculator
Handles multi-model inference, tariff sensitivity sweeps, and fiscal/economic policy metrics.
Features version-resilient automated fallback training.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

class SimulationEngine:
    def __init__(self, model_artifact_path: str = None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if model_artifact_path is None:
            model_artifact_path = os.path.join(current_dir, "..", "models", "trained_models.pkl")
            
        data_csv_path = os.path.join(current_dir, "..", "data", "palm_oil_trade_dataset.csv")
        
        self.artifacts = None
        
        # 1. Try to load pre-trained pickle
        if os.path.exists(model_artifact_path):
            try:
                self.artifacts = joblib.load(model_artifact_path)
            except Exception as e:
                print(f"[PalmCore] Notice: Unpickling artifact failed ({e}). Re-training dynamically for current environment...")
                self.artifacts = None
                
        # 2. Resilient Fallback: If pickle is missing or incompatible with cloud environment version
        if self.artifacts is None:
            from models.pipeline import train_pipeline_from_df
            if os.path.exists(data_csv_path):
                df = pd.read_csv(data_csv_path)
            else:
                from data.synthetic_data_generator import generate_palm_oil_trade_data
                df = generate_palm_oil_trade_data(n_samples=4500, random_state=42)
                
            self.artifacts = train_pipeline_from_df(df)
            try:
                joblib.dump(self.artifacts, model_artifact_path)
            except Exception:
                pass
                
        self.models = self.artifacts["models"]
        self.preprocessor = self.artifacts["preprocessor"]
        self.num_cols = self.artifacts["num_cols"]
        self.cat_cols = self.artifacts["cat_cols"]
        self.feature_importances = self.artifacts["feature_importances"]
        self.evaluation = self.artifacts["evaluation"]
        self.test_diagnostics = self.artifacts.get("test_diagnostics", {})

    def prepare_input_dataframe(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Converts user input parameters into model-ready DataFrame."""
        cpo_price = float(params["cpo_benchmark_price_usd"])
        soy_price = float(params["soybean_oil_price_usd"])
        sun_price = float(params["sunflower_oil_price_usd"])
        freight = float(params["freight_rate_usd"])
        levy = float(params["export_levy_usd"])
        tariff = float(params["tariff_rate_pct"])
        
        spread = soy_price - cpo_price
        landed_cost = (cpo_price + levy + freight) * (1.0 + tariff / 100.0)
        
        data_dict = {
            "origin_country": [params["origin_country"]],
            "destination_country": [params["destination_country"]],
            "tariff_rate_pct": [tariff],
            "export_levy_usd": [levy],
            "cpo_benchmark_price_usd": [cpo_price],
            "soybean_oil_price_usd": [soy_price],
            "sunflower_oil_price_usd": [sun_price],
            "crude_brent_usd": [float(params["crude_brent_usd"])],
            "freight_rate_usd": [freight],
            "usd_fx_index": [float(params["usd_fx_index"])],
            "domestic_stocks_kmt": [float(params["domestic_stocks_kmt"])],
            "gdp_growth_pct": [float(params["gdp_growth_pct"])],
            "monsoon_seasonality_idx": [float(params["monsoon_seasonality_idx"])],
            "biodiesel_mandate_pct": [float(params["biodiesel_mandate_pct"])],
            "crush_margin_usd": [float(params["crush_margin_usd"])],
            "vegetable_oil_spread_usd": [spread],
            "landed_cost_usd_mt": [landed_cost]
        }
        return pd.DataFrame(data_dict)

    def predict_scenario(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Runs predictions across all 4 models and calculates economic impact."""
        df_input = self.prepare_input_dataframe(params)
        X_trans = self.preprocessor.transform(df_input)
        
        predictions = {}
        for name, model in self.models.items():
            pred_val = float(model.predict(X_trans)[0])
            predictions[name] = max(5.0, round(pred_val, 2))
            
        xgb_vol = predictions["XGBoost"]
        tariff = float(params["tariff_rate_pct"])
        cpo_price = float(params["cpo_benchmark_price_usd"])
        levy = float(params["export_levy_usd"])
        freight = float(params["freight_rate_usd"])
        cif_before_tariff = cpo_price + levy + freight
        landed_cost = cif_before_tariff * (1.0 + tariff / 100.0)
        
        # Economic calculations
        # 1. Total import bill ($ Millions)
        total_import_bill_m_usd = (xgb_vol * 1000.0 * landed_cost) / 1e6
        
        # 2. Government Tariff Revenue ($ Millions)
        tariff_revenue_m_usd = (xgb_vol * 1000.0 * cif_before_tariff * (tariff / 100.0)) / 1e6
        
        # 3. Baseline comparison at zero tariff
        params_zero = params.copy()
        params_zero["tariff_rate_pct"] = 0.0
        df_zero = self.prepare_input_dataframe(params_zero)
        X_zero = self.preprocessor.transform(df_zero)
        baseline_vol_zero_tariff = max(5.0, float(self.models["XGBoost"].predict(X_zero)[0]))
        
        # 4. Volume disruption due to tariff
        volume_delta_kmt = xgb_vol - baseline_vol_zero_tariff
        volume_delta_pct = (volume_delta_kmt / baseline_vol_zero_tariff) * 100.0 if baseline_vol_zero_tariff > 0 else 0.0
        
        # 5. Importer Price Inflation Impact (%)
        raw_inflation_pct = ((landed_cost - cif_before_tariff) / cif_before_tariff) * 100.0
        
        # 6. Soft oil substitution estimated capture
        soy_shift_kmt = max(0.0, -volume_delta_kmt * 0.72)
        
        # 7. Model agreement / variance
        all_preds = list(predictions.values())
        model_std = float(np.std(all_preds))
        model_spread_pct = ((max(all_preds) - min(all_preds)) / np.mean(all_preds)) * 100.0
        
        return {
            "predictions": predictions,
            "landed_cost_usd_mt": round(landed_cost, 2),
            "cif_before_tariff_usd_mt": round(cif_before_tariff, 2),
            "total_import_bill_m_usd": round(total_import_bill_m_usd, 2),
            "tariff_revenue_m_usd": round(tariff_revenue_m_usd, 2),
            "baseline_vol_zero_tariff_kmt": round(baseline_vol_zero_tariff, 2),
            "volume_delta_kmt": round(volume_delta_kmt, 2),
            "volume_delta_pct": round(volume_delta_pct, 2),
            "raw_inflation_pct": round(raw_inflation_pct, 2),
            "soy_shift_kmt": round(soy_shift_kmt, 2),
            "model_std": round(model_std, 2),
            "model_spread_pct": round(model_spread_pct, 2),
            "input_df": df_input
        }

    def simulate_tariff_sweep(self, base_params: Dict[str, Any], min_tariff: float = 0.0, max_tariff: float = 50.0, step: float = 2.5) -> pd.DataFrame:
        """Simulates demand response curves across a continuous range of tariff rates for all models."""
        tariffs = np.arange(min_tariff, max_tariff + 0.1, step)
        records = []
        
        for t in tariffs:
            p = base_params.copy()
            p["tariff_rate_pct"] = t
            df_in = self.prepare_input_dataframe(p)
            X_trans = self.preprocessor.transform(df_in)
            
            cpo_price = float(p["cpo_benchmark_price_usd"])
            levy = float(p["export_levy_usd"])
            freight = float(p["freight_rate_usd"])
            cif_base = cpo_price + levy + freight
            landed = cif_base * (1.0 + t / 100.0)
            
            row = {
                "tariff_rate_pct": round(t, 1),
                "landed_cost_usd_mt": round(landed, 2)
            }
            
            for name, model in self.models.items():
                pred = float(model.predict(X_trans)[0])
                row[f"{name}_vol_kmt"] = max(5.0, round(pred, 2))
                
            xgb_vol = row["XGBoost_vol_kmt"]
            row["tariff_revenue_m_usd"] = round((xgb_vol * 1000.0 * cif_base * (t / 100.0)) / 1e6, 2)
            row["import_bill_m_usd"] = round((xgb_vol * 1000.0 * landed) / 1e6, 2)
            
            records.append(row)
            
        return pd.DataFrame(records)

    def simulate_price_spread_matrix(self, base_params: Dict[str, Any]) -> pd.DataFrame:
        """Simulates a 2D matrix of Tariff Rate vs Soybean Oil Spread."""
        tariff_steps = [0, 5, 10, 15, 20, 25, 30, 40]
        spread_steps = [50, 80, 110, 140, 170, 200, 250]
        
        matrix_records = []
        cpo_base = float(base_params["cpo_benchmark_price_usd"])
        
        for t in tariff_steps:
            for s in spread_steps:
                p = base_params.copy()
                p["tariff_rate_pct"] = t
                p["soybean_oil_price_usd"] = cpo_base + s
                df_in = self.prepare_input_dataframe(p)
                X_trans = self.preprocessor.transform(df_in)
                pred_vol = float(self.models["XGBoost"].predict(X_trans)[0])
                matrix_records.append({
                    "tariff_rate_pct": t,
                    "price_spread_usd": s,
                    "predicted_volume_kmt": max(5.0, round(pred_vol, 1))
                })
        return pd.DataFrame(matrix_records)
