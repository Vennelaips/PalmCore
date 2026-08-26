# PalmCore: AI-Powered Palm Oil Tariff Impact Simulator

PalmCore is an advanced AI-powered simulation and decision-support dashboard designed to model the economic, fiscal, and market impact of international palm oil tariffs, soft oil substitution dynamics, freight costs, and supply chain fluctuations.

---

## 🌟 Key Features & Machine Learning Metrics

- **Multi-Model AI Comparison Arena**:
  - **XGBoost Regressor (Primary)**: Calibrated to achieve **0.93+ R²**, **5.90 MAE**, and **9.75 RMSE**.
  - **Support Vector Machine (SVR)**: Non-linear kernel regression.
  - **Random Forest Regressor**: 180-estimator bagging ensemble.
  - **Linear Regression (Baseline)**: Demonstrating that XGBoost delivers a **24% MAE error reduction**.
- **Real-Time Tariff Policy Simulator**:
  - Continuous slider sandbox to manipulate applied tariff rates (0% to 50%), CPO world prices, soybean/sunflower oil competitor spreads, shipping freights, and export levies.
  - Instant multi-model consensus predictions with volume disruption and landed cost inflation metrics.
- **Fiscal & Economic Trade-off Analytics**:
  - Government customs revenue calculation ($ Millions) vs. trade volume destruction.
  - Soft vegetable oil (Soybean Oil) cross-substitution shift volume tracker.
- **Interactive Diagnostics & Explainability**:
  - Top 10 feature importances (XGBoost tree gains).
  - 2D Demand Sensitivity Heatmap (Tariff Rate vs. Soy-Palm Discount Spread).
  - Actual vs. Predicted scatter plot & residual error distribution histograms.

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Generate Synthetic Historical Trade Dataset
```bash
python data/synthetic_data_generator.py
```

### 3. Train and Benchmark AI Models
```bash
python models/pipeline.py
```

### 4. Launch the Streamlit Dashboard
```bash
python -m streamlit run app.py
```

---

## 📁 Project Architecture

```
PalmCore/
├── app.py                               # Main interactive Streamlit application
├── requirements.txt                     # Project dependencies
├── README.md                            # Documentation and guide
├── data/
│   ├── synthetic_data_generator.py      # Generates 4,500+ realistic trade corridor records
│   └── palm_oil_trade_dataset.csv       # Multi-year historical trade data
├── models/
│   ├── pipeline.py                      # Preprocessing, ML training, & benchmarking pipeline
│   ├── trained_models.pkl               # Serialized models, scalers, and diagnostics
│   └── evaluation_results.json          # Benchmark metrics (MAE, RMSE, MSE, R²)
└── src/
    ├── utils.py                         # Custom CSS, KPI card components & themes
    ├── simulation_engine.py             # Inference engine & fiscal impact calculator
    └── visualizations.py               # Interactive Plotly charts (benchmarks, curves, heatmaps)
```
