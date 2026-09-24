"""
hyperparameter_tuning_engine.py
Helper module for LoanGuard AI hyperparameter tuning analytics,
visualization of GridSearchCV vs RandomizedSearchCV gains,
and live inference using the tuned champion model.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent
TUNING_RESULTS_FILE = BASE_DIR / "hyperparameter_tuning_results.json"
TUNED_MODEL_FILE = BASE_DIR / "models" / "tuned_best_model.pkl"

FEATURE_NAMES = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
    "HasMortgage", "HasDependents"
]

def load_tuning_results():
    """Loads hyperparameter tuning benchmark results."""
    for p in [TUNING_RESULTS_FILE, Path("hyperparameter_tuning_results.json")]:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading tuning results from {p}: {e}")
    return None

def load_tuned_model():
    """Loads the serialized tuned champion model with robust fallback resolution."""
    candidate_paths = [
        TUNED_MODEL_FILE,
        Path("models/tuned_best_model.pkl"),
        Path("tuned_best_model.pkl"),
        BASE_DIR / "tuned_best_model.pkl"
    ]
    for p in candidate_paths:
        if p.exists():
            try:
                return joblib.load(p)
            except Exception:
                try:
                    import pickle
                    with open(p, "rb") as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Error loading tuned model from {p}: {e}")
    return None

def build_tuning_metrics_chart(tuning_data):
    """Builds an interactive Plotly grouped bar chart comparing Baseline vs GridSearchCV vs RandomizedSearchCV."""
    stages = ["Baseline (Untuned)", "GridSearchCV Tuned", "RandomizedSearchCV Tuned"]
    
    base_m = tuning_data["baseline"]["test_metrics"]
    grid_m = tuning_data["grid_search"]["test_metrics"]
    rand_m = tuning_data["randomized_search"]["test_metrics"]

    fig = go.Figure()

    # ROC-AUC trace
    fig.add_trace(go.Bar(
        name="ROC-AUC Score",
        x=stages,
        y=[base_m["ROC_AUC"], grid_m["ROC_AUC"], rand_m["ROC_AUC"]],
        marker_color="#8B5CF6",
        text=[f"{v:.4f}" for v in [base_m["ROC_AUC"], grid_m["ROC_AUC"], rand_m["ROC_AUC"]]],
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=11)
    ))

    # Test Accuracy trace
    fig.add_trace(go.Bar(
        name="Accuracy (%)",
        x=stages,
        y=[base_m["Accuracy"] * 100, grid_m["Accuracy"] * 100, rand_m["Accuracy"] * 100],
        marker_color="#3B82F6",
        text=[f"{v*100:.2f}%" for v in [base_m["Accuracy"], grid_m["Accuracy"], rand_m["Accuracy"]]],
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=11)
    ))

    # F1 Score trace
    fig.add_trace(go.Bar(
        name="F1-Score (Defaults)",
        x=stages,
        y=[base_m["F1_Score"] * 100, grid_m["F1_Score"] * 100, rand_m["F1_Score"] * 100],
        marker_color="#10B981",
        text=[f"{v*100:.2f}%" for v in [base_m["F1_Score"], grid_m["F1_Score"], rand_m["F1_Score"]]],
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=11)
    ))

    fig.update_layout(
        barmode="group",
        title=dict(
            text="<b>Performance Verification: Baseline vs GridSearchCV vs RandomizedSearchCV</b>",
            font=dict(size=15, color="#F8FAFC", family="Plus Jakarta Sans")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        margin=dict(l=40, r=30, t=50, b=40),
        height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#E2E8F0", size=11)
        ),
        yaxis=dict(
            title=dict(text="Score / Percentage", font=dict(color="#94A3B8", size=11)),
            gridcolor="rgba(255, 255, 255, 0.06)",
            tickfont=dict(color="#94A3B8")
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.03)",
            tickfont=dict(color="#CBD5E1")
        )
    )
    return fig

def predict_with_tuned_model(applicant_dict, tuned_model=None):
    """Executes dynamic real-time prediction using the tuned champion model."""
    if tuned_model is None:
        tuned_model = load_tuned_model()
    if tuned_model is None:
        return {
            "error": "Tuned model not loaded.",
            "prediction": 0,
            "probability": 0.0,
            "risk_tier": "Model Initializing",
            "risk_color": "#94A3B8",
            "decision": "MODEL INITIALIZING",
            "recommendation": "Tuned champion model weights are active and initializing."
        }

    row_data = {}
    for f in FEATURE_NAMES:
        val = applicant_dict.get(f, 0)
        if f in ["HasMortgage", "HasDependents"] and isinstance(val, str):
            val = 1 if val.strip().lower() in ["yes", "1", "true"] else 0
        row_data[f] = [float(val)]
    df_app = pd.DataFrame(row_data)
    raw_input = df_app[FEATURE_NAMES].values

    pred = int(tuned_model.predict(raw_input)[0])
    prob = float(tuned_model.predict_proba(raw_input)[0, 1])

    if prob < 0.15:
        tier = "Tier 1: Minimal Risk (Prime)"
        color = "#10B981"
        rec = "Auto-Approve with Prime Terms"
    elif prob < 0.35:
        tier = "Tier 2: Low Risk (Standard)"
        color = "#06B6D4"
        rec = "Standard Approval with Automated Verification"
    elif prob < 0.55:
        tier = "Tier 3: Moderate Risk (Borderline)"
        color = "#F59E0B"
        rec = "Manual Review & Stricter Collateral Verification"
    elif prob < 0.75:
        tier = "Tier 4: High Risk (Subprime)"
        color = "#F97316"
        rec = "Decline or Co-Signer Required"
    else:
        tier = "Tier 5: Critical Risk (Default Imminent)"
        color = "#EF4444"
        rec = "Strict Decline - High Default Risk"

    return {
        "prediction": pred,
        "probability": prob,
        "risk_tier": tier,
        "risk_color": color,
        "decision": "DECLINE / HIGH RISK" if pred == 1 else "APPROVE / LOW RISK",
        "recommendation": rec
    }
