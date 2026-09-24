# Implementation Plan - Modern Loan Default Prediction Application (LoanGuard AI)

Transform the existing basic Streamlit application into a modern, state-of-the-art AI Loan Risk Intelligence Platform (**LoanGuard AI**) with a sleek fintech dark-mode interface, rich multi-page navigation, interactive Plotly dashboards, model explainability, batch portfolio processing, and financial calculators.

---

## Proposed Architecture & Design

### 1. Design System & Aesthetics
- **Theme**: Fintech Dark/Slate theme with deep navy backdrop (`#0B0F19`), subtle card elevations (`#111827`, `#1E293B`), glassmorphic borders (`rgba(255, 255, 255, 0.08)`), and electric accent gradients (Indigo `#6366F1`, Emerald `#10B981`, Rose `#EF4444`, Amber `#F59E0B`, Cyan `#06B6D4`).
- **Typography**: Google Fonts (`Plus Jakarta Sans` & `Inter`) for clean, crisp financial dashboard aesthetics.
- **Micro-interactions**: Polished custom metrics, animated status badges, styled sliders/inputs, glowing risk indicators, and clean tabbed interfaces.

### 2. Multi-Page Navigation Structure

```
├── 🎯 Single Loan Risk Assessment
│   ├── Quick Profile Presets (Prime, Average, High-Risk, Young Professional)
│   ├── Interactive Underwriting Form (Loan, Financial, Demographic features)
│   ├── Real-time Plotly Risk Speedometer / Probability Gauge
│   ├── Risk Tier Badge (Minimal, Low, Moderate, High, Critical)
│   ├── Key Risk Factor Waterfall / Feature Contribution
│   ├── Automated Underwriter Recommendations & Approval Conditions
│   └── Interactive "What-If" Sensitivity Simulator
│
├── 📂 Batch Portfolio Risk Auditor
│   ├── CSV File Uploader with automatic column mapping & validation
│   ├── Instant "Load Sample Batch (500 records)" button for zero-effort demo
│   ├── Portfolio Risk Summary KPIs (Total Exposure, Predicted Defaults, At-Risk Volume)
│   ├── Risk Distribution Histogram & Credit Score vs Risk Scatter Plot
│   └── Searchable Scored Dataframe + Enriched CSV Export (Risk Tier + Decision)
│
├── 📊 Portfolio & Market Analytics (EDA)
│   ├── Real-time exploration of the 255k records dataset (cached)
│   ├── Default Rate breakdowns by Loan Purpose, Education, Employment Type
│   ├── Multi-variable correlation heatmap (DTI, Interest Rate, Credit Score, Income)
│   └── Interactive filters (Age range, Income bracket, Loan purpose)
│
├── 🧠 Model Diagnostics & Explainability
│   ├── Logistic Regression weights & Odds Ratio interpretations
│   ├── Feature Importance ranking
│   ├── Sigmoid / Log-Odds mathematical explanation & probability curves
│   └── Interactive Decision Threshold tuning (Precision vs Recall trade-off)
│
└── 🧮 EMI & Loan Affordability Calculator
    ├── Monthly Payment (EMI) & Total Interest Calculation
    ├── Amortization Schedule & Principal vs Interest Breakdown Chart
    └── Stress-Test Simulator (Rate hikes, Income shocks)
```

---

## User Review Required

> [!NOTE]
> All existing model assets (`model.pkl` and `scaler.pkl`) and 11 feature definitions (`Age`, `Income`, `LoanAmount`, `CreditScore`, `MonthsEmployed`, `NumCreditLines`, `InterestRate`, `LoanTerm`, `DTIRatio`, `HasMortgage`, `HasDependents`) will be fully preserved and utilized with exact compatibility.

---

## Proposed Changes

### Core File

#### [MODIFY] [app.py](file:///c:/Users/IndStar/OneDrive/Study/SEM%205/ML/ML_PROJECT/app.py)
- Replace the entire legacy styling and structure with the new multi-page architecture.
- Integrate Plotly charts for dynamic gauges, waterfall charts, risk matrices, and histograms.
- Implement responsive CSS styling with custom card containers, metric badges, and dark fintech theme tokens.
- Add preset loader state management, batch file processing with sample generator, explainability tabs, and EMI calculations.

---

## Verification Plan

### Automated Verification
- Run Python syntax checks on `app.py`: `python -m py_compile app.py`.
- Run headless validation tests simulating predictions, batch processing, and scaler transforms to verify 0 errors.

### Manual / Browser Verification
- Launch the Streamlit application via `streamlit run app.py` and inspect pages:
  - Verify Single Risk Assessment with presets and custom inputs.
  - Verify Plotly Gauge and Feature Waterfall charts render correctly.
  - Verify Batch Processing with uploaded CSV and built-in sample data.
  - Verify Portfolio EDA visualizations with the 255k dataset.
  - Verify Model Diagnostics & Affordability Calculator.
