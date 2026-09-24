# Walkthrough - LoanGuard AI: Modernized Loan Default Prediction Platform

The Streamlit user interface in [`app.py`](file:///c:/Users/IndStar/OneDrive/Study/SEM%205/ML/ML_PROJECT/app.py) has been completely redesigned and upgraded to a modern, enterprise-grade AI credit underwriting and risk intelligence application (**LoanGuard AI**).

---

## 🌟 Key Highlights of the Redesign

### 1. Modern Dark Fintech Design System
- **Visual Aesthetic**: Sleek glassmorphic dark theme (`#0A0E17` / `#111827` / `#1E293B`) with neon accents (Indigo `#6366F1`, Emerald `#10B981`, Rose `#EF4444`, Amber `#F59E0B`, Cyan `#38BDF8`).
- **Typography**: Integrated Google Fonts (`Plus Jakarta Sans` & `JetBrains Mono`) for modern financial dashboard readability.
- **Custom UI Components**: Metric containers with glow effects, dynamic color-coded risk tier badges, glassmorphic top navigation bar with live model health indicator, and stylized forms.

---

### 2. Multi-Page Architecture Overview

The application features 5 dedicated modules in the sidebar navigation:

| Page | Description | Key Features |
|---|---|---|
| **🎯 Single Loan Risk Assessment** | Real-time individual underwriting engine | • 5 Instant Archetype Presets (Prime, Average, High-Risk, Young Pro, High DTI)<br>• Dynamic Plotly Risk Speedometer / Probability Gauge<br>• 5-Tier Risk Classification & Automated Policy Verdict<br>• Feature Log-Odds Waterfall Impact Chart (Explainability)<br>• Automated Underwriting Policy Checklist<br>• Interactive "What-If" Sensitivity Simulator |
| **📁 Batch Portfolio Risk Auditor** | High-throughput portfolio risk auditor | • CSV Uploader + Built-in 500-Record Test Portfolio Button<br>• Total Loan Exposure & At-Risk Capital ($) KPIs<br>• Plotly Probability Distribution Histogram & Risk Tier Donut Chart<br>• Interactive Filterable Scored Applicant Ledger<br>• One-click export of enriched CSV with Risk Probabilities & Decisions |
| **📊 Portfolio & Market Analytics (EDA)** | Macro loan market intelligence | • Cached analysis of 255k loan dataset records<br>• Default rates by Education, Loan Purpose, and Employment Type<br>• Interactive Multi-Feature Correlation Matrix Heatmap<br>• Feature Distribution Boxplots (Defaults vs Non-Defaults) |
| **🧠 Model Diagnostics & Explainability** | Transparent AI & Model Governance | • Standardized Beta Coefficients & Odds Ratios ($e^{\beta}$) table with human-readable interpretations<br>• Feature Importance ranking chart<br>• Mathematical Sigmoid equation and probability transformation overview |
| **🧮 EMI & Loan Affordability Calculator** | Comprehensive borrower financial planner | • Equated Monthly Installment (EMI) and interest calculator<br>• Interactive Plotly Amortization Trajectory Chart (Principal vs Interest vs Balance)<br>• Detailed monthly amortization schedule table export |

---

## 🔍 Verification & Testing

1. **Python Compilation & Syntax**: Verified clean compilation (`python -m py_compile app.py` exited with 0 errors).
2. **End-to-End Model Pipeline**: Verified data scaling with fitted feature names, logistic regression probabilities, and metrics across all preset profiles without warnings.
3. **Streamlit Server**: Successfully launched and tested on `http://localhost:8501` returning HTTP 200.

---

## 🚀 How to Run the Application

To launch your modernized application at any time, run:

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.
