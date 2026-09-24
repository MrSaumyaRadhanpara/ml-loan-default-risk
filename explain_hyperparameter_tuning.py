"""
explain_hyperparameter_tuning.py
--------------------------------
A simple, short, and self-contained script demonstrating how Hyperparameter Tuning works.

Think of Hyperparameters as the settings/knobs on a machine:
- Model Parameters: Learned by the model DURING training (e.g., weights, decision splits).
- Hyperparameters: Set by YOU BEFORE training (e.g., tree depth, learning rate).

Hyperparameter tuning is simply finding the best knob combination to get the highest score!
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ==============================================================================
# STEP 1: Load a small sample of data for quick demonstration
# ==============================================================================
print("\n" + "=" * 60)
print("  STEP 1: PREPARING DATA")
print("=" * 60)

df = pd.read_csv('preprocessed_loan_default.csv')
features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'InterestRate', 'DTIRatio']
target = 'LoanDefault'

# Take a quick sample of 3,000 rows so this script runs in seconds!
df_sample = df.sample(n=3000, random_state=42)
X = df_sample[features]
y = df_sample[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")


# ==============================================================================
# STEP 2: Baseline (Untuned) Model
# ==============================================================================
print("\n" + "=" * 60)
print("  STEP 2: TRAIN BASELINE (UNTUNED) MODEL")
print("=" * 60)

# Untuned model with initial weak settings
baseline = GradientBoostingClassifier(n_estimators=10, max_depth=1, learning_rate=0.01, random_state=42)
baseline.fit(X_train, y_train)

base_roc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])
print(f"Baseline Settings: n_estimators=10, max_depth=1, learning_rate=0.01")
print(f"Baseline Test ROC-AUC Score: {base_roc:.4f}")


# ==============================================================================
# STEP 3: Method 1 - GridSearchCV (Try EVERY combination)
# ==============================================================================
print("\n" + "=" * 60)
print("  STEP 3: METHOD 1 - GridSearchCV (Exhaustive Search)")
print("  Concept: Like trying every key on a keyring until one fits.")
print("=" * 60)

# Define exact values to test
param_grid = {
    'n_estimators': [50, 100],      # 2 choices
    'max_depth': [2, 3],            # 2 choices
    'learning_rate': [0.05, 0.1]    # 2 choices
}
# Total combinations = 2 x 2 x 2 = 8 combinations.
# With 3-fold Cross-Validation = 8 x 3 = 24 total fits.

grid = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,               # 3-Fold Cross-Validation
    scoring='roc_auc',  # Metric to optimize
    n_jobs=-1
)
grid.fit(X_train, y_train)

grid_best_model = grid.best_estimator_
grid_roc = roc_auc_score(y_test, grid_best_model.predict_proba(X_test)[:, 1])

print(f"Total Combinations Tested: {len(grid.cv_results_['params'])}")
print(f"Best Parameters Found:     {grid.best_params_}")
print(f"GridSearchCV Test ROC-AUC: {grid_roc:.4f} (Improvement: +{grid_roc - base_roc:.4f})")


# ==============================================================================
# STEP 4: Method 2 - RandomizedSearchCV (Randomly Sample Combinations)
# ==============================================================================
print("\n" + "=" * 60)
print("  STEP 4: METHOD 2 - RandomizedSearchCV (Budgeted Search)")
print("  Concept: Like picking 5 lottery tickets instead of buying all 1,000.")
print("=" * 60)

# Define a wider pool of possibilities
param_dist = {
    'n_estimators': [30, 50, 70, 100, 120],
    'max_depth': [2, 3, 4, 5],
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
    'subsample': [0.8, 0.9, 1.0]
}
# Total possibilities = 5 x 4 x 5 x 3 = 300!
# But we set n_iter=6, so it ONLY tests 6 random picks (6 x 3 = 18 fits).

random_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=6,           # Only sample 6 random combinations
    cv=3,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)

random_best_model = random_search.best_estimator_
random_roc = roc_auc_score(y_test, random_best_model.predict_proba(X_test)[:, 1])

print(f"Random Combinations Tested: 6 (out of 300 possible)")
print(f"Best Parameters Found:      {random_search.best_params_}")
print(f"RandomSearch Test ROC-AUC:  {random_roc:.4f} (Improvement: +{random_roc - base_roc:.4f})")


# ==============================================================================
# STEP 5: Summary Comparison
# ==============================================================================
print("\n" + "=" * 60)
print("  SUMMARY: HOW HYPERPARAMETER TUNING IMPROVED PERFORMANCE")
print("=" * 60)
print(f"{'Method':<20} | {'Test ROC-AUC':<14} | {'Improvement':<12}")
print("-" * 52)
print(f"{'1. Baseline (Untuned)':<20} | {base_roc:<14.4f} | {'Baseline':<12}")
print(f"{'2. GridSearchCV':<20} | {grid_roc:<14.4f} | {f'+{grid_roc - base_roc:.4f}':<12}")
print(f"{'3. RandomizedSearchCV':<20} | {random_roc:<14.4f} | {f'+{random_roc - base_roc:.4f}':<12}")
print("=" * 60 + "\n")
