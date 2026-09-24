"""
model_characteristics.py - Dynamic Model Characteristics, Introspection & Evaluation Engine
Provides real-time hyperparameter introspection, holdout benchmark evaluation,
decision tree rule extraction, and neighbor analysis for LoanGuard AI.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

DEFAULT_FEATURE_NAMES = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
    "HasMortgage", "HasDependents"
]


def extract_model_characteristics(model_name, model_obj, feature_names=None):
    """
    Dynamically introspects any scikit-learn model object to extract its
    live hyperparameters, structural traits, architecture summary, and feature weights/importance.
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    info = {
        "model_name": model_name,
        "class_name": model_obj.__class__.__name__,
        "module": model_obj.__class__.__module__,
        "params": model_obj.get_params() if hasattr(model_obj, "get_params") else {},
        "architecture_subtitle": "",
        "key_metrics_label": "",
        "specs": {},
        "details": {}
    }

    class_lower = info["class_name"].lower()

    # -------------------------------------------------------------
    # Random Forest Classifier
    # -------------------------------------------------------------
    if "randomforest" in class_lower:
        n_est = getattr(model_obj, "n_estimators", 100)
        max_d = getattr(model_obj, "max_depth", None)
        class_wt = getattr(model_obj, "class_weight", None)
        crit = getattr(model_obj, "criterion", "gini")
        min_split = getattr(model_obj, "min_samples_split", 2)
        min_leaf = getattr(model_obj, "min_samples_leaf", 1)
        n_jobs = getattr(model_obj, "n_jobs", None)

        tree_depths = [e.get_depth() for e in model_obj.estimators_] if hasattr(model_obj, "estimators_") and model_obj.estimators_ else []
        tree_leaves = [e.get_n_leaves() for e in model_obj.estimators_] if hasattr(model_obj, "estimators_") and model_obj.estimators_ else []

        avg_depth = round(float(np.mean(tree_depths)), 1) if tree_depths else max_d
        max_actual_depth = int(np.max(tree_depths)) if tree_depths else max_d
        total_leaves = int(np.sum(tree_leaves)) if tree_leaves else 0

        info["architecture_subtitle"] = f"{n_est} Trees, Depth={max_d} ({'Balanced' if class_wt == 'balanced' else 'Standard'})"
        info["specs"] = {
            "Total Estimators": f"{n_est} Trees",
            "Configured Max Depth": f"{max_d} Levels" if max_d else "Unlimited",
            "Actual Avg Depth": f"{avg_depth} Levels" if avg_depth else "N/A",
            "Max Tree Depth": f"{max_actual_depth} Levels" if max_actual_depth else "N/A",
            "Class Weight": str(class_wt),
            "Splitting Criterion": crit.upper(),
            "Min Samples Leaf": str(min_leaf),
            "Min Samples Split": str(min_split),
            "Total Forest Leaves": f"{total_leaves:,}" if total_leaves else "N/A",
            "Parallel Workers": f"n_jobs={n_jobs}"
        }

        # Feature importances
        if hasattr(model_obj, "feature_importances_"):
            importances = model_obj.feature_importances_
            names = getattr(model_obj, "feature_names_in_", feature_names)
            df_imp = pd.DataFrame({
                "Feature": names,
                "Gini Importance (%)": (importances * 100).round(2),
            }).sort_values(by="Gini Importance (%)", ascending=False).reset_index(drop=True)
            df_imp["Rank"] = [f"#{i+1}" for i in range(len(df_imp))]
            df_imp["Cumulative (%)"] = df_imp["Gini Importance (%)"].cumsum().round(2)
            info["details"]["feature_importances"] = df_imp

    # -------------------------------------------------------------
    # K-Nearest Neighbors Classifier
    # -------------------------------------------------------------
    elif "kneighbors" in class_lower:
        k = getattr(model_obj, "n_neighbors", 5)
        weights = getattr(model_obj, "weights", "uniform")
        algo = getattr(model_obj, "algorithm", "auto")
        leaf_sz = getattr(model_obj, "leaf_size", 30)
        p = getattr(model_obj, "p", 2)
        metric = getattr(model_obj, "metric", "minkowski")
        n_samples_fit = getattr(model_obj, "n_samples_fit_", None)
        if n_samples_fit is None and hasattr(model_obj, "_fit_X"):
            n_samples_fit = model_obj._fit_X.shape[0]

        metric_desc = f"Minkowski (p={p}, Euclidean)" if metric == "minkowski" and p == 2 else f"{metric} (p={p})"

        info["architecture_subtitle"] = f"Instance Memory (K={k}, {str(weights).title()} Wt)"
        info["specs"] = {
            "K Neighbors": f"{k}",
            "Weighting Scheme": str(weights).title(),
            "Distance Metric": metric_desc,
            "Indexed Profiles": f"{n_samples_fit:,} Profiles" if n_samples_fit else "N/A",
            "Algorithm": str(algo).upper(),
            "Leaf Size": f"{leaf_sz}",
            "P-Norm Distance": f"p = {p}"
        }
        info["details"]["n_samples_fit"] = n_samples_fit
        info["details"]["k"] = k
        info["details"]["weights"] = weights

    # -------------------------------------------------------------
    # Logistic Regression
    # -------------------------------------------------------------
    elif "logistic" in class_lower:
        c_val = getattr(model_obj, "C", 1.0)
        penalty = getattr(model_obj, "penalty", "l2")
        solver = getattr(model_obj, "solver", "lbfgs")
        max_iter = getattr(model_obj, "max_iter", 100)
        class_wt = getattr(model_obj, "class_weight", "None")
        n_features = getattr(model_obj, "n_features_in_", len(feature_names))

        info["architecture_subtitle"] = f"Linear Model ({n_features} Beta Weights)"
        info["specs"] = {
            "Regularization (C)": f"C={c_val}",
            "Penalty Norm": str(penalty).upper() if penalty else "None",
            "Optimization Solver": str(solver).upper(),
            "Max Iterations": f"{max_iter}",
            "Class Weight": str(class_wt),
            "Fitted Coefficients": f"{n_features} Parameters"
        }

        if hasattr(model_obj, "coef_") and hasattr(model_obj, "intercept_"):
            coefs = model_obj.coef_[0]
            intercept = model_obj.intercept_[0]
            odds_ratios = np.exp(coefs)
            names = getattr(model_obj, "feature_names_in_", feature_names)

            df_weights = pd.DataFrame({
                "Feature": names,
                "Coefficient (Beta)": coefs.round(4),
                "Odds Ratio (exp(Beta))": odds_ratios.round(4),
                "Risk Direction": ["📈 Increases Default Risk" if c > 0 else "📉 Protects / Lowers Risk" for c in coefs],
                "Impact Summary": [
                    f"1 std-dev increase changes default odds by {((o-1)*100):+.1f}%" for o in odds_ratios
                ]
            }).sort_values(by="Coefficient (Beta)", ascending=False).reset_index(drop=True)

            info["details"]["intercept"] = intercept
            info["details"]["weights_table"] = df_weights

    # -------------------------------------------------------------
    # Decision Tree Classifier
    # -------------------------------------------------------------
    elif "decisiontree" in class_lower:
        max_d = getattr(model_obj, "max_depth", None)
        actual_d = model_obj.get_depth() if hasattr(model_obj, "get_depth") else max_d
        n_leaves = model_obj.get_n_leaves() if hasattr(model_obj, "get_n_leaves") else "N/A"
        class_wt = getattr(model_obj, "class_weight", "None")
        crit = getattr(model_obj, "criterion", "gini")
        node_cnt = model_obj.tree_.node_count if hasattr(model_obj, "tree_") else "N/A"

        info["architecture_subtitle"] = f"Balanced Weights (Depth={actual_d}, {n_leaves} Leaves)"
        info["specs"] = {
            "Configured Max Depth": f"{max_d}" if max_d else "Unlimited",
            "Actual Depth": f"{actual_d} Levels",
            "Splitting Criterion": str(crit).upper(),
            "Total Terminal Leaves": f"{n_leaves}",
            "Class Weight": str(class_wt),
            "Total Tree Nodes": f"{node_cnt}"
        }

    return info


def extract_tree_rules_table(dt_model, feature_names=None):
    """
    Dynamically inspects dt_model.tree_ to extract all internal splitting rules,
    thresholds, Gini impurities, and sample counts.
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    if not hasattr(dt_model, "tree_"):
        return pd.DataFrame()

    tree = dt_model.tree_
    node_count = tree.node_count
    rules = []

    for node_id in range(node_count):
        feat_idx = tree.feature[node_id]
        if feat_idx >= 0:  # Internal decision node
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature {feat_idx}"
            threshold = tree.threshold[node_id]
            impurity = tree.impurity[node_id]
            samples = tree.n_node_samples[node_id]
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]

            # Format real-world threshold based on feature
            if "Income" in feat_name or "LoanAmount" in feat_name:
                thresh_str = f"<= ${threshold:,.1f}"
            elif "Rate" in feat_name or "DTI" in feat_name:
                thresh_str = f"<= {threshold:.2f}%" if "Rate" in feat_name else f"<= {threshold:.2f}"
            elif "Age" in feat_name:
                thresh_str = f"<= {threshold:.1f} Yrs"
            else:
                thresh_str = f"<= {threshold:.2f}"

            rules.append({
                "Node ID": f"Node #{node_id}",
                "Splitting Feature": feat_name,
                "Decision Rule": thresh_str,
                "Gini Impurity": f"{impurity:.4f}",
                "Training Samples": f"{samples:,}",
                "Left Child (True)": f"Node #{left_child}",
                "Right Child (False)": f"Node #{right_child}"
            })

    return pd.DataFrame(rules)


def evaluate_dynamic_tree_path(dt_model, feature_dict, feature_names=None):
    """
    Dynamically traces an applicant profile through the decision tree
    using dt_model.decision_path without hardcoded node checks.
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    input_df = pd.DataFrame([feature_dict])[feature_names]
    tree = dt_model.tree_

    # Get traversed node IDs
    node_indicator = dt_model.decision_path(input_df)
    node_indices = node_indicator.indices

    pred_class = int(dt_model.predict(input_df)[0])
    prob = float(dt_model.predict_proba(input_df)[0][1]) * 100

    steps = []
    for step_num, node_id in enumerate(node_indices, start=1):
        feat_idx = tree.feature[node_id]
        if feat_idx >= 0:  # Decision node
            feat_name = feature_names[feat_idx]
            threshold = tree.threshold[node_id]
            applicant_val = input_df[feat_name].iloc[0]

            is_left = (applicant_val <= threshold)
            comp_sym = "<=" if is_left else ">"
            direction = "Left Branch (Threshold Met)" if is_left else "Right Branch (Threshold Exceeded)"

            if "Income" in feat_name or "LoanAmount" in feat_name:
                val_str = f"${applicant_val:,.0f}"
                thresh_str = f"${threshold:,.1f}"
            elif "Rate" in feat_name:
                val_str = f"{applicant_val:.2f}%"
                thresh_str = f"{threshold:.2f}%"
            elif "DTI" in feat_name:
                val_str = f"{applicant_val:.2f}"
                thresh_str = f"{threshold:.2f}"
            elif "Age" in feat_name:
                val_str = f"{applicant_val} Yrs"
                thresh_str = f"{threshold:.1f} Yrs"
            else:
                val_str = f"{applicant_val}"
                thresh_str = f"{threshold:.2f}"

            steps.append(
                f"**Step {step_num} (Node #{node_id}):** Evaluated **{feat_name}** = {val_str} {comp_sym} threshold {thresh_str}. Branched to **{direction}**."
            )
        else:
            # Leaf node reached
            leaf_samples = tree.n_node_samples[node_id]
            val_dist = tree.value[node_id][0]
            non_def_count = int(val_dist[0])
            def_count = int(val_dist[1])
            steps.append(
                f"**Terminal Leaf Reached (Node #{node_id}):** Training cohort contains {leaf_samples:,} applicants ({non_def_count:,} non-defaults vs {def_count:,} defaults)."
            )

    if pred_class == 1:
        outcome = f"🔴 Classified as High-Risk Default ({prob:.1f}% risk score, underwriting decline / collateral required)."
    else:
        outcome = f"🟢 Classified as Prime / Non-Default ({prob:.1f}% risk score, eligible for automated approval)."

    return steps, outcome, pred_class, prob


def evaluate_dynamic_knn_neighbors(feature_dict, knn_model, scaler, feature_names=None):
    """
    Dynamically queries knn_model with its own configured n_neighbors
    and extracts actual historical neighbor profiles, distances, and vote weights.
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    clean_dict = {}
    for f in feature_names:
        val = feature_dict.get(f, 0)
        if f in ["HasMortgage", "HasDependents"] and isinstance(val, str):
            val = 1 if val.strip().lower() in ["yes", "1", "true"] else 0
        clean_dict[f] = float(val) if val is not None else 0.0

    input_df = pd.DataFrame([clean_dict])[feature_names]
    scaled_vector = scaler.transform(input_df)

    k = knn_model.n_neighbors
    distances, indices = knn_model.kneighbors(scaled_vector, n_neighbors=k)

    dist_list = distances[0]
    idx_list = indices[0]

    # Retrieve historical outcomes if available
    labels = knn_model._y[idx_list] if hasattr(knn_model, "_y") else np.zeros(len(idx_list))

    weights = 1.0 / np.maximum(dist_list, 1e-6)
    normalized_weights = (weights / weights.sum()) * 100

    records = []
    for rank, (idx, d, lbl, w) in enumerate(zip(idx_list, dist_list, labels, normalized_weights), start=1):
        records.append({
            "Rank": f"#{rank}",
            "Cohort Index": f"ID {idx:,}",
            "Distance (Euclidean)": round(float(d), 3),
            "Vote Weight (%)": f"{w:.1f}%",
            "Historical Outcome": "🔴 Defaulted" if lbl == 1 else "🟢 Repaid on Time",
            "Status Code": int(lbl)
        })

    df_neighbors = pd.DataFrame(records)
    default_count = int((labels == 1).sum())
    repaid_count = int((labels == 0).sum())

    return df_neighbors, default_count, repaid_count, k


def compute_dynamic_benchmarks(
    models_dict,
    scaler,
    dataset_path,
    raw_dataset_path=None,
    test_size=0.2,
    knn_eval_size=5000,
    random_state=42,
    feature_names=None
):
    """
    Dynamically loads the dataset, executes holdout test evaluation across all models,
    computes all standard classification metrics, confusion matrices, empirical ROC curves,
    and dynamically determines top performers and winners.
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    target_path = dataset_path if dataset_path and Path(dataset_path).exists() else (
        raw_dataset_path if raw_dataset_path and Path(raw_dataset_path).exists() else None
    )

    if target_path is None or not Path(target_path).exists():
        raise FileNotFoundError("Dataset path not found for dynamic benchmarking.")

    # Load data
    df = pd.read_csv(target_path)
    if "LoanDefault" not in df.columns:
        raise ValueError("Target column 'LoanDefault' not present in dataset.")

    X = df[feature_names]
    y = df["LoanDefault"]

    # Reproducible stratified holdout test split (matching training notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_test_scaled = scaler.transform(X_test)
    total_test_samples = len(y_test)

    # Subsample for KNN to maintain instantaneous UI response
    knn_samples = min(knn_eval_size, total_test_samples)
    X_test_knn_scaled = X_test_scaled[:knn_samples]
    y_test_knn = y_test[:knn_samples]

    benchmarks = {}
    roc_curves = {}

    for name, model in models_dict.items():
        t0 = time.time()
        is_knn = "k-nearest" in name.lower() or "knn" in name.lower()
        is_lr = "logistic" in name.lower()

        # Choose appropriately scaled input
        if is_knn:
            X_eval = X_test_knn_scaled
            y_eval = y_test_knn
        elif is_lr:
            X_eval = X_test_scaled
            y_eval = y_test
        else:
            X_eval = X_test
            y_eval = y_test

        y_pred = model.predict(X_eval)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_eval)[:, 1]
        elif hasattr(model, "decision_function"):
            raw_scores = model.decision_function(X_eval)
            y_prob = 1 / (1 + np.exp(-raw_scores))
        else:
            y_prob = y_pred

        # Metrics
        acc = float(accuracy_score(y_eval, y_pred))
        rec_def = float(recall_score(y_eval, y_pred, pos_label=1, zero_division=0))
        prec_def = float(precision_score(y_eval, y_pred, pos_label=1, zero_division=0))
        f1_def = float(f1_score(y_eval, y_pred, pos_label=1, zero_division=0))

        rec_nondef = float(recall_score(y_eval, y_pred, pos_label=0, zero_division=0))
        prec_nondef = float(precision_score(y_eval, y_pred, pos_label=0, zero_division=0))
        f1_nondef = float(f1_score(y_eval, y_pred, pos_label=0, zero_division=0))

        macro_prec = float(precision_score(y_eval, y_pred, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_eval, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_eval, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_eval, y_pred, average="weighted", zero_division=0))

        try:
            auc = float(roc_auc_score(y_eval, y_prob))
        except Exception:
            auc = 0.5

        # Confusion Matrix
        cm = confusion_matrix(y_eval, y_pred)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # Empirical ROC Curve
        fpr, tpr, thresholds = roc_curve(y_eval, y_prob)
        # Downsample ROC curve points if very dense (> 100 points) for Plotly efficiency
        if len(fpr) > 100:
            step = max(1, len(fpr) // 80)
            fpr_sampled = list(fpr[::step]) + [1.0]
            tpr_sampled = list(tpr[::step]) + [1.0]
        else:
            fpr_sampled = list(fpr)
            tpr_sampled = list(tpr)

        roc_curves[name] = {
            "fpr": fpr_sampled,
            "tpr": tpr_sampled,
            "auc": auc
        }

        # Introspect model characteristics
        model_chars = extract_model_characteristics(name, model, feature_names)

        # Determine dynamic strengths and role
        if rec_def >= 0.60:
            role = "Primary Institutional Loss Mitigation Core (Highest Default Catch Rate)"
            strength = f"Catches {rec_def*100:.1f}% of defaults with best risk discrimination"
            weakness = "Requires explainability tooling to audit non-linear ensemble decisions"
        elif spec >= 0.98:
            role = "Conservative Tier-1 Prime Lending (Zero False Alarm Friction)"
            strength = f"Highest specificity ({spec*100:.1f}%) preserving profitable prime borrower relationships"
            weakness = f"Lower sensitivity to subprime defaults ({rec_def*100:.1f}% recall)"
        elif "decision" in name.lower():
            role = "Frontline Sieve & Instant Policy Pre-Screening"
            strength = f"Catches {rec_def*100:.1f}% of defaults with transparent explainable threshold rules"
            weakness = f"Elevated false positive rate on safe applicants ({(1-spec)*100:.1f}%)"
        else:
            role = "Peer-Cohort Risk Benchmarking & Case Auditing"
            strength = f"Local neighborhood distance comparison with {acc*100:.1f}% overall accuracy"
            weakness = "Sensitive to localized applicant cluster density"

        benchmarks[name] = {
            "Accuracy": acc,
            "Precision_Default": prec_def,
            "Recall_Default": rec_def,
            "F1_Default": f1_def,
            "Precision_NonDefault": prec_nondef,
            "Recall_NonDefault": rec_nondef,
            "F1_NonDefault": f1_nondef,
            "Macro_Precision": macro_prec,
            "Macro_Recall": macro_rec,
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1,
            "ROC_AUC": auc,
            "Specificity": spec,
            "True_Negatives": tn,
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Positives": tp,
            "Evaluation_Samples": len(y_eval),
            "Eval_Time_Sec": round(time.time() - t0, 3),
            "Type": model_chars["architecture_subtitle"],
            "Key_Strength": strength,
            "Key_Weakness": weakness,
            "Optimal_Role": role,
            "Specs": model_chars["specs"]
        }

    # Dynamically find winners across key metrics
    winners = {
        "Accuracy": max(benchmarks.items(), key=lambda x: x[1]["Accuracy"]),
        "Recall_Default": max(benchmarks.items(), key=lambda x: x[1]["Recall_Default"]),
        "Precision_Default": max(benchmarks.items(), key=lambda x: x[1]["Precision_Default"]),
        "F1_Default": max(benchmarks.items(), key=lambda x: x[1]["F1_Default"]),
        "Macro_F1": max(benchmarks.items(), key=lambda x: x[1]["Macro_F1"]),
        "Weighted_F1": max(benchmarks.items(), key=lambda x: x[1]["Weighted_F1"]),
        "ROC_AUC": max(benchmarks.items(), key=lambda x: x[1]["ROC_AUC"]),
        "Specificity": max(benchmarks.items(), key=lambda x: x[1]["Specificity"]),
        "Lowest_Misses": min(benchmarks.items(), key=lambda x: x[1]["False_Negatives"]),
    }

    metadata = {
        "total_holdout_samples": total_test_samples,
        "knn_samples": knn_samples,
        "dataset_path": str(target_path),
        "winners": winners,
        "computed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    return benchmarks, roc_curves, metadata
