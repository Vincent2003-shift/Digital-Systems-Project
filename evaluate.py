"""
evaluate.py — Evaluate all trained models and produce dissertation metrics.
Reads:  models/*.pkl
Writes: results/roc_curves.png, confusion_matrices.png,
        feature_importance.png, anomaly_score_distribution.png,
        metrics_summary.csv
"""
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.inspection import permutation_importance

# Features most relevant to the aimbot hypotheses — used for error inspection.
KEY_FEATURES = [
    "cv_dist_at_fire_mean", "angular_vel_at_fire_mean", "fire_interval_cv_mean",
    "on_target_ratio_mean", "cv_dist_mean_mean", "prefiring_cv_yaw_mean",
    "aim_jerk_yaw_mean", "max_angular_vel_mean", "first_fire_tick_mean",
    "firing_rate_mean",
]

ROOT_DIR    = Path(__file__).resolve().parent
MODEL_DIR   = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED       = 42
CLR_MAIN   = "#0072B2"  # colorblind-safe blue
CLR_ACCENT = "#D55E00"  # colorblind-safe orange

plt.rcParams.update({"font.size": 11, "figure.dpi": 140})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all():
    saved      = joblib.load(MODEL_DIR / "test_data.pkl")
    iso        = joblib.load(MODEL_DIR / "isolation_forest.pkl")
    iso_thr    = joblib.load(MODEL_DIR / "iso_threshold.pkl")
    rule_thr   = joblib.load(MODEL_DIR / "rule_thresholds.pkl")
    rf         = joblib.load(MODEL_DIR / "random_forest.pkl")
    clf        = joblib.load(MODEL_DIR / f"{saved['clf_name']}.pkl")
    try:
        iso_unsup = joblib.load(MODEL_DIR / "isolation_forest_unsup.pkl")
    except FileNotFoundError:
        iso_unsup = None
    return iso, iso_thr, iso_unsup, rule_thr, rf, clf, saved


def recall_at_fpr(y_true, y_score, fpr_target=0.01):
    """Recall at the highest threshold where FPR ≤ fpr_target."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = fpr <= fpr_target
    return float(tpr[valid].max()) if valid.any() else 0.0


def threshold_predict(score, threshold=0.5):
    return (score >= threshold).astype(int)


def export_error_analysis(y_test, score, X_test_scaled, scaler, feature_names,
                          results_dir, threshold=0.5):
    """
    Split test players into TP/FP/FN/TN for the best model, report their
    key feature values in ORIGINAL units, and save two CSVs:
      - error_groups_summary.csv : mean of each key feature per outcome group
      - error_players.csv        : every FP and FN individually
    """
    y_pred = (score >= threshold).astype(int)

    # Back to original units so the numbers in the report are interpretable
    X_orig = scaler.inverse_transform(X_test_scaled)
    df = pd.DataFrame(X_orig, columns=feature_names)
    df["true_label"] = y_test
    df["pred_label"] = y_pred
    df["model_score"] = score

    conditions = {
        "TP (caught cheaters)":   (y_test == 1) & (y_pred == 1),
        "FN (missed cheaters)":   (y_test == 1) & (y_pred == 0),
        "FP (wrongly flagged)":   (y_test == 0) & (y_pred == 1),
        "TN (correctly cleared)": (y_test == 0) & (y_pred == 0),
    }

    keys = [f for f in KEY_FEATURES if f in feature_names]  # defensive

    rows = []
    for name, mask in conditions.items():
        row = {"group": name, "n_players": int(mask.sum()),
               "mean_model_score": float(score[mask].mean()) if mask.any() else np.nan}
        for f in keys:
            row[f] = float(df.loc[mask, f].mean()) if mask.any() else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows).round(4)
    summary.to_csv(results_dir / "error_groups_summary.csv", index=False)

    errors = df[(y_test != y_pred)].copy()
    errors["error_type"] = np.where(errors["true_label"] == 1,
                                    "FN (missed cheater)", "FP (wrongly flagged)")
    cols = ["error_type", "model_score"] + keys
    errors[cols].round(4).to_csv(results_dir / "error_players.csv", index=False)

    print("\n--- Error analysis (XGBoost @ threshold %.2f) ---" % threshold)
    print(summary.to_string(index=False))
    print(f"Saved {results_dir / 'error_groups_summary.csv'}")
    print(f"Saved {results_dir / 'error_players.csv'}")


def threshold_use_case_table(y_test, score, results_dir,
                             thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    """
    Precision / recall / F1 / FPR of the best model at several operating
    thresholds — the 'what could this actually be used for' table (Table 3).
    """
    rows = []
    for t in thresholds:
        pred = (score >= t).astype(int)
        fpr = ((pred == 1) & (y_test == 0)).sum() / max((y_test == 0).sum(), 1)
        rows.append({
            "Threshold": t,
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "FPR": fpr,
            "Flagged players": int(pred.sum()),
        })
    table = pd.DataFrame(rows).round(4)
    table.to_csv(results_dir / "threshold_use_case_table.csv", index=False)
    print("\n--- Threshold / use-case table (XGBoost) ---")
    print(table.to_string(index=False))
    print(f"Saved {results_dir / 'threshold_use_case_table.csv'}")


def rule_based_predict(X_raw, feature_names, rule_thresholds):
    """
    Flag a player if angular_vel_at_fire_mean OR delta_yaw_std_std falls below
    the 5th percentile of clean training players. Both features are negatively
    correlated with cheating: aimbots fire with minimal movement (low angular
    velocity) and produce unnaturally consistent behaviour across engagements
    (low cross-engagement variance).
    """
    names  = list(feature_names)
    av_idx = names.index("angular_vel_at_fire_mean")
    dy_idx = names.index("delta_yaw_std_std")
    return (
        (X_raw[:, av_idx] < rule_thresholds["angular_vel_at_fire_mean"]) |
        (X_raw[:, dy_idx] < rule_thresholds["delta_yaw_std_std"])
    ).astype(int)


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(name, y_true, y_score, y_pred=None, threshold=0.5):
    if y_pred is None:
        y_pred = threshold_predict(y_score, threshold)
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    print(classification_report(y_true, y_pred,
                                target_names=["Clean", "Cheater"], digits=4))
    auc = roc_auc_score(y_true, y_score)
    ap  = average_precision_score(y_true, y_score)
    f1  = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    r1  = recall_at_fpr(y_true, y_score)
    print(f"ROC-AUC      : {auc:.4f}")
    print(f"Avg Precision: {ap:.4f}")
    print(f"Recall@1%FPR : {r1:.4f}")
    return {"Model": name, "ROC-AUC": auc, "Avg Precision": ap,
            "F1": f1, "Precision": pre, "Recall": rec, "Recall@1%FPR": r1}


def evaluate_rule_based(name, y_true, y_pred):
    """Rule-based models have no continuous score, so ROC-AUC is not applicable."""
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    print(classification_report(y_true, y_pred,
                                target_names=["Clean", "Cheater"], digits=4))
    f1  = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    return {"Model": name, "ROC-AUC": None, "Avg Precision": None,
            "F1": f1, "Precision": pre, "Recall": rec, "Recall@1%FPR": None}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_roc_curves(models_data, y_test):
    scored = [(n, s, p) for n, s, p in models_data if s is not None]
    fig, axes = plt.subplots(1, len(scored), figsize=(5 * len(scored), 5), sharey=True)
    for ax, (name, score, _) in zip(np.atleast_1d(axes), scored):
        disp = RocCurveDisplay.from_predictions(y_test, score, ax=ax, name=name)
        disp.line_.set_color(CLR_MAIN)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
        ax.set_title(name)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("ROC Curves — Cheat Detection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = RESULTS_DIR / "roc_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")


def plot_confusion_matrices(models_data, y_test):
    fig, axes = plt.subplots(1, len(models_data), figsize=(5 * len(models_data), 4))
    for ax, (name, _, pred) in zip(np.atleast_1d(axes), models_data):
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred,
            display_labels=["Clean", "Cheater"],
            colorbar=False, ax=ax, cmap="Blues",
        )
        ax.set_title(name)
    fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = RESULTS_DIR / "confusion_matrices.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_feature_importance(rf, X_test, y_test, feature_names, top_n=20):
    result    = permutation_importance(rf, X_test, y_test, n_repeats=10,
                                       scoring="roc_auc", random_state=SEED, n_jobs=-1)
    imps      = result.importances_mean
    idx       = np.argsort(imps)[-top_n:]
    names_top = [feature_names[i] for i in idx]
    vals_top  = imps[idx]

    colors = [CLR_ACCENT if any(k in n for k in ("cv_dist", "prefiring", "on_target"))
              else CLR_MAIN for n in names_top]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(names_top, vals_top, color=colors, edgecolor="white")
    ax.set_xlabel("Permutation Importance (ROC-AUC drop)")
    ax.set_title(f"Random Forest — Top {top_n} Features (Permutation Importance)\n"
                 "(orange = aim-accuracy features, blue = movement features)")
    plt.tight_layout()
    out = RESULTS_DIR / "feature_importance.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_anomaly_score_distribution(iso_scores, y_test, iso_thr):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, name, color in [(0, "Clean", CLR_MAIN), (1, "Cheater", CLR_ACCENT)]:
        vals = iso_scores[y_test == label]
        ax.hist(vals, bins=40, alpha=0.6, density=True, label=name, color=color)
    ax.axvline(iso_thr, color="black", linestyle="--", lw=1.2,
               label=f"Calibrated threshold ({iso_thr:.3f})")
    ax.set_xlabel("Isolation Forest Anomaly Score (higher = more suspicious)")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly Score Distribution: Clean vs Cheater\n"
                 "(threshold calibrated from 95th pct of clean training players)")
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "anomaly_score_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    iso, iso_thr, iso_unsup, rule_thr, rf, clf, saved = load_all()
    X_test        = saved["X_test"]
    X_test_raw    = saved["X_test_raw"]
    y_test        = saved["y_test"]
    feature_names = saved["feature_names"]
    clf_name      = saved["clf_name"].replace("_", " ").title()

    iso_score_arr = -iso.score_samples(X_test)   # higher = more suspicious
    rf_score      = rf.predict_proba(X_test)[:, 1]
    clf_score     = clf.predict_proba(X_test)[:, 1]

    iso_pred  = (iso_score_arr >= iso_thr).astype(int)
    rule_pred = rule_based_predict(X_test_raw, feature_names, rule_thr)

    models_data = [
        ("Isolation Forest (semi-supervised)", iso_score_arr, iso_pred),
        ("Random Forest",                      rf_score,      rf.predict(X_test)),
        (clf_name,                             clf_score,     clf.predict(X_test)),
        ("Rule-Based",                         None,          rule_pred),
    ]

    if iso_unsup is not None:
        iso_unsup_score = -iso_unsup.score_samples(X_test)
        iso_unsup_pred  = (iso_unsup.predict(X_test) == -1).astype(int)
        models_data.insert(1, ("Isolation Forest (unsupervised)", iso_unsup_score, iso_unsup_pred))

    rows = []
    for name, score, pred in models_data:
        if score is not None:
            rows.append(evaluate_model(name, y_test, score, y_pred=pred))
        else:
            rows.append(evaluate_rule_based(name, y_test, pred))

    plot_roc_curves(models_data, y_test)
    plot_confusion_matrices(models_data, y_test)
    plot_feature_importance(rf, X_test, y_test, feature_names)
    plot_anomaly_score_distribution(iso_score_arr, y_test, iso_thr)

    summary = pd.DataFrame(rows).round(4)
    summary.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    print("\n--- Summary -------------------------------------------")
    print(summary.to_string(index=False))

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    export_error_analysis(y_test, clf_score, X_test, scaler, feature_names, RESULTS_DIR)
    threshold_use_case_table(y_test, clf_score, RESULTS_DIR)

    print(f"\nAll results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
