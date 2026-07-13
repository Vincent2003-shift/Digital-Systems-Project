"""
train.py — Train cheat-detection models on extracted features.
Reads:  features.csv
Writes: models/scaler.pkl, isolation_forest.pkl, random_forest.pkl,
        xgboost.pkl (or gradient_boosting.pkl), test_data.pkl
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    _XGB = False
    print("XGBoost not found — falling back to GradientBoostingClassifier")

ROOT_DIR  = Path(__file__).resolve().parent
DATA_DIR  = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

SEED      = 42
TEST_SIZE = 0.20


def load_features():
    df   = pd.read_csv(DATA_DIR / "features.csv")
    cols = [c for c in df.columns if c != "label"]
    X    = df[cols].values.astype(np.float32)
    y    = df["label"].values.astype(int)
    return X, y, cols


def main():
    X, y, feature_names = load_features()
    print(f"Dataset: {X.shape[0]} players, {X.shape[1]} features")
    print(f"Class balance: {dict(zip(['clean','cheater'], np.bincount(y)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    scaler      = StandardScaler()
    X_train_s   = scaler.fit_transform(X_train)
    X_test_s    = scaler.transform(X_test)

 
    # Rule-based thresholds — derived from clean training players only
    # Both features are negatively correlated with cheating (lower = more suspicious)
  
    fn = list(feature_names)
    clean_train = X_train[y_train == 0]
    rule_thresholds = {
        "angular_vel_at_fire_mean": float(np.percentile(clean_train[:, fn.index("angular_vel_at_fire_mean")], 5)),
        "delta_yaw_std_std":        float(np.percentile(clean_train[:, fn.index("delta_yaw_std_std")], 5)),
    }
    joblib.dump(rule_thresholds, MODEL_DIR / "rule_thresholds.pkl")
    print(f"Rule thresholds (5th pct clean): {rule_thresholds}")

  
    # 1. Isolation Forest — unsupervised anomaly detection
    #    Trained ONLY on clean players; cheaters are never seen during fit.

    clean_mask = y_train == 0
    iso = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=SEED,
    )
    iso.fit(X_train_s[clean_mask])
    print(f"\nIsolation Forest (semi-supervised) trained on {clean_mask.sum()} clean players")

    # Calibrate decision threshold from clean training players — avoids test-set leakage.
    # Flag any player whose anomaly score exceeds the 95th percentile of clean players.
    clean_scores  = iso.score_samples(X_train_s[clean_mask])
    iso_threshold = -float(np.percentile(clean_scores, 5))  # negate: higher = more suspicious
    joblib.dump(iso_threshold, MODEL_DIR / "iso_threshold.pkl")
    print(f"  Calibrated threshold (95th pct clean anomaly score): {iso_threshold:.4f}")

  
    # Fully unsupervised IF — trained on all players, no label used

    iso_unsup = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED)
    iso_unsup.fit(X_train_s)
    joblib.dump(iso_unsup, MODEL_DIR / "isolation_forest_unsup.pkl")
    print(f"Isolation Forest (unsupervised) trained on all {len(X_train_s)} players")

  
    # 2. Random Forest — supervised baseline

    rf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)

    cv_rf = cross_val_score(rf, X_train_s, y_train,
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            scoring="roc_auc", n_jobs=-1)
    print(f"Random Forest   5-fold ROC-AUC: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")


    # 3. XGBoost / Gradient Boosting — second supervised model
  
    if _XGB:
        clf      = XGBClassifier(n_estimators=300, learning_rate=0.05,
                                 max_depth=6, subsample=0.8,
                                 eval_metric="logloss", random_state=SEED)
        clf_name = "xgboost"
    else:
        clf      = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                              max_depth=4, random_state=SEED)
        clf_name = "gradient_boosting"

    clf.fit(X_train_s, y_train)

    cv_clf = cross_val_score(clf, X_train_s, y_train,
                             cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                             scoring="roc_auc", n_jobs=-1)
    print(f"{clf_name.replace('_',' ').title():30s} 5-fold ROC-AUC: {cv_clf.mean():.4f} ± {cv_clf.std():.4f}")

  
    # Save everything
  
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(iso,    MODEL_DIR / "isolation_forest.pkl")
    joblib.dump(rf,     MODEL_DIR / "random_forest.pkl")
    joblib.dump(clf,    MODEL_DIR / f"{clf_name}.pkl")
    joblib.dump({
        "X_test":        X_test_s,
        "X_test_raw":    X_test,    # unscaled — needed for rule-based baseline
        "y_test":        y_test,
        "feature_names": feature_names,
        "clf_name":      clf_name,
    }, MODEL_DIR / "test_data.pkl")

    print(f"\nAll models saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
