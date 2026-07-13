"""
features.py — Extract per-player behavioural features from raw .npy data.
Reads:  legit.npy, cheaters.npy
Writes: features.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_raw():
    clean = np.load(DATA_DIR / "legit.npy")
    cheat = np.load(DATA_DIR / "cheaters.npy")
    data  = np.concatenate([clean, cheat], axis=0).astype(np.float32)
    data[~np.isfinite(data)] = 0.0
    y = np.concatenate([
        np.zeros(len(clean), dtype=int),
        np.ones(len(cheat),  dtype=int),
    ])
    print(f"Loaded {len(clean)} clean + {len(cheat)} cheater players")
    return data, y


def _engagement_features(eng):
    """
    Compute behavioural statistics for one engagement.
    eng: (192, 5) — [dYaw, dPitch, cvYaw, cvPitch, Firing]
    """
    dYaw, dPitch, cvYaw, cvPitch, firing = eng.T

    fired = firing.astype(bool)
    angular_vel = np.sqrt(dYaw**2 + dPitch**2)
    cv_dist     = np.sqrt(cvYaw**2 + cvPitch**2)
    fire_ticks  = np.where(fired)[0]
    # ticks strictly before the first shot (not all non-firing ticks)
    pre = np.arange(len(firing)) < fire_ticks[0] if len(fire_ticks) > 0 else np.zeros(len(firing), dtype=bool)

    return {
        # Raw aim movement
        "delta_yaw_mean":   float(dYaw.mean()),
        "delta_yaw_std":    float(dYaw.std()),
        "delta_yaw_max":    float(np.abs(dYaw).max()),
        "delta_pitch_mean": float(dPitch.mean()),
        "delta_pitch_std":  float(dPitch.std()),
        "delta_pitch_max":  float(np.abs(dPitch).max()),
        # Crosshair-to-victim offset (lower = more accurate)
        "cv_yaw_mean":   float(np.abs(cvYaw).mean()),
        "cv_yaw_std":    float(cvYaw.std()),
        "cv_pitch_mean": float(np.abs(cvPitch).mean()),
        "cv_pitch_std":  float(cvPitch.std()),
        "cv_dist_mean":  float(cv_dist.mean()),
        "cv_dist_std":   float(cv_dist.std()),
        # Pre-fire aim accuracy — aimbots lock onto target before pulling trigger
        "prefiring_cv_yaw":   float(np.abs(cvYaw[pre]).mean()) if pre.sum() > 0 else 0.0,
        "prefiring_cv_pitch": float(np.abs(cvPitch[pre]).mean()) if pre.sum() > 0 else 0.0,
        # Aim smoothness: low jerk means robotic, perfectly smooth movement
        "aim_jerk_yaw":   float(np.diff(dYaw).std()),
        "aim_jerk_pitch": float(np.diff(dPitch).std()),
        # Firing behaviour
        "firing_rate":     float(firing.mean()),
        "first_fire_tick": float(fire_ticks[0]) if len(fire_ticks) > 0 else float(len(firing)),
        # Angular velocity (flick detection)
        "max_angular_vel":  float(angular_vel.max()),
        "mean_angular_vel": float(angular_vel.mean()),
        # How often the crosshair was within 2° of the victim
        "on_target_ratio": float((cv_dist < 2.0).mean()),
        # Fire-moment features — measured only at ticks when the trigger is pulled
        "cv_dist_at_fire":       float(cv_dist[fire_ticks].mean())       if len(fire_ticks) > 0 else 0.0,
        "angular_vel_at_fire":   float(angular_vel[fire_ticks].mean())   if len(fire_ticks) > 0 else 0.0,
        # Regularity of inter-shot intervals: aimbots shoot at constant cadence (low CV)
        "fire_interval_cv": float(np.diff(fire_ticks).std() / (np.diff(fire_ticks).mean() + 1e-9))
                            if len(fire_ticks) > 1 else 0.0,
    }


def extract_features(data, y):
    """
    Aggregate per-engagement features into one row per player.
    data: (N, 30, 192, 5)
    Returns a DataFrame with 2 × n_engagement_features columns + label.
    """
    records = []
    n_engagements = data.shape[1]

    for p in range(len(data)):
        eng_rows = [_engagement_features(data[p, e]) for e in range(n_engagements)]
        eng_df   = pd.DataFrame(eng_rows)

        row = {"label": int(y[p])}
        for col in eng_df.columns:
            row[f"{col}_mean"] = eng_df[col].mean()
            row[f"{col}_std"]  = eng_df[col].std()
        records.append(row)

    return pd.DataFrame(records)


if __name__ == "__main__":
    data, y = load_raw()

    print("Extracting features...")
    df = extract_features(data, y)

    out = DATA_DIR / "features.csv"
    df.to_csv(out, index=False)

    n_feats = df.shape[1] - 1
    print(f"Saved {n_feats} features for {len(df)} players -> {out}")
    print(f"Class balance: {dict(zip(['clean','cheater'], np.bincount(y)))}")
    print(df.describe().round(4))
