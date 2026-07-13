"""
visualise.py — Dissertation-quality exploratory visualisations.
Reads:  features.csv
Writes: figures/feature_distributions.png, label_correlation.png,
        scatter_accuracy_smoothness.png, boxplots.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
FIG_DIR  = ROOT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

COLORS = {"Clean": "#0072B2", "Cheater": "#D55E00"}  # Wong colorblind-safe palette

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load():
    df = pd.read_csv(DATA_DIR / "features.csv")
    df["Player Type"] = df["label"].map({0: "Clean", 1: "Cheater"})
    return df


# ---------------------------------------------------------------------------
# 1. Feature distribution histograms
# ---------------------------------------------------------------------------

def plot_feature_distributions(df):
    features_of_interest = [
        ("cv_dist_mean_mean",       "Mean Crosshair Distance to Victim (°)"),
        ("prefiring_cv_yaw_mean",   "Pre-Fire Aim Offset — Yaw (°)"),
        ("prefiring_cv_pitch_mean", "Pre-Fire Aim Offset — Pitch (°)"),
        ("aim_jerk_yaw_mean",       "Aim Jerk / Smoothness — Yaw"),
        ("max_angular_vel_mean",    "Max Angular Velocity (flick speed)"),
        ("firing_rate_mean",        "Firing Rate (fraction of ticks)"),
        ("first_fire_tick_mean",    "Reaction Time (first fire tick)"),
        ("on_target_ratio_mean",    "On-Target Ratio (within 2°)"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (col, title) in zip(axes.flat, features_of_interest):
        for ptype, color in COLORS.items():
            vals = df.loc[df["Player Type"] == ptype, col].dropna()
            ax.hist(vals, bins=30, alpha=0.55, density=True,
                    label=ptype, color=color, edgecolor="none")
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_ylabel("Density")

    handles = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Behavioural Feature Distributions: Clean vs Cheater Players",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    out = FIG_DIR / "feature_distributions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# 2. Correlation bar chart
# ---------------------------------------------------------------------------

def plot_label_correlation(df):
    feat_cols = [c for c in df.columns if c not in ("label", "Player Type")]
    corr  = df[feat_cols + ["label"]].corr()["label"].drop("label").sort_values()
    top   = pd.concat([corr.head(10), corr.tail(10)])

    colors = [COLORS["Cheater"] if v > 0 else COLORS["Clean"] for v in top.values]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.index, top.values, color=colors, edgecolor="none")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson Correlation with Label (1 = Cheater)")
    ax.set_title("Top 20 Features by Correlation with Cheating\n"
                 "(red = positively correlated with cheating)",
                 fontweight="bold")
    handles = [mpatches.Patch(color=COLORS["Cheater"], label="→ More cheating"),
               mpatches.Patch(color=COLORS["Clean"],   label="→ More clean")]
    ax.legend(handles=handles, fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "label_correlation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# 3. Scatter: aim accuracy vs aim smoothness
# ---------------------------------------------------------------------------

def plot_scatter(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    for ptype, (marker, alpha) in [("Clean", ("o", 0.45)), ("Cheater", ("x", 0.65))]:
        sub = df[df["Player Type"] == ptype]
        ax.scatter(
            sub["cv_dist_mean_mean"],
            sub["aim_jerk_yaw_mean"],
            c=COLORS[ptype], marker=marker,
            alpha=alpha, s=35, label=ptype, linewidths=0.8,
        )
    ax.set_xlabel("Mean Crosshair Distance to Victim (°)\n← more accurate")
    ax.set_ylabel("Aim Jerk — Yaw\n← smoother / more robotic")
    ax.set_title("Aim Accuracy vs Movement Smoothness", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out = FIG_DIR / "scatter_accuracy_smoothness.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# 4. Box plots for key features
# ---------------------------------------------------------------------------

def plot_boxplots(df):
    features = [
        ("cv_dist_mean_mean",    "Crosshair Distance (°)"),
        ("on_target_ratio_mean", "On-Target Ratio"),
        ("aim_jerk_yaw_mean",    "Aim Jerk"),
        ("first_fire_tick_mean", "Reaction Time (ticks)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, (col, label) in zip(axes, features):
        data_by_group = [
            df.loc[df["Player Type"] == pt, col].dropna().values
            for pt in ["Clean", "Cheater"]
        ]
        bp = ax.boxplot(data_by_group, patch_artist=True,
                        medianprops=dict(color="white", lw=2),
                        whiskerprops=dict(lw=1.2),
                        capprops=dict(lw=1.2))
        for patch, color in zip(bp["boxes"], COLORS.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Clean", "Cheater"])
        ax.set_title(label, fontsize=10, fontweight="bold")

    fig.suptitle("Key Feature Distributions by Player Type", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "boxplots.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load()
    print(f"Loaded {len(df)} players  ({df['Player Type'].value_counts().to_dict()})")
    plot_feature_distributions(df)
    plot_label_correlation(df)
    plot_scatter(df)
    plot_boxplots(df)
    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
