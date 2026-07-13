# Exploratory script, not part of the pipeline (pipeline.py never calls this).
# Dumps the raw .npy arrays to CSV for ad-hoc inspection in Excel/pandas;
# features.py reads legit.npy/cheaters.npy directly and does its own cleaning,
# so nothing downstream depends on the CSVs this produces.
import os
import numpy as np
import pandas as pd
from pathlib import Path


# Paths

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

clean_path = DATA_DIR / "legit.npy"
cheat_path = DATA_DIR / "cheaters.npy"

flat_csv_path = DATA_DIR / "player_behaviour_flat.csv"
long_csv_path = DATA_DIR / "sample_long_format.csv"


# Load data

clean = np.load(clean_path)
cheat = np.load(cheat_path)

# Combine into one dataset
data = np.concatenate([clean, cheat], axis=0)

# Labels matching the loaded data
y = np.concatenate([
    np.zeros(clean.shape[0], dtype=int),  # 0 = clean
    np.ones(cheat.shape[0], dtype=int)    # 1 = cheater
])

print("Raw data shape:", data.shape, "dtype:", data.dtype)
print("Labels shape:", y.shape, "class counts:", np.bincount(y))


# Sanity checks

print("Any NaNs?", np.isnan(data).any())
print("Any inf?", np.isinf(data).any())
print("Min/Max:", np.nanmin(data), np.nanmax(data))

clean_mean = data[y == 0].mean()
cheater_mean = data[y == 1].mean()
print("Clean mean:", clean_mean)
print("Cheater mean:", cheater_mean)

# Expected shape: (players, 30, 192, 5)
assert data.ndim == 4, f"Unexpected data shape: {data.shape}"
assert data.shape[1:] == (30, 192, 5), f"Unexpected per-player shape: {data.shape[1:]}"


# Basic cleaning

# Replace NaN/inf with 0.0 (simple baseline)
data = data.astype(np.float32, copy=False)
data[~np.isfinite(data)] = 0.0

# Optional quick domain check: firing rate
firing_rate = data[..., 4].mean()
print("Average firing rate:", firing_rate)


# 1) Export flattened CSV (for ML models)

num_players = data.shape[0]
X = data.reshape(num_players, -1)  # (players, 30*192*5) = (players, 28800)

# Append label column
X_with_labels = np.column_stack([X, y])  # (players, 28801)


# This CSV will have 28801 columns and Excel will not display it properly.
# Still useful for ML pipelines / loading with pandas.
tmp_path = flat_csv_path.with_suffix(flat_csv_path.suffix + ".tmp")

with open(tmp_path, "w", newline="") as f:
    np.savetxt(f, X_with_labels, delimiter=",", fmt="%.6f")

# Atomic rename (prevents partial/empty files if something goes wrong)
if os.path.exists(flat_csv_path):
    os.remove(flat_csv_path)
os.rename(tmp_path, flat_csv_path)

print("\nSaved FLATTENED CSV to:", flat_csv_path)
print("Flattened file size (bytes):", os.path.getsize(flat_csv_path))

# Print preview (first line snippet)
with open(flat_csv_path, "r") as f:
    print("Flattened preview:", f.readline()[:200], "...")


# 2) Export long-format sample CSV (Excel-friendly)

# Export only a small number of players so the file is manageable.
players_to_export = min(5, num_players)  # change to 10 if you want
rows = []

for p in range(players_to_export):
    label = int(y[p])
    for e in range(30):
        for t in range(192):
            aDy, aDp, cVy, cVp, firing = data[p, e, t]
            rows.append([
                p, e, t,
                float(aDy), float(aDp), float(cVy), float(cVp),
                int(round(firing)),  # ensure 0/1
                label
            ])

df = pd.DataFrame(rows, columns=[
    "player_id", "engagement", "tick",
    "AttackerDeltaYaw", "AttackerDeltaPitch",
    "CrosshairToVictimYaw", "CrosshairToVictimPitch",
    "Firing", "label"
])

df.to_csv(long_csv_path, index=False)

print("\nSaved LONG-FORMAT sample CSV to:", long_csv_path)
print("Long-format rows:", len(df), "cols:", df.shape[1])
print("Long-format file size (bytes):", os.path.getsize(long_csv_path))
print(df.head())
