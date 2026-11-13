# %% [markdown]
# # CS04: Hierarchical segment profiling (Olist)
#
# Snapshot date: 2018-01-01. Ward linkage runs on a 5,000-customer sample (it needs O(n²) memory).
# Every customer is then labeled by the nearest Ward centroid, and a depth-3 decision tree
# turns the labels into rules that a non-technical reviewer can read.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

from hierseg.core import (
    FEATURES,
    RANDOM_STATE,
    SNAPSHOT_DATE,
    build_customer_features,
    choose_k,
    data_dir,
    load_tables,
    nearest_centroid_labels,
    scale_features,
    surrogate_tree,
    ward_labels,
)

FIG = Path.cwd().parent / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# %%
tables = load_tables()
print("data:", data_dir())
feats = build_customer_features(tables, SNAPSHOT_DATE)
print("customers:", len(feats))
print("share with exactly one order before snapshot:", round((feats["frequency"] == 1).mean(), 3))
feats.describe().round(2)

# %%
X = scale_features(feats)
sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(X), size=5000, replace=False)
X_sample = X[sample_idx]

k_table = choose_k(X_sample, k_range=range(3, 7))
k_table.round(3)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(k_table["k"], k_table["silhouette"], marker="o")
ax.set_xlabel("k")
ax.set_ylabel("silhouette (Ward, sample)")
ax.set_title("CS04: silhouette by number of Ward clusters")
fig.tight_layout()
fig.savefig(FIG / "cs04_silhouette_by_k.png", dpi=150)
plt.show()

# %%
K = int(k_table.loc[k_table["silhouette"].idxmax(), "k"])
print("chosen k:", K)
labels_sample, Z = ward_labels(X_sample, K)

fig, ax = plt.subplots(figsize=(10, 4))
dendrogram(Z, truncate_mode="lastp", p=40, ax=ax, no_labels=True, color_threshold=Z[-(K - 1), 2])
ax.set_title("CS04: Ward dendrogram (sample, last 40 merges)")
ax.set_ylabel("merge distance")
fig.tight_layout()
fig.savefig(FIG / "cs04_dendrogram.png", dpi=150)
plt.show()

# %%
labels_all = nearest_centroid_labels(X_sample, labels_sample, X)
feats = feats.assign(segment=labels_all)
shares = feats["segment"].value_counts(normalize=True).sort_index()
print("share by segment:")
print(shares.round(3))
print("smallest segment share:", round(shares.min(), 4), "(minimum is 0.02)")

# %%
profile = feats.groupby("segment").agg(
    customers=("frequency", "size"),
    median_recency_days=("recency_days", "median"),
    median_frequency=("frequency", "median"),
    median_monetary=("monetary", "median"),
    median_avg_item_price=("avg_item_price", "median"),
    median_review=("avg_review_score", "median"),
    median_delivery_days=("avg_delivery_days", "median"),
)
profile["share"] = profile["customers"] / profile["customers"].sum()
profile.round(2)

# %%
# Trees do not depend on monotone scaling, so the surrogate is fitted on raw values to keep thresholds readable.
tree, test_acc, rules = surrogate_tree(feats[FEATURES].to_numpy(), labels_all, FEATURES, max_depth=3)
print(f"held-out surrogate accuracy: {test_acc:.3f}")
print(rules)

# %%
fig, ax = plt.subplots(figsize=(8, 5))
heat = profile[["median_recency_days", "median_frequency", "median_monetary", "median_review", "median_delivery_days"]]
sns.heatmap(
    np.log1p(heat.T.clip(lower=0)),
    annot=heat.T.round(1).values,
    fmt="",
    cmap="viridis",
    cbar_kws={"label": "log1p of median"},
    ax=ax,
)
ax.set_xlabel("segment")
ax.set_title("CS04: segment profiles (annotated with raw medians)")
fig.tight_layout()
fig.savefig(FIG / "cs04_profile_heatmap.png", dpi=150)
plt.show()
