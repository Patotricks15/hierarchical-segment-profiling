"""Ward hierarchical segmentation and decision-tree rule surrogate on Olist (CS04)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier, export_text

RANDOM_STATE = 42
SNAPSHOT_DATE = pd.Timestamp("2018-01-01")
EXCLUDED_STATUSES = ("canceled", "unavailable")
DEFAULT_DATA_DIR = Path.home() / ".local/share/clv-data/olist"
FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_item_price",
    "avg_freight",
    "avg_review_score",
    "avg_delivery_days",
]
LOG_FEATURES = ["frequency", "monetary", "avg_item_price", "avg_freight"]


def data_dir() -> Path:
    return Path(os.environ.get("OLIST_DATA_DIR", DEFAULT_DATA_DIR))


def load_tables(directory: Path | None = None) -> dict[str, pd.DataFrame]:
    base = directory or data_dir()
    return {
        "orders": pd.read_csv(
            base / "olist_orders_dataset.csv",
            parse_dates=["order_purchase_timestamp", "order_delivered_customer_date"],
        ),
        "customers": pd.read_csv(base / "olist_customers_dataset.csv"),
        "items": pd.read_csv(base / "olist_order_items_dataset.csv"),
        "payments": pd.read_csv(base / "olist_order_payments_dataset.csv"),
        "reviews": pd.read_csv(base / "olist_order_reviews_dataset.csv", parse_dates=["review_creation_date"]),
    }


def build_customer_features(tables: dict[str, pd.DataFrame], snapshot: pd.Timestamp = SNAPSHOT_DATE) -> pd.DataFrame:
    """Customer features that use only facts known before `snapshot`.

    Customers are keyed by `customer_unique_id`, which merges repeat purchases made under
    different `customer_id` values.
    """
    orders = tables["orders"]
    orders = orders[(orders["order_purchase_timestamp"] < snapshot) & (~orders["order_status"].isin(EXCLUDED_STATUSES))]
    cust = tables["customers"][["customer_id", "customer_unique_id"]]
    orders = orders.merge(cust, on="customer_id", how="inner")

    payments = tables["payments"].groupby("order_id", as_index=False)["payment_value"].sum()
    items = tables["items"].groupby("order_id", as_index=False).agg(
        item_price=("price", "mean"), freight=("freight_value", "mean")
    )
    reviews = tables["reviews"][tables["reviews"]["review_creation_date"] < snapshot]
    reviews = reviews.groupby("order_id", as_index=False)["review_score"].mean()

    o = orders.merge(payments, on="order_id", how="left").merge(items, on="order_id", how="left").merge(
        reviews, on="order_id", how="left"
    )
    delivered_before = o["order_delivered_customer_date"] < snapshot
    o["delivery_days"] = (o["order_delivered_customer_date"] - o["order_purchase_timestamp"]).dt.days.where(
        delivered_before
    )

    g = o.groupby("customer_unique_id")
    feats = pd.DataFrame(
        {
            "recency_days": (snapshot - g["order_purchase_timestamp"].max()).dt.days,
            "frequency": g["order_id"].nunique(),
            "monetary": g["payment_value"].sum(),
            "avg_item_price": g["item_price"].mean(),
            "avg_freight": g["freight"].mean(),
            "avg_review_score": g["review_score"].mean(),
            "avg_delivery_days": g["delivery_days"].mean(),
        }
    )
    # Customers whose only orders have no item rows have no price data; they are dropped and counted in the notebook.
    feats = feats.dropna(subset=["avg_item_price", "avg_freight"])
    # Customers with no review or delivery before the snapshot get the population median.
    for col in ["avg_review_score", "avg_delivery_days"]:
        feats[col] = feats[col].fillna(feats[col].median())
    return feats[FEATURES]


def scale_features(feats: pd.DataFrame) -> np.ndarray:
    logged = np.log1p(feats[LOG_FEATURES].clip(lower=0))
    rest = feats[["recency_days", "avg_review_score", "avg_delivery_days"]]
    combined = pd.concat([logged, rest], axis=1)
    return RobustScaler().fit_transform(combined)


def ward_labels(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Flat Ward labels (1..k) and the linkage matrix for the dendrogram."""
    Z = linkage(X, method="ward")
    return fcluster(Z, t=k, criterion="maxclust"), Z


def choose_k(X: np.ndarray, k_range: range = range(3, 7)) -> pd.DataFrame:
    rows = []
    for k in k_range:
        labels, _ = ward_labels(X, k)
        sil = silhouette_score(X, labels, sample_size=min(len(X), 5000), random_state=RANDOM_STATE)
        counts = pd.Series(labels).value_counts(normalize=True)
        rows.append({"k": k, "silhouette": sil, "min_share": counts.min()})
    return pd.DataFrame(rows)


def nearest_centroid_labels(X_sample: np.ndarray, labels_sample: np.ndarray, X_all: np.ndarray) -> np.ndarray:
    """Assign each customer to the nearest Ward centroid computed on the clustered sample."""
    ks = np.unique(labels_sample)
    centroids = np.vstack([X_sample[labels_sample == k].mean(axis=0) for k in ks])
    dists = ((X_all[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return ks[dists.argmin(axis=1)]


def surrogate_tree(X: np.ndarray, labels: np.ndarray, feature_names: list[str], max_depth: int = 3):
    """Shallow tree fitted to cluster labels. Returns the tree, held-out fidelity, and rule text."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels)
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=50, random_state=RANDOM_STATE)
    tree.fit(X_tr, y_tr)
    return tree, tree.score(X_te, y_te), export_text(tree, feature_names=feature_names)
