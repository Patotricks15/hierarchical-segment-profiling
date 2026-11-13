import numpy as np
import pandas as pd

from hierseg.core import FEATURES, build_customer_features, nearest_centroid_labels, surrogate_tree

SNAP = pd.Timestamp("2018-01-01")


def _tables():
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "customer_id": ["c1", "c1", "c2", "c2"],
            "order_status": ["delivered", "delivered", "delivered", "canceled"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2017-06-01", "2018-03-01", "2017-12-01", "2017-11-01"]
            ),
            "order_delivered_customer_date": pd.to_datetime(
                ["2017-06-05", "2018-03-09", "2017-12-11", pd.NaT]
            ),
        }
    )
    customers = pd.DataFrame({"customer_id": ["c1", "c2"], "customer_unique_id": ["u1", "u2"]})
    items = pd.DataFrame({"order_id": ["o1", "o2", "o3", "o4"], "price": [10.0, 99.0, 50.0, 1.0], "freight_value": [1.0, 2.0, 3.0, 4.0]})
    payments = pd.DataFrame({"order_id": ["o1", "o2", "o3", "o4"], "payment_value": [10.0, 99.0, 50.0, 1000.0]})
    reviews = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "review_score": [5, 1, 3],
            "review_creation_date": pd.to_datetime(["2017-06-10", "2018-03-20", "2017-12-20"]),
        }
    )
    return {"orders": orders, "customers": customers, "items": items, "payments": payments, "reviews": reviews}


def test_features_exclude_post_snapshot_and_canceled_orders():
    feats = build_customer_features(_tables(), SNAP)
    assert list(feats.columns) == FEATURES
    # o2 is dated after the snapshot, so only o1 counts for u1.
    assert feats.loc["u1", "frequency"] == 1
    assert feats.loc["u1", "monetary"] == 10.0
    # u2: o3 counts, canceled o4 (price 1000) is excluded.
    assert feats.loc["u2", "frequency"] == 1
    assert feats.loc["u2", "monetary"] == 50.0


def test_review_after_snapshot_is_ignored():
    tables = _tables()
    feats = build_customer_features(tables, SNAP)
    # o2's review (score 1) is created after the snapshot, so it must not affect u1.
    assert feats.loc["u1", "avg_review_score"] == 5.0


def test_delivery_days_only_for_orders_delivered_before_snapshot():
    feats = build_customer_features(_tables(), SNAP)
    assert feats.loc["u1", "avg_delivery_days"] == 4.0


def test_nearest_centroid_assigns_each_point_to_closest_cluster():
    X_sample = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
    labels_sample = np.array([1, 1, 2, 2])
    X_all = np.array([[0.2, 0.1], [9.5, 9.8]])
    assert list(nearest_centroid_labels(X_sample, labels_sample, X_all)) == [1, 2]


def test_surrogate_tree_respects_depth_and_reports_rules():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = (X[:, 0] > 0).astype(int) + 1
    tree, acc, rules = surrogate_tree(X, y, ["a", "b"], max_depth=3)
    assert tree.get_depth() <= 3
    assert acc > 0.9
    assert "a <=" in rules
