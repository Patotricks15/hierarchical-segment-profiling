# Hierarchical Segment Profiling (CS04)

Ward agglomerative clustering with a depth-3 decision-tree surrogate, so that segments can be read as plain rules, on real Olist marketplace orders.

**Core question:** how do segments nest, and can a non-technical team read them?

## Setup

- Snapshot date: **2018-01-01**. Only orders, payments, items, and reviews dated before the snapshot are used. Canceled and unavailable orders are excluded.
- Customers are keyed by `customer_unique_id`, which merges repeat purchases made under different `customer_id` values.
- Dataset: Olist Brazilian E-Commerce (Kaggle `olistbr/brazilian-ecommerce`, **CC BY-NC-SA 4.0, non-commercial**). See [data/README.md](data/README.md).
- Features: recency, order count, total payment, mean item price, mean freight, mean review score, mean delivery days. Monetary-type features use `log1p`, then `RobustScaler`.
- Clustering: Ward linkage on a random sample of 5,000 customers (it needs O(n²) memory). All customers are then assigned to the nearest Ward centroid.
- Rules: `DecisionTreeClassifier(max_depth=3, min_samples_leaf=50)` fitted to the segment labels on raw feature values, so thresholds are in days, currency, and review points. Fidelity is measured on a 30% held-out split.

## Results

Customers: **43,315** (7 customers with no item rows were dropped, since they have no price data).

**Limitation:** 97.2% of customers have exactly one order before the snapshot. Recency and frequency therefore carry little information, and the segments separate mainly on price, review score, and delivery time.

Choice of k (silhouette on the sample, Ward):

| k | Silhouette | Smallest segment |
|---|---|---|
| 3 | **0.271** | 16.8% |
| 4 | 0.153 | 16.5% |
| 5 | 0.160 | 3.6% |
| 6 | 0.170 | 3.6% |

k = 3 is chosen. Every segment passes the 2% minimum size.

| Segment | Customers | Share | Median review | Median delivery (days) | Median item price | Median total payment |
|---|---|---|---|---|---|---|
| 1 | 7,274 | 16.8% | 1.0 | 11 | 69.9 | 104.8 |
| 2 | 9,166 | 21.2% | 5.0 | 14 | 159.0 | 208.7 |
| 3 | 26,875 | 62.0% | 5.0 | 9 | 59.9 | 84.4 |

Rules (surrogate tree, held-out accuracy **91.8%**; the majority-class baseline is 62.0%):

```
avg_review_score <= 3.12
  avg_freight <= 26.06        -> segment 1 (delivery splits do not change the class)
  avg_freight > 26.06
    avg_review_score <= 2.25  -> segment 1
    avg_review_score > 2.25   -> segment 2
avg_review_score > 3.12
  avg_freight <= 21.27        -> segment 3
  avg_freight > 21.27         -> segment 2
```

Findings:

- **Segment 1 is a low-satisfaction group.** Median review is 1.0 and delivery takes 11 days. The data does not show whether these customers return, so churn is not measured here.
- **Segment 2 has higher-value orders and freight.** Its median item price is 159.0, the highest of the three.
- **Segment 3 is the largest group (62%).** It has satisfied, low-ticket buyers with fast delivery.
- **The dendrogram** (sample, last 40 merges) shows how the three segments nest, but the nesting is not analyzed further.

## Figures

- [`reports/figures/cs04_silhouette_by_k.png`](reports/figures/cs04_silhouette_by_k.png): silhouette by k
- [`reports/figures/cs04_dendrogram.png`](reports/figures/cs04_dendrogram.png): Ward dendrogram of the sample
- [`reports/figures/cs04_profile_heatmap.png`](reports/figures/cs04_profile_heatmap.png): segment profiles

## Limitations

- The population is mostly one-time buyers, so behavioral segments based on repeat purchase are not identifiable here. The segments reflect order experience (review, delivery, freight), not loyalty.
- Ward is fitted on a 5,000-customer sample, and the other customers are assigned to the nearest centroid. Segments may differ from a full-population Ward fit.

## Reproduce

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest -q
cd notebooks && jupyter nbconvert --to notebook --execute --inplace hierarchical_segment_profiling.ipynb
```

The notebook reads from `OLIST_DATA_DIR` (default `~/.local/share/clv-data/olist`).
