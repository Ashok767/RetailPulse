"""Customer segmentation: RFM features + K-Means."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import SEED

NAMES = ["Lost", "At Risk", "Loyal", "Champions"]  # ascending composite score


def rfm(tx: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    t = tx[tx.date <= snapshot]
    orders = t.groupby(["order_id", "customer_id", "date"], as_index=False).revenue.sum()
    g = orders.groupby("customer_id")
    return pd.DataFrame({
        "recency": (snapshot - g.date.max()).dt.days,
        "frequency": g.size(),
        "monetary": g.revenue.sum().round(2),
    }).reset_index()


def segment(tx: pd.DataFrame, snapshot: pd.Timestamp, k: int = 4):
    f = rfm(tx, snapshot)
    X = np.log1p(f[["recency", "frequency", "monetary"]])
    Xs = StandardScaler().fit_transform(X)

    sil = {}
    for kk in range(3, 7):
        lab = KMeans(kk, n_init=10, random_state=SEED).fit_predict(Xs)
        sil[kk] = float(silhouette_score(Xs, lab, sample_size=min(2000, len(Xs)), random_state=SEED))

    km = KMeans(k, n_init=10, random_state=SEED).fit(Xs)
    f["cluster"] = km.labels_
    c = pd.DataFrame(km.cluster_centers_, columns=["recency", "frequency", "monetary"])
    score = c.frequency + c.monetary - c.recency          # high frequency/value, low recency = best
    order = score.sort_values().index.tolist()
    names = {cl: NAMES[i] if k == len(NAMES) else f"Segment {i + 1}" for i, cl in enumerate(order)}
    f["segment"] = f.cluster.map(names)

    summary = (f.groupby("segment").agg(customers=("customer_id", "count"),
                                        avg_recency_days=("recency", "mean"),
                                        avg_orders=("frequency", "mean"),
                                        avg_spend=("monetary", "mean"),
                                        total_revenue=("monetary", "sum")).round(2).reset_index())
    summary["revenue_share_pct"] = (100 * summary.total_revenue / summary.total_revenue.sum()).round(1)
    summary = summary.sort_values("avg_spend", ascending=False)
    return f.drop(columns="cluster"), summary, {"silhouette_by_k": sil, "silhouette_chosen": sil.get(k)}
