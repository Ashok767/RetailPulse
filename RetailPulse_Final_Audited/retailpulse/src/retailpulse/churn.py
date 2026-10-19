"""Churn prediction.

Setup (leak-free, time-based): pick a snapshot date S. Features use only orders on or before S. A customer who was
active (ordered in the last 90 days) is labelled churned if they place NO order in (S, S + 60 days].
We train at S = last_date - 60 days, where the label window is fully observed, then score customers at last_date.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ACTIVE_WINDOW, CHURN_WINDOW, MODELS, SEED

FEATURES = ["recency", "tenure", "frequency", "monetary", "avg_order_value", "avg_items", "avg_categories",
            "orders_30", "orders_prev30", "orders_90", "trend_30", "avg_gap", "recency_ratio"]


def orders_table(tx: pd.DataFrame) -> pd.DataFrame:
    return tx.groupby(["order_id", "customer_id", "date"], as_index=False).agg(
        revenue=("revenue", "sum"), n_items=("quantity", "sum"), n_categories=("category", "nunique"))


def customer_features(orders: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    o = orders[orders.date <= snapshot].copy()
    o["age"] = (snapshot - o.date).dt.days
    g = o.groupby("customer_id")
    f = pd.DataFrame({
        "recency": g.age.min(), "tenure": g.age.max(), "frequency": g.size(),
        "monetary": g.revenue.sum(), "avg_order_value": g.revenue.mean(),
        "avg_items": g.n_items.mean(), "avg_categories": g.n_categories.mean(),
    })
    f["orders_30"] = o[o.age < 30].groupby("customer_id").size()
    f["orders_prev30"] = o[(o.age >= 30) & (o.age < 60)].groupby("customer_id").size()
    f["orders_90"] = o[o.age < 90].groupby("customer_id").size()
    f = f.fillna({"orders_30": 0, "orders_prev30": 0, "orders_90": 0})
    f["trend_30"] = f.orders_30 - f.orders_prev30
    span = f.tenure - f.recency
    f["avg_gap"] = np.where(f.frequency > 1, span / (f.frequency - 1).clip(lower=1), 90.0)
    f["recency_ratio"] = f.recency / np.clip(f.avg_gap, 1, None)
    return f.reset_index()


def label_churn(feats: pd.DataFrame, orders: pd.DataFrame, snapshot: pd.Timestamp, window: int = CHURN_WINDOW):
    future = orders[(orders.date > snapshot) & (orders.date <= snapshot + pd.Timedelta(days=window))]
    returned = set(future.customer_id)
    return (~feats.customer_id.isin(returned)).astype(int)


def train(tx: pd.DataFrame):
    orders = orders_table(tx)
    last = orders.date.max()
    snap = last - pd.Timedelta(days=CHURN_WINDOW)
    feats = customer_features(orders, snap)
    feats = feats[feats.recency <= ACTIVE_WINDOW].reset_index(drop=True)
    y = label_churn(feats, orders, snap)

    Xtr, Xte, ytr, yte = train_test_split(feats[FEATURES], y, test_size=0.25, stratify=y, random_state=SEED)
    candidates = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "gradient_boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=8,
                                                           random_state=SEED),
    }
    results, fitted = {}, {}
    for name, m in candidates.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        results[name] = {"roc_auc": float(roc_auc_score(yte, p)),
                         "avg_precision": float(average_precision_score(yte, p)),
                         "precision": float(precision_score(yte, pred, zero_division=0)),
                         "recall": float(recall_score(yte, pred, zero_division=0))}
        fitted[name] = m
    best = max(results, key=lambda n: results[n]["roc_auc"])
    model = fitted[best]
    joblib.dump({"model": model, "features": FEATURES}, MODELS / "churn_model.joblib")

    imp = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=SEED)
    importance = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
                  .sort_values("importance", ascending=False).round(4))
    info = {"best_model": best, "models": results, "churn_rate": float(y.mean()),
            "n_customers": int(len(feats)), "train_snapshot": str(snap.date()), "label_window_days": CHURN_WINDOW}
    return model, info, importance, feats[FEATURES]


def score_current(model, tx: pd.DataFrame, segments: pd.DataFrame | None = None):
    orders = orders_table(tx)
    snap = orders.date.max()
    feats = customer_features(orders, snap)
    feats = feats[feats.recency <= ACTIVE_WINDOW].reset_index(drop=True)
    out = feats[["customer_id", "recency", "frequency", "monetary"]].copy()
    out["churn_probability"] = model.predict_proba(feats[FEATURES])[:, 1].round(4)
    out["risk_tier"] = pd.cut(out.churn_probability, [-0.01, 0.3, 0.6, 1.01], labels=["Low", "Medium", "High"])
    if segments is not None:
        out = out.merge(segments[["customer_id", "segment"]], on="customer_id", how="left")
    return out.sort_values("churn_probability", ascending=False).reset_index(drop=True), feats[FEATURES]
