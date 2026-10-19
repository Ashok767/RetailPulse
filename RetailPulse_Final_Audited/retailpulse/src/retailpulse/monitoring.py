"""Data / prediction drift monitoring: Population Stability Index (PSI) and a two-sample KS test."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import PSI_MAJOR, PSI_WARN


def psi(expected, actual, bins: int = 10) -> float:
    """PSI between a reference sample and a current sample. <0.1 stable, 0.1-0.25 moderate, >0.25 major shift."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    return "major drift" if value > PSI_MAJOR else "moderate drift" if value > PSI_WARN else "stable"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features) -> pd.DataFrame:
    rows = []
    for f in features:
        v = psi(reference[f], current[f])
        rows.append({"feature": f, "psi": round(v, 4), "ks_p_value": round(float(ks_2samp(reference[f], current[f]).pvalue), 4),
                     "status": status(v)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def demand_drift(daily: pd.DataFrame, recent_days: int = 60) -> dict:
    cut = daily.date.max() - pd.Timedelta(days=recent_days)
    ref, cur = daily[daily.date <= cut].units, daily[daily.date > cut].units
    v = psi(ref, cur)
    return {"psi": round(v, 4), "status": status(v), "recent_days": recent_days}
