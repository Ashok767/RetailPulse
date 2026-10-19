"""Inventory optimisation: safety stock, reorder point, EOQ, days of cover and a suggested order per product."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import CFG, HORIZON


def plan(forecast: pd.DataFrame, daily: pd.DataFrame, products: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    c = CFG["inventory"]
    z = norm.ppf(c["service_level"])
    review = c["review_period_days"]

    f = forecast.groupby("product_id").forecast.agg(forecast_units="sum", avg_daily_forecast="mean").reset_index()
    recent = daily[daily.date > daily.date.max() - pd.Timedelta(days=90)]
    sd = recent.groupby("product_id").units.std().rename("daily_std").reset_index()
    p = products.merge(f, on="product_id").merge(sd, on="product_id")

    p["safety_stock"] = z * p.daily_std * np.sqrt(p.lead_time_days + review)          # SS = z * sigma * sqrt(L + R)
    p["reorder_point"] = p.avg_daily_forecast * p.lead_time_days + p.safety_stock     # ROP = d * L + SS
    holding = (p.cost * c["holding_rate"]).clip(lower=0.01)
    p["eoq"] = np.ceil(np.sqrt(2 * p.avg_daily_forecast * 365 * c["ordering_cost"] / holding))  # EOQ = sqrt(2DS/H)
    p["days_of_cover"] = p.stock_on_hand / p.avg_daily_forecast.clip(lower=1e-9)
    need = p.forecast_units + p.safety_stock - p.stock_on_hand
    p["suggested_order_qty"] = np.ceil(need.clip(lower=0)).astype(int)

    lead_demand = p.avg_daily_forecast * p.lead_time_days
    p["status"] = np.select(
        [p.stock_on_hand < lead_demand, p.stock_on_hand < p.reorder_point, p.days_of_cover > 45],
        ["Stockout risk", "Reorder now", "Overstock"], default="OK")
    p["revenue_at_risk"] = (np.clip(lead_demand - p.stock_on_hand, 0, None) * p.price).round(2)
    cols = ["product_id", "category", "price", "stock_on_hand", "lead_time_days", "avg_daily_forecast", "forecast_units",
            "safety_stock", "reorder_point", "eoq", "days_of_cover", "suggested_order_qty", "status", "revenue_at_risk"]
    order = {"Stockout risk": 0, "Reorder now": 1, "OK": 2, "Overstock": 3}
    return (p[cols].round(2).assign(_o=p.status.map(order)).sort_values(["_o", "days_of_cover"])
            .drop(columns="_o").reset_index(drop=True))
