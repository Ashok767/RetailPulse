"""Synthetic retail data generator.

No real retailer data is available, so we simulate one: customers with individual purchase rates (some of them
gradually disengage and churn), products with category seasonality, a weekly pattern and a Nov/Dec peak.
A little dirt (duplicates, nulls, negative quantities) is injected so the ETL step has real work to do.
"""
import numpy as np
import pandas as pd

from .config import CFG, RAW, SEED, ensure_dirs

# category: (seasonal amplitude, peak day-of-year, min price, max price)
CATEGORIES = {
    "Beverages": (0.40, 200, 2.5, 6.0),
    "Snacks": (0.15, 340, 1.5, 4.0),
    "Dairy": (0.05, 0, 2.0, 5.0),
    "Personal Care": (0.10, 100, 4.0, 15.0),
    "Home & Kitchen": (0.20, 330, 8.0, 40.0),
    "Electronics": (0.30, 330, 25.0, 200.0),
}
CITIES = ["Hyderabad", "Bengaluru", "Chennai", "Mumbai", "Delhi", "Vijayawada"]
CHANNELS = ["Store", "Online", "App"]


def season_curve(dates):
    doy = dates.dayofyear.values
    dow = dates.dayofweek.values
    yearly = 1 + 0.15 * np.sin(2 * np.pi * (doy - 80) / 365)
    weekend = 1 + 0.35 * (dow >= 5)
    holiday = 1 + 0.40 * ((dates.month == 12) | ((dates.month == 11) & (dates.day >= 20)))
    return yearly * weekend * holiday


def generate(n_customers=None, n_days=None, start=None, seed=SEED, dirty=True, write=True):
    n_customers = n_customers or CFG["data"]["n_customers"]
    n_days = n_days or CFG["data"]["n_days"]
    start = start or CFG["data"]["start_date"]
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")

    # ---- products
    rows = []
    for cat, (amp, peak, lo, hi) in CATEGORIES.items():
        for _ in range(5):
            rows.append({"category": cat, "amp": amp, "peak": peak,
                         "price": round(float(rng.uniform(lo, hi)), 2)})
    products = pd.DataFrame(rows)
    products.insert(0, "product_id", [f"P{i + 1:03d}" for i in range(len(products))])
    products["cost"] = (products.price * rng.uniform(0.55, 0.8, len(products))).round(2)
    products["lead_time_days"] = rng.integers(3, 15, len(products))
    pop = rng.lognormal(0, 0.5, len(products))

    # ---- customers
    cid = np.array([f"C{i + 1:04d}" for i in range(n_customers)])
    signup_idx = np.where(rng.random(n_customers) < 0.6, 0,
                          rng.integers(0, max(n_days - 90, 1), n_customers))
    base_rate = rng.gamma(2.0, 0.02, n_customers)                      # mean ~0.04 orders/day
    churner = rng.random(n_customers) < 0.35
    churn_day = np.where(churner, rng.integers(150, n_days, n_customers), 10 ** 6)

    t = np.arange(n_days)[None, :]
    # gradual disengagement during the 45 days before churn, zero afterwards
    decay = np.clip((churn_day[:, None] - t) / 45.0, 0, 1)
    active = (t >= signup_idx[:, None])
    p = np.clip(base_rate[:, None] * decay * active * season_curve(dates)[None, :], 0, 0.95)
    purchases = rng.random(p.shape) < p
    ci, di = np.nonzero(purchases)

    n_lines = np.minimum(1 + rng.poisson(0.6, len(ci)), 4)
    order_no = np.repeat(np.arange(len(ci)), n_lines)
    cust_line = np.repeat(ci, n_lines)
    day_line = np.repeat(di, n_lines)

    doy = dates.dayofyear.values
    w = pop[None, :] * (1 + products.amp.values[None, :] *
                        np.sin(2 * np.pi * (doy[:, None] - products.peak.values[None, :]) / 365))
    w = w / w.sum(axis=1, keepdims=True)
    prod_line = np.empty(len(order_no), dtype=int)
    for d in np.unique(day_line):
        m = day_line == d
        prod_line[m] = rng.choice(len(products), size=int(m.sum()), p=w[d])

    qty = 1 + rng.poisson(0.7, len(order_no))
    qty = np.minimum(qty, 6)
    is_elec = products.category.values[prod_line] == "Electronics"
    qty = np.where(is_elec, np.minimum(qty, 2), qty)
    price = products.price.values[prod_line] * (1 + rng.normal(0, 0.03, len(order_no)))

    tx = pd.DataFrame({
        "order_id": [f"O{n + 1:06d}" for n in order_no],
        "date": dates[day_line],
        "customer_id": cid[cust_line],
        "product_id": products.product_id.values[prod_line],
        "quantity": qty,
        "unit_price": price.round(2),
    })

    # ---- inventory snapshot
    avg_daily = tx.groupby("product_id").quantity.sum() / n_days
    products["stock_on_hand"] = (products.product_id.map(avg_daily) * rng.uniform(4, 30, len(products))).round().astype(int)
    products = products.drop(columns=["amp", "peak"])

    customers = pd.DataFrame({
        "customer_id": cid,
        "signup_date": (pd.Timestamp(start) + pd.to_timedelta(signup_idx, unit="D")).strftime("%Y-%m-%d"),
        "city": rng.choice(CITIES, n_customers),
        "channel": rng.choice(CHANNELS, n_customers, p=[0.5, 0.3, 0.2]),
    })

    if dirty:  # data-quality problems for the ETL step to catch
        n = len(tx)
        dup = tx.sample(int(0.005 * n), random_state=seed)
        tx = pd.concat([tx, dup], ignore_index=True)
        bad = rng.choice(len(tx), int(0.003 * n), replace=False)
        tx.loc[bad, "quantity"] = -tx.loc[bad, "quantity"]
        nul = rng.choice(len(tx), int(0.003 * n), replace=False)
        tx.loc[nul, "customer_id"] = np.nan

    if write:
        ensure_dirs()
        tx.to_csv(RAW / "transactions.csv", index=False)
        products.to_csv(RAW / "products.csv", index=False)
        customers.to_csv(RAW / "customers.csv", index=False)
    return {"transactions": tx, "products": products, "customers": customers}
