"""Demand forecasting: direct 28-day-ahead gradient boosting per (product, day), with honest baselines.

Direct strategy: every lag feature is at least HORIZON days old, so a forecast made at origin T for any day in
(T, T+HORIZON] only uses information available at T. There is no recursive error build-up and no leakage.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import HORIZON, MODELS, SEED

FEATURES = ["lag_h", "lag_h7", "roll28", "roll7", "dow", "month", "doy_sin", "doy_cos",
            "is_weekend", "is_holiday_season", "product_code", "category_code", "price"]


def wape(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.abs(actual - pred).sum() / max(actual.sum(), 1e-9))


def build_features(daily: pd.DataFrame, products: pd.DataFrame, horizon: int = HORIZON, extend: bool = False):
    df = daily[["date", "product_id", "units"]].copy()
    if extend:  # append `horizon` future days (units unknown) so their features can be built
        fut_dates = pd.date_range(df.date.max() + pd.Timedelta(days=1), periods=horizon)
        fut = pd.MultiIndex.from_product([fut_dates, df.product_id.unique()],
                                         names=["date", "product_id"]).to_frame(index=False)
        fut["units"] = np.nan
        df = pd.concat([df, fut], ignore_index=True)
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    g = df.groupby("product_id")["units"]
    df["lag_h"] = g.shift(horizon)
    df["lag_h7"] = g.shift(horizon + 7)
    df["roll28"] = g.transform(lambda s: s.shift(horizon).rolling(28, min_periods=14).mean())
    df["roll7"] = g.transform(lambda s: s.shift(horizon).rolling(7, min_periods=4).mean())

    d = df["date"]
    df["dow"] = d.dt.dayofweek
    df["month"] = d.dt.month
    doy = d.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["is_weekend"] = (df.dow >= 5).astype(int)
    df["is_holiday_season"] = ((d.dt.month == 12) | ((d.dt.month == 11) & (d.dt.day >= 20))).astype(int)

    meta = products[["product_id", "category", "price"]].copy()
    meta["category_code"] = meta.category.astype("category").cat.codes
    meta["product_code"] = meta.product_id.astype("category").cat.codes
    return df.merge(meta[["product_id", "category_code", "product_code", "price"]], on="product_id", how="left")


def _model():
    return HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=300,
                                         max_leaf_nodes=15, random_state=SEED)


def backtest(daily, products, horizon: int = HORIZON):
    """Train on everything up to T = last_date - horizon, evaluate on the final `horizon` days."""
    df = build_features(daily, products, horizon).dropna(subset=FEATURES + ["units"])
    cutoff = df.date.max() - pd.Timedelta(days=horizon)
    train, test = df[df.date <= cutoff], df[df.date > cutoff].copy()
    model = _model().fit(train[FEATURES], train.units)
    test["pred"] = np.clip(model.predict(test[FEATURES]), 0, None)
    test["naive_lag"] = test.lag_h          # same weekday 28 days ago
    test["naive_mean"] = test.roll28        # trailing 28-day mean

    def block(frame):
        return {
            "model_wape": wape(frame.units, frame.pred),
            "naive_lag_wape": wape(frame.units, frame.naive_lag),
            "naive_mean_wape": wape(frame.units, frame.naive_mean),
            "model_mae": float(np.abs(frame.units - frame.pred).mean()),
        }

    metrics = block(test)
    total = test.groupby("date")[["units", "pred", "naive_lag", "naive_mean"]].sum().reset_index()
    metrics["total_level_wape"] = wape(total.units, total.pred)
    metrics["total_level_naive_mean_wape"] = wape(total.units, total.naive_mean)
    wk = test.assign(week=(test.date - test.date.min()).dt.days // 7).groupby(["product_id", "week"])[
        ["units", "pred", "naive_mean"]].sum()
    metrics["weekly_product_wape"] = wape(wk.units, wk.pred)
    metrics["weekly_product_naive_mean_wape"] = wape(wk.units, wk.naive_mean)
    metrics["test_start"] = str((cutoff + pd.Timedelta(days=1)).date())
    metrics["test_end"] = str(test.date.max().date())
    metrics["train_rows"] = int(len(train))

    per_product = (test.groupby("product_id")
                   .apply(lambda f: pd.Series(block(f)), include_groups=False).reset_index())
    resid = (test.units - test.pred).values
    return metrics, per_product, test[["date", "product_id", "units", "pred"]], resid


def final_forecast(daily, products, resid, horizon: int = HORIZON):
    """Refit on all data and forecast the next `horizon` days with an empirical 80% interval."""
    hist = build_features(daily, products, horizon).dropna(subset=FEATURES + ["units"])
    model = _model().fit(hist[FEATURES], hist.units)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / "demand_model.joblib")

    ext = build_features(daily, products, horizon, extend=True)
    fut = ext[ext.date > daily.date.max()].copy()
    fut["forecast"] = np.clip(model.predict(fut[FEATURES]), 0, None)
    q10, q90 = np.quantile(resid, [0.10, 0.90])
    fut["lower"] = np.clip(fut.forecast + q10, 0, None)
    fut["upper"] = fut.forecast + q90
    return fut[["date", "product_id", "forecast", "lower", "upper"]].reset_index(drop=True)


def _lstm_forecast(train_vals, horizon, window, epochs):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(0)
    y = np.asarray(train_vals, dtype="float32")
    mu, sd = y.mean(), y.std() + 1e-6
    z = (y - mu) / sd
    X = np.stack([z[i:i + window] for i in range(len(z) - window)])[..., None]
    model = tf.keras.Sequential([tf.keras.layers.Input((window, 1)), tf.keras.layers.LSTM(48),
                                 tf.keras.layers.Dense(1)])
    model.compile("adam", "mse")
    model.fit(X, z[window:], epochs=epochs, batch_size=32, verbose=0)
    buf, out = list(z[-window:]), []
    for _ in range(horizon):
        p = float(model.predict(np.array(buf[-window:])[None, :, None], verbose=0)[0, 0])
        out.append(p)
        buf.append(p)
    return np.clip(np.array(out) * sd + mu, 0, None)


def benchmark_total(daily, horizon: int = HORIZON, use_prophet=True, use_lstm=True, lstm_window=28, lstm_epochs=15):
    """WAPE on TOTAL daily demand over the same hold-out as backtest(): seasonal-naive always, Prophet and LSTM when
    their libraries are installed (otherwise they are skipped and listed under 'skipped')."""
    total = daily.groupby("date", as_index=False).units.sum()
    train, test = total.iloc[:-horizon], total.iloc[-horizon:]
    out, skipped = {}, []
    out["seasonal_naive_7d"] = wape(test.units, np.tile(train.units.values[-7:], horizon // 7 + 1)[:horizon])
    if use_prophet:
        try:
            from prophet import Prophet
            m = Prophet(weekly_seasonality=True, yearly_seasonality=True).fit(
                train.rename(columns={"date": "ds", "units": "y"}))
            pred = m.predict(test[["date"]].rename(columns={"date": "ds"}))["yhat"].clip(lower=0)
            out["prophet"] = wape(test.units, pred)
        except Exception:
            skipped.append("prophet")
    if use_lstm:
        try:
            out["lstm"] = wape(test.units, _lstm_forecast(train.units.values, horizon, lstm_window, lstm_epochs))
        except Exception:
            skipped.append("lstm")
    return out, skipped
