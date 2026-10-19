# RetailPulse - Architecture

```
 data/raw  transactions.csv, products.csv, customers.csv   (data_gen.py simulates them; replace with real exports)
     |
     v
  ETL (etl.py)  validate -> clean -> enrich -> dense daily demand      ---> data/processed
     |
     +--> Demand forecasting (forecasting.py)   direct 28-day GBM per product-day + baselines (+ optional Prophet / LSTM)
     +--> Segmentation       (segmentation.py)  RFM + K-Means
     +--> Churn              (churn.py)         snapshot features, 60-day label, LR vs GBM
     +--> Inventory          (inventory.py)     SS, ROP, EOQ, order suggestion  (uses the forecast)
     +--> Monitoring         (monitoring.py)    PSI + KS drift
              |
   Tracking (tracking.py: MLflow or JSON) ---> reports/*.csv + summary.json ---> Streamlit (app/streamlit_app.py)

 Docker image --> Kubernetes Deployment + HPA + nightly retrain CronJob --> GitHub Actions CI
```

## Design decisions
* **No leakage.** Forecast lags are at least one horizon (28 days) old, so a forecast made at day T uses only data up to T.
  Churn features use only orders on or before the snapshot; the label looks only at the next 60 days. Both are unit-tested.
* **Churn population.** Only customers active in the last 90 days are scored. Including long-lapsed customers makes the task
  trivial (recency alone gives AUC ~0.98) and the score meaningless.
* **Honest baselines.** Every forecast number is shown next to a naive baseline; optional models are skipped, not faked.
* **Config-driven.** `configs/config.yaml` is read by `src/retailpulse/config.py`; code defaults apply if it is missing.
* **Graceful degradation.** Prophet, TensorFlow and MLflow are optional.
* **Inventory formulas.** SS = z * sigma * sqrt(L + R); ROP = d * L + SS; EOQ = sqrt(2DS/H); z from the service level.
* **Drift.** PSI < 0.10 stable, 0.10-0.25 moderate, > 0.25 major (retrain).

## Using real company data
Put `transactions.csv` (order_id, date, customer_id, product_id, quantity, unit_price), `products.csv` (product_id, category,
price, cost, lead_time_days, stock_on_hand) and `customers.csv` in `data/raw/`, then call `retailpulse.etl.run_etl()` and the
modules in `pipeline.py` (comment out the `generate(...)` line).
