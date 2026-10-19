# RetailPulse - AI-Powered Customer Analytics & Demand Forecasting Platform

End-to-end Data Science / MLOps project for retail: **demand forecasting, customer segmentation, churn prediction,
inventory optimisation, drift monitoring and an interactive dashboard**.

**Stack:** Python, Pandas, Scikit-learn, Streamlit, MLflow, Docker, Kubernetes, GitHub Actions.
Optional benchmarks: Prophet and an LSTM (TensorFlow) - used automatically when installed.


## Premium dashboard experience

The Streamlit interface includes a responsive dark AI-analytics theme, subtle motion/hover effects, interactive Plotly charts, an operations brief derived from pipeline outputs, and sidebar exports for a ZIP report bundle and JSON KPI summary. Export contents are generated from the files available in `reports/` and `data/processed/`; missing optional report files are skipped.

To run locally on Windows PowerShell from the folder containing `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/run_pipeline.py
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The packaged dataset is for demonstration and is synthetic. Do not present its metrics as real company performance. Before publishing publicly, confirm that all included data is authorized for release.

## GitHub + live demo deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for exact Windows PowerShell commands to push this project to GitHub and deploy it on Streamlit Community Cloud. The Streamlit entry point is `app/streamlit_app.py`; the root `requirements.txt` includes dashboard dependencies.

**Important:** the bundled data and results are synthetic demonstration assets, not a real retailer's data. Do not upload private or employer-owned datasets to a public repository.

## Quick start

**Windows PowerShell** (from the project root):
```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/run_pipeline.py
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

**macOS/Linux**:
```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/run_pipeline.py
.venv/bin/python -m streamlit run app/streamlit_app.py
.venv/bin/python -m unittest discover -s tests -v
```

PowerShell activation is optional; using the virtual environment's Python executable directly avoids execution-policy errors. The normal pipeline preserves existing `data/raw/*.csv` files. `--quick` intentionally replaces them with a small synthetic dataset for smoke testing; back up raw files before using it.

Optional: `pip install -r requirements-optional.txt` adds Prophet + LSTM benchmarks and MLflow tracking (heavy install). The dashboard deployment does not need MLflow; JSON run logging is used as a fallback.
Docker: `docker build -t retailpulse:1.0.0 . && docker run -p 8501:8501 retailpulse:1.0.0`  |  Kubernetes: `kubectl apply -f k8s/`
MLflow UI: `mlflow ui --backend-store-uri models/mlruns`

> **Data note.** No real retailer data was available, so `data_gen.py` simulates 2 years of sales (1,500 customers,
> 30 products, 6 categories) with weekly + yearly seasonality, a Nov/Dec peak, customers who gradually disengage, and
> injected dirty rows. To use real data, back up the sample CSVs and put `transactions.csv`, `products.csv`, `customers.csv` in `data/raw/`. The normal pipeline now preserves existing files and uses them. It validates required columns and reports missing fields. `--quick` deliberately replaces raw files with synthetic test data; do not use it after adding real data. See `docs/ARCHITECTURE.md`.

## Input CSV schema

- `transactions.csv`: `order_id`, `date`, `customer_id`, `product_id`, `quantity`, `unit_price`
- `products.csv`: `product_id`, `category`, `price`, `cost`, `lead_time_days`, `stock_on_hand`
- `customers.csv`: `customer_id`, `signup_date`, `city`, `channel` (kept for completeness; current models derive customer behavior from transactions)

Use consistent product IDs between transactions and products. Dates should be parseable; quantities and prices should be positive. The ETL removes invalid transaction rows. Back up `data/raw/` before experimenting with different datasets.

## Modules
| Module | What it does | File |
|---|---|---|
| ETL | de-duplicate, drop nulls/invalid rows, enrich, build a zero-filled daily demand grid | `src/retailpulse/etl.py` |
| Demand forecasting | direct 28-day-ahead gradient boosting per product-day, naive baselines, 80% intervals, optional Prophet/LSTM benchmark | `forecasting.py` |
| Segmentation | RFM + K-Means (silhouette compared for k = 3..6) | `segmentation.py` |
| Churn prediction | time-based, leak-free; logistic regression vs gradient boosting; permutation importance | `churn.py` |
| Inventory | safety stock, reorder point, EOQ, days of cover, suggested order, stockout/overstock status | `inventory.py` |
| Monitoring | PSI + KS drift tests (customer features, demand) | `monitoring.py` |
| Tracking | MLflow (falls back to JSON files in `models/runs`) | `tracking.py` |
| Dashboard | 6-tab Streamlit app | `app/streamlit_app.py` |
| MLOps | Dockerfile, docker-compose, K8s Deployment + HPA + nightly retrain CronJob, GitHub Actions CI | root, `k8s/`, `.github/` |

Everything tunable (horizon, churn window, service level, PSI thresholds...) lives in `configs/config.yaml`.

## Results (included in `reports/`, seed 42)
**Forecasting** - 28-day hold-out, WAPE (lower is better):

| Level | Gradient boosting (ours) | Baseline |
|---|---|---|
| Product x week | **21.7%** | 22.3% (trailing 28-day mean) |
| Product x day | **55.2%** | 56.5% (trailing mean), 74.7% (same day 28 days ago) |
| Total demand x day | **14.2%** | 15.4% (trailing mean), 25.2% (repeat last 7 days) |

The gain over the best baseline is modest: one product sells only ~4 units/day, so daily demand is
mostly Poisson noise. Use the weekly and total-level numbers for planning. Prophet and LSTM were **not run** in the
build environment (not installed); they appear in the comparison table automatically once installed.

**Churn** - active customers (ordered in the last 90 days), label = no order in the next 60 days; 981 customers, 22% churn.
Best model logistic regression: **ROC AUC 0.75**, recall 0.70, precision 0.37 (gradient boosting 0.69). Main drivers: order
frequency, average gap between orders, orders in the last 30 days. Probabilities come from a class-balanced model, so
use them to *rank* customers (risk tiers: 296 High, 348 Medium, 288 Low) rather than as literal percentages.

**Segments** (silhouette 0.35): Champions 456 customers = 61.9% of revenue; At Risk 409 (25.5%); Loyal 263 (9.8%); Lost 271 (2.8%).
Names are assigned by a rule on cluster centres (frequency + spend - recency).

**Inventory** - 20 products OK, 8 "Reorder now", 2 "Stockout risk". EOQ uses assumed costs (order cost 50, holding 20%
of unit cost) set in `config.yaml`; change them to your real values.

**Monitoring** - demand PSI 0.014 (stable). `tenure` always shows major drift between two snapshots (every customer is
60 days older), so treat that one as expected.

## Verification status
Run and passing in the build environment: the full pipeline (also with `--quick` and from another directory), all 11 unit tests,
and the dashboard script executed against a stubbed Streamlit. **Not executed there** (no network / no Docker): a real
`streamlit run`, the Docker build, the Kubernetes manifests, MLflow, Prophet, LSTM. Run them once on your machine before submitting.

## Limitations
Simulated data (numbers show the system works, not real-world performance); modest forecasting lift over a strong baseline;
churn threshold fixed at 0.5 rather than tuned to a retention-cost model; no hierarchical reconciliation between product-level
and total forecasts.

*Mentor: Chandan Mishra - Submission deadline: 20 Oct 2026*
