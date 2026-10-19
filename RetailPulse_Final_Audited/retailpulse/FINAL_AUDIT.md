# RetailPulse AI — Final Audit Notes

## Verified in this package
- Python unit tests: **13 passed** in the local audit environment.
- Python source syntax compilation passed for the Streamlit app and core modules.
- Bundled ETL outputs, forecast reports, customer segments, churn scores, inventory recommendations, and drift reports are present.
- Streamlit pages included: Overview, Forecasting, Customer Segments, Churn Radar, Inventory Ops, Model Health.
- Dashboard uses synthetic demonstration data. Do not use its metrics to make real business decisions.

## Important limitations to disclose
- Automated tests validate core pipeline functions; they do not prove a full browser-based end-to-end test on every computer.
- Optional Prophet and LSTM benchmark models may be skipped when their optional dependencies are not installed. The core forecasting pipeline uses the included gradient-boosting model and a seasonal-naive benchmark.
- The sample data does not include a channel column, so Overview explicitly falls back to category revenue for that chart.
- Anomaly flags are rolling-baseline heuristics, not a separately trained anomaly detection model.
- Conversion rate is a demo indicator because session/visit funnel data is not included.
- MLflow is represented by local run metadata in this package; a production tracking server has not been configured.
- Docker/Kubernetes/CI configuration files are included, but cloud deployment was not executed as part of this local audit.

## Run on Windows
1. Create/activate a virtual environment.
2. Install packages from `requirements.txt`.
3. Start with `RUN_RETAILPULSE_WINDOWS.bat` or follow `README_WINDOWS_QUICKSTART.txt`.
4. Open the local Streamlit URL printed in the terminal.
