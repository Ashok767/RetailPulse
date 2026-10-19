# GitHub and Streamlit Community Cloud deployment

## Before publishing

- This repository currently includes **synthetic demonstration data** and derived reports/models. Do not publish confidential customer or company data.
- Do not commit passwords, API keys, access tokens, or private customer records.
- The dashboard numbers describe the bundled demonstration dataset until you replace the inputs with authorized data and rerun the pipeline.
- Keep the folder layout intact: `app/`, `src/`, `data/`, `reports/`, `models/`, `requirements.txt`.

## 1. Verify locally (Windows PowerShell)

Run commands from the inner `retailpulse` folder (the folder containing `requirements.txt`):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

If your terminal is one folder above the project root, first run `cd .\retailpulse`.

## 2. Create a GitHub repository and push

1. Sign in to GitHub and create a new **empty** repository named `RetailPulse` (do not initialize it with a README if you already have this README locally).
2. Open PowerShell in the inner project root, where `README.md` and `requirements.txt` are located.
3. Run the following, replacing the URL with your own repository URL:

```powershell
git init
git add .
git status
git commit -m "Prepare RetailPulse analytics dashboard for deployment"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/RetailPulse.git
git push -u origin main
```

Review `git status` before committing. Only push data you are authorized to publish. If Git says `git` is not recognized, install Git for Windows and reopen PowerShell.

## 3. Deploy free with Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in using GitHub.
2. Choose **Create app** / **Deploy an app**.
3. Select the `RetailPulse` repository, branch `main`, and main file path `app/streamlit_app.py`.
4. Deploy. Streamlit Cloud installs the root `requirements.txt` automatically.
5. Wait for build logs. If deployment fails, open **Manage app → Logs** and fix the first Python/package error shown.
6. Open the public URL and verify all six dashboard tabs.

The cloud app reads the bundled `data/`, `reports/`, and `models/` artifacts. If you update raw data or pipeline code, run the pipeline locally, test the result, commit the intended updated artifacts, and push again. Avoid running a full training pipeline on every page load.

## 4. Final live smoke check

- Executive overview charts load without errors.
- Forecast product selector works and forecast values render.
- Segment and churn tables load.
- Inventory filters and recommendations render.
- Monitoring values load and are labelled as demo data where appropriate.
- The public app does not expose private data or secrets.

## Troubleshooting

- `requirements.txt not found`: deploy from the repository root and confirm the file is committed.
- `No module named streamlit`: dependency installation failed or the wrong environment was used; inspect cloud build logs.
- `No pipeline outputs found`: commit the included `reports/` artifacts or run `python scripts/run_pipeline.py` locally, then commit the resulting reports.
- `ModuleNotFoundError: retailpulse`: ensure `src/retailpulse/` is present in the repository; the app adds `src/` to `sys.path`.
