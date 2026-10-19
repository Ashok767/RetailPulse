RETAILPULSE - WINDOWS QUICK START

EASIEST METHOD
1. Extract the ZIP fully (right-click the ZIP > Extract All).
2. Open the extracted retailpulse folder.
3. Double-click RUN_RETAILPULSE_WINDOWS.bat.
4. On first run, allow a few minutes for dependencies to install. Internet is required.
5. Keep the black terminal window open. The browser should open automatically.
6. If it does not, open http://localhost:8501 in Chrome.
7. To stop the dashboard, click the terminal window and press Ctrl+C.

MANUAL METHOD (VS CODE)
Open a terminal in this retailpulse folder, then run:
  .\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py

If the virtual environment does not exist, first run:
  py -3 -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt

TROUBLESHOOTING
- Do not run the command from the parent folder. Terminal's current folder must end with \retailpulse.
- If port 8501 is occupied, stop the old Streamlit process with Ctrl+C, or run manually with --server.port 8502 and open http://localhost:8502.
- If pip install fails, check internet access and share the complete terminal error.
- The bundled dataset is synthetic demo data. Do not present it as actual company/customer data.
