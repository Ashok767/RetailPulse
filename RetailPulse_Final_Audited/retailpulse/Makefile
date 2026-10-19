install:       ; pip install -r requirements.txt
install-core:  ; pip install -r requirements-core.txt
pipeline:      ; python scripts/run_pipeline.py
test:          ; python -m unittest discover -s tests -v
app:           ; streamlit run app/streamlit_app.py
mlflow-ui:     ; mlflow ui --backend-store-uri models/mlruns
docker:        ; docker build -t retailpulse:1.0.0 . && docker run -p 8501:8501 retailpulse:1.0.0
