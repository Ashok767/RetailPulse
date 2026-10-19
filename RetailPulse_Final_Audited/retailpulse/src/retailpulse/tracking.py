"""Experiment tracking: MLflow when installed (models/mlruns), otherwise one JSON file per run in models/runs."""
import json
import time

from .config import ROOT, RUNS, ensure_dirs


def log_run(run_name: str, params: dict, metrics: dict) -> str:
    try:
        import mlflow
        mlflow.set_tracking_uri((ROOT / 'models' / 'mlruns').as_uri())
        mlflow.set_experiment("retailpulse")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        return "mlflow"
    except Exception:
        ensure_dirs()
        rec = {"run": run_name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "params": params, "metrics": metrics}
        (RUNS / f"{run_name}.json").write_text(json.dumps(rec, indent=2, default=str))
        return "json"
