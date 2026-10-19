"""Central configuration: paths + values from configs/config.yaml (defaults used if the file or PyYAML is missing)."""
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "reports"
MODELS = ROOT / "models"
RUNS = ROOT / "models" / "runs"

DEFAULTS = {
    "seed": 42,
    "data": {"n_customers": 1500, "n_days": 730, "start_date": "2024-10-01"},
    "forecast": {"horizon_days": 28, "use_prophet": True, "use_lstm": True, "lstm_window": 28, "lstm_epochs": 15},
    "churn": {"label_window_days": 60, "active_window_days": 90},
    "inventory": {"service_level": 0.95, "review_period_days": 7, "ordering_cost": 50.0, "holding_rate": 0.20},
    "monitoring": {"psi_warn": 0.10, "psi_major": 0.25},
}


def _load() -> dict:
    cfg = deepcopy(DEFAULTS)
    try:
        import yaml
        path = ROOT / "configs" / "config.yaml"
        if path.exists():
            user = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for key, val in user.items():
                if isinstance(val, dict):
                    cfg.setdefault(key, {}).update(val)
                else:
                    cfg[key] = val
    except ImportError:
        pass
    return cfg


CFG = _load()
SEED = int(CFG["seed"])
HORIZON = int(CFG["forecast"]["horizon_days"])
CHURN_WINDOW = int(CFG["churn"]["label_window_days"])
ACTIVE_WINDOW = int(CFG["churn"]["active_window_days"])
PSI_WARN = float(CFG["monitoring"]["psi_warn"])
PSI_MAJOR = float(CFG["monitoring"]["psi_major"])


def ensure_dirs():
    for p in (RAW, PROC, OUT, MODELS, RUNS):
        p.mkdir(parents=True, exist_ok=True)
