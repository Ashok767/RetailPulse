"""End-to-end pipeline: generate data -> ETL -> forecast -> segments -> churn -> inventory -> monitoring.

    python scripts/run_pipeline.py            # full dataset (see configs/config.yaml)
    python scripts/run_pipeline.py --quick    # small dataset (CI / smoke test)
"""
import json
import logging
import time

from . import churn, forecasting, inventory, monitoring, segmentation, tracking
from .config import CFG, OUT, RAW, ensure_dirs
from .data_gen import generate
from .etl import run_etl

log = logging.getLogger("retailpulse")


REQUIRED_RAW_FILES = ("transactions.csv", "products.csv", "customers.csv")


def raw_inputs_available() -> bool:
    """Return True only when all required raw CSV inputs exist and are non-empty."""
    return all((RAW / name).is_file() and (RAW / name).stat().st_size > 0
               for name in REQUIRED_RAW_FILES)


def run(quick: bool = False, force_synthetic: bool = False) -> dict:
    """Run the pipeline.

    Existing raw CSVs are preserved by default. `quick=True` is intended for tests and
    explicitly creates a small synthetic dataset. Use `force_synthetic=True` only when
    you deliberately want to replace raw inputs with generated demo data.
    """
    t0 = time.time()
    ensure_dirs()
    metrics = {}

    if quick or force_synthetic or not raw_inputs_available():
        log.info("[1/7] generating synthetic data%s", " (quick mode)" if quick else "")
        generate(n_customers=400 if quick else None, n_days=600 if quick else None)
        data_source = "synthetic demo data" + (" (quick mode)" if quick else "")
    else:
        log.info("[1/7] using existing CSV inputs in data/raw; they will not be overwritten")
        data_source = "existing CSV inputs (confirm whether data is synthetic or real)"
    metrics["data_source"] = data_source

    log.info("[2/7] ETL")
    tx, daily, products, etl_report = run_etl()
    metrics["etl"] = etl_report

    log.info("[3/7] demand forecasting")
    fcfg = CFG["forecast"]
    fm, per_product, bt, resid = forecasting.backtest(daily, products)
    fc = forecasting.final_forecast(daily, products, resid)
    bench, skipped = forecasting.benchmark_total(daily, use_prophet=fcfg["use_prophet"], use_lstm=fcfg["use_lstm"],
                                                 lstm_window=fcfg["lstm_window"], lstm_epochs=fcfg["lstm_epochs"])
    bench = {"gradient_boosting (ours)": fm["total_level_wape"], **bench}
    fm["total_level_comparison_wape"] = bench
    fm["skipped_optional_models"] = skipped
    metrics["forecast"] = fm
    fc.to_csv(OUT / "demand_forecast.csv", index=False)
    bt.to_csv(OUT / "forecast_backtest.csv", index=False)
    per_product.round(4).to_csv(OUT / "forecast_product_metrics.csv", index=False)
    tracking.log_run("forecast", {"model": "HistGradientBoosting(poisson)", "horizon_days": fm["test_end"]}, fm)

    log.info("[4/7] customer segmentation")
    seg, seg_summary, seg_info = segmentation.segment(tx, tx.date.max())
    metrics["segmentation"] = seg_info
    seg.to_csv(OUT / "customer_segments.csv", index=False)
    seg_summary.to_csv(OUT / "segment_summary.csv", index=False)
    tracking.log_run("segmentation", {"k": 4}, {"silhouette": seg_info["silhouette_chosen"]})

    log.info("[5/7] churn prediction")
    model, cinfo, importance, train_feats = churn.train(tx)
    scores, current_feats = churn.score_current(model, tx, seg)
    metrics["churn"] = cinfo
    scores.to_csv(OUT / "churn_scores.csv", index=False)
    importance.to_csv(OUT / "churn_feature_importance.csv", index=False)
    tracking.log_run("churn", {"model": cinfo["best_model"]}, cinfo["models"][cinfo["best_model"]])

    log.info("[6/7] inventory optimisation")
    plan = inventory.plan(fc, daily, products)
    plan.to_csv(OUT / "inventory_recommendations.csv", index=False)
    metrics["inventory"] = plan.status.value_counts().to_dict()

    log.info("[7/7] monitoring")
    drift = monitoring.drift_report(train_feats, current_feats, churn.FEATURES)
    drift.to_csv(OUT / "drift_report.csv", index=False)
    metrics["drift"] = {"demand": monitoring.demand_drift(daily),
                        "features_with_major_drift": drift[drift.status == "major drift"].feature.tolist()}

    metrics["runtime_seconds"] = round(time.time() - t0, 1)
    (OUT / "summary.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    log.info("pipeline complete in %.1fs", metrics["runtime_seconds"])
    return metrics
