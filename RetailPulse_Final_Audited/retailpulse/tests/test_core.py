import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retailpulse import churn, forecasting, inventory, monitoring, segmentation  # noqa: E402
from retailpulse.data_gen import generate  # noqa: E402
from retailpulse.etl import clean_transactions, daily_demand, validate_input_columns  # noqa: E402

DATA = generate(n_customers=600, n_days=500, write=False)
TX, REPORT = clean_transactions(DATA["transactions"], DATA["products"])
DAILY = daily_demand(TX)


class TestETL(unittest.TestCase):
    def test_input_schema_validation_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "transactions.csv missing columns"):
            validate_input_columns(pd.DataFrame({"date": []}), pd.DataFrame({"product_id": []}))

    def test_dirty_rows_removed(self):
        self.assertGreater(REPORT["duplicates_removed"], 0)
        self.assertGreater(REPORT["null_rows_removed"], 0)
        self.assertGreater(REPORT["invalid_value_rows_removed"], 0)
        self.assertTrue((TX.quantity > 0).all() and TX.customer_id.notna().all())
        self.assertFalse(TX.duplicated().any())

    def test_daily_grid_is_dense(self):
        self.assertEqual(len(DAILY), DAILY.date.nunique() * DAILY.product_id.nunique())
        self.assertEqual(DAILY.units.sum(), TX.quantity.sum())


class TestForecasting(unittest.TestCase):
    def test_features_do_not_leak_the_future(self):
        h = forecasting.HORIZON
        f = forecasting.build_features(DAILY, DATA["products"], h)
        row = f[(f.product_id == "P001")].iloc[100]
        past = DAILY[(DAILY.product_id == "P001") & (DAILY.date == row.date - pd.Timedelta(days=h))].units.iloc[0]
        self.assertEqual(row.lag_h, past)

    def test_backtest_and_forecast_shape(self):
        m, per_product, bt, resid = forecasting.backtest(DAILY, DATA["products"])
        self.assertLess(m["weekly_product_wape"], 0.6)
        fc = forecasting.final_forecast(DAILY, DATA["products"], resid)
        self.assertEqual(len(fc), forecasting.HORIZON * DAILY.product_id.nunique())
        self.assertTrue((fc.forecast >= 0).all() and (fc.lower <= fc.forecast).all() and (fc.forecast <= fc.upper).all())
        self.assertEqual(fc.date.min(), DAILY.date.max() + pd.Timedelta(days=1))


class TestCustomers(unittest.TestCase):
    def test_segments_cover_all_customers(self):
        seg, summary, info = segmentation.segment(TX, TX.date.max())
        self.assertEqual(len(seg), TX.customer_id.nunique())
        self.assertEqual(summary.customers.sum(), len(seg))
        self.assertGreater(info["silhouette_chosen"], 0.1)

    def test_churn_label_uses_only_the_future_window(self):
        orders = churn.orders_table(TX)
        snap = orders.date.max() - pd.Timedelta(days=60)
        feats = churn.customer_features(orders, snap)
        self.assertTrue((feats.recency >= 0).all())
        y = churn.label_churn(feats, orders, snap)
        returned = set(orders[(orders.date > snap)].customer_id)
        self.assertTrue(((y == 0) == feats.customer_id.isin(returned)).all())

    def test_churn_model_beats_chance(self):
        _, info, _, _ = churn.train(TX)
        best = info["models"][info["best_model"]]
        self.assertGreater(best["roc_auc"], 0.6)


class TestInventoryAndMonitoring(unittest.TestCase):
    def test_inventory_plan(self):
        _, _, _, resid = forecasting.backtest(DAILY, DATA["products"])
        fc = forecasting.final_forecast(DAILY, DATA["products"], resid)
        plan = inventory.plan(fc, DAILY, DATA["products"])
        self.assertEqual(len(plan), len(DATA["products"]))
        self.assertTrue((plan.suggested_order_qty >= 0).all())
        self.assertTrue(plan.status.isin(["OK", "Reorder now", "Stockout risk", "Overstock"]).all())

    def test_psi(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 5000)
        self.assertLess(monitoring.psi(a, rng.normal(0, 1, 5000)), 0.1)
        self.assertGreater(monitoring.psi(a, rng.normal(1.5, 1, 5000)), 0.25)



class TestConfigAndPipeline(unittest.TestCase):
    def test_existing_raw_inputs_are_detected_without_generation(self):
        import tempfile
        from unittest.mock import patch
        from retailpulse import pipeline
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir)
            for filename in pipeline.REQUIRED_RAW_FILES:
                (raw_dir / filename).write_text("placeholder", encoding="utf-8")
            with patch.object(pipeline, "RAW", raw_dir):
                self.assertTrue(pipeline.raw_inputs_available())
                (raw_dir / "products.csv").write_text("", encoding="utf-8")
                self.assertFalse(pipeline.raw_inputs_available())

    def test_config_loaded_from_yaml(self):
        from retailpulse import config
        self.assertEqual(config.HORIZON, config.CFG["forecast"]["horizon_days"])
        self.assertEqual(config.CHURN_WINDOW, 60)

    def test_benchmark_total_always_has_a_baseline(self):
        out, skipped = forecasting.benchmark_total(DAILY, use_prophet=False, use_lstm=False)
        self.assertIn("seasonal_naive_7d", out)
        self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
