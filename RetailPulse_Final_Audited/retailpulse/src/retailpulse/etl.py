"""ETL: raw CSV -> validated, enriched sales table + dense daily demand table."""
import json

import pandas as pd

from .config import PROC, RAW, ensure_dirs


def clean_transactions(tx: pd.DataFrame, products: pd.DataFrame):
    """Pure function (no I/O). Returns (clean_df, report)."""
    report = {"rows_raw": int(len(tx))}
    tx = tx.copy()
    tx["date"] = pd.to_datetime(tx["date"], errors="coerce")

    before = len(tx)
    tx = tx.drop_duplicates()
    report["duplicates_removed"] = int(before - len(tx))

    before = len(tx)
    tx = tx.dropna(subset=["date", "customer_id", "product_id", "quantity", "unit_price"])
    report["null_rows_removed"] = int(before - len(tx))

    before = len(tx)
    tx = tx[(tx.quantity > 0) & (tx.unit_price > 0)]
    report["invalid_value_rows_removed"] = int(before - len(tx))

    before = len(tx)
    tx = tx[tx.product_id.isin(products.product_id)]
    report["unknown_product_rows_removed"] = int(before - len(tx))

    tx = tx.merge(products[["product_id", "category"]], on="product_id", how="left")
    tx["quantity"] = tx["quantity"].astype(int)
    tx["revenue"] = (tx.quantity * tx.unit_price).round(2)
    tx = tx.sort_values(["date", "order_id"]).reset_index(drop=True)
    report["rows_clean"] = int(len(tx))
    return tx, report


def daily_demand(tx: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, product) with zeros filled in (a day with no sales is demand 0, not 'missing')."""
    d = tx.groupby(["date", "product_id"], as_index=False).agg(units=("quantity", "sum"), revenue=("revenue", "sum"))
    idx = pd.MultiIndex.from_product(
        [pd.date_range(d.date.min(), d.date.max(), freq="D"), sorted(d.product_id.unique())],
        names=["date", "product_id"])
    return d.set_index(["date", "product_id"]).reindex(idx, fill_value=0).reset_index()


def validate_input_columns(tx: pd.DataFrame, products: pd.DataFrame) -> None:
    """Fail early with actionable messages when an input CSV has the wrong schema."""
    required_tx = {"order_id", "date", "customer_id", "product_id", "quantity", "unit_price"}
    required_products = {"product_id", "category", "price", "cost", "lead_time_days", "stock_on_hand"}
    missing_tx = sorted(required_tx - set(tx.columns))
    missing_products = sorted(required_products - set(products.columns))
    problems = []
    if missing_tx:
        problems.append("transactions.csv missing columns: " + ", ".join(missing_tx))
    if missing_products:
        problems.append("products.csv missing columns: " + ", ".join(missing_products))
    if problems:
        raise ValueError("Invalid RetailPulse input CSV schema. " + " | ".join(problems) +
                         ". See README.md for the expected columns.")
    if tx.empty:
        raise ValueError("transactions.csv has no data rows.")
    if products.empty:
        raise ValueError("products.csv has no data rows.")


def run_etl():
    ensure_dirs()
    tx = pd.read_csv(RAW / "transactions.csv")
    products = pd.read_csv(RAW / "products.csv")
    validate_input_columns(tx, products)
    clean, report = clean_transactions(tx, products)
    daily = daily_demand(clean)
    clean.to_csv(PROC / "sales_clean.csv", index=False)
    daily.to_csv(PROC / "daily_demand.csv", index=False)
    (PROC / "etl_report.json").write_text(json.dumps(report, indent=2))
    return clean, daily, products, report
