#!/usr/bin/env python
"""Run the full RetailPulse pipeline:  python scripts/run_pipeline.py [--quick]"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retailpulse.pipeline import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="generate a small synthetic dataset (fast smoke test; replaces data/raw CSVs)")
    ap.add_argument("--synthetic", action="store_true", help="explicitly replace data/raw CSVs with a full synthetic demo dataset")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run(args.quick, force_synthetic=args.synthetic), indent=2, default=str))
