"""Batch-score a day of transactions and write the alert queue.

    python -m fraud.score --input transactions.csv --budget 0.005 --out alerts.csv
Input columns: txn_id, account_id, ts, amount, mcc, device_id, ip_country, home_country, channel,
account_age_days (labels not needed).
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pandas as pd

from .features import build
from .models import RULES, RulesEngine


def score_frame(raw: pd.DataFrame, models_path: str = "reports/models.joblib", budget: float = 0.005) -> pd.DataFrame:
    raw = raw.copy()
    for col, default in [("is_fraud", 0), ("pattern", "unknown"), ("label_available_ts", 0)]:
        if col not in raw:
            raw[col] = default
    m = joblib.load(models_path)
    feats = build(raw)
    feats["risk_score"] = m["hybrid"].score(feats)
    hits = RulesEngine().hits(feats)
    feats["rules_hit"] = hits.apply(lambda r: ", ".join(c for c in RULES if r[c]), axis=1)
    k = max(1, int(round(budget * len(feats))))
    alerts = feats.nlargest(k, "risk_score")
    return alerts[["txn_id", "ts", "amount", "risk_score", "rules_hit"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="alerts.csv")
    ap.add_argument("--budget", type=float, default=0.005)
    a = ap.parse_args()
    t0 = time.perf_counter()
    alerts = score_frame(pd.read_csv(a.input), budget=a.budget)
    alerts.to_csv(a.out, index=False)
    print(f"Wrote {len(alerts):,} alerts to {a.out} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
