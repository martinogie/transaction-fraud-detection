"""Point-in-time behavioural features. Every feature uses only the account's past and present."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MCC_RISK

FEATURES = ["log_amount", "amount_vs_history", "hour", "is_night", "new_device", "foreign_ip",
            "txn_10m", "txn_1h", "txn_24h", "amount_24h_vs_history", "small_txn_1h", "secs_since_last",
            "account_age_days", "mcc_risk", "is_transfer", "is_online"]


def _window_count(key: np.ndarray, width: int) -> np.ndarray:
    """Number of earlier rows of the same account within `width` seconds (key is sorted)."""
    return np.arange(len(key)) - np.searchsorted(key, key - width, side="left")


def build(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["account_id", "ts", "txn_id"]).reset_index(drop=True)
    key = d["account_id"].to_numpy(np.int64) * 10_000_000 + d["ts"].to_numpy(np.int64)
    g = d.groupby("account_id", sort=False)

    n_prior = g.cumcount().to_numpy()
    cum = g["amount"].cumsum().to_numpy() - d["amount"].to_numpy()
    hist_mean = np.where(n_prior > 0, cum / np.maximum(n_prior, 1), np.nan)
    global_median = float(d["amount"].median())
    hist_mean = np.where(np.isnan(hist_mean), global_median, hist_mean)

    amt = d["amount"].to_numpy()
    csum = np.cumsum(amt)
    start24 = np.searchsorted(key, key - 86_400, side="left")
    amount_24h = csum - np.where(start24 > 0, csum[start24 - 1], 0.0)

    small = (amt < 5).astype(int)
    csmall = np.cumsum(small)
    start1h = np.searchsorted(key, key - 3_600, side="left")
    small_1h = csmall - np.where(start1h > 0, csmall[start1h - 1], 0)

    prev_ts = g["ts"].shift(1).to_numpy()
    first_device_use = d.groupby(["account_id", "device_id"], sort=False).cumcount().to_numpy() == 0

    hour = (d["ts"].to_numpy() % 86_400) / 3600
    out = pd.DataFrame({
        "txn_id": d["txn_id"], "log_amount": np.log1p(amt), "amount_vs_history": amt / hist_mean,
        "hour": hour, "is_night": ((hour < 5) | (hour >= 23)).astype(int),
        "new_device": (first_device_use & (n_prior > 0)).astype(int),
        "foreign_ip": (d["ip_country"] != d["home_country"]).astype(int).to_numpy(),
        "txn_10m": _window_count(key, 600), "txn_1h": _window_count(key, 3_600),
        "txn_24h": _window_count(key, 86_400), "amount_24h_vs_history": amount_24h / hist_mean,
        "small_txn_1h": small_1h, "secs_since_last": np.nan_to_num(d["ts"].to_numpy() - prev_ts, nan=30 * 86_400),
        "account_age_days": d["account_age_days"], "mcc_risk": d["mcc"].map(MCC_RISK).to_numpy(),
        "is_transfer": (d["channel"] == "transfer").astype(int).to_numpy(),
        "is_online": (d["channel"] == "online").astype(int).to_numpy(),
    })
    keep = ["txn_id", "ts", "amount", "is_fraud", "pattern", "label_available_ts"]
    return d[keep].merge(out, on="txn_id").sort_values("txn_id").reset_index(drop=True)
