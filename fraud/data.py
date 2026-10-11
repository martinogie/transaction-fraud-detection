"""Synthetic card/transfer transactions for a fintech: ~1.2M transactions over 30 days.

Legitimate behaviour: each account has a typical amount, home country, a few devices and a daily rhythm,
plus look-alikes that trip naive rules (travellers on new devices abroad, transit micro-purchases,
young accounts moving large sums legitimately).
Fraud is injected in three realistic patterns (about 0.2% of transactions):
  * account takeover (ATO): new device, foreign IP, burst of large transfers within hours;
  * card testing: many tiny charges across merchants in minutes, sometimes followed by a big one;
  * mule cash-out: young accounts moving unusually large amounts at odd hours.
Fraud labels arrive late (chargebacks/reports take 1-14 days), which the evaluation respects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS = 30
MCC_RISK = {"grocery": 0.2, "fuel": 0.3, "travel": 0.6, "electronics": 0.8, "gift_cards": 1.0,
            "crypto": 1.0, "utilities": 0.1, "restaurants": 0.2, "transfers": 0.7}


def generate(n_accounts: int = 40_000, days: int = DAYS, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    acct = np.arange(n_accounts)
    acct_age = rng.integers(1, 2000, n_accounts)                     # days since account opened
    typical = np.exp(rng.normal(3.6, 0.8, n_accounts))               # typical spend ~ $37
    rate = rng.gamma(2.0, 0.5, n_accounts)                           # transactions per day
    home = rng.choice(["US", "US", "US", "US", "CA", "MX", "GB"], n_accounts)

    # Legitimate transactions.
    n_tx = rng.poisson(rate * days)
    a = np.repeat(acct, n_tx)
    n = len(a)
    hour = np.clip(rng.normal(14, 4.5, n), 0, 23.99)
    t = rng.integers(0, days, n) * 86400 + (hour * 3600).astype(int) + rng.integers(0, 3600, n)
    mccs = np.array(list(MCC_RISK))
    mcc = rng.choice(mccs, n, p=[0.25, 0.12, 0.05, 0.06, 0.02, 0.01, 0.12, 0.27, 0.10])
    amount = np.round(typical[a] * np.exp(rng.normal(0, 0.6, n)) * np.where(mcc == "travel", 4, 1), 2)
    legit = pd.DataFrame({"account_id": a, "ts": t, "amount": amount, "mcc": mcc,
                          "device_id": a * 10 + rng.choice([0, 0, 0, 1, 2], n),
                          "ip_country": np.where(rng.random(n) < 0.02, rng.choice(["GB", "FR", "NG", "BR"], n), home[a]),
                          "channel": rng.choice(["card_present", "online", "transfer"], n, p=[0.45, 0.40, 0.15]),
                          "is_fraud": 0, "pattern": "none"})

    # Legitimate behaviour that looks suspicious, so detectors face realistic false positives.
    lookalikes = []
    travellers = rng.choice(acct, int(n_accounts * 0.03), replace=False)
    for v in travellers:
        k = rng.integers(3, 12)
        start = rng.integers(0, max(1, days * 86400 - 5 * 86400))
        legit_frames = pd.DataFrame({
            "account_id": v, "ts": start + np.sort(rng.integers(0, 5 * 86400, k)),
            "amount": np.round(typical[v] * np.exp(rng.normal(1.2, 0.7, k)), 2),
            "mcc": rng.choice(["travel", "restaurants", "electronics", "transfers"], k),
            "device_id": 7_000_000 + v if rng.random() < 0.4 else v * 10,
            "ip_country": rng.choice(["GB", "FR", "NG", "BR", "MX"]), "channel": rng.choice(["online", "card_present"], k),
            "is_fraud": 0, "pattern": "none"})
        lookalikes.append(legit_frames)
    commuters = rng.choice(acct, int(n_accounts * 0.02), replace=False)
    for v in commuters:
        k = rng.integers(4, 10)
        start = rng.integers(0, days * 86400 - 3600)
        lookalikes.append(pd.DataFrame({
            "account_id": v, "ts": start + np.sort(rng.integers(0, 50 * 60, k)),
            "amount": np.round(rng.uniform(1.0, 6.0, k), 2), "mcc": rng.choice(["fuel", "restaurants", "grocery"], k),
            "device_id": v * 10, "ip_country": home[v], "channel": "card_present", "is_fraud": 0, "pattern": "none"}))
    big_movers = rng.choice(acct[acct_age < 120], int(n_accounts * 0.01), replace=False)
    for v in big_movers:
        k = rng.integers(1, 4)
        lookalikes.append(pd.DataFrame({
            "account_id": v, "ts": rng.integers(0, days) * 86400 + rng.integers(0, 6 * 3600, k),
            "amount": np.round(rng.uniform(300, 3000, k), 2), "mcc": "transfers", "device_id": v * 10,
            "ip_country": home[v], "channel": "transfer", "is_fraud": 0, "pattern": "none"}))

    frames = [legit, *lookalikes]
    # Account takeover bursts.
    victims = rng.choice(acct, int(n_accounts * 0.006), replace=False)
    for v in victims:
        k = rng.integers(2, 7)
        start = rng.integers(0, days * 86400 - 12 * 3600)
        frames.append(pd.DataFrame({
            "account_id": v, "ts": start + np.sort(rng.integers(0, 10 * 3600, k)),
            "amount": np.round(typical[v] * rng.uniform(2.5, 15, k), 2),
            "mcc": rng.choice(["transfers", "electronics", "gift_cards", "crypto", "travel"], k),
            "device_id": 9_000_000 + v if rng.random() < 0.8 else v * 10,
            "ip_country": rng.choice(["NG", "RU", "BR", "VN"]) if rng.random() < 0.6 else home[v],
            "channel": rng.choice(["online", "transfer"], k), "is_fraud": 1, "pattern": "account_takeover"}))
    # Card testing.
    tested = rng.choice(acct, int(n_accounts * 0.003), replace=False)
    for v in tested:
        k = rng.integers(4, 15)
        start = rng.integers(0, days * 86400 - 3 * 3600)
        amt = np.round(rng.uniform(0.5, 8.0, k), 2)
        if rng.random() < 0.5:
            amt[-1] = round(typical[v] * rng.uniform(10, 30), 2)
        frames.append(pd.DataFrame({
            "account_id": v, "ts": start + np.sort(rng.integers(0, 2 * 3600, k)), "amount": amt,
            "mcc": rng.choice(mccs, k), "device_id": 8_000_000 + v if rng.random() < 0.7 else v * 10,
            "ip_country": home[v],
            "channel": "online", "is_fraud": 1, "pattern": "card_testing"}))
    # Mule cash-out from young accounts.
    young = acct[acct_age < 60]
    mules = rng.choice(young, min(len(young), int(n_accounts * 0.003)), replace=False)
    for v in mules:
        k = rng.integers(2, 6)
        day = rng.integers(0, days)
        night = rng.random() < 0.6
        frames.append(pd.DataFrame({
            "account_id": v, "ts": day * 86400 + (rng.integers(0, 5 * 3600, k) if night else rng.integers(9 * 3600, 20 * 3600, k)),
            "amount": np.round(rng.uniform(400, 4000, k), 2), "mcc": "transfers",
            "device_id": v * 10, "ip_country": home[v], "channel": "transfer",
            "is_fraud": 1, "pattern": "mule_cashout"}))

    df = pd.concat(frames, ignore_index=True).sort_values(["ts", "account_id"]).reset_index(drop=True)
    df["account_age_days"] = acct_age[df["account_id"]] + df["ts"] // 86400
    df["home_country"] = home[df["account_id"]]
    # Label delay: fraud is confirmed 1-14 days after the transaction.
    df["label_available_ts"] = df["ts"] + np.where(df["is_fraud"] == 1,
                                                   rng.integers(1, 15, len(df)) * 86400, 0)
    df.insert(0, "txn_id", np.arange(len(df)))
    return df
