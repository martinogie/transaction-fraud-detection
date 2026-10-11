"""Evaluation the way a fraud team runs it: a fixed daily alert budget, dollars saved, cost of reviews."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REVIEW_COST = 4.0   # analyst cost per reviewed alert, USD


def at_budget(y, amount, score, budget: float) -> dict:
    """Review the top `budget` share of transactions by score."""
    y, amount, score = np.asarray(y), np.asarray(amount), np.asarray(score)
    k = max(1, int(round(budget * len(y))))
    top = np.argpartition(-score, k - 1)[:k]
    caught = y[top] == 1
    fraud_usd = float((amount * y).sum())
    caught_usd = float(amount[top][caught].sum())
    return {"alerts": k, "precision": float(caught.mean()), "recall": float(caught.sum() / max(y.sum(), 1)),
            "fraud_usd_caught": caught_usd, "usd_recall": caught_usd / fraud_usd if fraud_usd else 0.0,
            "net_savings_usd": caught_usd - k * REVIEW_COST}


def summary(y, amount, score, budget: float = 0.005) -> dict:
    out = at_budget(y, amount, score, budget)
    out["pr_auc"] = float(average_precision_score(y, score))
    return out


def budget_curve(y, amount, score, budgets) -> pd.DataFrame:
    return pd.DataFrame([{"budget": b, **at_budget(y, amount, score, b)} for b in budgets])


def recall_by_pattern(df: pd.DataFrame, score, budget: float = 0.005) -> dict:
    k = max(1, int(round(budget * len(df))))
    flagged = np.zeros(len(df), bool)
    flagged[np.argpartition(-np.asarray(score), k - 1)[:k]] = True
    fr = df["is_fraud"].to_numpy() == 1
    return {p: float(flagged[fr & (df["pattern"].to_numpy() == p)].mean())
            for p in sorted(df.loc[fr, "pattern"].unique())}
