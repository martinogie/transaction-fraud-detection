"""Four detectors, from simplest to strongest.

1. RulesEngine       – hand-written rules an analyst team typically starts with.
2. AnomalyDetector   – Isolation Forest; needs no labels, catches behaviour that is simply unusual.
3. SupervisedModel   – gradient boosting trained on confirmed fraud labels.
4. HybridModel       – gradient boosting that also sees the rule hits and the anomaly score.
All expose `score(df) -> np.ndarray` where higher means more suspicious.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from .features import FEATURES

RULES = {
    "new_device_large_amount": lambda d: (d["new_device"] == 1) & (d["amount_vs_history"] > 5),
    "velocity_burst": lambda d: d["txn_10m"] >= 5,
    "card_testing": lambda d: d["small_txn_1h"] >= 4,
    "foreign_ip_large_amount": lambda d: (d["foreign_ip"] == 1) & (d["amount_vs_history"] > 3),
    "young_account_night_transfer": lambda d: (d["account_age_days"] < 90) & (d["is_night"] == 1)
                                              & (d["is_transfer"] == 1) & (d["amount_vs_history"] > 3),
}


class RulesEngine:
    def hits(self, d: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({name: rule(d).astype(int) for name, rule in RULES.items()})

    def score(self, d: pd.DataFrame) -> np.ndarray:
        # Rule count, with a small tie-break on how unusual the amount is.
        return self.hits(d).sum(axis=1).to_numpy() + np.tanh(np.log1p(d["amount_vs_history"].to_numpy())) * 0.1


class AnomalyDetector:
    def __init__(self, sample: int = 250_000, seed: int = 0):
        self.sample, self.seed = sample, seed

    def fit(self, d: pd.DataFrame) -> "AnomalyDetector":
        X = d[FEATURES].sample(min(self.sample, len(d)), random_state=self.seed)
        self.iso = IsolationForest(n_estimators=200, max_samples=4096, random_state=self.seed, n_jobs=-1).fit(X)
        return self

    def score(self, d: pd.DataFrame) -> np.ndarray:
        return -self.iso.score_samples(d[FEATURES])


class SupervisedModel:
    def __init__(self, extra: list[str] | None = None, seed: int = 0):
        self.cols = FEATURES + (extra or [])
        self.model = HistGradientBoostingClassifier(learning_rate=0.08, max_iter=300, max_leaf_nodes=31,
                                                    l2_regularization=1.0, class_weight="balanced",
                                                    early_stopping=True, random_state=seed)

    def fit(self, d: pd.DataFrame, y, value_weighted: bool = True) -> "SupervisedModel":
        """value_weighted: weight each fraud example by its amount, so the model ranks by dollars at risk
        rather than by count (card testing produces many tiny frauds)."""
        y = np.asarray(y)
        w = None
        if value_weighted:
            amt = np.expm1(d["log_amount"].to_numpy())
            w = np.where(y == 1, np.clip(amt / np.median(amt[y == 1]), 0.2, 20), 1.0)
        self.model.fit(d[self.cols], y, sample_weight=w)
        return self

    def score(self, d: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(d[self.cols])[:, 1]


class HybridModel:
    """Gradient boosting on behavioural features + rule hits + Isolation Forest score."""

    def __init__(self, anomaly: AnomalyDetector, seed: int = 0):
        self.anomaly, self.rules = anomaly, RulesEngine()
        self.sup = SupervisedModel(extra=list(RULES) + ["anomaly_score"], seed=seed)

    def _augment(self, d: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([d.reset_index(drop=True), self.rules.hits(d).reset_index(drop=True),
                          pd.Series(self.anomaly.score(d), name="anomaly_score")], axis=1)

    def fit(self, d: pd.DataFrame, y) -> "HybridModel":
        self.sup.fit(self._augment(d), y)
        return self

    def score(self, d: pd.DataFrame) -> np.ndarray:
        return self.sup.score(self._augment(d))
