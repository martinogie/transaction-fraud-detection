import unittest

import numpy as np
import pandas as pd

from fraud import evaluate as E
from fraud.data import generate
from fraud.features import build
from fraud.models import AnomalyDetector, HybridModel, RulesEngine, SupervisedModel


class TestFraud(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = generate(n_accounts=6000, days=30, seed=5)
        cls.df = build(cls.raw)
        cut = 20 * 86_400
        cls.train = cls.df[cls.df.ts < cut].reset_index(drop=True)
        cls.test = cls.df[cls.df.ts >= 22 * 86_400].reset_index(drop=True)
        cls.y_train = ((cls.train.is_fraud == 1) & (cls.train.label_available_ts <= cut)).astype(int)

    def test_fraud_is_rare_and_labels_are_delayed(self):
        self.assertTrue(0.0005 < self.raw.is_fraud.mean() < 0.01)
        fraud = self.raw[self.raw.is_fraud == 1]
        self.assertTrue((fraud.label_available_ts > fraud.ts).all())

    def test_features_use_only_the_past(self):
        # The first transaction of every account has no history: velocity counts must be zero.
        first = self.df.loc[self.df.groupby(self.raw.set_index("txn_id").loc[self.df.txn_id, "account_id"].to_numpy())
                            ["ts"].idxmin()]
        self.assertTrue((first[["txn_10m", "txn_1h", "txn_24h"]] == 0).all().all())

    def test_velocity_feature_counts_bursts(self):
        burst = self.df[self.df.pattern == "card_testing"]
        normal = self.df[self.df.pattern == "none"]
        self.assertGreater(burst.txn_1h.mean(), 5 * normal.txn_1h.mean())

    def test_training_ignores_unconfirmed_fraud(self):
        self.assertLess(self.y_train.sum(), self.train.is_fraud.sum())

    def test_models_rank_fraud_above_random(self):
        anomaly = AnomalyDetector(sample=50_000).fit(self.train)
        hybrid = HybridModel(anomaly).fit(self.train, self.y_train)
        y, amt = self.test.is_fraud, self.test.amount
        base = y.mean()
        for scorer in (RulesEngine(), anomaly, SupervisedModel().fit(self.train, self.y_train), hybrid):
            self.assertGreater(E.summary(y, amt, scorer.score(self.test))["pr_auc"], 5 * base)

    def test_budget_metrics(self):
        y = np.r_[np.ones(10), np.zeros(990)]
        amt = np.full(1000, 100.0)
        perfect = E.at_budget(y, amt, y.astype(float), 0.01)
        self.assertEqual(perfect["alerts"], 10)
        self.assertEqual(perfect["precision"], 1.0)
        self.assertEqual(perfect["recall"], 1.0)
        self.assertAlmostEqual(perfect["net_savings_usd"], 1000 - 10 * E.REVIEW_COST)


if __name__ == "__main__":
    unittest.main()
