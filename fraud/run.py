"""End-to-end: generate data -> features -> four detectors -> evaluation -> reports.

    python -m fraud.run
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import evaluate as E
from .data import generate
from .features import build
from .models import AnomalyDetector, HybridModel, RulesEngine, SupervisedModel

OUT = Path("reports")
TRAIN_END_DAY, TEST_START_DAY = 20, 22   # 2-day gap; labels only as known on day 20
BUDGET = 0.005                            # review the riskiest 0.5% of transactions
BUDGETS = [round(x, 4) for x in np.r_[np.linspace(0.0005, 0.005, 10), np.linspace(0.006, 0.03, 25)]]


def main(seed: int = 21):
    OUT.mkdir(exist_ok=True)
    raw = generate(seed=seed)
    t0 = time.perf_counter()
    df = build(raw)
    feat_s = time.perf_counter() - t0

    cutoff = TRAIN_END_DAY * 86_400
    train = df[df["ts"] < cutoff].reset_index(drop=True)
    test = df[df["ts"] >= TEST_START_DAY * 86_400].reset_index(drop=True)
    # Only fraud confirmed by the cutoff counts as fraud in training (label delay).
    y_train = ((train["is_fraud"] == 1) & (train["label_available_ts"] <= cutoff)).astype(int)

    rules = RulesEngine()
    anomaly = AnomalyDetector(seed=seed).fit(train)
    supervised = SupervisedModel(seed=seed).fit(train, y_train)
    hybrid = HybridModel(anomaly, seed=seed).fit(train, y_train)
    models = {"Rules engine": rules, "Isolation Forest": anomaly, "Supervised GBM": supervised, "Hybrid": hybrid}

    y, amt = test["is_fraud"].to_numpy(), test["amount"].to_numpy()
    scores = {name: m.score(test) for name, m in models.items()}
    results = {name: E.summary(y, amt, s, BUDGET) for name, s in scores.items()}
    by_pattern = {name: E.recall_by_pattern(test, s, BUDGET) for name, s in scores.items()}
    curves = {name: E.budget_curve(y, amt, s, BUDGETS) for name, s in scores.items()}

    # Throughput: featurise + score the full ~1.2M-transaction month with the hybrid model.
    t0 = time.perf_counter()
    hybrid.score(build(raw))
    full_s = time.perf_counter() - t0

    meta = {"transactions": len(df), "accounts": int(raw["account_id"].nunique()),
            "fraud_rate": float(df["is_fraud"].mean()), "train_transactions": len(train),
            "train_fraud_labels_known": int(y_train.sum()), "train_fraud_actual": int(train["is_fraud"].sum()),
            "test_transactions": len(test), "test_fraud": int(y.sum()), "test_fraud_usd": float((amt * y).sum()),
            "alert_budget": BUDGET, "review_cost_usd": E.REVIEW_COST,
            "feature_seconds_1_2m": round(feat_s, 1), "featurise_and_score_seconds_1_2m": round(full_s, 1)}
    report = {"meta": meta, "at_budget": results, "recall_by_pattern": by_pattern}
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))
    (OUT / "budget_curves.json").write_text(json.dumps(
        {name: c[["budget", "recall", "usd_recall", "precision", "net_savings_usd"]].round(4).to_dict("list")
         for name, c in curves.items()}))
    joblib.dump({"anomaly": anomaly, "hybrid": hybrid}, OUT / "models.joblib")
    _plots(curves, by_pattern)
    _report(report)

    for name, r in results.items():
        print(f"{name:17s} PR-AUC {r['pr_auc']:.3f}  @0.5%: precision {r['precision']:.2f}  recall {r['recall']:.2f}  "
              f"$ recall {r['usd_recall']:.2f}  net ${r['net_savings_usd']:,.0f}")
    print(f"Featurise + score 1.2M transactions: {full_s:.1f}s; labels known at training cutoff "
          f"{meta['train_fraud_labels_known']}/{meta['train_fraud_actual']}")


COLORS = {"Rules engine": "#9AA5B1", "Isolation Forest": "#C08B2C", "Supervised GBM": "#4F7CAC", "Hybrid": "#B4372E"}


def _plots(curves, by_pattern):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for name, c in curves.items():
        ax[0].plot(c["budget"] * 100, c["usd_recall"] * 100, label=name, color=COLORS[name], lw=2)
    ax[0].axvline(BUDGET * 100, ls="--", color="grey", lw=1)
    ax[0].set(title="Fraud dollars caught vs. alert budget (test week)", xlabel="Transactions reviewed (%)",
              ylabel="Fraud $ caught (%)", ylim=(0, 100))
    ax[0].legend()
    patterns = list(next(iter(by_pattern.values())))
    x = np.arange(len(patterns))
    for i, (name, rec) in enumerate(by_pattern.items()):
        ax[1].bar(x + (i - 1.5) * 0.2, [rec[p] * 100 for p in patterns], 0.2, label=name, color=COLORS[name])
    ax[1].set_xticks(x, [p.replace("_", " ") for p in patterns])
    ax[1].set(title="Recall by fraud pattern at 0.5% budget", ylabel="Fraud transactions caught (%)", ylim=(0, 100))
    fig.tight_layout()
    fig.savefig(OUT / "results.png", dpi=130)
    plt.close(fig)


def _report(r):
    m = r["meta"]
    lines = ["# Transaction Fraud Detection – Results", "",
             f"{m['transactions']:,} transactions · {m['accounts']:,} accounts · fraud rate {m['fraud_rate']:.2%} · "
             f"test week: {m['test_transactions']:,} transactions, {m['test_fraud']:,} fraudulent (${m['test_fraud_usd']:,.0f}).",
             "", f"At a {m['alert_budget']:.1%} review budget (${m['review_cost_usd']:.0f} per review):", "",
             "| Detector | PR-AUC | Precision | Recall | Fraud $ caught | Net savings |", "|---|---|---|---|---|---|"]
    for name, x in r["at_budget"].items():
        lines.append(f"| {name} | {x['pr_auc']:.3f} | {x['precision']:.2f} | {x['recall']:.2f} | "
                     f"{x['usd_recall']:.0%} | ${x['net_savings_usd']:,.0f} |")
    lines += ["", "![results](results.png)"]
    (OUT / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
