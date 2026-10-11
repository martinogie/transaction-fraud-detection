# Transaction Fraud Detection – Results

1,212,492 transactions · 39,857 accounts · fraud rate 0.20% · test week: 324,037 transactions, 637 fraudulent ($416,079).

At a 0.5% review budget ($4 per review):

| Detector | PR-AUC | Precision | Recall | Fraud $ caught | Net savings |
|---|---|---|---|---|---|
| Rules engine | 0.125 | 0.16 | 0.40 | 57% | $230,977 |
| Isolation Forest | 0.244 | 0.24 | 0.61 | 76% | $309,532 |
| Supervised GBM | 0.585 | 0.36 | 0.92 | 95% | $387,821 |
| Hybrid | 0.540 | 0.35 | 0.90 | 97% | $396,442 |

![results](results.png)