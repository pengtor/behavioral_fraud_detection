# Behavioral Fraud Detection with Sequence Models

Detecting fraudulent transactions by modeling each user's transaction history, comparing a feature-engineered LightGBM baseline against a sequence model (GRU/LSTM).

## Data

Data comes from the Kaggle competition
[Behavioral Fraud Detection](https://www.kaggle.com/competitions/behavioral-fraud-detection).

The dataset is synthetic: each simulated user has a home country, typical spending range, preferred merchant category, and known devices. Fraud labels are assigned based on deviations from a user's own history (spending spikes, unfamiliar locations, new devices, unusual merchants, bursty timing) rather than any single attribute, which makes it well suited to sequence modeling.

| File      | Rows    | Notes                         |
|-----------|---------|-------------------------------|
| train.csv | 182,125 | includes `label` (1 = fraud)  |
| test.csv  | 45,531  | same columns, no `label`      |

Each row is one transaction: user, timestamp, amount, merchant category, country, device, channel (0/1, meaning not documented), and the time since the user's previous transaction.

### Key findings (see `notebooks/01_eda.ipynb`)

- **Heavily imbalanced:** only 1.64% of transactions are fraud. A model that always predicts "not fraud" is 98.4% accurate, so models are evaluated with PR-AUC instead of accuracy.
- **Clean data:** no missing values; all columns are numeric.
- **Consistent user histories:** 5,000 users with 18–60 transactions each in train (median 36). Train and test cover the same time range (timestamps 6 to 2,874,525; units not documented).
- **`hours_since_prev_txn` is misnamed:** it matches the gap between consecutive timestamps, so it's in timestamp units, not hours. A user's first transaction is filled with a placeholder value of 999.
- **Train and test split each user's timeline:** test transactions are interleaved in time with train ones. For example, user 1818 has a test transaction between two of their train transactions. Full user histories therefore require combining both files.

## How to run

```bash
nix develop
uv sync
kaggle competitions download -c behavioral-fraud-detection -p data/raw
unzip data/raw/behavioral-fraud-detection.zip -d data/raw
jupyter lab
```

Accept the competition rules on Kaggle before downloading.

## Lessons learned

- **Verify column meanings:** I assumed `hours_since_prev_txn` was in hours until recomputing gaps from the timestamps showed it uses timestamp units. Now I check columns against the raw data before trusting their names.
- **Investigate odd values:** A gap of `999` looked like real data, but it only appears on a user's first transaction, where no gap exists. I now look for placeholder values before feeding columns to a model.
- **Understand the split:** Some gaps didn't match train alone. Tracing one back led to a test transaction sitting between two train ones, which showed each user's timeline is split across both files. I now check how train/test were created before designing validation.
- **Choose the right metric:** With only 1.64% fraud, accuracy rewards a model that never flags anything. I'll evaluate with PR-AUC instead.
