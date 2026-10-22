# Behavioral Fraud Detection with Sequence Models

Detecting fraudulent transactions by modeling each user's transaction history, comparing a feature-engineered LightGBM baseline against a sequence model (GRU/LSTM).

---

## Data

Data comes from the Kaggle competition:
[Behavioral Fraud Detection](https://www.kaggle.com/competitions/behavioral-fraud-detection).

The dataset is synthetic: each simulated user has a home country, typical spending range, preferred merchant category, and known devices. Fraud labels are assigned based on deviations from a user's own history (spending spikes, unfamiliar locations, new devices, unusual merchants, bursty timing) rather than any single attribute, which makes it well suited to sequence modeling.

| File      | Rows    | Notes                         |
|-----------|---------|-------------------------------|
| train.csv | 182,125 | includes `label` (1 = fraud)  |
| test.csv  | 45,531  | same columns, no `label`      |

Each row is one transaction: `user`, `timestamp`, `amount`, `merchant category`, `country`, `device`, `channel` (0/1, meaning not documented), and the time since the user's previous transaction.

--- 

### Key findings (`notebooks/eda.ipynb`)

- **Heavily imbalanced:** only **1.64%** of transactions are fraud. A model that always predicts "not fraud" is **98.4%** accurate, so models are evaluated with PR-AUC instead of accuracy.
- **Clean data:** no missing values; all columns are numeric.
- **Consistent user histories:** 5,000 users with 18–60 transactions each in train (median 36). Train and test cover the same time range (timestamps 6 to 2,874,525; units not documented).
- **`hours_since_prev_txn` is misnamed:** it matches the gap between consecutive timestamps, so it's in timestamp units, not hours. A user's first transaction is filled with a placeholder value of `999`.
- **Train and test split each user's timeline:** test transactions are interleaved in time with train ones. For example, **user 1818** has a test transaction between two of their train transactions. Full user histories therefore require combining both files.

---

## Baseline: LightGBM with behavioral features s (`notebooks/baseline.ipynb`) 

**Goal:** set a score for the sequence model to beat, using a standard model on features.

**Setup.** Train and test were merged and sorted by user and time, so features see each user's full history. Every feature looks only backward. Each one targets a fraud pattern from the dataset description:

| Feature           | Fraud pattern                        |
|-------------------|--------------------------------------|
| `amount_ratio`    | spending spike vs. user's past average |
| `is_new_device`   | unfamiliar device                    |
| `is_new_country`  | unfamiliar location                  |
| `is_new_merchant` | unusual merchant category            |
| `avg_gap_last3`   | bursty timing                        |

Labeled rows were split 80/20 (stratified) for validation.

**Results.**

| Model                  | Train PR-AUC | Validation PR-AUC |
|------------------------|--------------|-------------------|
| Random guessing        | ~            | 0.016             |
| LightGBM (default)     | 0.30         | 0.019             |
| LightGBM (regularized) | 0.075        | 0.021             |
| LSTM | 0.024 | ~ |

**What went wrong, and why.** The baseline barely beats random. The features do not separate the classes: average values for fraud and non-fraud are nearly identical, and new devices are a bit *rarer* in fraud. The only signal found was in the extremes, where the top 1% of spending spikes is ***2.5×** more likely to be fraud*. The default model overfit **(0.30 vs 0.019)**; regularizing closed the gap without improving validation, which points toward weak features rather than bad settings.

**The open question:** The dataset says fraud depends on patterns *across* transactions. Single-row features may be too blunt to capture that. A sequence model sees the full ordered history, which is the next thing to test.

*The sequence models win, by a small but consistent margin.* Across four random seeds, the GRU scored **0.0238–0.0240**, beating the baseline each time. Among the **1%** of transactions each model found most suspicious, **6.0%** were fraud for the GRU vs. **4.4%** for LightGBM.

*GRU vs. LSTM:* An LSTM (`notebooks/lstm.ipynb`) trained with identical settings scored **0.0237–0.0244** across the same four seeds, overlapping the GRU's range. With short 20-step sequences, the LSTM's extra memory did not help, so the simpler and more consistent GRU stays the main model.

*However, all models remain close to random.* Even the GRU's top picks are mostly false alarms. Combined with the EDA (fraud and normal transactions have nearly identical feature averages), this tells us that the fraud labels in this dataset carry only a weak behavioral signal.

---

## Sequence model: GRU (`notebooks/sequence.ipynb`)

**Goal:** test whether a model that reads each user's raw transaction history can find patterns the built features missed.
**Setup:** For every transaction, the input is the user's last 20 transactions *(including the current one)*, oldest first. Each step has 9 values:

| Input | Notes |
|---|---|
| `amount`, time gap, `amount_ratio` | log-transformed, then standardized |
| `channel` | 0/1 |
| `is_first_txn`, `is_new_device`, `is_new_country`, `is_new_merchant` | 0/1 flags |
| `present` | 1 for a transaction, 0 for padding |

Users with fewer than 20 past transactions are padded with zeros at the start; the `present` flag lets the model tell padding from real data. Validation uses the exact same rows as the baseline notebook. 

**Model:** A GRU (32 hidden units) reads the sequence step by step, and a linear layer turns its final state into a fraud score. Trained with Adam (*learning rate **0.001**, batch size **512***) and a weighted loss that counts each fraud case **~60×** more, to offset the class imbalance.

**Results:**

| Model | Validation PR-AUC |
|---|---|
| Random guessing | 0.016 |
| LightGBM baseline | 0.021 |
| GRU | 0.024 |

- **Training longer didn't help:** PR-AUC rose for the first 5 one full pass through the training data, then stabilized around **0.023–0.024** through 15 passes.
- **The improvement is consistent:** across 4 random seeds, the GRU scored **0.0238–0.0240**, beating the baseline every time.
- **Both models stay close to random.** The sequence model extracts a little more signal, but fraud in this dataset leaves only a small trace in user behavior.

---

## Comparison

| Model | Validation PR-AUC | Fraud rate in top 1% flagged |
|---|---|---|
| Random guessing | 0.016 | 1.6% |
| LightGBM baseline | 0.021 | 4.4% |
| GRU | 0.024 | 6.0% |

*The sequence model wins, by a small yet consistent margin.* Across the four random seeds, the GRU scored **0.0238–0.0240**, beating the baseline every time. Among the **1%** of transactions each model found most suspicious, **6.0%** were fraud for the GRU vs. **4.4%** for LightGBM. *However both models remain close to random.* Even the GRU's top picks are mostly false alarms. Combined with the EDA (fraud and normal transactions have nearly identical feature averages), this tells us that the fraud labels in this dataset carry only a weak behavioral signal.

--- 

## How to run

```bash
nix develop
uv sync
kaggle competitions download -c behavioral-fraud-detection -p data/raw
unzip data/raw/behavioral-fraud-detection.zip -d data/raw
jupyter lab
```

---

## Lessons learned

- **Verify column meanings:** I assumed `hours_since_prev_txn` was in hours until recomputing gaps from the timestamps showed it uses timestamp units. Now I check columns against the raw data before trusting their names.
- **Investigate odd values:** A gap of `999` looked like real data, but it only appears on a user's first transaction, where no gap exists. I now look for placeholder values before feeding columns to a model.
- **Understand the split:** Some gaps didn't match train alone. Tracing one back led to a test transaction sitting between two train ones, which showed each user's timeline is split across both files. I now check how train/test were created before designing validation.
- **Choosing the right metric:** With only **1.64%** fraud, accuracy rewards a model that never flags anything. I'll evaluate with PR-AUC instead.
- **The averages can hide extremes:** Fraud and normal transactions had almost identical feature averages, which suggested the features were useless. Looking at the top **1%** of spending spikes showed a **2.5×** higher fraud rate the averages had flattened out. I now check the tails of a distribution, not just the mean.
- **Big train/validation gap means overfitting:** The first `LightGBM` model scored **0.30** on *training data* but **0.019** on *validation data*, so it was simpily memorizing noise rather than learning. Regularizing closed the gap but that did not raise the validation score, which showed the real problem was weak signal, not the model's settings.
- **Neural networks need scaled inputs:** Some features, like time gaps, spanned very large ranges while others were simple 0/1 flags, so the larger-valued features would have dominated training. Log transforms and standardization brought every input onto a comparable scale before training the GRU.

---


