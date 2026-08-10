# Detecting Money Laundering in Transaction Networks with Graph Neural Networks

MSc dissertation implementation. Three families of model are compared on the same data, the same
chronological split and the same metrics: classic tabular machine learning, a static graph
convolutional network, and a temporal GNN that gives every account a recurrent memory.

The premise is straightforward. A transaction is not an isolated record. It is an edge between two
accounts, and the laundering patterns that matter in practice (layering chains, fan-in and fan-out
structures, circular flows) only become visible across several edges at once. Conventional
transaction monitoring scores each payment as an independent row in a table, which throws that
structure away before the model ever sees it.

---

## Dataset

[IBM synthetic AML benchmark](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
(Altman et al., 2023), file `HI-Small_Trans.csv`.

Using synthetic data was not really a choice. Real transaction data with verified laundering
labels cannot be released, and in a live portfolio an unlabelled transaction might be legitimate
or might be laundering that nobody ever caught. That ambiguity makes recall impossible to measure
honestly. The IBM generator gives you complete ground truth instead.

The CSV is **not** in this repository, since it is well past GitHub's file size limit. Download it
from the link above and put it at `data/HI-Small_Trans.csv`.

Working sample after day-based sampling at `SAMPLE_FRAC = 0.05`:

| | |
|---|---|
| Transactions | 1,048,033 |
| Unique accounts (graph nodes) | 423,310 |
| Laundering transactions | 305 |
| Class imbalance | about 3,435 : 1 |
| Period covered | 2022-09-01 |

Sampling happens by whole days rather than by random rows. Dropping a random 95 per cent of
transactions would leave the transaction graph full of holes: laundering chains lose their middle
hops, and the degree distribution stops looking like anything real. The entire argument here rests
on graph structure carrying signal, so a sampling scheme that destroys the structure would have
undermined the comparison before it began.

---

## Method

**Transaction features:** `log_amount`, `currency_enc`, `format_enc`, `hour`, `same_bank`.

**Account features**, computed from the training period only: out-degree, in-degree, log total
sent, log total received. Degree is paired with value on purpose. Structuring tends to produce
accounts with high degree and low average value, placement often produces the reverse, and neither
pattern shows up if you look at only one of the two.

**Split:** chronological, at the 80th percentile of time order. A random split would scatter
transactions from the same laundering chain across both sides of the partition, and it would let
the model see the future of an account's neighbourhood while predicting its past. Both inflate the
numbers, and neither is something a deployed system could do.

| | Rows | Period | Positives |
|---|---:|---|---:|
| Train | 838,426 | 00:00 to 15:45 | 207 |
| Test | 209,607 | 15:45 to 23:13 | 98 |

**Models:**

| Family | Model | Notes |
|---|---|---|
| Classic ML | Logistic Regression | `class_weight="balanced"` |
| Classic ML | Random Forest | 200 trees, balanced weights |
| Classic ML | XGBoost | 300 rounds, `scale_pos_weight` about 4,050 |
| Graph | Static GCN | 2 × `GCNConv`, then an edge MLP over the two endpoint embeddings |
| Graph | Temporal GNN | `GCNConv` plus a `GRUCell` memory per account, rolled forward over snapshots |

Imbalance is handled by weighting the loss rather than by SMOTE-style resampling. Interpolating
new minority samples would invent transactions between accounts that never dealt with each other,
manufacturing the exact structure the graph models are supposed to be learning.

---

## Results

Test period: 209,607 transactions, 98 of them laundering. Threshold fixed at 0.5.

| Model | ROC-AUC | Recall | Precision | Caught | False positives | Alert rate |
|---|---:|---:|---:|---:|---:|---:|
| **Temporal GNN** | **0.891** | 0.337 | 0.0024 | 33 / 98 | 13,548 | 6.5 % |
| Logistic Regression | 0.877 | 0.898 | 0.0010 | 88 / 98 | 85,604 | 40.9 % |
| Static GCN | 0.818 | 0.571 | 0.0020 | 56 / 98 | 27,410 | 13.1 % |
| XGBoost | 0.811 | 0.010 | 0.0007 | 1 / 98 | 1,409 | 0.7 % |
| Random Forest | 0.541 | 0.000 | 0.000 | 0 / 98 | 20 | 0.0 % |

### Reading the table

The graph models rank better than the tree ensembles. Temporal GNN at 0.891 and static GCN at
0.818 both come out ahead of XGBoost (0.811), and the random forest at 0.541 is barely above
random ordering. Every model saw the same five transaction features; only the graph models also
saw where an account sits in the network. That is the difference.

The two tree ensembles fail in opposite ways, which is worth separating. The random forest never
learns anything, and an AUC of 0.541 says so independently of any threshold. XGBoost ranks
perfectly respectably but converts almost none of that into detections, because `scale_pos_weight`
leaves its probabilities uncalibrated and a 0.5 cut-off then discards the ranking. Its row in the
table sells the model short.

Accuracy inverts here. The random forest posts the highest accuracy in the study (0.9994) and
catches nothing at all, while logistic regression posts the lowest (0.5915) and catches the most.
At 3,435 : 1 the metric carries no information, and it appears in the notebook mainly to show why
it should be left out of the conclusions.

Precision is unusable for all five models. The test window is 7.5 hours, so working through the
temporal GNN's 13,581 alerts inside it would mean closing roughly 30 cases a minute. That is a
consequence of the default 0.5 threshold rather than a ceiling on what the models can do. An AUC
of 0.891 means the ranking is informative; 0.5 is just the wrong place to cut it.

### How much of this is noise

With only 98 positives, recall estimates come with wide error bars. Wilson 95 per cent intervals:

| Model | Recall | 95 % CI |
|---|---:|---|
| Logistic Regression | 0.898 | 0.822 to 0.944 |
| Static GCN | 0.571 | 0.473 to 0.665 |
| Temporal GNN | 0.337 | 0.251 to 0.435 |

The static and temporal GNN intervals do not overlap, so that gap is real. The 0.014 AUC
difference between the temporal GNN and logistic regression almost certainly is not. What can be
claimed safely is that the temporal GNN matches the strongest baseline on ranking while raising
one sixth of the alerts.

---

## Known limitations

These shape how the results should be read, so they belong up front rather than in a footnote.

**The temporal component is not really being tested yet.** At `SAMPLE_FRAC = 0.05` the retained
data covers one calendar day, so daily snapshot binning yields exactly one snapshot and the GRU
has nothing to carry information across. The architecture works and the result is real, but the
Objective 3 hypothesis cannot be evaluated at this sample size. Two fixes: raise `SAMPLE_FRAC`, or
bin hourly with `.dt.floor('1h')`, which gives roughly 16 snapshots from the current data.

**Only 98 positives in the test set**, which is where the confidence intervals above come from.

**The split lands inside a single day**, so training covers hours 0 to 15 and testing covers 15 to
23. Whatever the models learn about `hour` gets applied to values they never really trained on.

**The 0.5 threshold was never chosen.** It is the default, not a decision taken from a
precision-recall curve.

**Label encoding of currency and format** implies an ordering that does not exist, which the
linear and neural models take at face value. One-hot encoding or learned embeddings would be the
right treatment.

**Encoders are fitted before the split**, so they have seen the test period's category vocabulary.
The impact is negligible because the category set is small and closed, but it should still be
corrected.

**No validation split, no early stopping.** The GCN loss was still falling at epoch 30, so that
result is a lower bound on the architecture.

**Transductive setting.** Accounts that appear only in the test period end up with uninformative
features, so anything involving freshly created mule accounts sits outside what this
implementation can show.

---

## Where this goes next

1. Raise `SAMPLE_FRAC` so several days are retained. This gives the GRU a real sequence, raises
   the positive count, and pushes the train/test boundary between days so `hour` becomes usable
   again. Highest value change by some margin.
2. Choose the threshold from a precision-recall curve on a validation period, driven by an alert
   budget such as the top 500 transactions per day.
3. Add average precision. Under imbalance this severe it is the metric the literature prefers, and
   it is a one-line addition to the evaluation function.
4. Try graph attention (GAT) so neighbours are weighted rather than averaged, which addresses the
   hub over-smoothing problem the EDA turned up. GraphSAGE with neighbour sampling would cover the
   inductive case.
5. Add motif counts for fan-in, fan-out and short cycles, so the model gets direct access to the
   structures that define the typologies instead of having to infer them.
6. Score the models on cost, pricing a missed case against an analyst hour, rather than on an
   arbitrary threshold.

---

## Repository structure

```
.
├── notebooks/
│   └── TEJESH_DISSERTATION_Annotated.ipynb    # full annotated pipeline
├── data/
│   └── .gitkeep                               # HI-Small_Trans.csv goes here
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Running it

```bash
python -m venv .venv && source .venv/Scripts/activate
```

```bash
pip install -r requirements.txt
```

PyTorch Geometric has to match your PyTorch and CUDA build. If `pip install torch-geometric` fails,
work through the
[official install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

Put `HI-Small_Trans.csv` in `data/`, point `DATA_PATH` at it in the configuration cell, then run
the notebook top to bottom from a fresh kernel. Restart and Run All, not piecemeal, since some
cells depend on state created earlier.

Expect 20 to 40 minutes on CPU at `SAMPLE_FRAC = 0.05`. Raising the fraction will push you toward
a GPU and toward neighbour sampling, because both GNNs currently train full-batch across all
838,426 training edges in one pass.

---

## References

Altman, E., Blanuša, J., von Niederhäusern, L., Egressy, B., Anghel, A. and Atasu, K. (2023)
*Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*. Advances in Neural
Information Processing Systems 36.

Kipf, T.N. and Welling, M. (2017) *Semi-Supervised Classification with Graph Convolutional
Networks*. ICLR.

---

## Author

[Your Name], MSc [Course], [University], 2026.

Academic work submitted for assessment. Please do not reuse it for your own submission.
