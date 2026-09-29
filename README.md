# AML Transaction Monitoring with XGBoost

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/raja-reivan/Money-Laundering-Detection)

**Live Demo:** Explore the Money Laundering Detection model on [Hugging Face Spaces](https://huggingface.co/spaces/raja-reivan/Money-Laundering-Detection).

Machine learning project for detecting potentially fraudulent / money-laundering transactions using **XGBoost** with extensive transaction-level feature engineering and **stateful streaming inference**.

The project includes a **Streamlit/Gradio application** that simulates how an AML transaction monitoring model can process transactions arriving sequentially over time. Instead of generating synthetic transactions, the application uses the **test set as future incoming transactions**.

The number of future transactions processed by the model can be controlled directly from the Streamlit interface.

---

## Project Overview

Money laundering detection is a highly imbalanced classification problem because legitimate transactions vastly outnumber laundering transactions.

This project approaches AML detection by combining:

* XGBoost binary classification
* Transaction and temporal feature engineering
* Sender behavioral history
* Receiver behavioral history
* Sender-receiver pair behavior
* Rolling transaction statistics
* `scale_pos_weight` for class imbalance
* Bayesian hyperparameter optimization with Optuna
* Probability threshold optimization
* Stateful inference
* Streamlit-based transaction monitoring simulation

The main idea is not only to evaluate whether a transaction looks suspicious based on its own attributes, but also to consider the **behavioral context of the sender and receiver**.

---

# Dataset

## 📊 Data Overview

This project utilizes the **Synthetic Anti-Money Laundering Dataset (SAML-D)** for model training, testing, and evaluation. SAML-D is a typology-based synthetic dataset designed specifically to simulate complex transaction patterns and advance transaction monitoring systems.

* **Dataset Source:** Available on [Kaggle - Synthetic Transaction Monitoring Dataset (AML)](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)
* **Official Paper:** [IEEE Xplore (Oztas et al., 2023)](https://ieeexplore.ieee.org/document/10356193)

### Dataset Attribution & Reference
This project relies on the dataset created by Berkan Oztas et al. If you reference the data or building upon this repository, please credit and cite the original paper:

> B. Oztas, D. Cetinkaya, F. Adedoyin, M. Budka, H. Dogan and G. Aksu, *"Enhancing Anti-Money Laundering: Development of a Synthetic Transaction Monitoring Dataset,"* 2023 IEEE International Conference on e-Business Engineering (ICEBE), Sydney, Australia, 2023, pp. 47-54, doi: [10.1109/ICEBE59045.2023.00028](https://doi.org/10.1109/ICEBE59045.2023.00028).

The original dataset contains the following features:

```text
Time
Date
Sender_account
Receiver_account
Amount
Payment_currency
Received_currency
Sender_bank_location
Receiver_bank_location
Payment_type
Is_laundering
Laundering_type
```

The target variable is:

```text
Is_laundering
```

The original transaction data contains relatively few laundering transactions compared with legitimate transactions, making this a **highly imbalanced classification problem**.

For the final evaluation:

* Total transactions: **1,425,728**
* Laundering transactions: **1,697**
* Laundering prevalence: **0.119%**

This means that only about 1 out of every 840 transactions is actually labeled as laundering.

Because of this extreme class imbalance, accuracy alone is not sufficient to evaluate model performance.

---

# Feature Engineering

The original dataset contains relatively simple transaction attributes. Most of the predictive information used by the final model comes from feature engineering.

The feature engineering process creates three main groups of features:

1. Transaction features
2. Temporal features
3. Behavioral history features

These features are calculated separately for senders, receivers, and sender-receiver pairs.

---

## Final Feature Set

### Numerical Features

```python
numeric_features = [
    # Transaction
    "Amount",
    "log_amount",

    # Temporal
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",

    # Sender history
    "sender_tx_count_history",
    "sender_total_amount_history",
    "sender_avg_amount_history",
    "sender_amount_vs_history_ratio",
    "sender_unique_receivers_history",
    "sender_time_since_last_tx",
    "sender_tx_count_1h",
    "sender_total_amount_1h",
    "sender_avg_amount_1h",
    "sender_tx_count_24h",
    "sender_total_amount_24h",
    "sender_avg_amount_24h",
    "sender_amount_vs_1h_avg",
    "sender_amount_vs_24h_avg",

    # Receiver history
    "receiver_tx_count_history",
    "receiver_total_amount_history",
    "receiver_avg_amount_history",
    "receiver_amount_vs_history_ratio",
    "receiver_unique_senders_history",
    "receiver_time_since_last_tx",
    "receiver_tx_count_1h",
    "receiver_total_amount_1h",
    "receiver_avg_amount_1h",
    "receiver_tx_count_24h",
    "receiver_total_amount_24h",
    "receiver_avg_amount_24h",
    "receiver_amount_vs_1h_avg",
    "receiver_amount_vs_24h_avg",

    # Pair behavior
    "pair_tx_count_history"
]
```

### Categorical Features

```python
categorical_features = [
    # Transaction characteristics
    "is_cross_border",
    "is_currency_conversion",

    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type"
]
```

The final feature set is:

```python
feature_cols = numeric_features + categorical_features
```

---

# Feature Engineering Details

## Transaction Features

The raw transaction amount is transformed using:

```text
Amount
log_amount
```

`log_amount` is used to reduce the effect of extremely large transaction values and make the distribution easier for the model to learn.

---

## Temporal Features

Instead of directly using raw hour and day-of-week values, cyclical encoding is used.

```text
hour_sin
hour_cos
dow_sin
dow_cos
```

This allows the model to understand that:

```text
23:00 → 00:00
```

are close to each other in time rather than treating them as distant numerical values.

---

# Sender Behavioral Features

The model tracks historical behavior for each sender.

Examples include:

```text
sender_tx_count_history
sender_total_amount_history
sender_avg_amount_history
sender_amount_vs_history_ratio
sender_unique_receivers_history
sender_time_since_last_tx
```

These features describe the sender's historical activity.

For example:

```text
sender_amount_vs_history_ratio
```

compares the current transaction amount against the sender's historical behavior.

A transaction that is significantly larger than a sender's usual transaction size may therefore provide a different signal than a transaction consistent with the sender's historical behavior.

---

## Rolling Sender Features

Recent transaction activity is also tracked using time windows:

### 1-hour window

```text
sender_tx_count_1h
sender_total_amount_1h
sender_avg_amount_1h
sender_amount_vs_1h_avg
```

### 24-hour window

```text
sender_tx_count_24h
sender_total_amount_24h
sender_avg_amount_24h
sender_amount_vs_24h_avg
```

These features allow the model to detect changes in transaction frequency and amount over shorter periods.

---

# Receiver Behavioral Features

The same concept is applied to receivers.

Historical receiver behavior includes:

```text
receiver_tx_count_history
receiver_total_amount_history
receiver_avg_amount_history
receiver_amount_vs_history_ratio
receiver_unique_senders_history
receiver_time_since_last_tx
```

Recent activity is measured using:

```text
receiver_tx_count_1h
receiver_total_amount_1h
receiver_avg_amount_1h

receiver_tx_count_24h
receiver_total_amount_24h
receiver_avg_amount_24h
```

Additional relative features compare the current transaction with the receiver's recent behavior:

```text
receiver_amount_vs_1h_avg
receiver_amount_vs_24h_avg
```

---

# Sender-Receiver Pair Behavior

The model also tracks the historical relationship between a sender and receiver:

```text
pair_tx_count_history
```

This feature represents how many times a particular sender-receiver pair has interacted historically.

The idea is to provide the model with relational context rather than looking at the sender and receiver independently.

---

# Handling Class Imbalance

The dataset is extremely imbalanced:

```text
Laundering:      1,697
Non-laundering:  1,424,031
```

Therefore, directly optimizing a standard classifier without considering the class distribution can cause the model to focus excessively on the majority class.

Instead of relying on random oversampling or synthetic laundering examples, the model uses the XGBoost parameter:

```text
scale_pos_weight
```

This parameter increases the relative importance of the minority class during training.

The value of `scale_pos_weight` is treated as part of the model optimization process.

---

# Hyperparameter Optimization

Model hyperparameters are optimized using **Optuna** with a **Bayesian optimization / sequential search approach**.

Instead of manually testing a small number of combinations, Optuna explores the hyperparameter space and uses information from previous trials to guide subsequent trials.

The optimization process includes model parameters relevant to XGBoost classification, including the class imbalance weighting represented by:

```text
scale_pos_weight
```

The objective is based on performance appropriate for the highly imbalanced AML classification problem, with emphasis on **Precision-Recall AUC (PR-AUC)** rather than relying solely on ROC-AUC or accuracy.

---

# Classification Threshold

XGBoost produces a probability for the positive class:

```text
P(Is_laundering = 1)
```

The probability is then converted into the final class prediction using a threshold.

The final threshold used by the application is:

```text
0.75
```

A transaction is classified as laundering when:

```text
predicted_probability >= 0.75
```

The threshold was selected using cross-validation on the development dataset (`X_dev`) by examining the trade-off between **precision and recall**.

A threshold of `0.75` was selected as the operating point that provided the desired compromise between:

* detecting as many laundering transactions as possible
* minimizing false alarms

This threshold is therefore separate from the model's learned parameters.

---

# Model Evaluation

The final XGBoost model was evaluated on:

```text
1,425,728 transactions
```

using a classification threshold of:

```text
0.75
```

Only:

```text
1,697 transactions (0.119%)
```

were actual laundering transactions.

## Metrics

| Metric    |       Score |
| --------- | ----------: |
| PR-AUC    |  **0.9936** |
| ROC-AUC   |  **1.0000** |
| Precision |  **0.9958** |
| Recall    |  **0.9859** |
| F1-score  |  **0.9908** |
| Accuracy  | **1.0000*** |

### PR-AUC

```text
0.9936
```

PR-AUC is particularly informative for this problem because the positive class is extremely rare.

Unlike accuracy, Precision-Recall metrics focus specifically on the model's ability to identify the minority class while controlling false positives.

---

### ROC-AUC

```text
1.0000
```

The ROC-AUC indicates extremely strong class separation on the evaluation dataset.

However, because the dataset is highly imbalanced, PR-AUC is a more informative metric for understanding minority-class performance.

---

### Precision

```text
0.9958
```

Approximately **99.58% of transactions flagged by the model as laundering were actually laundering**.

This is important for AML monitoring because false alerts can increase the workload for investigators.

---

### Recall

```text
0.9859
```

The model detected:

```text
1,673 / 1,697
```

actual laundering transactions.

That means:

```text
24
```

laundering transactions were missed by the model.

---

### F1-score

```text
0.9908
```

The F1-score summarizes the balance between precision and recall for the laundering class.

The high F1-score reflects the combination of:

```text
Precision = 0.9958
Recall    = 0.9859
```

---

# Confusion Matrix

The evaluation produced the following confusion matrix:

|                           | Predicted Non-Laundering | Predicted Laundering |
| ------------------------- | -----------------------: | -------------------: |
| **Actual Non-Laundering** |                1,424,024 |                    7 |
| **Actual Laundering**     |                       24 |                1,673 |

Therefore:

```text
True Negative  = 1,424,024
False Positive = 7
False Negative = 24
True Positive  = 1,673
```

The model generated only:

```text
7 false positives
```

while detecting:

```text
1,673 laundering transactions
```

---

## Why Accuracy is Shown as 1.0000

The reported accuracy is rounded to four decimal places:

```text
1.0000
```

However, the actual accuracy is slightly below 100%.

Using the confusion matrix:

```text
Accuracy =
(TN + TP) / (TN + FP + FN + TP)

= (1,424,024 + 1,673) / 1,425,728

≈ 0.9999783
```

or approximately:

```text
99.9978%
```

The value becomes:

```text
1.0000
```

after rounding to four decimal places.

This is another reason why accuracy should not be interpreted in isolation for this dataset.

---

# Stateful Inference

One of the main objectives of the project is to simulate a more realistic AML monitoring scenario.

In a real transaction monitoring system, transactions arrive over time.

The model should therefore not assume that every prediction is independent.

Instead, the application maintains transaction state.

The state contains historical information needed to construct behavioral features, such as:

```text
Sender history
Receiver history
Recent 1-hour activity
Recent 24-hour activity
Sender-receiver interaction history
```

When a new batch of transactions is processed:

```text
1. Read the next transactions
2. Build features using the current state
3. Run model inference
4. Store the transactions into the state
5. Move to the next batch
```

This means that transactions processed later can use information generated by transactions processed earlier.

---

# Streamlit Application

The project includes a Streamlit application that simulates streaming AML inference.

Instead of generating synthetic transactions, the application uses the **test dataset as the future transaction stream**.

The user can select how many rows from the test set should be processed.

For example:

```text
n_rows = 100
```

means that the application takes the next 100 transactions from the test set and sends them to the model as if they were newly arriving transactions.

The important part of the simulation is the order of operations.

### Step 1 — Select Future Transactions

The application selects:

```text
n_rows
```

transactions from the test set.

### Step 2 — Feature Construction

Features are generated using the state accumulated from previously processed transactions.

### Step 3 — Inference

The transactions are passed to the trained XGBoost model.

The model outputs a laundering probability for each transaction.

### Step 4 — State Update

Only **after inference is completed**, the processed transactions are added to the historical state.

This is important because the current transaction should not use its own information as part of its historical features.

Conceptually:

```text
Current State
     ↓
Build Features
     ↓
Model Inference
     ↓
Prediction
     ↓
Update State
     ↓
Next Batch
```

This allows the Streamlit application to simulate a sequential transaction monitoring environment.

---

# Why Stateful Inference Matters

Consider a sender that has historically made:

```text
5 transactions
```

The next transaction from that sender can be represented using:

```text
previous transaction count = 5
```

After processing the new transaction, the state becomes:

```text
previous transaction count = 6
```

The following transaction can therefore use the updated state.

This is fundamentally different from treating every transaction as an independent row.

The Streamlit application is therefore intended to demonstrate the difference between:

```text
Static batch prediction
```

and:

```text
Stateful sequential inference
```

---

# Inference Architecture

The simplified inference flow is:

```text
                    ┌────────────────────┐
                    │     Test Set       │
                    │ Future Transactions │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Select n_rows      │
                    │ from test stream   │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Current State      │
                    │                    │
                    │ Sender State       │
                    │ Receiver State     │
                    │ Recent State       │
                    │ Pair State         │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Feature Engineering│
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ XGBoost Model      │
                    │ Threshold = 0.75   │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Predictions        │
                    │                    │
                    │ Laundering /       │
                    │ Non-Laundering     │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Update State       │
                    └─────────┬──────────┘
                              │
                              ▼
                       Next n_rows
```

---

# Important Data Leakage Consideration

Because the model uses historical and rolling transaction features, temporal ordering is important.

A transaction should only use information that would have been available **before that transaction occurred**.

The stateful inference design follows this principle by:

```text
Feature calculation
        ↓
Prediction
        ↓
State update
```

rather than updating the state before calculating the current transaction's features.

This prevents the current transaction from contributing to its own historical features during inference.

For a production system, historical and rolling state would typically be maintained in a dedicated transactional or analytical data infrastructure rather than recalculated from the entire dataset for every inference request.

---

# Project Structure

The project structure is:

```text
AML_detection/
├── aml_inference.py
├── gradio_app.py
├── streamlit_app.py
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── README.md
├── Summary_EDA.txt
├── artifacts/
│   ├── initial_state.pkl
│   ├── test_transactions.parquet
│   └── model/
│       └── aml_xgb_model.pkl
├── assets/
│   └── figures/
│       └── evaluation/
│           ├── confusion_matrix.jpg
│           ├── PR-AUC.png
│           ├── ROC-AUC.png
│           ├── threshold_evaluation_for_LR.png
│           └── threshold_evaluation_XGboost.png
├── notebooks/
│   ├── EDA-part.ipynb
│   ├── feature-engineering.ipynb
│   ├── inference.ipynb
│   └── model-training-and-evaluation.ipynb
└── utils/
    ├── __init__.py
    └── recent_state.py
```

---

# Technology Stack

```text
Python
Pandas
NumPy
Scikit-learn
XGBoost
Optuna
Joblib
Streamlit
Gradio
```

---

# Key Concepts Demonstrated

This project demonstrates several concepts commonly used in production-oriented machine learning systems:

### Machine Learning

* Binary classification
* XGBoost
* Class imbalance
* Precision / Recall trade-off
* PR-AUC
* ROC-AUC
* Probability thresholding

### Feature Engineering

* Log transformation
* Cyclical time encoding
* Historical behavioral features
* Rolling-window features
* Entity-level aggregation
* Pair-level behavioral features

### Model Optimization

* Bayesian hyperparameter optimization
* Optuna
* `scale_pos_weight`
* Cross-validation
* Threshold optimization

### ML Inference

* Stateful inference
* Sequential transaction processing
* Batch / micro-batch inference
* Historical state management
* Test-set streaming simulation

### Application

* Streamlit
* Interactive AML monitoring
* Real-time-like inference simulation
* Prediction and state progression

---

# Limitations

Although the evaluation results are very strong, the results should not automatically be interpreted as proof of production-level AML performance.

Several factors are important to consider:

1. The dataset is highly imbalanced, so a small number of false positives or false negatives can significantly affect operational behavior.

2. The evaluation results depend on the characteristics of the available dataset and its temporal structure.

3. Stateful behavioral features can introduce leakage if historical state is constructed incorrectly.

4. The Streamlit application simulates streaming inference using the test set. It is not a live production transaction-processing system.

5. A production AML system would require additional components such as alert investigation workflows, case management, model monitoring, drift detection, data quality monitoring, access control, and audit logging.

---

# Future Improvements

Potential extensions include:

```text
- Model monitoring and drift detection
- Feature drift monitoring
- Probability calibration
- Threshold monitoring
- SHAP-based explainability
- MLflow experiment tracking
- Model Registry integration
- REST API deployment
- Real-time message queue integration
- Persistent state storage
- Feature store integration
- AML alert prioritization
- Investigator dashboard
```

---

# Results Summary

The final XGBoost model demonstrates strong performance on the evaluation dataset:

```text
Evaluation transactions : 1,425,728
Actual laundering       : 1,697
Laundering prevalence   : 0.119%

Threshold               : 0.75

PR-AUC                  : 0.9936
ROC-AUC                 : 1.0000
Precision               : 0.9958
Recall                  : 0.9859
F1-score                : 0.9908
Accuracy                : ~99.9978%

True Positive           : 1,673
False Positive          : 7
True Negative           : 1,424,024
False Negative          : 24
```

The project combines **behavioral feature engineering, class-imbalance handling, Bayesian hyperparameter optimization, threshold selection, and stateful inference** to simulate an AML transaction monitoring workflow.

---

# Demo

The Streamlit or Gradio application allows users to simulate the arrival of future transactions by selecting how many rows from the test set should be processed at each step.

Each batch is:

```text
Test Set
   ↓
Select n_rows
   ↓
Generate historical/rolling features
   ↓
XGBoost inference
   ↓
Apply threshold = 0.75
   ↓
Display predictions
   ↓
Update state
   ↓
Process next batch
```

This provides an interactive demonstration of how an AML machine learning model can operate in a sequential transaction monitoring environment.
