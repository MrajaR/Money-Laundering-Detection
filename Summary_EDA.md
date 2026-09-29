
## 1. Dataset Overview

* The dataset contains **9,504,852 transaction observations**.
* The original dataset contains **12 columns**:

  * `Time`
  * `Date`
  * `Sender_account`
  * `Receiver_account`
  * `Amount`
  * `Payment_currency`
  * `Received_currency`
  * `Sender_bank_location`
  * `Receiver_bank_location`
  * `Payment_type`
  * `Is_laundering`
  * `Laundering_type`
* After feature engineering, the number of variables increased to **19 columns**.
* The data covers the period from **October 7, 2022 – August 23, 2023**.
* There are **86,400 unique time values**, effectively covering every time combination within a day.
* The dataset has a very large transaction structure, making computational efficiency an important consideration; memory usage increased to approximately **1.1 GB** after feature engineering. 

---

# 2. Data Quality

### Missing Values

* There are **no missing values** across all variables.
* All 9.5 million observations have complete values.

### Duplicates

* There are **no duplicate rows**.
* `df.duplicated().sum() = 0`.

### Invalid Amount

* There are no transactions with `Amount < 0`.
* There are no transactions with `Amount = 0`.
* Therefore, all transaction amounts are positive.

### Datetime Validity

* `Date` was successfully converted to datetime without producing any `NaT` values.
* `Time` was also successfully converted without producing any `NaT` values.
* Combining the two produces a valid `Datetime`.

### Temporal Ordering

* The data is already **chronologically sorted**:
  `Datetime.is_monotonic_increasing = True`.

### Self-Transfer

* There are no transactions where:
  `Sender_account = Receiver_account`.

**Data quality conclusion:** the dataset is relatively clean and does not require direct handling of missing values or duplicates. 

---

# 3. Categorical Variable Structure

Several variables have the following numbers of categories:

* `Payment_currency`: **13**
* `Received_currency`: **13**
* `Sender_bank_location`: **18**
* `Receiver_bank_location`: **18**
* `Payment_type`: **7**
* `Is_laundering`: **2**
* `Laundering_type`: **28**

For accounts:

* **292,715 unique sender accounts**
* **652,266 unique receiver accounts**
* **855,460 unique accounts** overall. 

This indicates that the data has a **behavioral/network** dimension, rather than consisting solely of individual transaction characteristics.

---

# 4. Target Variable: `Is_laundering`

This is the **most important finding from the EDA**.

Target distribution:

| Status         | Transactions | Percentage |
| -------------- | --------: | ---------: |
| Normal (0)     | 9,494,979 |   99.8961% |
| Laundering (1) |     9,873 |    0.1039% |

Therefore:

* only **9,873 transactions** are laundering transactions;
* approximately **99.9% of transactions are normal**;
* the dataset has **extreme class imbalance**.

Roughly, only about **1 in 964 transactions** is a laundering transaction.

### Implications

Accuracy **is not sufficient** for evaluating the model.

For example, if a model always predicts `0`, its accuracy would already be around 99.9%, but the model would fail to detect any laundering transactions.

Therefore, model evaluation should place greater emphasis on:

* Recall
* Precision
* F1-score
* PR-AUC / Average Precision
* Confusion matrix

---

# 5. `Laundering_type` Indicates Potential Target Leakage

This is the second very important finding.

The crosstab results show:

* all `Normal_*` types → `Is_laundering = 0`
* all laundering types → `Is_laundering = 1`

For example:

* `Structuring` → 1,870 transactions → **100% laundering**
* `Smurfing` → 932 → **100% laundering**
* `Cash_Withdrawal` → 1,334 → **100% laundering**
* `Layered_Fan_In` → 656 → **100% laundering**
* `Cycle` → 382 → **100% laundering**
* and so on.

Conversely:

* `Normal_Fan_Out`
* `Normal_Fan_In`
* `Normal_Small_Fan_Out`
* `Normal_Cash_Deposits`
* etc.

all have label **0**.

### Conclusion

`Laundering_type` almost directly reveals the target.

Therefore, **this variable must not be used as a model input if the task is to predict `Is_laundering`**, because it would cause **target leakage**.

In other words:

> the model is not actually "discovering" laundering transactions; it is simply reading information already provided by `Laundering_type`.

---

# 6. `Amount` Distribution

Transaction amounts show a **strongly right-skewed** distribution.

### Overall Data

* Mean ≈ **8,762.97**
* Median ≈ **6,113.72**
* Std ≈ **25,614.95**
* Minimum = **3.73**
* Maximum = **12,618,498.40**

The mean being much higher than the median indicates the presence of transactions with very large amounts.

### By Target

#### Normal

* Mean ≈ **8,729.88**
* Median ≈ **6,114.63**
* Maximum ≈ **999,962.19**

#### Laundering

* Mean ≈ **40,587.67**
* Median ≈ **5,322.79**
* Maximum ≈ **12,618,498.40**

Interestingly:

> the median laundering transaction amount is actually not larger than the median normal transaction amount, but its mean is much higher.

This suggests that the laundering distribution is likely strongly influenced by a **small number of transactions with extreme amounts**.

### Implications

`Amount` has the potential to be an important feature, but its distribution is highly skewed.

A transformation such as:

`log1p(Amount)`

is worth considering to make the extreme amount scale more manageable.

---

# 7. Payment Type

The laundering rate differs noticeably across payment types.

| Payment type    | Laundering rate |
| --------------- | --------------: |
| Cash Deposit    |      **0.624%** |
| Cash Withdrawal |      **0.444%** |
| Cross-border    |      **0.281%** |
| ACH             |          0.058% |
| Credit card     |          0.056% |
| Debit card      |          0.056% |
| Cheque          |          0.054% |

### Findings

* **Cash Deposit** has the highest laundering rate.
* **Cash Withdrawal** has the second-highest rate.
* `Cheque`, `Debit card`, and `Credit card` have much lower rates.

This indicates that `Payment_type` could be an informative **predictive feature**.

---

# 8. Payment Currency

There are **13 payment currencies**.

Highest laundering rates:

1. Moroccan dirham — **0.354%**
2. Dirham — **0.326%**
3. Swiss franc — **0.298%**
4. Indian rupee — **0.276%**
5. Mexican Peso — **0.274%**

Lowest:

* UK pounds — **0.097%**

Thus, there is variation in laundering rates across payment currencies.

However, interpretation requires caution:

> differences in laundering rates across categories **do not automatically mean that the currency causally causes laundering**.

This only indicates an association within the dataset.

---

# 9. Received Currency

For `Received_currency`, the variation is even more pronounced.

Highest laundering rates:

1. **Naira — 0.639%**
2. Moroccan dirham — **0.623%**
3. Albanian lek — **0.573%**
4. Dirham — **0.526%**
5. Mexican Peso — **0.519%**

Sedangkan:

* UK pounds — **0.079%**

Therefore, `Received_currency` also has the potential to be an informative feature.

---

# 10. Sender Bank Location

Highest laundering rates by sender location:

1. **Morocco — 0.298%**
2. **Mexico — 0.249%**
3. **Netherlands — 0.244%**
4. India — 0.239%
5. Italy — 0.238%

Lowest:

* **UK — 0.101%**

This indicates differences in laundering rates across sender locations.

---

# 11. Receiver Bank Location

The differences by receiver location are even larger.

Highest laundering rates:

1. **Nigeria — 0.656%**
2. **Morocco — 0.654%**
3. **Albania — 0.606%**
4. **UAE — 0.543%**
5. **Mexico — 0.513%**

Lowest:

* **UK — 0.082%**

This suggests that **receiver location may contain a stronger signal** than sender location.

---

# 12. Cross-Border Transactions

The following feature was created:

`is_cross_border = Sender_bank_location != Receiver_bank_location`

Hasil:

| Transaction  | Laundering rate |
| ------------ | --------------: |
| Domestic     |     **0.0797%** |
| Cross-border |     **0.3259%** |

Therefore, the laundering rate for cross-border transactions is approximately **4.1×** higher than for domestic transactions.

This is one of the stronger findings from the EDA.

### Implications

`is_cross_border` is highly worth considering as a feature.

---

# 13. Currency Conversion

The following feature was created:

`is_currency_conversion = Payment_currency != Received_currency`

Hasil:

| Transaction         | Laundering rate |
| ------------------- | --------------: |
| No conversion       |     **0.0738%** |
| Currency conversion |     **0.3368%** |

Therefore, transactions involving currency conversion have a laundering rate approximately **4.6×** higher.

This is also a highly promising feature.

---

# 14. Temporal Pattern

The dataset was then decomposed into:

* `hour`
* `day_of_week`
* `day_of_month`
* `date_only`

### Hour

There is variation in laundering rates by hour.

The hour with the highest laundering rate in the notebook:

> **07:00**

This suggests that transaction time may provide behavioral information.

However, differences across hours should not immediately be considered significant without statistical testing or model-based validation.

---

# 15. Day of Week

Laundering rates are relatively similar across days:

| Day   | Laundering rate |
| ----- | --------------: |
| Monday  |          0.106% |
| Tuesday |          0.104% |
| Wednesday   |          0.105% |
| Thursday  |          0.108% |
| Friday  |      **0.108%** |
| Saturday  |          0.098% |
| Sunday  |          0.098% |

Findings:

* Friday has the highest rate.
* The weekend has slightly lower rates.
* However, the differences are **relatively small**.

Therefore, `day_of_week` likely has a weaker signal than features such as cross-border transactions or currency conversion.

---

# 16. Daily / Monthly Pattern

The notebook also shows variation in transaction volume and the number of laundering transactions by date.

There are indications of a **monthly/temporal seasonal pattern**.

This means:

> laundering transactions are not distributed identically throughout the entire observation period.

This opens up the possibility of using temporal features to capture behavioral changes.

---

# 17. Account Behavior

The data contains:

* **292,715 senders**
* **652,266 receivers**
* **855,460 unique accounts**

Activity analysis shows that some accounts conduct far more transactions than others.

Some of the top senders conduct approximately **700+ transactions** during the dataset period.

Examples of behavioral features that have already begun to be explored:

* `transaction_count`
* `total_amount`
* `avg_amount`
* `unique_receivers`
* `unique_senders`

### Implications

Account-level information can be highly important.

A transaction that appears "normal" individually may become suspicious when viewed in the context of account activity.

---

# 18. Fan-Out Behavior

- There are 4,950 suspicious sender accounts with laundering_rate > 0.

- Some senders show very high laundering rates, reaching as high as 100%.
Extreme examples:

5917350547: 9 unique receivers → 100% laundering rate
2247370907: 2 unique receivers → 95.83%
4159678387: 3 unique receivers → 92.50%
2488893433: 4 unique receivers → 89.66%
5008453588: 4 unique receivers → 89.29%

- Therefore, a high laundering rate does not always require a large fan-out. Some senders with only 2–4 unique receivers already show very high laundering rates.

- Conversely, having many unique receivers does not by itself indicate laundering. Fan-out should be considered together with other transaction characteristics.

---

# 19. Suspicious Receivers / Fan-In

Found:

> **4,074 receiver accounts** with a laundering rate > 0.

Some receivers even have a **100%** laundering rate for the transactions they receive.

Some receivers also receive transactions from many different senders.

This indicates a potential **fan-in pattern**:

> many senders → one receiver.

Patterns like this are highly relevant for network-based feature engineering.

---

# 20. Transaction Network

When senders and receivers are treated as nodes and transaction relationships as edges:

* there are **887,497 unique sender → receiver edges**.
* some of the same edges occur repeatedly, up to **84 transactions**.

This means the dataset has a fairly rich graph structure.

Therefore, in addition to standard transaction features, the following can be developed:

* sender degree
* receiver degree
* fan-in
* fan-out
* edge frequency
* unique counterparties
* transaction velocity
* historical account laundering rate
* network/community features

This has the potential to be one of the most interesting directions for model development.

---

# 21. `Laundering_type` Distribution

There are **28 laundering types**.

Some of the most frequent types are:

* `Normal_Small_Fan_Out` — 3,477,717
* `Normal_Fan_Out` — 2,302,220
* `Normal_Fan_In` — 2,104,285
* `Normal_Group` — 528,351
* `Normal_Cash_Withdrawal` — 305,031
* `Normal_Cash_Deposits` — 223,801

Laundering patterns include:

* `Structuring` — 1,870
* `Cash_Withdrawal` — 1,334
* `Deposit-Send` — 945
* `Smurfing` — 932
* `Layered_Fan_In` — 656
* `Layered_Fan_Out` — 529
* `Stacked Bipartite` — 506
* `Behavioural_Change_1` — 394
* `Bipartite` — 383
* `Cycle` — 382
* `Gather-Scatter` — 354
* `Behavioural_Change_2` — 345
* `Scatter-Gather` — 338
* `Single_large` — 250
* `Fan_Out` — 237
* `Over-Invoicing` — 54

However, because `Laundering_type` is information that is very closely related to the target, this variable is **better treated as a descriptive/analysis variable rather than a model input**. 

---

# 22. Most Important Insights from the Entire EDA

If all of the findings above are distilled into the **core findings**, there are approximately **10 key points**:

1. **The dataset is very large**, with 9.5 million transactions and hundreds of thousands of accounts.
2. **Data quality is very good**: there are no missing values, duplicates, invalid datetimes, zero/negative amounts, or self-transfers.
3. **Extreme class imbalance** is the main challenge: laundering transactions account for only **0.104%** of all transactions.
4. **`Laundering_type` creates potential target leakage** and should not be used as a predictor for `Is_laundering`.
5. **Amount is highly right-skewed** and contains extremely large transactions; a log transformation is worth considering.
6. `Payment_type`, `Payment_currency`, `Received_currency`, and bank locations show **differences in laundering rates across categories**.
7. **Cross-border transactions** have a laundering rate approximately **4.1×** that of domestic transactions.
8. **Currency conversion transactions** have a laundering rate approximately **4.6×** that of transactions without conversion.
9. There are **temporal patterns**, particularly variation by hour and indications of seasonal/daily patterns.
10. The dataset has a **strong network structure**: fan-in, fan-out, repeated edges, and suspicious accounts can provide highly promising sources for feature engineering.
