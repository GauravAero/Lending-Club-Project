# Lending Club: Loan Default Prediction (Keras)

Predicts whether a LendingClub borrower will fully repay a loan or charge off, using ~396k historical loans, feature engineering, and a dense neural network in TensorFlow/Keras.

**Notebook:** [`Keras-Project.ipynb`](Keras-Project.ipynb)

## Results

| Metric | Value |
|---|---|
| Accuracy (test, 118,566 loans) | **0.89** vs 0.80 majority-class baseline |
| Charged-off class: precision / recall / F1 | 0.95 / 0.45 / 0.61 |
| Fully-paid class: precision / recall / F1 | 0.88 / 0.99 / 0.93 |
| Final loss (train / validation) | 0.256 / 0.261 (no overfitting) |

Confusion matrix (rows = actual): of 23,363 actual charge-offs, 10,575 were caught and 12,788 were missed; only 541 of 95,203 repaid loans were wrongly flagged.

## Data

`lending_club_loan_two.csv.gz`: a 396,030-row, 27-column subset of LendingClub loans (gzip-compressed; pandas reads it directly). Column descriptions are in `lending_club_info.csv`. The label is `loan_status` (Fully Paid / Charged Off), mapped to `loan_repaid` (1/0). Classes are imbalanced at roughly 80/20.

## Approach

**Exploratory analysis:** label distribution, loan amounts, correlation heatmap, and charge-off rates by grade, sub-grade and employment length. `installment` is almost perfectly correlated with `loan_amnt` (r = 0.95), as expected.

**Missing data and feature selection**
- `emp_title`: dropped (173k unique values, too many to encode).
- `emp_length`: dropped, since the charge-off rate is flat at about 18–21% across all tenure buckets.
- `title`: dropped as a free-text duplicate of `purpose`.
- `mort_acc`: filled with the mean `mort_acc` for each `total_acc` value, its most correlated feature.
- Rows missing `revol_util` or `pub_rec_bankruptcies` (under 0.5% of data): dropped.
- `issue_d`: dropped to prevent data leakage (the funding date isn't known when deciding on a new applicant).

**Encoding and feature engineering**
- `term` mapped to 36 / 60; `grade` dropped because it is contained in `sub_grade`.
- One-hot encoding for `sub_grade`, `verification_status`, `application_type`, `initial_list_status`, `purpose`, and `home_ownership` (NONE and ANY merged into OTHER).
- Zip code extracted from `address`, then one-hot encoded; `earliest_cr_line` reduced to its year.
- Result: 78 numeric features.

**Model**
- 70/30 train/test split; `MinMaxScaler` fit on the training set only.
- Dense 78 → 39 → 19 → 1 with ReLU, Dropout(0.2) after each hidden layer, sigmoid output.
- Binary cross-entropy, Adam, 25 epochs, batch size 256.

## Limitations and next steps

- **Low recall on defaults (0.45):** add class weights, tune the decision threshold with a precision-recall curve, and report ROC-AUC and PR-AUC instead of accuracy.
- **Partial leakage:** `int_rate` and `sub_grade` are LendingClub's own risk assessment, so the model partly re-learns their scoring.
- **Imputation before the split:** the `mort_acc` group means use the full dataset; a scikit-learn `Pipeline` would fit them on training data only.
- **Baseline comparison:** compare against gradient-boosted trees (XGBoost/LightGBM), which are strong on tabular data.
- **Selection bias:** the data contains only approved loans, so the model never sees rejected applicants (the reject-inference problem).

## Run it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
jupyter notebook Keras-Project.ipynb
```
