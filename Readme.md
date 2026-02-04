# Decision Tree ROC/AUC Benchmark (OpenML)

This project benchmarks **Decision Tree classifiers** using **Gini impurity** vs **Entropy** splitting criteria on multiple **OpenML binary-classification datasets**.  
The objective is to compare model behavior using **ROC curves** and **AUC** under **Stratified Cross-Validation**.

---

## Project Highlights
- Compared **Gini vs Entropy** splitting criteria in Decision Trees
- Used **GridSearchCV** for hyperparameter tuning (`min_samples_leaf`)
- Evaluated models using **ROC curves and AUC**
- Applied **Stratified Cross-Validation** for reliable evaluation
- Built a clean, reproducible **ML benchmarking pipeline** in Python

---

## Datasets
The experiments were conducted on two binary-classification datasets from **OpenML**:

- **OpenML 42717 – Click_prediction_small**  
  Predicts whether a user clicks on an online advertisement based on numerical features.

- **OpenML 1462 – Banknote Authentication**  
  Binary classification task to distinguish genuine vs forged banknotes using numeric attributes.

These datasets were chosen to demonstrate model evaluation techniques while keeping runtime practical on a standard laptop.

---

## Methodology
For each dataset, the following steps were performed:

1. Load dataset using `fetch_openml`
2. Handle missing values using median imputation
3. Train Decision Tree models with:
   - Gini impurity
   - Entropy
4. Tune `min_samples_leaf` using **GridSearchCV**
5. Generate cross-validated prediction probabilities
6. Plot **ROC curves** and compute **AUC**

Evaluation uses **Stratified Cross-Validation** to preserve class balance across folds.

---

## Results
ROC curves are saved in the `outputs/` directory:

- `outputs/roc_dataset_42717.png`
- `outputs/roc_dataset_1462.png`

Observations:
- Both datasets show comparable performance between Gini and Entropy criteria
- Differences in AUC are dataset-dependent but generally small
- Results confirm that both criteria perform similarly in practice

(Note: Exact AUC values may vary slightly due to cross-validation randomness.)

---

## How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
