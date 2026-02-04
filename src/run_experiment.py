import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def load_openml_dataset(data_id: str):
    X, y = fetch_openml(data_id=int(data_id), return_X_y=True, as_frame=False)

    # Robust target handling:
    # - If y is strings, encode.
    # - If y is numeric but not {0,1}, still allow ROC with pos_label=1.
    if y.dtype.kind in {"U", "S", "O"}:
        le = LabelEncoder()
        y = le.fit_transform(y)
        # After encoding, "positive class" becomes 1 only if there are 2 classes.
        # This is OK for binary classification, but we’ll still treat label 1 as positive.
    else:
        y = y.astype(int)

    return X, y


def run_experiment(data_id: str, out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)

    X, y = load_openml_dataset(data_id)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    param_grid = {"model__min_samples_leaf": [1, 5, 10, 20, 50]}

    results = []

    for criterion in ["gini", "entropy"]:
        model = DecisionTreeClassifier(criterion=criterion, random_state=42)

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ])

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )
        grid.fit(X, y)
        best_params = grid.best_params_

        # Cross-validated predicted probabilities using best params
        best_pipe = pipe.set_params(**best_params)
        probas = cross_val_predict(best_pipe, X, y, cv=cv, method="predict_proba")

        # Ensure we use the probability for class "1" as the positive class.
        # For sklearn, classes_ is stored in the fitted final estimator.
        # But cross_val_predict returns aligned columns for the estimator's classes.
        # In binary classification, pick class 1 if present; else pick the "greater" class.
        # (This keeps it safe even if labels are {0,1} or {-1,1}.)
        # We’ll compute ROC with pos_label=1.
        if set(np.unique(y)) == {-1, 1}:
            # Convert {-1,1} to {0,1} for cleaner ROC behavior
            y_roc = (y == 1).astype(int)
            pos_proba = probas[:, list(np.sort(np.unique(y))).index(1)] if probas.shape[1] == 2 else probas[:, -1]
            fpr, tpr, _ = roc_curve(y_roc, pos_proba, pos_label=1)
        else:
            # Typical {0,1}
            # Determine which column corresponds to class 1 (fallback to last column)
            pos_col = 1 if probas.shape[1] == 2 else -1
            fpr, tpr, _ = roc_curve(y, probas[:, pos_col], pos_label=1)

        roc_auc = auc(fpr, tpr)

        results.append({
            "criterion": criterion,
            "best_params": best_params,
            "auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr
        })

    # Plot and save ROC curves
    plt.figure(figsize=(9, 6))
    for r in results:
        plt.plot(r["fpr"], r["tpr"], label=f'{r["criterion"]} (AUC={r["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], "k--", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - OpenML Dataset {data_id}")
    plt.legend()
    out_path = os.path.join(out_dir, f"roc_dataset_{data_id}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return results


if __name__ == "__main__":
    for data_id in [42717, 1462]:
        res = run_experiment(data_id)
        print(f"\nDataset {data_id} results:")
        for r in res:
            print(f'  - {r["criterion"]}: AUC={r["auc"]:.3f}, best={r["best_params"]}')
