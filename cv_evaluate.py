import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import time

# Import your actual preprocessing function to guarantee identical processing!
from solution import preprocess


def evaluate_5fold():
    print("Loading data...")
    df = pd.read_csv("train.csv")

    print("Applying your solution.py preprocessing...")
    processed_df = preprocess(df)

    # Separate target and features
    y = processed_df["Purchased_Coverage_Bundle"]
    X = processed_df.drop(
        columns=["User_ID", "Purchased_Coverage_Bundle"], errors="ignore"
    )

    print(
        f"Starting 5-Fold Stratified CV on {len(X)} rows and {len(X.columns)} features...\n"
    )

    # Stratified ensures those rare 8 and 9 bundles are in every single fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    # Our optimized hackathon parameters
    params = {
        "n_estimators": 150,
        "learning_rate": 0.08,
        "num_leaves": 31,
        "min_child_samples": 20,
        "class_weight": "balanced",  # Crucial for Macro F1
        "random_state": 42,
        "n_jobs": -1,  # Using all cores here just for fast local evaluation
        "verbose": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        start_time = time.time()

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        score = f1_score(y_val, preds, average="macro")
        fold_scores.append(score)

        elapsed = time.time() - start_time
        print(f"Fold {fold} - Macro F1: {score:.4f} (Took {elapsed:.1f}s)")

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    print("\n" + "=" * 45)
    print(f"🏆 FINAL 5-FOLD CV RESULTS")
    print(f"Average Macro F1: {mean_score:.4f} ± {std_score:.4f}")
    print("=" * 45)


if __name__ == "__main__":
    evaluate_5fold()
