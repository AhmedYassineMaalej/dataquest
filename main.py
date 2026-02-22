import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OrdinalEncoder

# 1. Load the dataset
df = pd.read_csv("train.csv")

# 2. Define features and target (using 'Policy_Cancelled_Post_Purchase')
target = "Policy_Cancelled_Post_Purchase"

# Drop User_ID (irrelevant) and Employer_ID (too many missing values)
X = df.drop(columns=[target, "User_ID", "Employer_ID"])
y = df[target]

# 3. Preprocess categorical features using OrdinalEncoder
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])

# 4. Train-Test Split (with stratification to handle class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Iteratively train models across different learning rates
learning_rates = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.25]
f1_scores = []

for lr in learning_rates:
    # Using HistGradientBoosting because it's fast and handles missing values natively
    clf = HistGradientBoostingClassifier(
        learning_rate=lr,
        max_iter=100,  # Number of trees
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Calculate Macro F1
    score = f1_score(y_test, y_pred, average="macro")
    f1_scores.append(score)

# 6. Plot the results using Matplotlib
plt.figure(figsize=(8, 5))
plt.plot(learning_rates, f1_scores, marker="o", linestyle="-", color="b", linewidth=2)
plt.title("Macro F1 Score vs. Learning Rate", fontsize=14)
plt.xlabel("Learning Rate (Log Scale)", fontsize=12)
plt.ylabel("Macro F1 Score", fontsize=12)
plt.xscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save the plot
plt.tight_layout()
plt.savefig("f1_vs_learning_rate.png")
