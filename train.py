# train.py
import pandas as pd
from lightgbm import LGBMClassifier  # <--- NEW IMPORT
import joblib

# Import the shared preprocessing logic!
from solution import preprocess


def train_and_save_model():
    print("Loading training data...")
    df = pd.read_csv("train.csv")

    print("Preprocessing data using solution.py pipeline...")
    # This guarantees the exact same transformations happen!
    processed_df = preprocess(df)

    # Separate features and target
    y = processed_df["Purchased_Coverage_Bundle"]
    X = processed_df.drop(columns=["User_ID", "Purchased_Coverage_Bundle"])

    print("Training LightGBM model with tuned hyperparameters...")

    # --- TUNED LightGBM Parameters ---
    model = LGBMClassifier(
        n_estimators=300,  # Increased from 100 (More trees)
        learning_rate=0.05,  # Added (Smaller steps per tree)
        num_leaves=50,  # Added (Allows more complex patterns, default is 31)
        max_depth=10,  # Keeps the trees from getting too deep to prevent overfitting
        class_weight="balanced",  # Crucial: Keep this to handle rare bundles!
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    joblib.dump(model, "model.joblib")
    print("✅ Model successfully trained and saved as model.joblib")


if __name__ == "__main__":
    train_and_save_model()
