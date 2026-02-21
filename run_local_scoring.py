import pandas as pd
import time
import os
from sklearn.metrics import f1_score
import solution  # Imports your solution.py file


def run_scoring_pipeline():
    # 1. Load a dataset that has the ground truth answers
    # (e.g., a 20% holdout set you saved from train.csv)
    dataset_path = "local_val.csv"
    print(f"Loading {dataset_path}...")

    try:
        val_df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(
            f"❌ ERROR: '{dataset_path}' not found! Please create a validation set from your training data."
        )
        return

    if "Purchased_Coverage_Bundle" not in val_df.columns:
        print(
            "❌ ERROR: Your validation dataset must contain the 'Purchased_Coverage_Bundle' column to calculate a score."
        )
        return

    # Extract the ground truth and remove it from the dataframe to simulate the test environment
    y_true = val_df["Purchased_Coverage_Bundle"].copy()
    test_df = val_df.drop(columns=["Purchased_Coverage_Bundle"])

    print(f"Dataset loaded successfully with {len(test_df)} rows.")
    print("\nStarting inference pipeline...")

    try:
        # 2. Preprocess
        print("Running preprocess()...")
        processed_df = solution.preprocess(test_df)

        # 3. Load Model
        print("Running load_model()...")
        model = solution.load_model()

        # 4. Predict & Time it (Only predict() is timed for the latency penalty)
        print("Running predict()...")
        start_time = time.time()
        predictions_df = solution.predict(processed_df, model)
        latency = time.time() - start_time

        y_pred = predictions_df["Purchased_Coverage_Bundle"]

        # --- SCORING CALCULATIONS ---
        print("\n" + "=" * 40)
        print("🏆 HACKATHON SCORE CALCULATION 🏆")
        print("=" * 40)

        # A. Base Metric: Macro F1 Score
        # Calculates F1 independently for all 10 classes and averages them
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"Raw Macro F1 Score: {macro_f1:.4f}")

        # B. Latency Penalty
        # Inference latency up to 10s is penalized linearly; floors at 0.5
        latency_multiplier = max(0.5, 1 - (latency / 10))
        print(
            f"Inference Latency:  {latency:.4f} seconds (Multiplier: {latency_multiplier:.4f})"
        )

        # C. Size Penalty
        # Models up to 200 MB are penalized linearly; floors at 0.5
        model_size_bytes = os.path.getsize("model.joblib")
        model_size_mb = model_size_bytes / (1024 * 1024)
        size_multiplier = max(0.5, 1 - (model_size_mb / 200))
        print(
            f"Model File Size:    {model_size_mb:.2f} MB (Multiplier: {size_multiplier:.4f})"
        )

        # D. Final Score Calculation
        # Final Score = Macro F1 * Size Multiplier * Latency Multiplier
        final_score = macro_f1 * size_multiplier * latency_multiplier

        print("-" * 40)
        print(f"🚀 ESTIMATED FINAL SCORE: {final_score:.4f} 🚀")
        print("-" * 40)

        if latency_multiplier <= 0.5:
            print(
                "⚠️ WARNING: Your latency is over 10s! You have hit the maximum latency penalty floor of 50%."
            )
        if size_multiplier <= 0.5:
            print(
                "⚠️ WARNING: Your model file is over 200MB! You have hit the maximum size penalty floor of 50%."
            )

    except Exception as e:
        print(f"\n❌ EXECUTION ERROR: {e}")


if __name__ == "__main__":
    run_scoring_pipeline()
