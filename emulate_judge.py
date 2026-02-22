import pandas as pd
import time
import os
from sklearn.metrics import f1_score
import solution


def calculate_multipliers(latency, file_size_mb):
    # Standard Hackathon Penalty Logic
    # Latency Penalty (Baseline 0.1s, capped at 10s)
    latency_multiplier = max(0.5, 1.0 - (latency / 10.0))
    # Size Penalty (Baseline 5MB, capped at 500MB)
    size_multiplier = max(0.5, 1.0 - (file_size_mb / 500.0))
    return latency_multiplier, size_multiplier


def judge_submission():
    print("🚀 [JUDGE] Starting Local Evaluation...")

    try:
        # 1. Load Ground Truth
        data = pd.read_csv("local_val.csv")
        y_true = data["Purchased_Coverage_Bundle"]
        X_test = data.drop(columns=["Purchased_Coverage_Bundle"])

        # 2. Measure Model Size
        model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        # 3. Load Model
        model = solution.load_model()

        # 4. Measure Inference Latency
        start_time = time.time()
        processed_df = solution.preprocess(X_test)
        predictions_df = solution.predict(processed_df, model)
        end_time = time.time()

        latency = (end_time - start_time) / len(X_test)
        y_pred = predictions_df["Purchased_Coverage_Bundle"]

        # 5. Calculate Metrics
        raw_f1 = f1_score(y_true, y_pred, average="macro")
        lat_mult, size_mult = calculate_multipliers(latency, file_size_mb)
        final_score = raw_f1 * lat_mult * size_mult

        # 6. Display Results
        print("\n" + "=" * 40)
        print(f"📊 RAW MACRO F1: {raw_f1:.4f}")
        print(f"⏱️  LATENCY:     {latency:.4f}s (Mult: {lat_mult:.4f})")
        print(f"📁 FILE SIZE:   {file_size_mb:.2f}MB (Mult: {size_mult:.4f})")
        print("-" * 40)
        print(f"🏆 ESTIMATED FINAL SCORE: {final_score:.4f}")
        print("=" * 40 + "\n")

    except Exception as e:
        print(f"❌ [JUDGE] EVALUATION FAILED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    judge_submission()
