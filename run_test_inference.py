import pandas as pd
import time
import solution  # Imports your solution.py file


def run_inference():
    print("Loading test.csv...")
    try:
        # Load the official test dataset (15,218 rows)
        test_df = pd.read_csv("test.csv")
    except FileNotFoundError:
        print("❌ ERROR: 'test.csv' not found! Make sure it is in the same folder.")
        return

    original_user_ids = test_df["User_ID"].tolist()

    print(f"Dataset loaded successfully with {len(test_df)} rows.")
    print("\nStarting inference pipeline...")

    try:
        # 1. Preprocess
        print("Running preprocess()...")
        processed_df = solution.preprocess(test_df)

        # 2. Load Model
        print("Running load_model()...")
        model = solution.load_model()

        # 3. Predict & Time it
        print("Running predict()...")
        start_time = time.time()
        predictions_df = solution.predict(processed_df, model)
        latency = time.time() - start_time

        # --- VALIDATION CHECKS ---
        print("\nValidating output against Hackathon rules...")

        # Check columns
        expected_columns = ["User_ID", "Purchased_Coverage_Bundle"]
        assert list(predictions_df.columns) == expected_columns, (
            f"Output must have exactly these columns: {expected_columns}"
        )

        # Check rows and missing IDs
        assert len(predictions_df) == len(test_df), (
            f"Expected {len(test_df)} rows, but got {len(predictions_df)}"
        )
        assert set(predictions_df["User_ID"]) == set(original_user_ids), (
            "Missing User_IDs! All test User_IDs must be present."
        )

        # Check data types and boundaries
        bundle_col = predictions_df["Purchased_Coverage_Bundle"]
        assert pd.api.types.is_integer_dtype(bundle_col), (
            "Purchased_Coverage_Bundle values must be integers."
        )
        assert bundle_col.min() >= 0 and bundle_col.max() <= 9, (
            "Purchased_Coverage_Bundle values must be between 0 and 9."
        )

        print("✅ SUCCESS: Output perfectly matches the judge's requirements!")
        print(f"⏱️ Inference Latency: {latency:.4f} seconds.")
        if latency > 10:
            print(
                "⚠️ WARNING: Latency is over 10s. You will incur a latency penalty[cite: 80]."
            )

        # Optional: Save predictions to a CSV so you can inspect them visually
        output_file = "my_predictions.csv"
        predictions_df.to_csv(output_file, index=False)
        print(f"\n📁 Predictions saved to {output_file} for your review.")

    except AssertionError as e:
        print(f"\n❌ FORMATTING ERROR: {e}")
    except Exception as e:
        print(f"\n❌ EXECUTION ERROR: {e}")


if __name__ == "__main__":
    run_inference()
