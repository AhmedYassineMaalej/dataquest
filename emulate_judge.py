import pandas as pd
import time
import sys
import traceback
import solution


def judge_submission():
    print("🚀 [JUDGE] Container started. Initializing test...")

    try:
        # Load test data (we use your local validation set, dropping the answer key)
        df = pd.read_csv("local_val.csv")
        if "Purchased_Coverage_Bundle" in df.columns:
            df = df.drop(columns=["Purchased_Coverage_Bundle"])

        original_user_ids = df["User_ID"].tolist()
        print(f"✅ [JUDGE] Loaded test dataset ({len(df)} rows).")

        # Step 1: Preprocess
        print("⏳ [JUDGE] Running solution.preprocess()...")
        processed_df = solution.preprocess(df)

        # Step 2: Load Model
        print("⏳ [JUDGE] Running solution.load_model()...")
        model = solution.load_model()

        # Step 3: Predict
        print("⏳ [JUDGE] Running solution.predict()...")
        predictions = solution.predict(processed_df, model)

        # Step 4: Strict Validation
        print("🔍 [JUDGE] Validating output format...")

        if not isinstance(predictions, pd.DataFrame):
            raise TypeError("predict() did not return a pandas DataFrame.")

        expected_cols = ["User_ID", "Purchased_Coverage_Bundle"]
        if list(predictions.columns) != expected_cols:
            raise ValueError(
                f"Columns mismatch. Expected {expected_cols}, got {list(predictions.columns)}"
            )

        if len(predictions) != len(df):
            raise ValueError(
                f"Row count mismatch. Expected {len(df)}, got {len(predictions)}"
            )

        if set(predictions["User_ID"]) != set(original_user_ids):
            raise ValueError("Missing or altered User_IDs in the output.")

        print(
            "🎉 [JUDGE] SUCCESS! Your code executed perfectly in the isolated container."
        )
        sys.exit(0)

    except Exception as e:
        print("\n❌ [JUDGE] CRITICAL FAILURE:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    judge_submission()
