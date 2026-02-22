import pandas as pd
import time
import solution  # This imports your solution.py file


def run_test():
    print("Starting local validation test...\n")

    # 1. Load actual data to ensure ALL features are present
    print("Loading test data...")
    try:
        # Read the first 10 rows of your actual training data to use as a test
        test_df = pd.read_csv("train.csv").head(10)
    except FileNotFoundError:
        print("❌ ERROR: 'train.csv' not found! Make sure it is in the same folder.")
        return

    # 2. Simulate the judge's environment
    # The judge's test set will NOT have the target column, so we must drop it
    # here to perfectly simulate the real hackathon environment.
    if "Purchased_Coverage_Bundle" in test_df.columns:
        test_df = test_df.drop(columns=["Purchased_Coverage_Bundle"])

    original_user_ids = test_df["User_ID"].tolist()

    try:
        # 3. Test preprocess()
        print("Testing preprocess()...")
        processed_df = solution.preprocess(test_df)

        # 4. Test load_model()
        print("Testing load_model()...")
        model = solution.load_model()

        # 5. Test predict() and measure latency
        print("Testing predict()...")
        start_time = time.time()
        predictions_df = solution.predict(processed_df, model)
        print(predictions_df)
        latency = time.time() - start_time

        # 6. Validate Output Format Requirements
        print("\nValidating output formatting...")

        # Check columns
        expected_columns = ["User_ID", "Purchased_Coverage_Bundle"]
        assert list(predictions_df.columns) == expected_columns, (
            f"Output must have exactly these columns: {expected_columns}"
        )

        # Check if all User_IDs are present
        assert len(predictions_df) == len(test_df), (
            "Number of predictions must match the input rows"
        )
        assert set(predictions_df["User_ID"]) == set(original_user_ids), (
            "Missing User_IDs in predictions! All test User_IDs must be present"
        )

        # Check data types and values
        bundle_col = predictions_df["Purchased_Coverage_Bundle"]
        assert pd.api.types.is_integer_dtype(bundle_col), (
            "Purchased_Coverage_Bundle values must be integers"
        )
        assert bundle_col.min() >= 0 and bundle_col.max() <= 9, (
            "Purchased_Coverage_Bundle values must be between 0 and 9"
        )

        print("\n✅ SUCCESS: Your solution.py passes all local formatting checks!")
        print(f"⏱️ Inference Latency: {latency:.4f} seconds.")

    except AssertionError as e:
        print(f"\n❌ FAILED CHECK: {e}")
    except Exception as e:
        print(f"\n❌ EXECUTION ERROR: {e}")


if __name__ == "__main__":
    run_test()
