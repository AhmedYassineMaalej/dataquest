# solution.py
import pandas as pd
import joblib
import os


def preprocess(df):
    processed_df = df.copy()

    # --- 1. FEATURE ENGINEERING ---
    # Combine dependents into a single feature
    if all(
        c in processed_df.columns
        for c in ["Adult_Dependents", "Child_Dependents", "Infant_Dependents"]
    ):
        processed_df["Total_Dependents"] = (
            processed_df["Adult_Dependents"]
            + processed_df["Child_Dependents"]
            + processed_df["Infant_Dependents"]
        )

        # Create a wealth metric (adding 1 to avoid division by zero)
        processed_df["Income_Per_Dependent"] = processed_df[
            "Estimated_Annual_Income"
        ] / (processed_df["Total_Dependents"] + 1)

        # Drop the original redundant columns
        processed_df = processed_df.drop(
            columns=["Adult_Dependents", "Child_Dependents", "Infant_Dependents"]
        )

    # --- 2. DROP UNNECESSARY FEATURES ---
    # Drop high-cardinality IDs (keeping User_ID for the predict function later)
    cols_to_drop = ["Broker_ID", "Employer_ID"]

    # Drop features that might have zero variance or are purely redundant
    cols_to_drop.extend(["Policy_Start_Year", "Existing_Policyholder"])

    # Safely drop if they exist in the dataset
    processed_df = processed_df.drop(
        columns=[c for c in cols_to_drop if c in processed_df.columns]
    )

    # --- 3. STANDARD CLEANUP (Missing values & Encoding) ---
    num_cols = processed_df.select_dtypes(include=["number"]).columns
    num_cols = num_cols.drop(["User_ID", "Purchased_Coverage_Bundle"], errors="ignore")
    processed_df[num_cols] = processed_df[num_cols].fillna(0)

    cat_cols = processed_df.select_dtypes(exclude=["number"]).columns
    cat_cols = cat_cols.drop(["User_ID", "Purchased_Coverage_Bundle"], errors="ignore")
    for col in cat_cols:
        processed_df[col] = processed_df[col].astype("category").cat.codes

    return processed_df


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    return joblib.load(model_path)


def predict(df, model):
    # Force the model to use only 1 thread to match the container constraint
    model.set_params(n_jobs=1)

    features = df.drop(columns=["User_ID"])
    predictions = model.predict(features)

    output_df = pd.DataFrame(
        {"User_ID": df["User_ID"], "Purchased_Coverage_Bundle": predictions}
    )
    output_df["Purchased_Coverage_Bundle"] = output_df[
        "Purchased_Coverage_Bundle"
    ].astype(int)
    return output_df
