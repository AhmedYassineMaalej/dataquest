import pandas as pd
import joblib


def preprocess(df):
    """
    Cleans the data and applies Feature Engineering.
    Runs before the latency timer starts!
    """
    # 1. Drop unnecessary columns (Do NOT drop User_ID yet)
    cols_to_drop = ["Employer_ID"]
    df_cleaned = df.drop(
        columns=[col for col in cols_to_drop if col in df.columns]
    ).copy()

    # 2. Feature Engineering (Exact same logic as train.py)
    df_cleaned["Total_Family_Size"] = (
        df_cleaned["Adult_Dependents"].fillna(0)
        + df_cleaned["Child_Dependents"].fillna(0)
        + df_cleaned["Infant_Dependents"].fillna(0)
        + 1
    )

    df_cleaned["Income_Per_Family_Member"] = (
        df_cleaned["Estimated_Annual_Income"].fillna(0)
        / df_cleaned["Total_Family_Size"]
    )
    df_cleaned["Risk_Score"] = df_cleaned["Previous_Claims_Filed"].fillna(0) / (
        df_cleaned["Years_Without_Claims"].fillna(0) + 1.0
    )

    return df_cleaned


def load_model():
    """
    Loads and returns the trained model object.
    """
    return joblib.load("model.joblib")


def predict(df, model):
    """
    Takes the preprocessed DataFrame and the loaded model,
    returns predictions mapping User_ID to Purchased_Coverage_Bundle.
    """
    # Extract User_ID for the final output format
    user_ids = df["User_ID"]

    # Drop User_ID right before inference
    X = df.drop(columns=["User_ID"])

    # Generate predictions
    preds = model.predict(X)

    # Construct the final DataFrame with exactly the two required columns
    submission_df = pd.DataFrame(
        {"User_ID": user_ids, "Purchased_Coverage_Bundle": preds.astype(int)}
    )

    return submission_df
