import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


def engineer_features(df):
    """
    Creates new business-logic features to capture the 20 Feature Engineering points.
    We use .fillna() safely just for the math, letting the model handle native NaNs elsewhere.
    """
    # 1. Total Family Size (+1 for the primary applicant)
    df["Total_Family_Size"] = (
        df["Adult_Dependents"].fillna(0)
        + df["Child_Dependents"].fillna(0)
        + df["Infant_Dependents"].fillna(0)
        + 1
    )

    # 2. Wealth / Income Per Family Member
    df["Income_Per_Family_Member"] = (
        df["Estimated_Annual_Income"].fillna(0) / df["Total_Family_Size"]
    )

    # 3. Risk Score: High claims in a short time means higher risk
    df["Risk_Score"] = df["Previous_Claims_Filed"].fillna(0) / (
        df["Years_Without_Claims"].fillna(0) + 1.0
    )

    return df


def train_model():
    print("Loading training data...")
    df = pd.read_csv("modified.csv")

    print("Applying Feature Engineering...")
    # Apply our engineered features
    df = engineer_features(df)

    # Set the target based on the hackathon document
    target = "Purchased_Coverage_Bundle"

    # Drop target and identifiers from features
    X = df.drop(columns=[target, "User_ID", "Employer_ID"])
    y = df[target]

    # Calculate sample weights to fix the extreme class imbalance (Bundles 8 & 9)
    print("Calculating balanced class weights...")
    sample_weights = compute_sample_weight(class_weight="balanced", y=y)

    # Identify categorical columns for encoding
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print("Building pipeline...")
    # Step 1: Encode categories
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cat_cols,
            )
        ],
        remainder="passthrough",
    )

    # Step 2: Model (Optimized Version)
    model = HistGradientBoostingClassifier(
        learning_rate=0.2,
        max_iter=300,
        max_leaf_nodes=127,  # Allows trees to grow more complex
        l2_regularization=0.1,  # Prevents overfitting on that extra complexity
        random_state=42,
    )

    # Combine into Pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    print("Training model (this will take a moment with 300 iterations)...")
    # We pass the sample_weights directly into the classifier step of the pipeline
    pipeline.fit(X, y, classifier__sample_weight=sample_weights)

    print("Saving model to model.joblib...")
    joblib.dump(pipeline, "model.joblib")
    print("Done! Model is ready for submission.")


if __name__ == "__main__":
    train_model()
