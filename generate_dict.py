import pandas as pd

# Load your training data
df = pd.read_csv("train.csv")

# Identify categorical columns (excluding IDs and Target)
cat_cols = df.select_dtypes(exclude=["number"]).columns
cat_cols = cat_cols.drop(["User_ID", "Purchased_Coverage_Bundle"], errors="ignore")

print("--- COPY THIS INTO YOUR SOLUTION.PY ---")
print("CATEGORY_MAPPINGS = {")
for col in cat_cols:
    # Convert everything to string and replace NaNs with "Unknown"
    cleaned_series = df[col].astype(str).replace("nan", "Unknown")
    unique_vals = sorted(cleaned_series.unique().tolist())

    mapping = {val: i for i, val in enumerate(unique_vals)}
    print(f"    '{col}': {mapping},")
print("}")
