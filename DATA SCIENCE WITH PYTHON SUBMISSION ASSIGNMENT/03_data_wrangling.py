"""
Data Science with Python
File 03: Data Wrangling

Purpose:
This section checks for missing values, fills missing LoanAmount values
using the mean, checks data types, and shows examples of combining data
using concat and merge.
"""

import pandas as pd


CSV_FILE = "loan_p_train.csv"


# Load the dataset.
df = pd.read_csv(CSV_FILE)

# Check for missing values in each column.
# This helps us know which columns need cleaning.
missing_values = df.apply(lambda x: sum(x.isnull()), axis=0)
print("Missing values before cleaning:")
print(missing_values)

# Fill missing values in the LoanAmount column using the mean.
# The mean is used so that the missing values will not be empty anymore.
if "LoanAmount" in df.columns:
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].mean())

# Check the data types of each column.
# This helps us know if the columns are numeric, text, or another type.
print("\nData types:")
print(df.dtypes)

# Example of concatenation:
# This combines rows from two DataFrames into one DataFrame.
df_first_rows = df.head(3)
df_last_rows = df.tail(3)
combined_df = pd.concat([df_first_rows, df_last_rows])
print("\nExample of concatenated DataFrame:")
print(combined_df)

# Example of merging:
# This merges two small DataFrames using a common key column.
# I used simple sample data here just to show how merge works.
left = pd.DataFrame({
    "key": [1, 2, 3],
    "name": ["John", "Ana", "Mark"]
})

right = pd.DataFrame({
    "key": [1, 2, 3],
    "score": [85, 90, 88]
})

merged_df = pd.merge(left, right, on="key")
print("\nExample of merged DataFrame:")
print(merged_df)

# Save the cleaned data into a new CSV file.
df.to_csv("loan_p_train_cleaned.csv", index=False)
print("\nCleaned CSV file saved as loan_p_train_cleaned.csv")
