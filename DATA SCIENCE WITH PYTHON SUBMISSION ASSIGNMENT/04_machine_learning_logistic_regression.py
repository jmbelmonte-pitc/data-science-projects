"""
Data Science with Python
File 04: Machine Learning with Scikit-Learn

Purpose:
This section trains a Logistic Regression model to predict loan approval.
The target column used is Loan_Status, and the model accuracy is measured
using accuracy_score.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


CSV_FILE = "loan_p_train.csv"


# Load the dataset.
df = pd.read_csv(CSV_FILE)

# Check if the target column exists.
# Loan_Status is commonly used in the loan prediction dataset.
if "Loan_Status" not in df.columns:
    raise ValueError("The column 'Loan_Status' was not found in the CSV file.")

# Fill missing values before training the model.
# For numeric columns, missing values are filled with the mean.
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].mean())

# For text/categorical columns, missing values are filled with the most common value.
categorical_columns = df.select_dtypes(include=["object"]).columns
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Convert the target column into numbers.
# Y becomes 1 and N becomes 0.
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Remove columns that are not useful as input features.
# Loan_ID is only an ID, so it should not be used to train the model.
columns_to_drop = ["Loan_Status"]
if "Loan_ID" in df.columns:
    columns_to_drop.append("Loan_ID")

# Separate the input features (x) and target output (y).
x = df.drop(columns=columns_to_drop)
y = df["Loan_Status"]

# Convert categorical columns into numeric columns using one-hot encoding.
# Machine learning models need numeric input, so this step is important.
x = pd.get_dummies(x, drop_first=True)

# Split the data into training and testing sets.
# 75% is used for training and 25% is used for testing.
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=0
)

# Create the Logistic Regression model.
classifier = LogisticRegression(random_state=0, max_iter=1000)

# Train the model using the training data.
classifier.fit(x_train, y_train)

# Use the model to predict the test data.
y_predict = classifier.predict(x_test)

# Measure and print the accuracy of the model.
accuracy = accuracy_score(y_test, y_predict)
print("Model Accuracy:", accuracy)
