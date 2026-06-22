# Insurance Linear Regression Assignment
# This script loads the insuranceData.csv file, encodes categorical data,
# performs linear regression, and saves the result to a new CSV file.

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load the insurance CSV file.
insurance_data = pd.read_csv("insuranceData.csv")

# 2. View the first few rows to understand the data.
print("First 5 rows of the dataset:")
print(insurance_data.head())


# 3. Apply LabelEncoder to the categorical column.
# The smoker column contains yes/no values, so I converted it into numbers.
label_encoder = LabelEncoder()
insurance_data["smokerLabel"] = label_encoder.fit_transform(insurance_data["smoker"])


# 4. Apply OneHotEncoder to the categorical column.
# This creates separate columns for each smoker category.
try:
    onehot_encoder = OneHotEncoder(sparse_output=False, drop=None)
except TypeError:
    onehot_encoder = OneHotEncoder(sparse=False, drop=None)

encoded_array = onehot_encoder.fit_transform(insurance_data[["smoker"]])
encoded_columns = onehot_encoder.get_feature_names_out(["smoker"])

encoded_data = pd.DataFrame(
    encoded_array,
    columns=encoded_columns,
    index=insurance_data.index
)

# Combine the original data and encoded columns.
insurance_data_encoded = pd.concat([insurance_data, encoded_data], axis=1)


# 5. Select the features and target column for linear regression.
# I used age, bmi, children, and smokerLabel to predict charges.
X = insurance_data_encoded[["age", "bmi", "children", "smokerLabel"]]
y = insurance_data_encoded["charges"]


# 6. Split the data into training and testing data.
# The assignment requires the test size to be 15%.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42
)


# 7. Create and train the Linear Regression model.
model = LinearRegression()
model.fit(X_train, y_train)


# 8. Make predictions using the test data.
y_pred = model.predict(X_test)


# 9. Evaluate the model performance.
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Results:")
print("Test Size: 15%")
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)


# 10. Save predictions back to a result CSV file.
insurance_data_encoded["predictedCharges"] = model.predict(X)
insurance_data_encoded["residual"] = insurance_data_encoded["charges"] - insurance_data_encoded["predictedCharges"]
insurance_data_encoded["dataSplit"] = "train"
insurance_data_encoded.loc[X_test.index, "dataSplit"] = "test"

insurance_data_encoded.to_csv("insuranceData_result.csv", index=False)

print("\nThe result CSV file was saved as insuranceData_result.csv")
