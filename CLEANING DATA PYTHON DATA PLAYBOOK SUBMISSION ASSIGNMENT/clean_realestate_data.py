# Cleaning Data Python Data Playbook Submission Assignment
# This script cleans the Real Estate dataset and saves the changes into a new CSV file.

import pandas as pd

# 1. Load the Real Estate CSV file
# I used pandas to read the CSV file so I can clean and update the dataset.
file_path = "Realestate.csv"
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)
print("Original columns:", df.columns.tolist())

# 2 and 3. Normalize the houseAge column and save it in a new column
# I used min-max normalization here. This changes the houseAge values into a scale from 0 to 1.
# Formula: (value - minimum value) / (maximum value - minimum value)
house_age_min = df["houseAge"].min()
house_age_max = df["houseAge"].max()
df["houseAgeStandardized"] = (df["houseAge"] - house_age_min) / (house_age_max - house_age_min)

# 4. Drop the numberOfConvenienceStores column
# This removes the column from the dataset and keeps the change in the dataframe.
df.drop(columns=["numberOfConvenienceStores"], inplace=True)

# 5. Rename the transaction column to transactionDate
# This makes the column name more clear and easier to understand.
df.rename(columns={"transaction": "transactionDate"}, inplace=True)

# 6. Use .loc[] to display all rows from 0 to 10
# .loc includes the ending row, so this will display rows 0 through 10.
print("\nRows 0 to 10 using .loc[]:")
print(df.loc[0:10])

# 7. Use .iloc[] to display the first 10 rows
# .iloc does not include the ending number, so 0:10 displays rows 0 through 9.
print("\nFirst 10 rows using .iloc[]:")
print(df.iloc[0:10])

# 8. Find and remove duplicate rows
# I checked the number of duplicate rows first, then removed them.
duplicate_count = df.duplicated().sum()
print("\nDuplicate rows found:", duplicate_count)
df.drop_duplicates(inplace=True)
print("Dataset shape after removing duplicates:", df.shape)

# 9. Find missing values and fill them with the mean
# I checked missing values and then filled numeric missing values using the mean of each column.
print("\nMissing values before filling:")
print(df.isnull().sum())

numeric_columns = df.select_dtypes(include=["number"]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

print("\nMissing values after filling:")
print(df.isnull().sum())

# 10. Comments were added for each section explaining what the code is doing.

# 11. Save the cleaned dataset to a new CSV file
# This makes all changes persist in the cleaned CSV file.
output_file = "Realestate_cleaned.csv"
df.to_csv(output_file, index=False)

print("\nCleaned dataset saved as:", output_file)
print("Final columns:", df.columns.tolist())
print("Final dataset shape:", df.shape)
