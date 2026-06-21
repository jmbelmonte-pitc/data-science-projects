"""
Data Science with Python
File 01: Library Imports and Data Loading

Purpose:
This section imports the main Python libraries used for data science
and loads the loan training CSV file into a pandas DataFrame.
"""

# pandas is used for reading and working with data tables.
import pandas as pd

# numpy is used for numerical operations.
import numpy as np

# matplotlib is used for creating charts and visualizations.
import matplotlib.pyplot as plt


# The CSV file should be in the same folder as this Python file.
CSV_FILE = "loan_p_train.csv"


def load_data(csv_file=CSV_FILE):
    """Load the CSV file and show the first rows and summary statistics."""

    # Read the CSV file into a DataFrame.
    df = pd.read_csv(csv_file)

    # Display the first five rows so we can quickly check the data.
    print("First five rows of the data:")
    print(df.head())

    # Display summary statistics for the numeric columns.
    print("\nSummary statistics:")
    print(df.describe())

    return df


if __name__ == "__main__":
    # Run the loading function when this file is executed directly.
    data = load_data()
