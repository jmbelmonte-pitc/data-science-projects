"""
Data Science with Python
File 02: Exploratory Data Analysis and Visualization

Purpose:
This section creates a histogram for the LoanAmount column.
The histogram helps us see the distribution of loan amounts and notice
possible outliers or unusual values.
"""

import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "loan_p_train.csv"


# Load the dataset.
df = pd.read_csv(CSV_FILE)

# Check that the LoanAmount column exists before trying to graph it.
if "LoanAmount" not in df.columns:
    raise ValueError("The column 'LoanAmount' was not found in the CSV file.")

# Create a histogram for the LoanAmount column.
# bins=50 means the values will be grouped into 50 intervals.
df["LoanAmount"].hist(bins=50)

# Add chart labels so the graph is easier to understand.
plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")

# Save the chart as an image file.
plt.savefig("loan_amount_histogram.png", bbox_inches="tight")

# Show the chart.
plt.show()
