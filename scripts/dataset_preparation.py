# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.stats import zscore  # Ensure scipy is installed

# Load the dataset
file_path = 'daily_cp_activity_dataset.csv'
data = pd.read_csv(file_path)

# Inspect the dataset
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Information:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())
print("\nMissing Values in Each Column:")
print(data.isnull().sum())

# Handle missing values
# For numerical columns, fill missing values with the median
numerical_columns = data.select_dtypes(include=[np.number]).columns
for column in numerical_columns:
    if data[column].isnull().sum() > 0:
        data[column] = data[column].fillna(data[column].median())

# Convert date column to datetime format
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    print("\nDate column converted to datetime format.")

# Handle outliers using z-score
data = data[(np.abs(zscore(data.select_dtypes(include=[np.number]))) < 3).all(axis=1)]

# Save the cleaned dataset
output_file_path = 'cleaned_daily_cp_activity.csv'
data.to_csv(output_file_path, index=False)
print(f"\nCleaned dataset saved to: {output_file_path}")

# Final Confirmation
print("\nCleaned Data Preview:")
print(data.head())
