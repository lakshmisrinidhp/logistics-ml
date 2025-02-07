# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'cleaned_daily_cp_activity.csv'  # Ensure this file exists from previous steps
data = pd.read_csv(file_path)

# Visualize the data
# Box plot for 'courier_partners_online'
sns.boxplot(data['courier_partners_online'])
plt.title("Box Plot for Courier Partners Online")
plt.show()

# Scatter plot for 'courier_partners_online' vs. 'date'
plt.figure(figsize=(10, 6))
plt.scatter(data['date'], data['courier_partners_online'], alpha=0.6)
plt.title("Courier Partners Online Over Time")
plt.xlabel("Date")
plt.ylabel("Courier Partners Online")
plt.xticks(rotation=45)
plt.show()

# Detect Outliers using IQR
# Calculate IQR for 'courier_partners_online'
Q1 = data['courier_partners_online'].quantile(0.25)
Q3 = data['courier_partners_online'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier thresholds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Print thresholds
print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

# Identify outliers
outliers = data[(data['courier_partners_online'] < lower_bound) | (data['courier_partners_online'] > upper_bound)]
print(f"Number of Outliers: {len(outliers)}")

# Filter out outliers
data_cleaned = data[(data['courier_partners_online'] >= lower_bound) & (data['courier_partners_online'] <= upper_bound)]

# Confirm Cleaned Data
print(f"Original Dataset Size: {len(data)}")
print(f"Cleaned Dataset Size: {len(data_cleaned)}")

# Save the cleaned dataset
output_file_path = 'cleaned_daily_cp_activity_no_outliers.csv'
data_cleaned.to_csv(output_file_path, index=False)
print(f"Cleaned dataset saved to: {output_file_path}")
