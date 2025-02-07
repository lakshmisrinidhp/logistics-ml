# Import necessary libraries
import pandas as pd

# Load the cleaned dataset
file_path = 'cleaned_daily_cp_activity_no_outliers.csv'  # Ensure this file exists
data = pd.read_csv(file_path)

# Convert the 'date' column to datetime format (if not already)
data['date'] = pd.to_datetime(data['date'])

# Feature 1: Day of the Week
data['day_of_week'] = data['date'].dt.day_name()

# Feature 2: Is Weekend (True for Saturday and Sunday, False otherwise)
data['is_weekend'] = data['day_of_week'].isin(['Saturday', 'Sunday'])

# Feature 3: Season (based on months)
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

data['season'] = data['date'].dt.month.apply(get_season)

# Feature 4: Categorized Temperature (Cold, Moderate, Hot)
def categorize_temperature(temp):
    if temp < 10:
        return 'Cold'
    elif 10 <= temp <= 25:
        return 'Moderate'
    else:
        return 'Hot'

data['temperature_category'] = data['temperature'].apply(categorize_temperature)

# Feature 5: Rain Intensity (Categorized Precipitation)
def categorize_precipitation(precip):
    if precip == 0:
        return 'No Rain'
    elif precip < 2.5:
        return 'Light Rain'
    elif precip < 7.6:
        return 'Moderate Rain'
    else:
        return 'Heavy Rain'

data['rain_intensity'] = data['precipitation'].apply(categorize_precipitation)

# Save the dataset with new features
output_file_path = 'engineered_daily_cp_activity.csv'
data.to_csv(output_file_path, index=False)
print(f"Dataset with new features saved to: {output_file_path}")

# Display the first few rows of the updated dataset
print("\nPreview of Dataset with New Features:")
print(data.head())
