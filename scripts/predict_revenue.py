import pandas as pd
import joblib
import numpy as np

# Load the cleaned dataset
data = pd.read_csv('cleaned_daily_cp_activity.csv')

# Load the trained model
model = joblib.load('best_model.pkl')

# Ensure the feature names match the trained model
feature_names = joblib.load('trained_feature_names.pkl')  # Save and load feature names during model training

# Generate data for tomorrow based on the latest entry
latest_entry = data.iloc[-1].copy()

# Create a new DataFrame for tomorrow's features
tomorrow_features = pd.DataFrame([latest_entry])
tomorrow_features['date'] = pd.to_datetime(tomorrow_features['date']) + pd.Timedelta(days=1)

# Encode day of the week, season, and rain intensity for tomorrow
tomorrow_features['day_of_week'] = tomorrow_features['date'].dt.day_name()
tomorrow_features['season'] = 'Winter'  # Example: Adjust based on the date
tomorrow_features['rain_intensity'] = 'No Rain'  # Example: Adjust based on weather forecast

# One-hot encode categorical columns (ensure the same encoding as during training)
encoded_features = pd.get_dummies(
    tomorrow_features[['day_of_week', 'season', 'rain_intensity']],
    drop_first=True
)

# Merge with numerical features
numerical_features = tomorrow_features[['temperature', 'relative_humidity', 'precipitation']]
tomorrow_features = pd.concat([numerical_features, encoded_features], axis=1)

# Add missing columns with zeros and reorder columns to match the trained model
for feature in feature_names:
    if feature not in tomorrow_features.columns:
        tomorrow_features[feature] = 0
tomorrow_features = tomorrow_features[feature_names]

# Predict the number of courier partners for tomorrow
predicted_couriers = model.predict(tomorrow_features)

# Calculate revenue using an assumed average revenue per courier
average_revenue_per_courier = 100  # Example value
predicted_revenue_tomorrow = predicted_couriers[0] * average_revenue_per_courier

print(f"Predicted Revenue for Tomorrow: ${predicted_revenue_tomorrow:.2f}")
