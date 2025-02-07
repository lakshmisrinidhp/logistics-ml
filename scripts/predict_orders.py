import pandas as pd
import joblib

# Load the cleaned dataset and trained model
data = pd.read_csv('cleaned_daily_cp_activity.csv')
model = joblib.load('best_model.pkl')

# Load the training dataset to get exact feature names
X_train = pd.read_csv('X_train.csv')  # Ensure this file exists in your working directory
feature_names = X_train.columns

# Define tomorrow's features dynamically (replace values with forecasts)
tomorrow_features = pd.DataFrame({
    'temperature': [20],
    'relative_humidity': [0.6],
    'precipitation': [0.1],
    'season_Spring': [1],
    'season_Summer': [0],
    'season_Winter': [0],
    'day_of_week_Monday': [0],
    'day_of_week_Tuesday': [1],  # Assuming tomorrow is Tuesday
    'day_of_week_Wednesday': [0],
    'day_of_week_Thursday': [0],
    'day_of_week_Saturday': [0],
    'day_of_week_Sunday': [0],
    'rain_intensity_No Rain': [1],
    'rain_intensity_Moderate Rain': [0]
})

# Align features with the model's expectations
tomorrow_features = tomorrow_features.reindex(columns=feature_names, fill_value=0)

# Predict orders for tomorrow
predicted_orders_tomorrow = model.predict(tomorrow_features)
print(f"Predicted Orders for Tomorrow: {predicted_orders_tomorrow[0]:.2f}")

# Define features for the next week
weekly_features = pd.DataFrame({
    'temperature': [20, 22, 18, 19, 25, 26, 24],  # Replace with weekly forecast
    'relative_humidity': [0.6, 0.7, 0.5, 0.6, 0.5, 0.4, 0.5],
    'precipitation': [0.1, 0.0, 0.3, 0.2, 0.1, 0.0, 0.0],
    'season_Spring': [1, 1, 1, 1, 1, 1, 1],
    'season_Summer': [0, 0, 0, 0, 0, 0, 0],
    'season_Winter': [0, 0, 0, 0, 0, 0, 0],
    'day_of_week_Monday': [1, 0, 0, 0, 0, 0, 0],
    'day_of_week_Tuesday': [0, 1, 0, 0, 0, 0, 0],
    'day_of_week_Wednesday': [0, 0, 1, 0, 0, 0, 0],
    'day_of_week_Thursday': [0, 0, 0, 1, 0, 0, 0],
    'day_of_week_Saturday': [0, 0, 0, 0, 1, 0, 0],
    'day_of_week_Sunday': [0, 0, 0, 0, 0, 1, 0],
    'rain_intensity_No Rain': [1, 1, 0, 1, 1, 1, 1],
    'rain_intensity_Moderate Rain': [0, 0, 1, 0, 0, 0, 0]
})

# Align weekly features with the model's expectations
weekly_features = weekly_features.reindex(columns=feature_names, fill_value=0)

# Predict orders for the week
weekly_predictions = model.predict(weekly_features)
print(f"Predicted Orders for the Next Week: {weekly_predictions}")
