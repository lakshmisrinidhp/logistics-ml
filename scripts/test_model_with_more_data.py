import joblib
import pandas as pd

# Load the saved model
model_filename = 'best_model.pkl'
print(f"Loading the model from {model_filename}...")
model = joblib.load(model_filename)

# Load X_train to get the correct feature names and order
X_train = pd.read_csv('X_train.csv')  # Ensure this is the same dataset used for training
feature_names = X_train.columns.tolist()

# Define additional test cases
new_data = pd.DataFrame([
    {'temperature': 15, 'relative_humidity': 0.8, 'precipitation': 0.0, 'season_Spring': 1,
     'season_Summer': 0, 'season_Winter': 0, 'day_of_week_Monday': 1, 'day_of_week_Tuesday': 0,
     'day_of_week_Wednesday': 0, 'day_of_week_Thursday': 0, 'day_of_week_Saturday': 0,
     'day_of_week_Sunday': 0, 'rain_intensity_No Rain': 1, 'rain_intensity_Moderate Rain': 0},

    {'temperature': 30, 'relative_humidity': 0.6, 'precipitation': 0.3, 'season_Spring': 0,
     'season_Summer': 1, 'season_Winter': 0, 'day_of_week_Monday': 0, 'day_of_week_Tuesday': 1,
     'day_of_week_Wednesday': 0, 'day_of_week_Thursday': 0, 'day_of_week_Saturday': 0,
     'day_of_week_Sunday': 0, 'rain_intensity_No Rain': 0, 'rain_intensity_Moderate Rain': 1},

    {'temperature': 22, 'relative_humidity': 0.4, 'precipitation': 0.0, 'season_Spring': 0,
     'season_Summer': 0, 'season_Winter': 1, 'day_of_week_Monday': 0, 'day_of_week_Tuesday': 0,
     'day_of_week_Wednesday': 1, 'day_of_week_Thursday': 0, 'day_of_week_Saturday': 0,
     'day_of_week_Sunday': 1, 'rain_intensity_No Rain': 1, 'rain_intensity_Moderate Rain': 0},

    {'temperature': 35, 'relative_humidity': 0.9, 'precipitation': 1.2, 'season_Spring': 0,
     'season_Summer': 0, 'season_Winter': 0, 'day_of_week_Monday': 0, 'day_of_week_Tuesday': 0,
     'day_of_week_Wednesday': 0, 'day_of_week_Thursday': 1, 'day_of_week_Saturday': 1,
     'day_of_week_Sunday': 0, 'rain_intensity_No Rain': 0, 'rain_intensity_Moderate Rain': 1},

    {'temperature': 25, 'relative_humidity': 0.5, 'precipitation': 0.1, 'season_Spring': 1,
     'season_Summer': 0, 'season_Winter': 0, 'day_of_week_Monday': 0, 'day_of_week_Tuesday': 0,
     'day_of_week_Wednesday': 0, 'day_of_week_Thursday': 0, 'day_of_week_Saturday': 1,
     'day_of_week_Sunday': 0, 'rain_intensity_No Rain': 1, 'rain_intensity_Moderate Rain': 0}
])

# Ensure new_data columns match the feature order of X_train
new_data = new_data[feature_names]

# Make predictions
print("Making predictions for the following data:")
print(new_data)

predictions = model.predict(new_data)

# Output the predictions
for i, pred in enumerate(predictions, start=1):
    print(f"Scenario {i}: Estimated Courier Partners Online = {pred:.2f}")


# Predict orders or courier partners online for tomorrow
tomorrow_features = pd.DataFrame({
    'temperature': [20],  # Replace with tomorrow's forecast
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
tomorrow_features = tomorrow_features[feature_names]  # Ensure column alignment
predicted_orders_tomorrow = model.predict(tomorrow_features)
print(f"Predicted Orders for Tomorrow: {predicted_orders_tomorrow[0]:.2f}")

# Predict for the next week
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
weekly_features = weekly_features[feature_names]
weekly_predictions = model.predict(weekly_features)
print(f"Predicted Orders for the Next Week: {weekly_predictions}")

