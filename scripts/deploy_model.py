import joblib
import pandas as pd

# Load the saved Linear Regression model
model_filename = 'best_model.pkl'
print(f"Loading the model from {model_filename}...")
model = joblib.load(model_filename)

# Load X_train to get the correct feature names and order
X_train = pd.read_csv('X_train.csv')  # Ensure this is the same dataset used for training
feature_names = X_train.columns.tolist()

# Define new input data (ensure feature names match X_train)
new_data = pd.DataFrame([
    {'temperature': 22, 'relative_humidity': 0.5, 'precipitation': 0.0, 'season_Spring': 1,
     'season_Summer': 0, 'season_Winter': 0, 'day_of_week_Monday': 1, 'day_of_week_Tuesday': 0,
     'day_of_week_Wednesday': 0, 'day_of_week_Thursday': 0, 'day_of_week_Saturday': 0,
     'day_of_week_Sunday': 0, 'rain_intensity_No Rain': 1, 'rain_intensity_Moderate Rain': 0},
    {'temperature': 28, 'relative_humidity': 0.7, 'precipitation': 0.2, 'season_Spring': 0,
     'season_Summer': 1, 'season_Winter': 0, 'day_of_week_Monday': 0, 'day_of_week_Tuesday': 1,
     'day_of_week_Wednesday': 0, 'day_of_week_Thursday': 0, 'day_of_week_Saturday': 0,
     'day_of_week_Sunday': 0, 'rain_intensity_No Rain': 1, 'rain_intensity_Moderate Rain': 0}
], columns=feature_names)

# Ensure the new_data columns match the order of X_train
new_data = new_data[feature_names]

# Make predictions
print("Making predictions for the following data:")
print(new_data)

predictions = model.predict(new_data)

# Output the predictions
for i, pred in enumerate(predictions, start=1):
    print(f"Prediction {i}: Estimated Courier Partners Online = {pred:.2f}")
