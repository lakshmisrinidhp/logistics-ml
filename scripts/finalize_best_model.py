import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load the train and test datasets
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()  # Convert to Series
y_test = pd.read_csv('y_test.csv').squeeze()

# Train the Linear Regression model
print("Training the Linear Regression Model...")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Evaluate the model
y_pred = linear_model.predict(X_test)
print("\nLinear Regression Performance on Test Set:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Save the trained model
model_filename = 'best_model.pkl'
joblib.dump(linear_model, model_filename)
print(f"\nLinear Regression model saved as '{model_filename}'")

# Verify the saved model
print("\nVerifying the saved model...")
loaded_model = joblib.load(model_filename)

# Predict with the loaded model
sample_data = X_test.iloc[:5]  # Use first 5 rows from X_test as a sample
sample_predictions = loaded_model.predict(sample_data)
print("Sample Predictions:", sample_predictions)
