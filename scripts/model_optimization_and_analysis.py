# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load predictions and dataset
predictions = pd.read_csv('model_predictions.csv')

# Residual Analysis for Linear Regression
print("\nPerforming Residual Analysis for Linear Regression...")
residuals = predictions['Actual'] - predictions['Linear_Predicted']
plt.scatter(predictions['Actual'], residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals of Linear Regression Model")
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.show()

# Feature Importance for Random Forest
print("\nAnalyzing Feature Importance for Random Forest...")
# Load the training dataset (X_train from previous steps)
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').squeeze()

# Train a Random Forest with default hyperparameters
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances
importances = rf_model.feature_importances_
feature_names = X_train.columns
sorted_importances = sorted(zip(importances, feature_names), reverse=True)
print("\nFeature Importances (Random Forest):")
for importance, feature in sorted_importances:
    print(f"{feature}: {importance:.4f}")

# Fine-Tune Random Forest
print("\nFine-Tuning Random Forest...")
rf_optimized = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_optimized.fit(X_train, y_train)

# Evaluate the optimized Random Forest
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').squeeze()
y_pred_rf_optimized = rf_optimized.predict(X_test)
print("\nOptimized Random Forest Performance:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf_optimized)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf_optimized):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_rf_optimized):.2f}")

# Train Linear Regression Model
print("\nRetraining Linear Regression Model...")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Save the Better Model
if r2_score(y_test, y_pred_rf_optimized) > r2_score(y_test, predictions['Linear_Predicted']):
    print("\nOptimized Random Forest performs better. Saving the model...")
    joblib.dump(rf_optimized, 'best_model.pkl')
else:
    print("\nLinear Regression performs better. Saving the model...")
    joblib.dump(linear_model, 'best_model.pkl')

# Save Updated Predictions for Analysis
updated_predictions = pd.DataFrame({
    'Actual': y_test,
    'Optimized_RF_Predicted': y_pred_rf_optimized
})
updated_predictions.to_csv('optimized_model_predictions.csv', index=False)
print("\nUpdated predictions saved to: optimized_model_predictions.csv")
