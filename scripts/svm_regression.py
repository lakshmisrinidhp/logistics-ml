# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the train and test datasets
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()  # Convert to Series
y_test = pd.read_csv('y_test.csv').squeeze()

# Train an SVM model
print("Training SVM Regression Model...")
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)  # Customize hyperparameters
svr_model.fit(X_train, y_train)

# Predict on the test set
y_pred_svr = svr_model.predict(X_test)

# Evaluate the model
print("\nSVM Regression Performance:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_svr)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_svr):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_svr):.2f}")

# Save the model
import joblib
joblib.dump(svr_model, 'svm_model.pkl')
print("\nSVM model saved to: svm_model.pkl")

# Plot Actual vs Predicted
plt.scatter(y_test, y_pred_svr, alpha=0.5, color='blue', label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("SVM Regression: Actual vs Predicted")
plt.legend()
plt.show()

# Save predictions for analysis
output = pd.DataFrame({'Actual': y_test, 'SVM_Predicted': y_pred_svr})
output.to_csv('svm_predictions.csv', index=False)
print("\nPredictions saved to: svm_predictions.csv")
