import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the cleaned dataset
data = pd.read_csv('cleaned_daily_cp_activity.csv')

# Split features and target
X = data.drop(['courier_partners_online', 'date'], axis=1)
y = data['courier_partners_online']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the feature names
joblib.dump(X_train.columns.tolist(), 'trained_feature_names.pkl')

# Train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(linear_model, 'best_model.pkl')
