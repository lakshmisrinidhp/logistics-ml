# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the engineered dataset
file_path = 'engineered_daily_cp_activity.csv'
data = pd.read_csv(file_path)

# Convert categorical columns to one-hot encoding
categorical_columns = ['season', 'day_of_week', 'rain_intensity']
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop 'first' to avoid dummy variable trap
encoded_features = encoder.fit_transform(data[categorical_columns])

# Create a DataFrame for the encoded features
encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder.get_feature_names_out(categorical_columns)
)

# Combine the encoded features with numerical features
numerical_features = ['temperature', 'relative_humidity', 'precipitation']
X = pd.concat([data[numerical_features], encoded_df], axis=1)

# Define the target variable
y = data['courier_partners_online']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the split data for future use
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Output confirmation
print("Feature selection and dataset splitting completed.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
