from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
data = pd.read_csv('cleaned_daily_cp_activity.csv')

# Select clustering features
clustering_features = data[['temperature', 'relative_humidity', 'precipitation']]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(clustering_features)

# Visualize clusters
plt.scatter(data['temperature'], data['precipitation'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Temperature')
plt.ylabel('Precipitation')
plt.title('Delivery Demand Clusters')
plt.show()

# Predict delivery region for a new scenario
new_scenario = pd.DataFrame({'temperature': [25], 'relative_humidity': [0.5], 'precipitation': [0.1]})
predicted_cluster = kmeans.predict(new_scenario)
print(f"Predicted Delivery Cluster: {predicted_cluster[0]}")
