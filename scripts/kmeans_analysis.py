from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('cleaned_daily_cp_activity.csv')

# Select relevant features for clustering
features = data[['temperature', 'precipitation', 'relative_humidity']]

# Train the KMeans model
n_clusters = 5  # Specify the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(features)

# Analyze cluster centroids
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features.columns)
print("Cluster Centroids:")
print(centroids)

for i, centroid in centroids.iterrows():
    print(f"\nCluster {i + 1} Characteristics:")
    for feature, value in centroid.items():
        print(f"  {feature}: {value:.2f}")

# Assign cluster labels to the data
data['Cluster'] = kmeans.labels_

# Save the clustered dataset
clustered_data_filename = 'clustered_daily_cp_activity.csv'
data.to_csv(clustered_data_filename, index=False)
print(f"Clustered dataset saved to: {clustered_data_filename}")

# Visualize the clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    features['temperature'],
    features['precipitation'],
    c=kmeans.labels_,
    cmap='viridis',
    alpha=0.7,
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    color='red',
    marker='x',
    s=200,
    label='Centroids',
)
plt.title("Clusters and Centroids")
plt.xlabel("Temperature")
plt.ylabel("Precipitation")
plt.legend()
plt.colorbar(scatter, label='Cluster Labels')
plt.show()
