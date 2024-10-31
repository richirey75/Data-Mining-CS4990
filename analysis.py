import clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

## NOTE: Comment out EACH section depending on what you want to test (Ctrl + /) ##

# K-Means Clustering
# Load Spotify dataset - change path to get the tracks (if needed)
data = pd.read_csv('C:/Users/tiffm/OneDrive/Documents/Clustering/Data-Mining-CS4990/Red Hot Chili Peppers_tracks_info.csv')

# Select relevant features - danceability, energy, and valence for clustering
columns_to_cluster = ["danceability", "energy", "valence"]
data_cluster = data[columns_to_cluster].to_numpy()  # Convert to NumPy array for easier processing

# Function to compute SSE for each k
def calculate_sse(data, max_k, columns):
    sse = []
    for k in range(1, max_k + 1):
        centers = clustering.kmeans(data, k, columns)
        sse_value = sum(min([clustering.dist(center, instance, columns) for center in centers]) ** 2 for instance in data)
        sse.append(sse_value)
    return sse

# Set the range of k values to test
max_k = 10
sse = calculate_sse(data_cluster, max_k, range(len(columns_to_cluster)))

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_k + 1), sse, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.title("Elbow Method for Optimal k")
plt.show()

# Determine the optimal k based on the elbow
optimal_k = 3  # Adjust based on observed elbow point

# Run k-means with the optimal k and get cluster centers
final_centers = clustering.kmeans(data_cluster, optimal_k, range(len(columns_to_cluster)))

# Assign each point to a cluster and visualize
labels = []
for instance in data_cluster:
    distances = [clustering.dist(center, instance, range(len(columns_to_cluster))) for center in final_centers]
    labels.append(np.argmin(distances))

# Calculate silhouette score for k=3
silhouette_avg = silhouette_score(data_cluster, labels)
print(f"Silhouette Score for k={optimal_k}: {silhouette_avg:.4f}")

# Plot the clusters
plt.figure(figsize=(8, 6))
for cluster_num in range(optimal_k):
    cluster_points = data_cluster[np.array(labels) == cluster_num]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_num + 1}")
    
# Plot final cluster centers
for center in final_centers:
    plt.scatter(center[0], center[1], color="black", marker="X", s=100, edgecolor="w")

plt.xlabel("Danceability")
plt.ylabel("Energy")
plt.title("K-Means Clustering of Red Hot Chili Peppers Tracks")
plt.legend()
plt.show()

##################################################################################################################################
# DBSCAN 
# load spotify dataset - change path to get the tracks (if needed)
data = pd.read_csv('C:/Users/tiffm/OneDrive/Documents/Clustering/Data-Mining-CS4990/combined_tracks_info_ArijitSingh_KK.csv')

# select relevant features - danceability, energy, and valence for clustering
columns_to_cluster = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
X = data[columns_to_cluster].values  # convert to numpy array for clustering

# Normalize the selected columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# set parameters
eps = 0.6 # Radius to consider neighbors
min_samples = 7 # Minimum number of neighbors to form a core point

# run DBSCAN algorithm
clusters, noise = clustering.dbscan([list(row) for row in X_scaled], list(range(len(columns_to_cluster))), eps, min_samples)

# Concatenate the clusters to a single array
cluster_labels = [-1] * len(X_scaled)
for i, cluster in enumerate(clusters):
  for point in cluster:
    idx = np.where((X_scaled == point).all(axis=1))[0][0]
    cluster_labels[idx] = i

# Calculate the silhouette score
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print("The average silhouette score is :", silhouette_avg)

# Visualize the clusters (example using PCA for dimensionality reduction)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
for i, cluster in enumerate(clusters):
    cluster_points = np.array([X_pca[np.where((X_scaled == point).all(axis=1))[0][0]] for point in cluster])
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')

noise_points = np.array([X_pca[np.where((X_scaled == point).all(axis=1))[0][0]] for point in noise])
plt.scatter(noise_points[:, 0], noise_points[:, 1], c='black', label='Noise')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('DBSCAN Clustering Visualization (PCA)')
plt.legend()
plt.show()

# ##########################################################################################################################
# DBSCAN - Iris Dataset (Additional Dataset for Testing)
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
df.head()

columns_to_cluster = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = df[columns_to_cluster].values  # convert to numpy array for clustering

# Normalize the selected columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# set parameters
eps = 0.6 # Radius to consider neighbors
min_samples = 7 # Minimum number of neighbors to form a core point

# run DBSCAN algorithm
clusters, noise = clustering.dbscan([list(row) for row in X_scaled], list(range(len(columns_to_cluster))), eps, min_samples)

# Concatenate the clusters to a single array
cluster_labels = [-1] * len(X_scaled)
for i, cluster in enumerate(clusters):
  for point in cluster:
    idx = np.where((X_scaled == point).all(axis=1))[0][0]
    cluster_labels[idx] = i

# Calculate the silhouette score
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print("The average silhouette score is :", silhouette_avg)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
for i, cluster in enumerate(clusters):
    cluster_points = np.array([X_pca[np.where((X_scaled == point).all(axis=1))[0][0]] for point in cluster])
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')

noise_points = np.array([X_pca[np.where((X_scaled == point).all(axis=1))[0][0]] for point in noise])
plt.scatter(noise_points[:, 0], noise_points[:, 1], c='black', label='Noise')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('DBSCAN Clustering Visualization (PCA)')
plt.legend()
plt.show()