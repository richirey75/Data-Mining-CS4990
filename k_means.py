import pandas as pd
import numpy as np
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output


def kmeans(data, k, columns, n=100):

  selected_data = data[columns]

  # Normalize the data
  def normalize(data):
    normalized_data = (data - data.min()) / (data.max() - data.min()) * 9 + 1
    return normalized_data
  normalized_data = normalize(selected_data)

  # Generate random centroids
  def random_centroids(data, k):
    centroids = []
    for col in range(k):
      centroid = data.apply(lambda x: float(x.sample()))
      centroids.append(centroid)
    return pd.concat(centroids, axis=1)
  centroids = random_centroids(normalized_data, k)  # Using normalized data

  # Get cluster labels
  def get_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data - x)** 2).sum(axis=1)))
    return distances.idxmin(axis=1)
  labels = get_labels(normalized_data, centroids)   # Using normalized data

  # Calculate new centroids
  def new_centroids(data, labels, k):
    return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
  new_center = new_centroids(normalized_data, labels, k)  # Using normalized data

  def plot_clusters(data, labels, centroids, iteration):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait=True)
    plt.title(f'Iteration {iteration}')
    plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=labels)
    plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1])
    plt.show()

  centroids = random_centroids(normalized_data, k)
  old_centroids = pd.DataFrame()
  iteration = 1

  while iteration < n and not centroids.equals(old_centroids):
      old_centroids = centroids.copy()  # Store current centroids as old_centroids

      labels = get_labels(normalized_data, centroids)
      centroids = new_centroids(normalized_data, labels, k)

      plot_clusters(normalized_data, labels, centroids, iteration)
      time.sleep(1)
      iteration += 1

  return normalized_data.describe(), centroids, labels, new_center



# HOW TO RUN THE CODE

# Make sure that your data is in this data framework 
features = ['danceability', 'energy', 'valence']
data = tracks[features]

# Assign the features you want into a variable and pass that variable 
# to the function as a parameter
columns_to_use = ['danceability', 'energy', 'valence']
kmeans(data, 3, columns_to_use)





