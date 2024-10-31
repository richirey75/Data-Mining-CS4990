import clusteringtiff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# load spotify dataset 
data = pd.read_csv('Backstreet Boys_tracks_info.csv')

# select relevant features - danceability, energy, and valence for clustering
columns_to_cluster = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
X = data[columns_to_cluster].values  # convert to numpy array for clustering

# Normalize the selected columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# set parameters 
eps = 0.8 # Radius to consider neighbors
min_samples = 4 # Minimum number of neighbors to form a core point

# run DBSCAN algorithm
clusters, noise = clusteringtiff.dbscan([list(row) for row in X_scaled], list(range(len(columns_to_cluster))), eps, min_samples)