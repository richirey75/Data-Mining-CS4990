import random
import math
import numpy as np
import pandas as pd
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output

# DO NOT CHANGE THE FOLLOWING LINE
def kmeans(data, k, columns, centers=None, n=None, eps=None):
# DO NOT CHANGE THE PRECEDING LINE
    # This function has to return a list of k cluster centers (lists of floats of the same length as columns)
    # Initialize cluster centers
    if centers is None:
        # Randomly choose initial centers if not provided
        centers = [data[i] for i in np.random.choice(len(data), k, replace=False)]
    else:
        # Ensure the provided centers are only using specified columns
        centers = [[center[i] for i in columns] for center in centers]
    
    # Run the K-Means algorithm
    for iteration in range(n if n is not None else 100):  # Default to 100 iterations if n is None
        # Step 1: Assign each data point to the nearest center
        labels = []
        for instance in data:
            distances = [dist(centers[j], instance, columns) for j in range(k)]
            labels.append(np.argmin(distances))  # Assign to the closest center
        
        # Step 2: Update centers
        new_centers = []
        for i in range(k):
            # Gather all instances assigned to cluster i
            cluster_points = [data[j] for j in range(len(data)) if labels[j] == i]
            if cluster_points:
                # Compute the mean for each column used
                new_center = np.mean([[point[col] for col in columns] for point in cluster_points], axis=0)
            else:
                # Handle empty cluster by keeping the previous center
                new_center = np.array(centers[i])
            new_centers.append(new_center)
        
        # Step 3: Check for convergence
        center_shift = np.sum([np.linalg.norm(np.array(new_centers[i]) - np.array(centers[i])) for i in range(k)])
        
        # If eps condition is met (center shift is below threshold) or we reach max iterations, stop
        if (eps is not None and center_shift < eps) or (n is not None and iteration >= n - 1):
            break
        
        centers = new_centers  # Update centers for the next iteration
    
    # Return final centers as lists
    return [center.tolist() for center in centers]

def dist(center, instance, columns):
    # Calculate the Euclidean distance between a cluster center and a data instance using specific columns
    return np.linalg.norm(np.array([center[i] for i in columns]) - np.array([instance[i] for i in columns]))
    

# DO NOT CHANGE THE FOLLOWING LINE
def dbscan(data, columns, eps, min_samples):
# DO NOT CHANGE THE PRECEDING LINE
    # This function has to return a list of cluster centers (lists of floats of the same length as columns)
    """
    Perform DBSCAN clustering on the specified columns of a dataset.

    Parameters:
      data        : Iterable containing data instances.
      columns     : List of column indices to use for clustering.
      eps         : Radius to consider neighbors.
      min_samples : Minimum number of neighbors to be a core point.

    Returns:
      List of clusters (each cluster is a list of data instances) and a list of noise points.
    """

    # Create a data subset based on specified columns for clustering
    # Changed line: Use advanced indexing with columns list
    D = np.array([instance for instance in data])
    D = D[:, columns]
    labels = [0] * len(D)  # Initialize all labels to 0 (unvisited)
    C = 0  # Cluster ID

    for P in range(len(D)):
        if labels[P] != 0:  # Skip if the point has already been processed
            continue

        NeighborPts = region_query(D, P, eps)

        # Check if the point is a core point
        if len(NeighborPts) < min_samples:
            labels[P] = -1  # Mark as noise
        else:
            C += 1  # Increment cluster ID
            grow_cluster(D, labels, P, NeighborPts, C, eps, min_samples)

    # Organize points into clusters and noise
    clusters = [[] for _ in range(C)]
    noise = []
    for i, label in enumerate(labels):
        if label == -1:
            noise.append(data[i])
        else:
            clusters[label - 1].append(data[i])

    return clusters, noise

def grow_cluster(D, labels, P, NeighborPts, C, eps, min_samples):
    """
    Grow a cluster from the seed point `P`.

    Parameters:
      D           : Data subset with relevant columns
      labels      : List of point labels
      P           : Seed point index
      NeighborPts : List of neighbors of the seed point
      C           : Cluster ID
      eps         : Radius to consider neighbors
      min_samples : Minimum number of neighbors to be a core point
    """
    labels[P] = C  # Assign the cluster label to the seed point

    i = 0
    while i < len(NeighborPts):
        Pn = NeighborPts[i]

        if labels[Pn] == -1:
            labels[Pn] = C  # Convert noise to border point

        elif labels[Pn] == 0:
            labels[Pn] = C
            PnNeighborPts = region_query(D, Pn, eps)
            if len(PnNeighborPts) >= min_samples:
                NeighborPts += PnNeighborPts

        i += 1

def region_query(D, P, eps):
    """
    Find neighbors within `eps` distance from point `P`.

    Parameters:
      D   : Data subset with relevant columns
      P   : Index of the point in question
      eps : Radius to consider neighbors

    Returns:
      List of indices of points within `eps` distance of `P`.
    """
    neighbors = []
    for Pn in range(len(D)):
        if np.linalg.norm(D[P] - D[Pn]) < eps:
            neighbors.append(Pn)

    return neighbors

    
# DO NOT CHANGE THE FOLLOWING LINE
def kmedoids(data, k, distance, centers=None, n=None, eps=None):
# DO NOT CHANGE THE PRECEDING LINE
    # This function has to return a list of k cluster centroids (data instances!)
    pass