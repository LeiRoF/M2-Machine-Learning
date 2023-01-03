from config import *
import numpy as np

data = np.loadtxt(DATA, delimiter=delimiter)
eps = 1e-10
nb_dim = data.shape[1]

if ignore_last_column:
    nb_dim -= 1
    data = data[:,:nb_dim]

# Get the min and max coordinates for each dimension
mins = []
maxs = []
for dim in range(nb_dim):
    mins.append(data[:,dim].min())
    maxs.append(data[:,dim].max())
mins = np.array(mins)
maxs = np.array(maxs)

lock = True
while lock:
    try:
        k = int(input("Maximum number of clusters: "))
        lock = False
    except ValueError:
        print("Please enter an integer")

# Create a list of k centroids
centroids = []
for i in range(k):
    centroid = []
    for dim in range(nb_dim):
        centroid.append(np.random.uniform(mins[dim], maxs[dim]))
    centroids.append(centroid)
centroids = np.array(centroids)

def generation(centroids):

    # Assign each data point to the closest centroid
    clusters = []
    for point in data:
        distances = []
        for centroid in centroids:
            distances.append(np.linalg.norm(point - centroid))
        # Assigning each point to a centroid
        clusters.append(np.argmin(distances))
    clusters = np.array(clusters)

    # Update the centroids to be the mean of the assigned points
    for i in range(k):
        centroids[i] = np.mean(data[clusters == i,:], axis=0)
    
    return centroids, clusters

# Repeat until convergence
last_centroids = np.zeros_like(centroids)
while (np.abs(centroids - last_centroids) > eps).any():
    last_centroids = np.copy(centroids)
    centroids, clusters = generation(centroids)

# Save the final centroids
np.savetxt(KMEANS, centroids)

res = np.zeros((data.shape[0], data.shape[1] + 1))
res[:,:-1] = data
res[:,-1] = clusters
np.savetxt(RESULTS, res, fmt='%.1f')