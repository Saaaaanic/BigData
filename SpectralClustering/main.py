import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import make_moons, make_circles, make_blobs

# Transformed blob dataset
# (taken from https://www.youtube.com/watch?v=YHz0PHcuJnk&t=175s&ab_channel=Dr.DataScience)
X, y = make_blobs(n_samples=1500, random_state=170)
transformation = ([0.6083459, -0.63667341], [-0.40887718, 0.85253229])
X = np.dot(X, transformation)

datasets = {
    "Moons": (make_moons(n_samples=300, noise=0.05), 2),
    "Circles": (make_circles(n_samples=300, noise=0.05, factor=0.5), 2),
    "Blobs": (make_blobs(n_samples=300, centers=2, cluster_std=1.0), 2),
    "Transformed blobs": ((X, y), 3)
}

fig, axs = plt.subplots(len(datasets), 2, figsize=(12, 12))
fig.suptitle("Comparison of KMeans and Optimized Spectral Clustering", fontsize=16)

for i, (name, ((X, _), n_clusters)) in enumerate(datasets.items()):
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans_labels = kmeans.fit_predict(X)

    # Spectral Clustering
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
    )
    spectral_labels = spectral.fit_predict(X)

    # Plot KMeans
    axs[i, 0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
    axs[i, 0].set_title(f"KMeans on {name}")

    # Plot Spectral Clustering
    axs[i, 1].scatter(X[:, 0], X[:, 1], c=spectral_labels, cmap='viridis')
    axs[i, 1].set_title(f"Optimized Spectral Clustering on {name}")

plt.tight_layout()
plt.show()
