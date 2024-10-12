import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans


def generate_datasets():
    """Generates convex and non-convex datasets."""
    X_blobs, y_blobs = make_blobs(n_samples=500, centers=5, cluster_std=0.60, random_state=0)

    X_moons, y_moons = make_moons(n_samples=500, noise=0.05, random_state=0)

    return (X_blobs, y_blobs), (X_moons, y_moons)


def plot_clusters(X, labels, centers=None, title='Cluster Visualization'):
    """Plots the clustered data with optional cluster centers."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', alpha=0.6)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def analyze_parameters(X):
    """Analyzes the influence of different K-Means parameters on inertia and execution time."""
    inertias = []
    times = []
    k_values = range(1, 10)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        start_time = time.time()
        kmeans.fit(X)
        end_time = time.time()
        inertias.append(kmeans.inertia_)
        times.append(end_time - start_time)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Inertia vs Number of Clusters')

    plt.subplot(1, 2, 2)
    plt.plot(k_values, times, 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs Number of Clusters')

    plt.tight_layout()
    plt.show()


def analyze_init_methods(X):
    """Analyzes the influence of different initialization methods on inertia and execution time."""
    inertias_init = {}
    times_init = {}
    init_methods = ['k-means++', 'random']

    for method in init_methods:
        kmeans = KMeans(n_clusters=4, init=method, n_init=10, random_state=0)
        start_time = time.time()
        kmeans.fit(X)
        end_time = time.time()
        inertias_init[method] = kmeans.inertia_
        times_init[method] = end_time - start_time

    for method in init_methods:
        print(f"Initialization Method: {method}")
        print(f" Inertia: {inertias_init[method]:.2f}")
        print(f" Execution Time: {times_init[method]:.4f} seconds\n")


def analyze_n_init_max_iter(X):
    """Analyzes the influence of n_init and max_iter parameters on inertia and execution time."""
    n_init_values = [10, 20, 50]
    max_iter_values = [100, 300, 500]

    inertias_n_init = []
    times_n_init = []

    for n_init in n_init_values:
        kmeans = KMeans(n_clusters=4, init='k-means++', n_init=n_init, max_iter=300, random_state=0)
        start_time = time.time()
        kmeans.fit(X)
        end_time = time.time()
        inertias_n_init.append(kmeans.inertia_)
        times_n_init.append(end_time - start_time)

    inertias_max_iter = []
    times_max_iter = []

    for max_iter in max_iter_values:
        kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=max_iter, random_state=0)
        start_time = time.time()
        kmeans.fit(X)
        end_time = time.time()
        inertias_max_iter.append(kmeans.inertia_)
        times_max_iter.append(end_time - start_time)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].plot(n_init_values, inertias_n_init, 'bo-')
    axes[0, 0].set_xlabel('n_init')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].set_title('Inertia vs n_init')
    axes[0, 0].grid(True)

    axes[0, 1].plot(n_init_values, times_n_init, 'go-')
    axes[0, 1].set_xlabel('n_init')
    axes[0, 1].set_ylabel('Execution Time (seconds)')
    axes[0, 1].set_title('Execution Time vs n_init')
    axes[0, 1].grid(True)

    axes[1, 0].plot(max_iter_values, inertias_max_iter, 'ro-')
    axes[1, 0].set_xlabel('max_iter')
    axes[1, 0].set_ylabel('Inertia')
    axes[1, 0].set_title('Inertia vs max_iter')
    axes[1, 0].grid(True)

    axes[1, 1].plot(max_iter_values, times_max_iter, 'mo-')
    axes[1, 1].set_xlabel('max_iter')
    axes[1, 1].set_ylabel('Execution Time (seconds)')
    axes[1, 1].set_title('Execution Time vs max_iter')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    # Generate Datasets
    (X_blobs, y_blobs), (X_moons, y_moons) = generate_datasets()

    # Clustering on Convex and Non-Convex Datasets
    kmeans_blobs = KMeans(n_clusters=5, random_state=0)
    y_kmeans_blobs = kmeans_blobs.fit_predict(X_blobs)
    plot_clusters(X_blobs, y_kmeans_blobs, kmeans_blobs.cluster_centers_, 'K-Means on Convex Clusters')

    kmeans_moons = KMeans(n_clusters=2, random_state=0)
    y_kmeans_moons = kmeans_moons.fit_predict(X_moons)
    plot_clusters(X_moons, y_kmeans_moons, kmeans_moons.cluster_centers_, 'K-Means on Non-Convex Clusters')

    # Parameter Analysis on Convex Dataset
    analyze_parameters(X_blobs)
    analyze_init_methods(X_blobs)
    analyze_n_init_max_iter(X_blobs)


if __name__ == "__main__":
    main()
