import numpy as np


class CustomPCA:
    def __init__(self, n_components=None, variance_threshold=0.90):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # Calculate the covariance matrix
        cov_matrix = np.cov(X, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort the eigenvalues and eigenvectors in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_idx]
        sorted_eigenvectors = eigenvectors[:, sorted_idx]

        # Calculate explained variance ratio
        self.explained_variance_ratio_ = sorted_eigenvalues / np.sum(sorted_eigenvalues)
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)

        if self.n_components is None:
            self.n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1

        # Select the top k eigenvectors
        self.components_ = sorted_eigenvectors[:, :self.n_components]

    def transform(self, X):
        if self.components_ is None:
            raise RuntimeError("You must fit the PCA before transforming data!")

        return np.dot(X, self.components_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
