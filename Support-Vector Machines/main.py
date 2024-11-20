import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0)

X_train, X_test, y_train, y_test = train_test_split(X, y)

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

grid = list(ParameterGrid(param_grid))

num_params = len(grid)
num_kernels = len(param_grid['kernel'])
rows = len(param_grid['C'])
cols = len(param_grid['gamma']) * num_kernels

# Create subplots
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
fig.suptitle('SVM Decision Boundaries for Various Kernels and Parameters', fontsize=16)

# Iterate over each combination and plot
for idx, params in enumerate(grid):
    C = params['C']
    gamma = params['gamma']
    kernel = params['kernel']

    # Initialize the SVM model with current parameters
    clf = svm.SVC(C=C, gamma=gamma, kernel=kernel, degree=3, coef0=0)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Rows correspond to different C values
    # Columns correspond to combinations of gamma and kernel
    row = param_grid['C'].index(C)
    kernel_idx = param_grid['kernel'].index(kernel)
    col = kernel_idx * len(param_grid['gamma']) + param_grid['gamma'].index(gamma)

    ax = axes[row, col]

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    ax.set_title(f"Kernel: {kernel}\nC: {C}, gamma: {gamma}\nAccuracy: {accuracy:.2f}")

# Adjust layout
plt.tight_layout()
plt.show()
