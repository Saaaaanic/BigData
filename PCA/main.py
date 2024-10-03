import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PCA.CustomPCA import CustomPCA
from sklearn.decomposition import PCA


def plot_pca_comparison(X_custom, X_sk, y):
    """
    Plots a comparison between CustomPCA and sklearn PCA.
    """
    plt.figure(figsize=(14, 6))

    # Plot CustomPCA
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_custom[:, 0], X_custom[:, 1], c=y, cmap='viridis', alpha=0.5)
    plt.xlabel('PC1 (CustomPCA)')
    plt.ylabel('PC2 (CustomPCA)')
    plt.title('Custom PCA Implementation')
    plt.colorbar(scatter, label='Price in Euros')
    plt.grid(True)

    # Plot scikit-learn's PCA
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_sk[:, 0], X_sk[:, 1], c=y, cmap='viridis', alpha=0.5)
    plt.xlabel('PC1 (scikit-learn PCA)')
    plt.ylabel('PC2 (scikit-learn PCA)')
    plt.title('scikit-learn PCA Implementation')
    plt.colorbar(scatter, label='Price in Euros')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def analyze_pca_loadings(custom_pca, sk_pca, features, k):
    # CustomPCA loadings
    loadings_custom = custom_pca.components_
    loading_df_custom = pd.DataFrame(loadings_custom, index=features, columns=[f'PC{i}' for i in range(1, k+1)])

    # scikit-learn PCA loadings
    loadings_sk = sk_pca.components_.T
    loading_df_sk = pd.DataFrame(loadings_sk, index=features, columns=[f'PC{i}' for i in range(1, k+1)])

    for i in range(1, min(k+1, 3)):
        # Sort loadings by absolute value in descending order
        sorted_custom = loading_df_custom[f'PC{i}'].abs().sort_values(ascending=False)
        sorted_sk = loading_df_sk[f'PC{i}'].abs().sort_values(ascending=False)

        plt.figure(figsize=(14, 7))
        plt.bar(sorted_custom.index, sorted_custom.values, alpha=0.6, label='CustomPCA')
        plt.bar(sorted_sk.index, sorted_sk.values, alpha=0.6, label='scikit-learn PCA', linestyle='--')
        plt.xlabel('Features')
        plt.ylabel('Absolute Loading Value')
        plt.title(f'Sorted Feature Loadings for Principal Component {i}')
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.show()


df = pd.read_csv('laptop_prices.csv')

target = 'Price_euros'

numeric_features = ['Inches', 'Ram', 'Weight', 'ScreenW', 'ScreenH', 'CPU_freq',
                    'PrimaryStorage', 'SecondaryStorage']
categorical_features = ['Company', 'Product', 'TypeName', 'OS', 'Screen',
                        'Touchscreen', 'IPSpanel', 'RetinaDisplay',
                        'CPU_company', 'CPU_model', 'PrimaryStorageType',
                        'SecondaryStorageType', 'GPU_company', 'GPU_model']

# Encode categorical variables using Label Encoding
le = LabelEncoder()
for col in categorical_features:
    df[col] = le.fit_transform(df[col].astype(str))

features = numeric_features + categorical_features
X = df[features].values
y = df[target].values

# Data Normalization using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

custom_pca = CustomPCA()
X_pca_custom, k_custom = custom_pca.fit_transform(X_scaled), custom_pca.n_components

sk_pca = PCA(n_components=k_custom)
X_pca_sk = sk_pca.fit_transform(X_scaled)

plot_pca_comparison(X_pca_custom[:, :2], X_pca_sk[:, :2], y)

analyze_pca_loadings(custom_pca, sk_pca, features, k_custom)
