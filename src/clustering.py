from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def perform_kmeans(features, n_clusters=10, random_state=42):
    """K-Means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20, max_iter=500)
    labels = kmeans.fit_predict(features)
    return labels, kmeans

def perform_agglomerative(features, n_clusters=10, linkage='ward'):
    """Agglomerative clustering"""
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(features)
    return labels

def perform_dbscan(features, eps=0.5, min_samples=5):
    """DBSCAN clustering"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)
    return labels

def pca_baseline(features, n_components=32):
    """PCA for dimensionality reduction"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features_scaled)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    return reduced_features, pca
