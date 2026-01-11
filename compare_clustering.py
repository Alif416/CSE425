import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from src.clustering import perform_kmeans, perform_agglomerative, perform_dbscan
from src.evaluation import evaluate_clustering

def compare_all_clustering_methods():
    """Compare K-Means, Agglomerative, and DBSCAN"""
    
    print("\n" + "="*80)
    print("COMPARING MULTIPLE CLUSTERING ALGORITHMS")
    print("="*80)
    
    # Load data
    print("\nLoading features...")
    audio_data = np.load('data/audio_features.npz', allow_pickle=True)
    genres = audio_data['genres']
    
    # Load VAE latent features
    latent_vae = np.load('results/latent_features_vae.npy')
    
    print(f"✓ Latent features shape: {latent_vae.shape}")
    print(f"✓ Genres: {np.unique(genres)}")
    
    # Map genres to numeric
    genre_to_idx = {g: i for i, g in enumerate(np.unique(genres))}
    true_labels = np.array([genre_to_idx[g] for g in genres])
    
    n_clusters = len(np.unique(genres))
    
    # Test different clustering algorithms
    results = {}
    
    # 1. K-Means
    print("\n[1/3] K-Means clustering...")
    labels_kmeans, _ = perform_kmeans(latent_vae, n_clusters=n_clusters)
    metrics_kmeans = evaluate_clustering(latent_vae, labels_kmeans, true_labels, 'K-Means')
    results['K-Means'] = metrics_kmeans
    print(f"✓ K-Means: Silhouette = {metrics_kmeans.get('silhouette', 0):.4f}")
    
    # 2. Agglomerative Clustering
    print("\n[2/3] Agglomerative clustering...")
    labels_agg = perform_agglomerative(latent_vae, n_clusters=n_clusters, linkage='ward')
    metrics_agg = evaluate_clustering(latent_vae, labels_agg, true_labels, 'Agglomerative')
    results['Agglomerative'] = metrics_agg
    print(f"✓ Agglomerative: Silhouette = {metrics_agg.get('silhouette', 0):.4f}")
    
    # 3. DBSCAN (we need to find good eps)
    print("\n[3/3] DBSCAN clustering...")
    from sklearn.neighbors import NearestNeighbors
    
    # Find optimal eps using k-distance
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors.fit(latent_vae)
    distances, indices = neighbors.kneighbors(latent_vae)
    distances = np.sort(distances[:, -1], axis=0)
    eps = np.percentile(distances, 90)
    
    print(f"  Using eps={eps:.3f}, min_samples=5")
    labels_dbscan = perform_dbscan(latent_vae, eps=eps, min_samples=5)
    metrics_dbscan = evaluate_clustering(latent_vae, labels_dbscan, true_labels, 'DBSCAN')
    results['DBSCAN'] = metrics_dbscan
    print(f"✓ DBSCAN: Found {len(np.unique(labels_dbscan[labels_dbscan!=-1]))} clusters, {np.sum(labels_dbscan==-1)} noise points")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).T
    
    print("\n" + "="*80)
    print("CLUSTERING ALGORITHM COMPARISON")
    print("="*80)
    print(results_df.to_string())
    print("="*80)
    
    # Save results
    results_df.to_csv('results/clustering_algorithms_comparison.csv')
    print("\n✓ Saved to: results/clustering_algorithms_comparison.csv")
    
    # Save labels for visualization
    np.savez('results/all_clustering_labels.npz',
             kmeans=labels_kmeans,
             agglomerative=labels_agg,
             dbscan=labels_dbscan)
    print("✓ Saved cluster labels to: results/all_clustering_labels.npz")
    
    return results_df

if __name__ == '__main__':
    results = compare_all_clustering_methods()
    
    print("\n✅ MEDIUM TASK - Clustering Comparison Complete!")
    print("\nKey metrics computed:")
    print("  - Silhouette Score ✓")
    print("  - Calinski-Harabasz Index ✓")
    print("  - Davies-Bouldin Index ✓")
    print("  - Adjusted Rand Index (ARI) ✓")
    print("  - Normalized Mutual Information (NMI) ✓")
    print("  - Cluster Purity ✓")
