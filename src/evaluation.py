from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                            davies_bouldin_score, adjusted_rand_score,
                            normalized_mutual_info_score)
import numpy as np
import pandas as pd

def evaluate_clustering(features, labels, true_labels=None, method_name='Method'):
    """Compute all clustering metrics"""
    results = {}
    
    # Handle noise points (DBSCAN label -1)
    mask = labels != -1
    if mask.sum() == 0:
        print(f"Warning: All points labeled as noise for {method_name}")
        return results
    
    features_filtered = features[mask]
    labels_filtered = labels[mask]
    
    unique_labels = len(np.unique(labels_filtered))
    
    if unique_labels > 1 and unique_labels < len(labels_filtered):
        try:
            results['silhouette'] = silhouette_score(features_filtered, labels_filtered)
            results['calinski_harabasz'] = calinski_harabasz_score(features_filtered, labels_filtered)
            results['davies_bouldin'] = davies_bouldin_score(features_filtered, labels_filtered)
        except Exception as e:
            print(f"Error computing metrics for {method_name}: {e}")
    
    if true_labels is not None:
        true_labels_filtered = true_labels[mask]
        try:
            results['ari'] = adjusted_rand_score(true_labels_filtered, labels_filtered)
            results['nmi'] = normalized_mutual_info_score(true_labels_filtered, labels_filtered)
            results['purity'] = cluster_purity(true_labels_filtered, labels_filtered)
        except Exception as e:
            print(f"Error computing supervised metrics: {e}")
    
    results['n_clusters'] = unique_labels
    results['noise_points'] = (labels == -1).sum()
    
    return results

def cluster_purity(true_labels, pred_labels):
    """Calculate cluster purity"""
    contingency_matrix = np.zeros((len(np.unique(true_labels)), len(np.unique(pred_labels))))
    
    for i, true_label in enumerate(np.unique(true_labels)):
        for j, pred_label in enumerate(np.unique(pred_labels)):
            contingency_matrix[i, j] = np.sum((true_labels == true_label) & (pred_labels == pred_label))
    
    purity = np.sum(np.max(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    return purity

def compare_methods(features_dict, n_clusters=10, true_labels=None):
    """Compare different feature extraction methods"""
    from src.clustering import perform_kmeans
    
    results = {}
    
    for method_name, features in features_dict.items():
        print(f"\nEvaluating {method_name}...")
        labels, _ = perform_kmeans(features, n_clusters)
        metrics = evaluate_clustering(features, labels, true_labels, method_name)
        results[method_name] = metrics
    
    results_df = pd.DataFrame(results).T
    return results_df
