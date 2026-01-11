import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import numpy as np
import os

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300

def visualize_latent_space(latent_features, labels, method='tsne', title='Latent Space', save_path=None):
    """Visualize latent space using t-SNE or UMAP"""
    print(f"Visualizing using {method.upper()}...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_features)-1))
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    
    embedded = reducer.fit_transform(latent_features)
    
    plt.figure(figsize=(12, 9))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                   c=[colors[idx]], label=f'Cluster {label}', 
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    plt.title(f'{title} - {method.upper()}', fontsize=16, fontweight='bold')
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    plt.close()

def visualize_clusters_by_genre(latent_features, cluster_labels, true_genres, method='tsne', save_path=None):
    """Compare predicted clusters vs true genres"""
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_features)-1))
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    
    embedded = reducer.fit_transform(latent_features)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot by predicted clusters
    unique_clusters = np.unique(cluster_labels)
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        axes[0].scatter(embedded[mask, 0], embedded[mask, 1], 
                       label=f'Cluster {cluster}', alpha=0.6, s=50)
    axes[0].set_title('Predicted Clusters', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot by true genres
    unique_genres = np.unique(true_genres)
    for genre in unique_genres:
        mask = true_genres == genre
        axes[1].scatter(embedded[mask, 0], embedded[mask, 1], 
                       label=genre, alpha=0.6, s=50)
    axes[1].set_title('True Genres', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    plt.close()

def plot_metrics_comparison(results_df, save_path='results/figures/metrics_comparison.png'):
    """Compare metrics across methods"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    metrics_to_plot = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    available_metrics = [m for m in metrics_to_plot if m in results_df.columns]
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available_metrics):
        values = results_df[metric].dropna()
        axes[idx].bar(values.index, values.values, color='steelblue', edgecolor='black', linewidth=1.5)
        axes[idx].set_title(metric.replace('_', ' ').title(), fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('Score', fontsize=12)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values.values):
            axes[idx].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def plot_cluster_distribution(cluster_labels, true_genres, save_path='results/figures/cluster_distribution.png'):
    """Heatmap of genre distribution across clusters"""
    from sklearn.metrics import confusion_matrix
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert genres to numeric if they're strings
    if isinstance(true_genres[0], str):
        unique_genres = np.unique(true_genres)
        genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
        true_genres_numeric = np.array([genre_to_idx[g] for g in true_genres])
    else:
        true_genres_numeric = true_genres
        unique_genres = np.unique(true_genres)
    
    cm = confusion_matrix(true_genres_numeric, cluster_labels)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=[f'Cluster {i}' for i in np.unique(cluster_labels)],
                yticklabels=unique_genres,
                cbar_kws={'label': 'Count'})
    plt.title('Genre Distribution Across Clusters', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Cluster', fontsize=12)
    plt.ylabel('True Genre', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
