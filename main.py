import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath('.'))

from src.dataset import MusicDataset
from src.vae import VAE
from src.clustering import perform_kmeans, pca_baseline
from src.evaluation import evaluate_clustering, compare_methods
from src.visualization import (visualize_latent_space, visualize_clusters_by_genre,
                               plot_metrics_comparison, plot_cluster_distribution)
from src.train_vae import extract_latent_features

def main():
    print("\n" + "="*80)
    print("VAE FOR HYBRID LANGUAGE MUSIC CLUSTERING - COMPLETE PIPELINE")
    print("="*80)
    
    # Configuration
    config = {
        'latent_dim': 32,
        'hidden_dim': 512,
        'n_clusters': 4,  # 4 genres in your dataset
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Device: {config['device']}")
    print(f"Number of clusters: {config['n_clusters']}")
    
    # Step 1: Load preprocessed features
    print("\n[1/7] Loading features...")
    audio_data = np.load('data/audio_features.npz', allow_pickle=True)
    audio_features = audio_data['features']
    genres = audio_data['genres']
    
    lyrics_data = np.load('data/lyrics_features.npz')
    lyrics_features = lyrics_data['embeddings']
    
    print(f"✓ Audio features: {audio_features.shape}")
    print(f"✓ Lyrics features: {lyrics_features.shape}")
    print(f"✓ Genres: {np.unique(genres)} (n={len(genres)})")
    
    # Step 2: Load trained VAE model
    print("\n[2/7] Loading trained VAE model...")
    full_dataset = MusicDataset(audio_features, lyrics_features, normalize=True)
    
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    
    # Load VAE model
    vae_full = VAE(input_dim=full_dataset.features.shape[1], 
                   hidden_dim=config['hidden_dim'], 
                   latent_dim=config['latent_dim'])
    vae_full.load_state_dict(torch.load('results/models/vae_final.pth', map_location=config['device']))
    vae_full.to(config['device'])
    print("✓ Loaded VAE (Audio + Lyrics)")
    
    # Step 3: Extract latent features
    print("\n[3/7] Extracting latent features...")
    latent_vae_full = extract_latent_features(vae_full, full_loader, config['device'])
    
    # PCA baseline
    print("\nComputing PCA baseline...")
    pca_features, _ = pca_baseline(audio_features, n_components=config['latent_dim'])
    
    print(f"✓ VAE latent shape: {latent_vae_full.shape}")
    print(f"✓ PCA shape: {pca_features.shape}")
    
    # Step 4: Perform clustering
    print("\n[4/7] Performing clustering...")
    
    # Map genres to numeric labels
    genre_to_idx = {g: i for i, g in enumerate(np.unique(genres))}
    true_labels = np.array([genre_to_idx[g] for g in genres])
    
    labels_vae, _ = perform_kmeans(latent_vae_full, config['n_clusters'])
    labels_pca, _ = perform_kmeans(pca_features, config['n_clusters'])
    
    print(f"✓ Clustered {len(labels_vae)} samples into {config['n_clusters']} clusters")
    
    # Step 5: Evaluate clustering
    print("\n[5/7] Evaluating clustering...")
    
    features_dict = {
        'VAE (Audio + Lyrics)': latent_vae_full,
        'PCA Baseline': pca_features
    }
    
    results_df = compare_methods(features_dict, config['n_clusters'], true_labels)
    
    print("\n" + "="*80)
    print("CLUSTERING RESULTS")
    print("="*80)
    print(results_df.to_string())
    print("="*80)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/clustering_metrics.csv')
    print("\n✓ Saved results to: results/clustering_metrics.csv")
    
    # Step 6: Generate visualizations
    print("\n[6/7] Generating visualizations...")
    
    os.makedirs('results/latent_visualization', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    # t-SNE visualizations
    print("Creating t-SNE plots...")
    visualize_latent_space(latent_vae_full, labels_vae, method='tsne',
                          title='VAE (Audio + Lyrics)',
                          save_path='results/latent_visualization/vae_full_tsne.png')
    
    visualize_latent_space(pca_features, labels_pca, method='tsne',
                          title='PCA Baseline',
                          save_path='results/latent_visualization/pca_tsne.png')
    
    # UMAP visualizations
    print("Creating UMAP plots...")
    visualize_latent_space(latent_vae_full, labels_vae, method='umap',
                          title='VAE (Audio + Lyrics)',
                          save_path='results/latent_visualization/vae_full_umap.png')
    
    # Cluster vs Genre comparison
    print("Creating cluster-genre comparison...")
    visualize_clusters_by_genre(latent_vae_full, labels_vae, genres,
                                method='tsne',
                                save_path='results/figures/cluster_vs_genre.png')
    
    plot_cluster_distribution(labels_vae, genres,
                             save_path='results/figures/cluster_distribution.png')
    
    # Metrics comparison
    print("Creating metrics comparison...")
    plot_metrics_comparison(results_df, save_path='results/figures/metrics_comparison.png')
    
    # Step 7: Generate report summary
    print("\n[7/7] Generating report summary...")
    
    with open('results/report_summary.txt', 'w') as f:
        f.write("VAE FOR HYBRID LANGUAGE MUSIC CLUSTERING\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASET INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Total samples: {len(genres)}\n")
        f.write(f"Genres: {np.unique(genres)}\n")
        f.write(f"Genre distribution:\n")
        for genre in np.unique(genres):
            count = np.sum(genres == genre)
            f.write(f"  - {genre}: {count}\n")
        f.write("\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*80 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("CLUSTERING RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(results_df.to_string())
        f.write("\n\n")
        
        f.write("KEY FINDINGS\n")
        f.write("-"*80 + "\n")
        best_method = results_df['silhouette'].idxmax()
        f.write(f"Best performing method: {best_method}\n")
        f.write(f"Silhouette Score: {results_df.loc[best_method, 'silhouette']:.4f}\n")
        
        if 'ari' in results_df.columns:
            f.write(f"Adjusted Rand Index: {results_df.loc[best_method, 'ari']:.4f}\n")
        if 'nmi' in results_df.columns:
            f.write(f"Normalized Mutual Information: {results_df.loc[best_method, 'nmi']:.4f}\n")
        if 'purity' in results_df.columns:
            f.write(f"Cluster Purity: {results_df.loc[best_method, 'purity']:.4f}\n")
    
    print("✓ Saved report summary to: results/report_summary.txt")
    
    print("\n" + "="*80)
    print("✅ COMPLETE PIPELINE FINISHED!")
    print("="*80)
    print("\nGenerated files:")
    print("  📊 results/clustering_metrics.csv")
    print("  📈 results/figures/metrics_comparison.png")
    print("  📈 results/figures/cluster_vs_genre.png")
    print("  📈 results/figures/cluster_distribution.png")
    print("  🎨 results/latent_visualization/vae_full_tsne.png")
    print("  🎨 results/latent_visualization/vae_full_umap.png")
    print("  🎨 results/latent_visualization/pca_tsne.png")
    print("  📝 results/report_summary.txt")
    print("\n📌 Next steps:")
    print("  1. Review visualizations in results/ folder")
    print("  2. Check clustering_metrics.csv for quantitative results")
    print("  3. Write your NeurIPS paper using these results")
    print("  4. Push code to GitHub")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
