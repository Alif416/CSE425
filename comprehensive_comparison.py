import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
# ..

sys.path.insert(0, os.path.abspath('.'))

from src.dataset import MusicDataset
from src.clustering import perform_kmeans, pca_baseline
from src.evaluation import evaluate_clustering

# Simple Autoencoder (for comparison)
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, latent_dim=32):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x):
        return self.encoder(x)

def train_autoencoder():
    """Train simple Autoencoder for comparison"""
    print("\n" + "="*80)
    print("TRAINING AUTOENCODER BASELINE")
    print("="*80)
    
    # Load data
    print("\nLoading features...")
    audio_data = np.load('data/audio_features.npz', allow_pickle=True)
    audio_features = audio_data['features']
    
    lyrics_data = np.load('data/lyrics_features.npz')
    lyrics_features = lyrics_data['embeddings']
    
    # Create dataset
    dataset = MusicDataset(audio_features, lyrics_features, normalize=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    input_dim = dataset.features.shape[1]
    model = Autoencoder(input_dim=input_dim, hidden_dim=512, latent_dim=32)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    print(f"✓ Using device: {device}")
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    epochs = 30  # Fewer epochs for AE
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            batch = batch.to(device)
            
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}')
    
    # Save model
    torch.save(model.state_dict(), 'results/models/autoencoder_final.pth')
    print("✓ Saved model: results/models/autoencoder_final.pth")
    
    # Extract latent features
    print("\nExtracting latent features...")
    model.eval()
    latent_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting'):
            batch = batch.to(device)
            latent = model.encode(batch)
            latent_features.append(latent.cpu().numpy())
    
    latent_features = np.concatenate(latent_features, axis=0)
    np.save('results/latent_features_autoencoder.npy', latent_features)
    print(f"✓ Saved latent features: {latent_features.shape}")
    
    return model, latent_features

def comprehensive_comparison():
    """Compare all methods: VAE, Conv VAE, Beta-VAE, CVAE, Autoencoder, PCA, Raw"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    # Load data
    print("\nLoading data and all latent features...")
    audio_data = np.load('data/audio_features.npz', allow_pickle=True)
    audio_features = audio_data['features']
    genres = audio_data['genres']
    
    # Map genres to numeric
    genre_to_idx = {g: i for i, g in enumerate(np.unique(genres))}
    true_labels = np.array([genre_to_idx[g] for g in genres])
    n_clusters = len(np.unique(genres))
    
    # Load all latent representations
    features_dict = {}
    
    # 1. Raw audio features
    print("✓ Raw audio features")
    features_dict['Raw Audio'] = audio_features
    
    # 2. PCA baseline
    print("✓ Computing PCA...")
    pca_features, _ = pca_baseline(audio_features, n_components=32)
    features_dict['PCA'] = pca_features
    
    # 3. Autoencoder
    if os.path.exists('results/latent_features_autoencoder.npy'):
        print("✓ Autoencoder")
        features_dict['Autoencoder'] = np.load('results/latent_features_autoencoder.npy')
    
    # 4. VAE (original)
    if os.path.exists('results/latent_features_vae.npy'):
        print("✓ VAE (Audio + Lyrics)")
        features_dict['VAE'] = np.load('results/latent_features_vae.npy')
    
    # 5. Convolutional VAE
    if os.path.exists('results/latent_features_conv_vae.npy'):
        print("✓ Convolutional VAE")
        features_dict['Conv-VAE'] = np.load('results/latent_features_conv_vae.npy')
    
    # 6. Beta-VAE
    if os.path.exists('results/latent_features_beta_vae_beta4.0.npy'):
        print("✓ Beta-VAE (β=4)")
        features_dict['Beta-VAE'] = np.load('results/latent_features_beta_vae_beta4.0.npy')
    
    # 7. CVAE
    if os.path.exists('results/latent_features_cvae.npy'):
        print("✓ CVAE")
        features_dict['CVAE'] = np.load('results/latent_features_cvae.npy')
    
    print(f"\nTotal methods to compare: {len(features_dict)}")
    
    # Evaluate all methods
    print("\n" + "="*80)
    print("EVALUATING ALL METHODS")
    print("="*80)
    
    results = {}
    
    for method_name, features in features_dict.items():
        print(f"\nEvaluating {method_name}...")
        
        # Perform K-Means clustering
        labels, _ = perform_kmeans(features, n_clusters=n_clusters)
        
        # Compute metrics
        metrics = evaluate_clustering(features, labels, true_labels, method_name)
        results[method_name] = metrics
        
        print(f"  Silhouette: {metrics.get('silhouette', 0):.4f}")
        print(f"  ARI: {metrics.get('ari', 0):.4f}")
        print(f"  NMI: {metrics.get('nmi', 0):.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).T
    
    print("\n" + "="*80)
    print("FINAL COMPARISON RESULTS")
    print("="*80)
    print(results_df.to_string())
    print("="*80)
    
    # Save results
    results_df.to_csv('results/comprehensive_comparison.csv')
    print("\n✓ Saved to: results/comprehensive_comparison.csv")
    
    # Visualize comparison
    print("\nGenerating comparison visualizations...")
    
    # 1. Metrics comparison bar chart
    metrics_to_plot = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'ari', 'nmi', 'purity']
    available_metrics = [m for m in metrics_to_plot if m in results_df.columns and results_df[m].notna().any()]
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(available_metrics):
        if idx < len(axes):
            values = results_df[metric].dropna().sort_values(ascending=(metric != 'davies_bouldin'))
            
            colors = ['red' if 'Raw' in x or 'PCA' in x else 'green' if 'VAE' in x or 'CVAE' in x or 'Beta' in x else 'blue' for x in values.index]
            
            axes[idx].barh(range(len(values)), values.values, color=colors, edgecolor='black', linewidth=1.5)
            axes[idx].set_yticks(range(len(values)))
            axes[idx].set_yticklabels(values.index, fontsize=10)
            axes[idx].set_xlabel('Score', fontsize=11)
            axes[idx].set_title(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
            axes[idx].grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(values.values):
                axes[idx].text(v, i, f' {v:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Comprehensive Method Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('results/figures/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/figures/comprehensive_comparison.png")
    plt.close()
    
    # 2. Ranking table visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Compute rankings
    rankings = pd.DataFrame()
    for metric in available_metrics:
        if metric in results_df.columns:
            ascending = (metric == 'davies_bouldin')  # Lower is better for DB
            rankings[metric] = results_df[metric].rank(ascending=ascending, na_option='bottom')
    
    # Average rank
    rankings['Average Rank'] = rankings.mean(axis=1)
    rankings = rankings.sort_values('Average Rank')
    
    # Display as heatmap
    sns.heatmap(rankings, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Rank (lower is better)'}, 
                linewidths=0.5, ax=ax)
    ax.set_title('Method Rankings Across All Metrics\n(1 = Best)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/figures/method_rankings.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/figures/method_rankings.png")
    plt.close()
    
    # Print best methods
    print("\n" + "="*80)
    print("BEST PERFORMING METHODS")
    print("="*80)
    for metric in available_metrics:
        if metric in results_df.columns:
            ascending = (metric == 'davies_bouldin')
            best = results_df[metric].idxmin() if ascending else results_df[metric].idxmax()
            best_value = results_df.loc[best, metric]
            print(f"{metric.replace('_', ' ').title():30s}: {best:20s} ({best_value:.4f})")
    
    print("\n" + "="*80)
    print("OVERALL BEST METHOD (by average rank):")
    print("="*80)
    best_overall = rankings['Average Rank'].idxmin()
    print(f"🏆 {best_overall} (Average Rank: {rankings.loc[best_overall, 'Average Rank']:.2f})")
    print("="*80)
    
    return results_df

if __name__ == '__main__':
    # Train Autoencoder
    print("Step 1: Training Autoencoder baseline...")
    ae_model, ae_latent = train_autoencoder()
    
    # Comprehensive comparison
    print("\n\nStep 2: Comparing all methods...")
    results_df = comprehensive_comparison()
    
    print("\n" + "="*80)
    print("✅ COMPREHENSIVE COMPARISON COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/models/autoencoder_final.pth")
    print("  - results/latent_features_autoencoder.npy")
    print("  - results/comprehensive_comparison.csv")
    print("  - results/figures/comprehensive_comparison.png")
    print("  - results/figures/method_rankings.png")
    print("\n🎉 ALL HARD TASK REQUIREMENTS COMPLETED!")
