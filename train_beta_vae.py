import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

from src.vae import VAE, vae_loss
from src.dataset import MusicDataset

def train_beta_vae(beta=4.0):
    """Train Beta-VAE with specified beta parameter"""
    
    print("\n" + "="*80)
    print(f"TRAINING BETA-VAE (β={beta})")
    print("="*80)
    print("Beta-VAE encourages disentangled representations by weighting KL divergence")
    
    # Load data
    print("\nLoading features...")
    audio_data = np.load('data/audio_features.npz', allow_pickle=True)
    audio_features = audio_data['features']
    genres = audio_data['genres']
    
    lyrics_data = np.load('data/lyrics_features.npz')
    lyrics_features = lyrics_data['embeddings']
    
    print(f"✓ Audio features: {audio_features.shape}")
    print(f"✓ Lyrics features: {lyrics_features.shape}")
    
    # Create dataset
    dataset = MusicDataset(audio_features, lyrics_features, normalize=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize Beta-VAE (same architecture as VAE, but different beta)
    input_dim = dataset.features.shape[1]
    model = VAE(input_dim=input_dim, hidden_dim=512, latent_dim=32)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    print(f"✓ Model initialized with latent_dim=32")
    print(f"✓ Using device: {device}")
    print(f"✓ Beta parameter: {beta}")
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    epochs = 50
    losses = []
    recon_losses = []
    kl_losses = []
    
    print(f"\nTraining for {epochs} epochs with β={beta}...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            batch = batch.to(device)
            
            # Forward pass
            x_recon, mu, logvar = model(batch)
            loss, recon, kl = vae_loss(x_recon, batch, mu, logvar, beta=beta)  # Beta applied here
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            
            pbar.set_postfix({'loss': f'{loss.item()/len(batch):.4f}'})
        
        avg_loss = epoch_loss / len(dataloader.dataset)
        avg_recon = epoch_recon / len(dataloader.dataset)
        avg_kl = epoch_kl / len(dataloader.dataset)
        
        losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)
        
        print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}')
        
        scheduler.step(avg_loss)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'results/models/beta_vae_epoch_{epoch+1}.pth')
            print(f"  ✓ Saved checkpoint")
    
    # Save final model
    model_path = f'results/models/beta_vae_beta{beta}_final.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Saved final model: {model_path}")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(losses, linewidth=2)
    axes[0].set_title(f'Total Loss (β={beta})', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(recon_losses, linewidth=2, color='orange')
    axes[1].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(kl_losses, linewidth=2, color='green')
    axes[2].set_title(f'KL Divergence (weighted by β={beta})', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/figures/beta_vae_training_curves_beta{beta}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved training curves")
    plt.close()
    
    # Extract latent features
    print("\nExtracting latent features...")
    model.eval()
    latent_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting'):
            batch = batch.to(device)
            mu = model.encode(batch)
            latent_features.append(mu.cpu().numpy())
    
    latent_features = np.concatenate(latent_features, axis=0)
    np.save(f'results/latent_features_beta_vae_beta{beta}.npy', latent_features)
    print(f"✓ Saved latent features: {latent_features.shape}")
    
    # Analyze latent space disentanglement
    print("\nAnalyzing latent space disentanglement...")
    latent_std = latent_features.std(axis=0)
    latent_mean = latent_features.mean(axis=0)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(32), latent_std, color='steelblue', edgecolor='black')
    plt.title(f'Latent Dimension Std Dev (β={beta})', fontsize=14, fontweight='bold')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Standard Deviation')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.corrcoef(latent_features.T), cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title(f'Latent Dimension Correlation (β={beta})', fontsize=14, fontweight='bold')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Latent Dimension')
    
    plt.tight_layout()
    plt.savefig(f'results/figures/beta_vae_disentanglement_beta{beta}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved disentanglement analysis")
    plt.close()
    
    print("\n" + "="*80)
    print(f"✅ BETA-VAE (β={beta}) TRAINING COMPLETE!")
    print("="*80)
    print(f"\nKey characteristics of β={beta}:")
    if beta > 1:
        print(f"  - Higher β emphasizes disentanglement over reconstruction")
        print(f"  - Expected: More independent latent dimensions")
        print(f"  - Trade-off: Slightly worse reconstruction quality")
    
    return model, latent_features

if __name__ == '__main__':
    # Train Beta-VAE with beta=4
    print("Training Beta-VAE with β=4 for disentangled representations...")
    model_beta4, latent_beta4 = train_beta_vae(beta=4.0)
    
    print("\n" + "="*80)
    print("BETA-VAE TRAINING COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/models/beta_vae_beta4.0_final.pth")
    print("  - results/latent_features_beta_vae_beta4.0.npy")
    print("  - results/figures/beta_vae_training_curves_beta4.0.png")
    print("  - results/figures/beta_vae_disentanglement_beta4.0.png")
