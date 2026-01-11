import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vae import VAE, vae_loss
from src.dataset import MusicDataset

def train_vae(model, dataloader, epochs=50, lr=1e-3, beta=1.0, device='cuda', save_path='results/models'):
    """Train VAE model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    model.to(device)
    
    os.makedirs(save_path, exist_ok=True)
    
    losses = []
    recon_losses = []
    kl_losses = []
    
    print(f"\n{'='*60}")
    print(f"Training VAE on {device}")
    print(f"Epochs: {epochs} | Batch size: {dataloader.batch_size} | LR: {lr}")
    print(f"{'='*60}\n")
    
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
            loss, recon, kl = vae_loss(x_recon, batch, mu, logvar, beta)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item()/len(batch):.4f}'})
        
        avg_loss = epoch_loss / len(dataloader.dataset)
        avg_recon = epoch_recon / len(dataloader.dataset)
        avg_kl = epoch_kl / len(dataloader.dataset)
        
        losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)
        
        print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}')
        
        scheduler.step(avg_loss)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_path, f'vae_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")
    
    return model, {'total_loss': losses, 'recon_loss': recon_losses, 'kl_loss': kl_losses}

def extract_latent_features(model, dataloader, device='cuda'):
    """Extract latent representations from trained VAE"""
    model.eval()
    latent_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting latent features'):
            batch = batch.to(device)
            mu = model.encode(batch)
            latent_features.append(mu.cpu().numpy())
    
    return np.concatenate(latent_features, axis=0)

def plot_training_curves(losses_dict, save_path='results/figures/training_curves.png'):
    """Plot training curves"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(losses_dict['total_loss'], linewidth=2)
    axes[0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(losses_dict['recon_loss'], linewidth=2, color='orange')
    axes[1].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(losses_dict['kl_loss'], linewidth=2, color='green')
    axes[2].set_title('KL Divergence', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training curves: {save_path}")
    plt.close()

if __name__ == '__main__':
    from src.dataset import MusicDataset
    
    print("\n" + "="*60)
    print("VAE TRAINING")
    print("="*60)
    
    # Load data
    print("\nLoading features...")
    audio_data = np.load('data/audio_features.npz', allow_pickle=True)
    audio_features = audio_data['features']
    genres = audio_data['genres']
    
    lyrics_data = np.load('data/lyrics_features.npz')
    lyrics_features = lyrics_data['embeddings']
    
    print(f"✓ Audio features: {audio_features.shape}")
    print(f"✓ Lyrics features: {lyrics_features.shape}")
    print(f"✓ Genres: {np.unique(genres)}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = MusicDataset(audio_features, lyrics_features, normalize=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    print(f"✓ Dataset size: {len(dataset)}")
    print(f"✓ Feature dimension: {dataset.features.shape[1]}")
    print(f"✓ Batch size: 32")
    
    # Initialize VAE
    input_dim = dataset.features.shape[1]
    model = VAE(input_dim=input_dim, hidden_dim=512, latent_dim=32)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Using device: {device}")
    
    if device == 'cpu':
        print("⚠ Warning: Training on CPU will be slow.")
    
    # Train VAE
    trained_model, losses = train_vae(
        model, 
        dataloader, 
        epochs=50,
        lr=1e-3,
        beta=1.0,
        device=device,
        save_path='results/models'
    )
    
    # Save final model
    final_model_path = 'results/models/vae_final.pth'
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"\n✓ Saved final model: {final_model_path}")
    
    # Plot training curves
    plot_training_curves(losses, save_path='results/figures/training_curves.png')
    
    # Extract latent features
    print("\nExtracting latent features...")
    latent_features = extract_latent_features(trained_model, dataloader, device)
    
    latent_save_path = 'results/latent_features_vae.npy'
    np.save(latent_save_path, latent_features)
    print(f"✓ Saved latent features: {latent_save_path}")
    print(f"✓ Latent features shape: {latent_features.shape}")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - results/models/vae_final.pth")
    print("  - results/latent_features_vae.npy")
    print("  - results/figures/training_curves.png")
    print("\nNext step: Clustering and evaluation")
    print("="*60 + "\n")
