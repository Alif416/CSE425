import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

from src.vae import vae_loss

# Conditional VAE Architecture
class ConditionalEncoder(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):
        super(ConditionalEncoder, self).__init__()
        # Concatenate input with one-hot encoded condition
        self.fc1 = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, condition):
        # Concatenate input and condition
        x_cond = torch.cat([x, condition], dim=1)
        h = F.relu(self.bn1(self.fc1(x_cond)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.fc2(h)))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dim, output_dim):
        super(ConditionalDecoder, self).__init__()
        # Concatenate latent with condition
        self.fc1 = nn.Linear(latent_dim + condition_dim, hidden_dim // 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, z, condition):
        z_cond = torch.cat([z, condition], dim=1)
        h = F.relu(self.bn1(self.fc1(z_cond)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.fc2(h)))
        x_recon = self.fc3(h)
        return x_recon

class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim=512, latent_dim=32):
        super(CVAE, self).__init__()
        self.encoder = ConditionalEncoder(input_dim, condition_dim, hidden_dim, latent_dim)
        self.decoder = ConditionalDecoder(latent_dim, condition_dim, hidden_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, condition):
        mu, logvar = self.encoder(x, condition)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, condition)
        return x_recon, mu, logvar
    
    def encode(self, x, condition):
        mu, _ = self.encoder(x, condition)
        return mu
    
    def generate(self, z, condition):
        """Generate samples given latent code and condition"""
        return self.decoder(z, condition)

# Conditional Dataset
class ConditionalMusicDataset(Dataset):
    def __init__(self, audio_features, lyrics_features, genres, normalize=True):
        # Normalize features
        if normalize:
            audio_mean = audio_features.mean(axis=0)
            audio_std = audio_features.std(axis=0) + 1e-8
            audio_features = (audio_features - audio_mean) / audio_std
            
            lyrics_mean = lyrics_features.mean(axis=0)
            lyrics_std = lyrics_features.std(axis=0) + 1e-8
            lyrics_features = (lyrics_features - lyrics_mean) / lyrics_std
        
        # Concatenate audio + lyrics
        combined = np.concatenate([audio_features, lyrics_features], axis=1)
        self.features = torch.FloatTensor(combined)
        
        # One-hot encode genres
        unique_genres = np.unique(genres)
        self.genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
        genre_indices = [self.genre_to_idx[g] for g in genres]
        
        self.conditions = torch.zeros(len(genres), len(unique_genres))
        for i, idx in enumerate(genre_indices):
            self.conditions[i, idx] = 1.0
        
        self.genre_names = genres
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.conditions[idx]

def train_cvae():
    print("\n" + "="*80)
    print("TRAINING CONDITIONAL VAE (CVAE)")
    print("="*80)
    print("CVAE conditions generation on genre labels")
    
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
    
    # Create conditional dataset
    dataset = ConditionalMusicDataset(audio_features, lyrics_features, genres, normalize=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    input_dim = dataset.features.shape[1]
    condition_dim = dataset.conditions.shape[1]
    
    print(f"✓ Input dimension: {input_dim}")
    print(f"✓ Condition dimension: {condition_dim} (number of genres)")
    
    # Initialize CVAE
    model = CVAE(input_dim=input_dim, condition_dim=condition_dim, hidden_dim=512, latent_dim=32)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    print(f"✓ Using device: {device}")
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    epochs = 50
    losses = []
    recon_losses = []
    kl_losses = []
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for features, conditions in pbar:
            features = features.to(device)
            conditions = conditions.to(device)
            
            # Forward pass
            x_recon, mu, logvar = model(features, conditions)
            loss, recon, kl = vae_loss(x_recon, features, mu, logvar, beta=1.0)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            
            pbar.set_postfix({'loss': f'{loss.item()/len(features):.4f}'})
        
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
            torch.save(model.state_dict(), f'results/models/cvae_epoch_{epoch+1}.pth')
            print(f"  ✓ Saved checkpoint")
    
    # Save final model
    torch.save(model.state_dict(), 'results/models/cvae_final.pth')
    print("\n✓ Saved final model: results/models/cvae_final.pth")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(losses, linewidth=2, color='purple')
    axes[0].set_title('CVAE Total Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(recon_losses, linewidth=2, color='orange')
    axes[1].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(kl_losses, linewidth=2, color='green')
    axes[2].set_title('KL Divergence', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/cvae_training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved training curves")
    plt.close()
    
    # Extract latent features
    print("\nExtracting conditional latent features...")
    model.eval()
    latent_features = []
    genre_labels = []
    
    with torch.no_grad():
        for features, conditions in tqdm(dataloader, desc='Extracting'):
            features = features.to(device)
            conditions = conditions.to(device)
            mu = model.encode(features, conditions)
            latent_features.append(mu.cpu().numpy())
            genre_labels.append(torch.argmax(conditions, dim=1).cpu().numpy())
    
    latent_features = np.concatenate(latent_features, axis=0)
    genre_labels = np.concatenate(genre_labels, axis=0)
    
    np.save('results/latent_features_cvae.npy', latent_features)
    np.save('results/cvae_genre_labels.npy', genre_labels)
    print(f"✓ Saved latent features: {latent_features.shape}")
    
    # Demonstrate conditional generation
    print("\nDemonstrating conditional generation...")
    model.eval()
    
    # Generate samples for each genre
    n_genres = condition_dim
    n_samples = 5
    
    with torch.no_grad():
        fig, axes = plt.subplots(n_genres, n_samples, figsize=(15, 8))
        
        for genre_idx in range(n_genres):
            # Create condition vector
            condition = torch.zeros(n_samples, n_genres).to(device)
            condition[:, genre_idx] = 1.0
            
            # Sample from prior
            z = torch.randn(n_samples, 32).to(device)
            
            # Generate
            generated = model.generate(z, condition)
            
            for sample_idx in range(n_samples):
                # Reshape for visualization (if applicable)
                axes[genre_idx, sample_idx].plot(generated[sample_idx].cpu().numpy()[:100])
                axes[genre_idx, sample_idx].set_title(f'Genre {genre_idx}', fontsize=8)
                axes[genre_idx, sample_idx].axis('off')
        
        plt.suptitle('CVAE Conditional Generation by Genre', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/figures/cvae_conditional_generation.png', dpi=300, bbox_inches='tight')
        print("✓ Saved conditional generation examples")
        plt.close()
    
    print("\n" + "="*80)
    print("✅ CVAE TRAINING COMPLETE!")
    print("="*80)
    
    return model, latent_features, dataset

if __name__ == '__main__':
    model, latent_features, dataset = train_cvae()
    
    print("\n" + "="*80)
    print("CVAE COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/models/cvae_final.pth")
    print("  - results/latent_features_cvae.npy")
    print("  - results/cvae_genre_labels.npy")
    print("  - results/figures/cvae_training_curves.png")
    print("  - results/figures/cvae_conditional_generation.png")
    print("\nCVAE allows controlled generation conditioned on genre!")
