import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

from src.vae import vae_loss

# Convolutional VAE Architecture
class ConvEncoder(torch.nn.Module):
    def __init__(self, latent_dim=32):
        super(ConvEncoder, self).__init__()
        # Input: (batch, 1, 40, 130) - MFCC as image
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        
        # After convolutions: (batch, 128, 5, 17)
        self.fc_mu = torch.nn.Linear(128 * 5 * 17, latent_dim)
        self.fc_logvar = torch.nn.Linear(128 * 5 * 17, latent_dim)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class ConvDecoder(torch.nn.Module):
    def __init__(self, latent_dim=32):
        super(ConvDecoder, self).__init__()
        self.fc = torch.nn.Linear(latent_dim, 128 * 5 * 17)
        
        self.deconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.deconv2 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.deconv3 = torch.nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 5, 17)
        x = torch.nn.functional.relu(self.bn1(self.deconv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)
        # Resize to match input (40, 130)
        x = torch.nn.functional.interpolate(x, size=(40, 130), mode='bilinear', align_corners=False)
        return x

class ConvVAE(torch.nn.Module):
    def __init__(self, latent_dim=32):
        super(ConvVAE, self).__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu

# Dataset for 2D spectrograms
class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        # Reshape MFCC to 2D: (N, 5200) -> (N, 1, 40, 130)
        self.spectrograms = features.reshape(-1, 40, 130)
        
        # Normalize
        mean = self.spectrograms.mean()
        std = self.spectrograms.std()
        self.spectrograms = (self.spectrograms - mean) / (std + 1e-8)
        
        # Add channel dimension
        self.spectrograms = torch.FloatTensor(self.spectrograms).unsqueeze(1)
        
    def __len__(self):
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        return self.spectrograms[idx]

def train_conv_vae():
    print("\n" + "="*80)
    print("TRAINING CONVOLUTIONAL VAE FOR SPECTROGRAMS")
    print("="*80)
    
    # Load data
    print("\nLoading audio features...")
    audio_data = np.load('data/audio_features.npz', allow_pickle=True)
    audio_features = audio_data['features']
    
    print(f"✓ Audio features shape: {audio_features.shape}")
    
    # Create dataset
    dataset = SpectrogramDataset(audio_features)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"✓ Spectrogram shape: {dataset.spectrograms.shape}")
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConvVAE(latent_dim=32)
    model.to(device)
    
    print(f"✓ Using device: {device}")
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 50
    
    losses = []
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            batch = batch.to(device)
            
            # Forward pass
            x_recon, mu, logvar = model(batch)
            loss, recon, kl = vae_loss(x_recon, batch, mu, logvar, beta=1.0)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item()/len(batch):.4f}'})
        
        avg_loss = epoch_loss / len(dataloader.dataset)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'results/models/conv_vae_epoch_{epoch+1}.pth')
            print(f"  ✓ Saved checkpoint")
    
    # Save final model
    torch.save(model.state_dict(), 'results/models/conv_vae_final.pth')
    print("\n✓ Saved final model: results/models/conv_vae_final.pth")
    
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
    np.save('results/latent_features_conv_vae.npy', latent_features)
    print(f"✓ Saved latent features: {latent_features.shape}")
    
    # Visualize reconstruction
    print("\nGenerating reconstruction examples...")
    model.eval()
    with torch.no_grad():
        sample_idx = np.random.choice(len(dataset), 5, replace=False)
        samples = torch.stack([dataset[i] for i in sample_idx]).to(device)
        recon, _, _ = model(samples)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(5):
            axes[0, i].imshow(samples[i].cpu().squeeze(), cmap='viridis', aspect='auto')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(recon[i].cpu().squeeze(), cmap='viridis', aspect='auto')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/figures/conv_vae_reconstruction.png', dpi=300, bbox_inches='tight')
        print("✓ Saved reconstruction: results/figures/conv_vae_reconstruction.png")
        plt.close()
    
    print("\n" + "="*80)
    print("✅ CONVOLUTIONAL VAE TRAINING COMPLETE!")
    print("="*80)
    
    return model, latent_features

if __name__ == '__main__':
    model, latent_features = train_conv_vae()
