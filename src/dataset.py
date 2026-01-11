import torch
from torch.utils.data import Dataset
import numpy as np

class MusicDataset(Dataset):
    def __init__(self, audio_features, lyrics_features=None, normalize=True):
        # Normalize features
        if normalize:
            audio_mean = audio_features.mean(axis=0)
            audio_std = audio_features.std(axis=0) + 1e-8
            audio_features = (audio_features - audio_mean) / audio_std
            
            if lyrics_features is not None:
                lyrics_mean = lyrics_features.mean(axis=0)
                lyrics_std = lyrics_features.std(axis=0) + 1e-8
                lyrics_features = (lyrics_features - lyrics_mean) / lyrics_std
        
        self.audio_features = torch.FloatTensor(audio_features)
        
        if lyrics_features is not None:
            self.lyrics_features = torch.FloatTensor(lyrics_features)
            # Concatenate features
            self.features = torch.cat([self.audio_features, self.lyrics_features], dim=1)
        else:
            self.features = self.audio_features
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]
