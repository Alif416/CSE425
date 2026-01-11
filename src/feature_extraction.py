import librosa
import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def extract_audio_features(file_path, n_mfcc=40, max_len=130):
    """Extract MFCC features from audio file"""
    try:
        # Load audio file (30 seconds)
        y, sr = librosa.load(file_path, duration=30, sr=22050, mono=True)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Pad or truncate to fixed length
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        
        return mfcc.flatten()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_gtzan_dataset(genres_folder, output_file='data/audio_features.npz'):
    """
    Process GTZAN dataset
    genres_folder: path to genres_original folder
    """
    features = []
    labels = []
    genres = []
    
    print(f"Looking for genres in: {genres_folder}")
    
    # Get all genre folders
    genre_folders = [f for f in os.listdir(genres_folder) 
                    if os.path.isdir(os.path.join(genres_folder, f))]
    
    genre_folders.sort()  # Sort for consistency
    
    print(f"Found {len(genre_folders)} genres: {genre_folders}")
    
    for genre in tqdm(genre_folders, desc="Processing genres"):
        genre_path = os.path.join(genres_folder, genre)
        audio_files = [f for f in os.listdir(genre_path) 
                      if f.endswith(('.wav', '.mp3', '.au'))]
        
        print(f"\n{genre}: {len(audio_files)} files")
        
        for filename in tqdm(audio_files, desc=f"{genre}", leave=False):
            file_path = os.path.join(genre_path, filename)
            
            feature = extract_audio_features(file_path)
            
            if feature is not None:
                features.append(feature)
                labels.append(filename)
                genres.append(genre)
    
    # Convert to numpy arrays
    features = np.array(features)
    
    # Save features
    np.savez(output_file, 
             features=features,
             labels=labels,
             genres=genres)
    
    print(f"\n{'='*60}")
    print(f"✓ Feature extraction complete!")
    print(f"Total samples: {len(features)}")
    print(f"Feature shape: {features.shape}")
    print(f"Genres: {np.unique(genres)}")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}")
    
    return features, labels, genres

def create_dummy_lyrics(n_samples, output_file='data/lyrics_features.npz'):
    """Create dummy lyrics embeddings (384-dimensional)"""
    print(f"\nCreating {n_samples} dummy lyrics embeddings...")
    dummy_embeddings = np.random.randn(n_samples, 384)
    
    np.savez(output_file,
             embeddings=dummy_embeddings,
             song_ids=np.arange(n_samples))
    
    print(f"✓ Saved dummy lyrics to: {output_file}")
    return dummy_embeddings

if __name__ == '__main__':
    # Path to your genres_original folder
    GENRES_FOLDER = 'data\genres_original'
    
    print("="*60)
    print("GTZAN FEATURE EXTRACTION")
    print("="*60)
    print(f"Dataset path: {GENRES_FOLDER}\n")
    
    # Check if path exists
    if not os.path.exists(GENRES_FOLDER):
        print(f"❌ Error: {GENRES_FOLDER} not found!")
        print("Please check your dataset location.")
        exit(1)
    
    # Extract audio features
    audio_features, labels, genres = process_gtzan_dataset(
        genres_folder=GENRES_FOLDER,
        output_file='data/audio_features.npz'
    )
    
    # Create dummy lyrics (since we don't have real lyrics)
    lyrics_features = create_dummy_lyrics(
        n_samples=len(audio_features),
        output_file='data/lyrics_features.npz'
    )
    
    print("\n" + "="*60)
    print("✅ ALL FEATURES EXTRACTED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  - data/audio_features.npz")
    print("  - data/lyrics_features.npz")
    print("\nNext step: python src/train_vae.py")
    print("="*60)
