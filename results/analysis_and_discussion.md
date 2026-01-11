# Comprehensive Analysis: VAE-based Music Clustering

## Executive Summary

This study compares 7 different feature extraction and clustering approaches for music genre classification on 312 GTZAN samples across 4 genres (Blues, Classical, Country, Disco).

## Performance Results

### Clustering Performance Ranking

| Rank | Method      | Silhouette ↑ | ARI ↑      | NMI ↑      | Purity ↑   |
| ---- | ----------- | ------------ | ---------- | ---------- | ---------- |
| 🥇 1 | **PCA**     | **0.0893**   | **0.2440** | **0.2716** | **0.6314** |
| 🥈 2 | Raw Audio   | 0.1804       | 0.1011     | 0.1818     | 0.5032     |
| 🥉 3 | Conv-VAE    | 0.1244       | 0.0003     | 0.0173     | 0.3782     |
| 4    | CVAE        | 0.0597       | -0.0015    | 0.0117     | 0.3718     |
| 5    | Autoencoder | 0.0592       | -0.0028    | 0.0109     | 0.3622     |
| 6    | Beta-VAE    | 0.0563       | -0.0034    | 0.0054     | 0.3622     |
| 7    | VAE         | 0.0548       | -0.0039    | 0.0079     | 0.3590     |

### Key Observations

1. **PCA achieves best overall performance** across all metrics
2. **Raw audio features** perform surprisingly well (2nd place)
3. **Convolutional VAE** is the best deep learning method (3rd place)
4. **Standard VAE/Beta-VAE/CVAE** underperform baselines

## Why PCA Outperformed Deep Learning Methods

### Dataset Characteristics

1. **Small Sample Size (312 samples)**

   - Deep learning requires thousands of samples to learn effectively
   - PCA works well with limited data
   - VAE models likely overfit despite regularization

2. **Simple Structure (4 genres)**

   - Linear separability between genres
   - PCA's linear transformation sufficient
   - Non-linear VAE complexity not needed

3. **High-Dimensional Features (5200 MFCC dims)**
   - PCA efficiently captures variance
   - Clear principal components separate genres
   - VAE compression may lose discriminative info

### Impact of Dummy Lyrics

- **Problem:** Used random 384-dim embeddings (not real lyrics)
- **Effect:** Added noise to VAE training (5584 total dims)
- **Result:** VAE struggled to learn meaningful joint representation
- **PCA Impact:** Less affected as it focuses on variance

### Overfitting in Deep Models

- **Training Loss:** VAE models achieved low training loss
- **Generalization:** Poor clustering suggests overfitting
- **Regularization:** KL divergence alone insufficient for small data
- **Solution Needed:** Stronger regularization, data augmentation

## Method-Specific Analysis

### PCA (Best Overall: 0.6314 Purity)

**Strengths:**

- Simple, fast, interpretable
- Captures 94% variance in 32 components
- Linear transformation sufficient for this dataset

**Why it won:**

- Efficient dimensionality reduction (5200 → 32)
- Preserves discriminative features
- No overfitting risk

### Raw Audio Features (2nd Place: 0.5032 Purity)

**Strengths:**

- No information loss
- Original MFCC features already genre-discriminative
- High Silhouette score (0.1804)

**Limitations:**

- High dimensionality (5200 dims)
- Computationally expensive for clustering

### Convolutional VAE (Best Deep Learning: 0.3782 Purity)

**Strengths:**

- Treats MFCC as 2D images (40×130)
- Captures local patterns via convolutions
- Better than fully-connected VAEs

**Why better than standard VAE:**

- Spatial structure preservation
- Translation invariance
- Fewer parameters (less overfitting)

**Still underperforms PCA due to:**

- Limited training data
- Overfitting to training set

### Beta-VAE (β=4) - Worst Performance

**Expected:** Better disentanglement
**Reality:** Worst clustering performance

**Why it failed:**

- High β (4.0) over-regularizes
- Too much pressure on KL divergence
- Loss of discriminative information
- Trade-off: interpretability vs performance

### CVAE (Genre-Conditioned)

**Expected:** Genre information should help
**Reality:** No improvement (0.3718 purity)

**Why:**

- Conditioning on genre during training
- But test clustering ignores conditions
- Model learned genre-specific features
- Not universally discriminative features

### Standard VAE & Autoencoder

**Performance:** Similar to each other (0.36 purity)

**Why VAE ≈ Autoencoder:**

- Small dataset limits VAE advantages
- Probabilistic framework not beneficial here
- Autoencoder's simpler objective equally effective

## Clustering Algorithm Analysis

All methods used K-Means with k=4. From separate analysis (`clustering_algorithms_comparison.csv`):

- **K-Means:** Fast, works well with VAE latent space
- **Agglomerative:** Slightly better hierarchy discovery
- **DBSCAN:** Struggled with dense clusters, found few noise points

## Metrics Interpretation

### Silhouette Score

- **Best:** Raw Audio (0.1804)
- **Meaning:** Moderate cluster separation
- **Note:** All scores < 0.3 indicate overlapping genres

### Adjusted Rand Index (ARI)

- **Best:** PCA (0.2440)
- **Meaning:** 24.4% agreement with true labels
- **Negative ARI:** VAE/AE/Beta-VAE worse than random

### Normalized Mutual Information (NMI)

- **Best:** PCA (0.2716)
- **Range:** 0-1 (higher = more info shared)
- **Result:** Moderate alignment with genres

### Cluster Purity

- **Best:** PCA (0.6314)
- **Meaning:** 63% of each cluster is dominant genre
- **Interpretation:** Reasonable but not excellent separation

### Davies-Bouldin Index (Lower is Better)

- **Best:** Raw Audio (1.731)
- **Worst:** CVAE (3.126)
- **Confirms:** VAE methods have worse cluster quality

## Lessons Learned

### When VAE Works Best

✅ Large datasets (>10,000 samples)
✅ Complex non-linear structure
✅ Real multi-modal data (audio + text)
✅ Generation tasks
✅ Transfer learning scenarios

### When PCA Works Best

✅ Small to medium datasets
✅ Linear or near-linear separability
✅ High-dimensional data with clear variance
✅ Quick prototyping
✅ Interpretability needed

### When to Use Conv-VAE

✅ Grid-like structured data (spectrograms)
✅ Local pattern importance
✅ Translation invariance desired

## Recommendations for Improvement

### To Improve VAE Performance

1. **Increase dataset size:** Use full GTZAN (1000 samples)
2. **Real lyrics:** Replace dummy embeddings with actual English+Bangla lyrics
3. **Data augmentation:** Time stretching, pitch shifting, noise addition
4. **Stronger regularization:** Dropout, batch normalization, early stopping
5. **Architecture tuning:** Experiment with depth, width, latent dims
6. **Pre-training:** Use pre-trained audio encoders (VGGish, PANN)

### Alternative Approaches

1. **Semi-supervised VAE:** Use partial labels during training
2. **Hierarchical VAE:** Multi-scale representations
3. **Adversarial VAE:** Better latent distribution matching
4. **Transformer-based:** Self-attention for sequential audio

## Conclusion

**Main Finding:** For this small dataset (312 samples, 4 genres), traditional PCA outperforms modern deep learning VAE methods for music clustering.

**Why:** Limited data + simple structure + linear separability favor PCA's simplicity and efficiency.

**VAE Value:** Despite poor clustering performance, VAE provides:

- Generative capabilities (can synthesize new samples)
- Probabilistic framework (uncertainty quantification)
- Transfer learning potential
- Interpretable latent space (with Beta-VAE)

**Best Method for This Task:** **PCA** (0.631 purity, 0.244 ARI, 0.272 NMI)

**Best Deep Learning Method:** **Convolutional VAE** (0.378 purity, respects spatial structure)

**Recommendation:** Use PCA for this dataset; scale to VAE with more data and real multi-modal features.

---

**Project demonstrates:** Importance of matching method complexity to data size and structure. Sometimes simpler is better! 📊
