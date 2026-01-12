

***

```markdown
#  VAE for Hybrid Language Music Clustering


---


---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/vae-music-clustering.git
cd vae-music-clustering
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
librosa>=0.9.0
sentence-transformers>=2.2.0
umap-learn>=0.5.0
tqdm>=4.62.0
```

---

## ⚡ Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
# Extract features (first time only)
python src/feature_extraction.py

# Train VAE model
python src/train_vae.py

# Run complete pipeline (clustering, evaluation, visualization)
python main.py
```

### Option 2: Run Individual Components

```bash
# 1. Extract audio and lyrics features
python src/feature_extraction.py

# 2. Train different VAE variants
python src/train_vae.py              # Standard VAE
python train_conv_vae.py             # Convolutional VAE
python train_beta_vae.py             # Beta-VAE (β=4)
python train_cvae.py                 # Conditional VAE

# 3. Compare clustering algorithms
python compare_clustering.py

# 4. Run comprehensive comparison
python comprehensive_comparison.py
```

### Option 3: Explore with Jupyter

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---


## 📝 Detailed Documentation

- **Analysis Document:** `results/analysis_and_discussion.md`
- **Summary Report:** `results/report_summary.txt`
- **Metrics CSV:** `results/comprehensive_comparison.csv`
- **NeurIPS Paper:** Available in Overleaf

---

## 🔮 Future Work

1. **Larger Dataset:** Expand to full GTZAN (1000 samples, 10 genres)
2. **Real Lyrics:** Integrate actual English + Bangla song lyrics
3. **Data Augmentation:** Time stretching, pitch shifting, noise injection
4. **Pre-trained Models:** Use VGGish, PANN for audio encoding
5. **Semi-supervised VAE:** Leverage partial labels
6. **Hierarchical VAE:** Multi-scale representations

---

## 🤝 Acknowledgments

- **Course:** CSE425 - Neural Networks
- **Dataset:** GTZAN Genre Collection
- **Frameworks:** PyTorch, scikit-learn, librosa
- **Inspiration:** Kingma & Welling (2013), Higgins et al. (2017)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Labibul Ahsan Alif**  
Computer Science Student  
Neural Networks Course Project  

📧 Email: labibalif2001@gmail.com


---

## 🌟 Star this repository if you found it helpful!

