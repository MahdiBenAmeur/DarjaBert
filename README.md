# DarjaBERT: From-Scratch Transformer for Tunisian Darja

This repository hosts a Jupyter notebook where **I rebuilt the BERT encoder architecture literally from scratch** and trained it on Tunisian Dialect (Darja). This educational project deep-dives into the inner workings of masked language modeling and transformer encoders, with fully custom implementations.

## 🚀 Highlights
- **Custom Encoder-Only MLM**: All core functions (tokenization, BPE merges, batching, masking) coded by hand for end-to-end transparency.
- **Detailed Transformer Blocks**: Embedding + positional encodings (dim=256, window=512), scaled dot-product attention, multi-head (8 heads), layer normalization, and feed-forward networks, stacked 6 times.
- **Full Training Loop**: Manual forward/backward passes, loss computation (Cross-Entropy), optimizer updates, and **progressive checkpoint saving** to recover from interruptions.
- **Week-Long Training**: Trained on a personal machine; loss dropped from **9.8 → 3.6** after hyperparameter tuning.
- **Compact Model**: Final model is **10× smaller than original BERT**, yet achieves solid semantic understanding.
- **Fine-Tuning**: Adapted the pretrained DarjaBERT for **sentiment analysis** on Tunisian Darja, demonstrating transfer learning with medium-good accuracy.

> **P.S.** These implementations reflect my intuition-driven experiments. They may not be fully optimized or production-ready, but they walk through every concept.

---

## 📚 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Setup & Installation](#setup--installation)
3. [Data Loading & Cleaning](#data-loading--cleaning)
4. [Tokenization & BPE Vocabulary](#tokenization--bpe-vocabulary)
5. [Building Transformer Layers](#building-transformer-layers)
6. [Masked Language Model Training](#masked-language-model-training)
7. [Checkpointing & Hyperparameter Tuning](#checkpointing--hyperparameter-tuning)
8. [Fine-Tuning for Sentiment Analysis](#fine-tuning-for-sentiment-analysis)
9. [Results & Metrics](#results--metrics)
10. [Next Steps & Improvements](#next-steps--improvements)
11. [Contact & License](#contact--license)

---

## 🛠 Prerequisites
- Python 3.8+
- PyTorch
- `datasets` (Hugging Face)
- scikit-learn
- pandas
- matplotlib
- tqdm

Install with:
```bash
pip install torch datasets scikit-learn pandas matplotlib tqdm
```

---

## 🔧 Setup & Installation
1. Clone this repo:
   ```bash
git clone https://github.com/yourusername/DarjaBERT.git
cd DarjaBERT
```
2. (Optional) Create a virtual environment:
   ```bash
python3 -m venv env
source env/bin/activate  # macOS/Linux
env\Scripts\activate      # Windows
```
3. Install requirements:
   ```bash
pip install -r requirements.txt
```
4. Open and run the notebook:
   ```bash
jupyter notebook DarjaBert_NB.ipynb
```

---

## 🗂 Data Loading & Cleaning
- Dataset: `khaled123/Tunisian_Dialectic_English_Derja` via Hugging Face
- Cleaning steps:
  - Remove non-Arabic/Latin characters
  - Lowercase and whitespace normalization

```python
from datasets import load_dataset
ds = load_dataset("khaled123/Tunisian_Dialectic_English_Derja", cache_dir="darija_datasets")
```

---

## 🔡 Tokenization & BPE Vocabulary
1. **Word tokenization** and **BPE merging** from scratch.
2. Build vocab dictionary and save as JSON for reuse.
3. Utilities:
   - `tokenize_batch()`
   - `learn_bpe_merges()`
   - `encode_with_bpe()`

---

## 🏗 Building Transformer Layers
- **Embedding + Positional Encoding** (256-d, max length=512)
- **Scaled Dot-Product Attention**
- **Multi-Head Attention** (8 heads)
- **LayerNorm** and **Feed-Forward Network**
- **Masking Function** for MLM
- **EncoderBlock**: stacks all components
- **DarjaModel**: stacks 6 `EncoderBlock`s

```python
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, ...):
        # init attention, ffn, norm layers
```

---

## 🎓 Masked Language Model Training
- Mask ~15% tokens per batch
- Cross-entropy loss on masked positions
- Adam optimizer, learning rate scheduling
- Training loop:
  1. `mask_tokens()` → inputs, labels
  2. Forward pass
  3. Compute loss
  4. Backward + optimizer step
  5. Save checkpoint every N steps

---

## 🔄 Checkpointing & Hyperparameter Tuning
- Progressive saving: resume training if interrupted
- Trained over **1+ week** on a personal machine
- Loss reduction from **9.8 → 3.6** via grid search on lr, batch size, etc.

---

## ✏️ Fine-Tuning for Sentiment Analysis
- Custom `sentiment_dataset` for CSV data
- Text cleaning function `clean_sa_text()`
- Classification head on top of encoder
- Training & evaluation on Darja sentiment data

```python
sa_model = DarjaSentimentModel(base_model)
# train and evaluate
evaluate_accuracy(sa_model, ...)
```

---

## 📊 Results & Metrics
- **MLM Convergence**: Loss curves in notebook
- **Sentiment Accuracy**: Train/Test reported at end
- Final test accuracy: *medium-good*, demonstrating transfer learning potential.

---

## 🌱 Next Steps & Improvements
- Optimize BPE and tokenization speed
- Scale up model depth/width
- Experiment with dynamic masking
- Explore other downstream tasks: NER, translation

---

## 📬 Contact & License
- **Author**: Mahdi Ben Ameur
- **LinkedIn**: https://www.linkedin.com/in/mahdi-ben-ameur-5089ba240/
- **License**: MIT (see [LICENSE](LICENSE))

Feel free to open issues or reach out for questions or suggestions!

