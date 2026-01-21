# Enhanced English-to-Tamazight Neural Machine Translation

A character-level Seq2Seq neural machine translation model for translating English to Tamazight (Berber/Amazigh) with both Latin and Tifinagh script support.

## Overview

This project implements an enhanced neural machine translation system specifically designed for the low-resource English→Tamazight language pair. The model achieves competitive performance (targeting BLEU 12-18+) through several advanced techniques including attention mechanisms, learning rate scheduling, and beam search decoding.

## Features

### Core Capabilities
- **Bidirectional Translation Support**: Translates English text to Tamazight
- **Dual Script Output**: Generates translations in both Latin script and Tifinagh (ⵜⵉⴼⵉⵏⴰⵖ)
- **Character-Level Architecture**: Handles morphologically rich Tamazight effectively
- **Low-Resource Optimizations**: Specialized techniques for limited training data

### Advanced Enhancements
- **Attention Mechanism**: Improves translation quality for longer sequences
- **Cosine Learning Rate Scheduling**: Includes warmup period for stable training
- **Validation Split & Early Stopping**: Prevents overfitting on limited data
- **Beam Search Decoding**: Explores multiple translation possibilities for better quality
- **Automatic Model Checkpointing**: Saves the best performing model during training

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Required Packages
```bash
pip install torch torchvision torchaudio
pip install datasets
pip install numpy
```

### Optional (for evaluation)
```bash
pip install sacrebleu  # For BLEU score calculation
pip install huggingface-hub  # For dataset authentication
```

## Dataset

The model trains on two Hugging Face datasets:

1. **Weblate-Translations** (`Tamazight-NLP/Weblate-Translations`)
   - No authentication required
   - Software localization translations
   
2. **Beni-Mellal-Tamazight** (`Tamazight-NLP/Beni-Mellal-Tamazight`)
   - May require Hugging Face authentication
   - General domain parallel corpus

### Authentication Setup

If using the Beni-Mellal dataset, authenticate with Hugging Face:

```bash
# Method 1: CLI
huggingface-cli login

# Method 2: Python
from huggingface_hub import login
login()
```

## Usage

### Basic Training

Run the script with default settings:

```python
python enhanced_tamazight_model.py
```

This will:
1. Load both datasets (if available)
2. Train with enhanced features for 15 epochs
3. Save the best model to `tamazight_model_best.pt`
4. Test translations on sample phrases

### Configuration Options

Edit the main block to customize training:

```python
# Use enhanced training (recommended)
USE_ENHANCED_TRAINING = True

# Dataset selection: 'both', 'weblate', or 'beni-mellal'
source_texts, target_texts = load_and_preprocess_dataset(dataset_choice='both')

# Training parameters
model, src_vocab, tgt_vocab, train_losses, val_losses = train_model_enhanced(
    source_texts, target_texts,
    epochs=15,              # Number of training epochs
    batch_size=32,          # Training batch size
    learning_rate=0.001,    # Initial learning rate
    use_attention=True,     # Use attention mechanism
    max_samples=None        # Limit training samples (None = use all)
)
```

### Quick Testing (Limited Data)

For rapid prototyping:

```python
# Train on subset for faster iteration
train_model_enhanced(
    source_texts, target_texts,
    epochs=5,
    batch_size=32,
    max_samples=1000  # Use only 1000 samples
)
```
