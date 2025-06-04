# XPUF-GAN: Generative Adversarial Networks for XOR PUF Modeling

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning framework for generating synthetic XOR-PUF (Physically Unclonable Function) challenge-response pairs using Generative Adversarial Networks.

## 🔐 Overview

XOR-PUFs are hardware security primitives that create unique device fingerprints. This GAN-based approach generates realistic synthetic PUF datasets while preserving statistical properties and security characteristics of real hardware.

## ✨ Features

- **Adaptive Architecture**: Auto-detects PUF parameters from input data
- **Comprehensive Validation**: Hamming weight, uniqueness, and predictability metrics
- **ML Attack Testing**: Built-in resistance analysis against neural network attacks
- **Reliability Analysis**: Noise tolerance testing with configurable levels
- **Multiple Formats**: Binary and hexadecimal output support

## 🚀 Quick Start

### Installation
```bash
pip install tensorflow numpy matplotlib scikit-learn pandas
```

### Usage
```python
from xpuf_gan import XPUF_GAN, load_xpuf_dataset

# Load your PUF dataset (format: HEX_CHALLENGE;BINARY_RESPONSE)
challenges, responses = load_xpuf_dataset('your_puf_data.txt')

# Initialize and train GAN
gan = XPUF_GAN(challenge_dim=64, response_dim=1)
gan.train(challenges, responses, epochs=5000, batch_size=128)

# Generate synthetic dataset
synthetic_challenges, synthetic_responses, metadata = gan.generate_synthetic_dataset(10000)
```

## 📊 Validation Metrics

- **Security Analysis**: ML attack success rates
- **Statistical Properties**: Hamming weight distribution
- **Uniqueness**: Inter-response distance analysis  
- **Reliability**: Performance under noise conditions

## 🏗️ Architecture

- **Generator**: Noise + Challenge → Synthetic Response
- **Discriminator**: Challenge-Response pairs → Real/Fake classification
- **Training**: Adversarial learning with comprehensive validation

## 📁 Output Structure

```
├── model_checkpoints/       # Saved model weights
├── results/                 # Training plots and metrics
├── synthetic_dataset/       # Generated PUF data
└── *.json                  # Validation results
```

## 🔬 Applications

- Security research without hardware access
- Dataset augmentation for limited PUF data
- Privacy-preserving data sharing
- Educational demonstrations of PUF concepts

## 📋 Requirements

- Python 3.7+
- TensorFlow 2.x
- GPU recommended for training
- Input format: `HEX_CHALLENGE;BINARY_RESPONSE`

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional PUF types (Ring Oscillator, SRAM)
- Advanced GAN architectures
- Real-time visualization
- Distributed training support

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## ⚠️ Disclaimer

For research and educational purposes only. Users responsible for compliance with applicable laws and regulations.
