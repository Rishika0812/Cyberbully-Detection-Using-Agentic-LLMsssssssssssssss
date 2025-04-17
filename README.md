# Cyberbullying Tweet Recognition System

## Overview
This system is a sophisticated deep learning-based solution for detecting and classifying cyberbullying in social media texts, with a particular focus on Twitter content. The system employs a multi-task learning approach combining cyberbullying classification, sentiment analysis, and context understanding.

## Features
- Multi-class cyberbullying detection (6 categories)
- Sentiment analysis integration
- Contextual understanding
- Real-time text processing
- High accuracy and low latency
- Production-ready API endpoints

## System Architecture

### Core Components
1. **Text Processing Pipeline**
   - Word Embeddings (GloVe, 300d)
   - Positional Encoding
   - Feature Extraction Network

2. **Neural Network Architecture**
   - BiLSTM Layer (256 units)
   - Multi-Head Attention (8 heads)
   - CNN Layer with Multiple Kernels
   - Dense Classification Layers

3. **Auxiliary Networks**
   - BERT-based Sentiment Analysis
   - Context Understanding Module

## Requirements

### Hardware Requirements
- RAM: 8GB minimum, 16GB recommended
- GPU: NVIDIA GPU with 6GB VRAM (recommended)
- Storage: 5GB minimum

### Software Dependencies
```
torch>=1.9.0
transformers>=4.5.0
numpy>=1.19.5
pandas>=1.3.0
scikit-learn>=0.24.2
nltk>=3.6.2
tensorflow>=2.5.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cyberbullying-detection.git
cd cyberbullying-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pre-trained models:
```bash
python scripts/download_models.py
```

## Usage

### Training the Model

```python
from cyberbullying_detector import CyberbullyingDetector
from cyberbullying_detector.training import train_model

# Initialize model
model = CyberbullyingDetector()

# Train model
train_model(model, train_data, val_data, epochs=100)
```

### Making Predictions

```python
from cyberbullying_detector import CyberbullyingDetector

# Load trained model
model = CyberbullyingDetector.load_from_checkpoint('path/to/checkpoint')

# Make prediction
text = "Your example text here"
prediction = model.predict(text)
print(prediction)
```

### API Usage

```python
from cyberbullying_detector.api import CyberbullyingAPI

api = CyberbullyingAPI()
response = api.analyze_text("Your text here")
```

## Model Performance

### Classification Metrics
- Overall Accuracy: 82.90%
- Macro F1-Score: 0.81

### Per-Class Performance
- Age-related: 0.84 F1-Score
- Ethnicity: 0.83 F1-Score
- Gender: 0.85 F1-Score
- Religion: 0.82 F1-Score
- Other: 0.79 F1-Score

## API Documentation

### REST API Endpoints

#### POST /api/v1/analyze
Analyzes text for cyberbullying content.

Request:
```json
{
    "text": "Your text to analyze",
    "include_sentiment": true,
    "include_context": true
}
```

Response:
```json
{
    "cyberbullying_type": "harassment",
    "confidence": 0.85,
    "sentiment": "negative",
    "context": "direct_attack",
    "recommendations": [
        "Consider reporting this content",
        "Block user if pattern continues"
    ]
}
```

## Contributing
We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Model Training Details

### Data Preprocessing
- Text cleaning and normalization
- Tokenization
- Embedding generation
- Sequence padding

### Training Parameters
- Batch Size: 32
- Learning Rate: 0.001
- Optimizer: Adam
- Loss Function: Combined (CrossEntropy + Focal Loss)
- Epochs: 100

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this system in your research, please cite:

```bibtex
@software{cyberbullying_detector_2024,
    title={Cyberbullying Tweet Recognition System},
    author={Your Name},
    year={2024},
    version={1.0.0}
}
```

## Acknowledgments
- Thanks to the open-source community
- Special thanks to contributors and maintainers
- Dataset providers and research partners

## Contact
- Project Link: https://github.com/Rishika0812/Cyberbully-Detection-Using-Agentic-LLMsssssssssssssss
- Documentation: https://docs.google.com/document/d/1aEr2ReshpTUukRkCKx-svYuupCArVNQNVVZFUDsFxYs/edit?tab=t.0

## Roadmap
- [ ] Add support for more languages
- [ ] Implement real-time monitoring
- [ ] Enhance context understanding
- [ ] Add automated response suggestions
- [ ] Improve model efficiency 
