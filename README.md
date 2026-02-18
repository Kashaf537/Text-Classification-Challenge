Text Classification Web App using Transformer

## 📄 Project Overview

This project implements a **movie review sentiment classification system** using a **Transformer-based neural network**, deployed as a web application with **FastAPI** and a simple **HTML front-end**.  

The system predicts whether a review is **Positive** or **Negative**.  

### Workflow Summary

1. **Data Preprocessing & Cleaning**  
   - Loading the dataset.  
   - Removing duplicate entries and irrelevant columns.  
   - Lowercasing, tokenization, and punctuation removal.  
   - Padding/truncation to a fixed length.  
   - Building a vocabulary mapping words to integer IDs.

2. **Model Implementation (`model.py`)**  
   - Transformer implemented **from scratch** using PyTorch.  
   - Includes **Embedding Layer**, **Multi-Head Self-Attention**, **Feed-Forward Networks**, **Residual Connections**, and **Layer Normalization**.  
   - All classes and modules are defined in `model.py` for modularity.

3. **Training & Hyperparameter Tuning**  
   - Multiple configurations tested for `d_model`, `num_heads`, `hidden_dim`, and `learning_rate`.  
   - Evaluation metrics (Accuracy, Precision, Recall, F1 Score) used to select the best configuration.  
   - Retrain final model using best parameters.  
   - Save `model.pth` and `vocab.pth`.

4. **Evaluation**  
   - Confusion Matrix visualization.  
   - Detailed metrics logged for analysis.  

5. **Deployment with FastAPI & HTML Front-End**  
   - FastAPI serves the trained model via API endpoints.  
   - HTML front-end allows users to input reviews and get real-time predictions.  
   - Probabilities displayed along with predicted labels.

---

## 🧰 Dependencies

- Python ≥ 3.8  
- PyTorch  
- pandas  
- scikit-learn  
- matplotlib  
- seaborn  
- FastAPI  
- Uvicorn  

Install all dependencies:

```bash
pip install torch pandas scikit-learn matplotlib seaborn fastapi uvicorn

.
├── model.py             # Transformer model classes (Embedding, Attention, FFN)
├── train_model.py       # Script for training and hyperparameter tuning
├── evaluate.py          # Evaluation function
├── predict.py           # Inference script for new reviews
├── app.py               # FastAPI backend
├── templates/
│   └── index.html       # Front-end HTML for user input
├── vocab.pth            # Saved vocabulary
├── model.pth            # Trained model weights
├── README.md            # Documentation
└── dataset/
    └── imdb_reviews.csv # Raw dataset

Data Preprocessing & Cleaning
1. Load Dataset
import pandas as pd
df = pd.read_csv("dataset/imdb_reviews.csv")

2.Remove Duplicates and NaNs
df.drop_duplicates(subset="review", inplace=True)
df.dropna(subset=["review", "label"], inplace=True)

3. Text Cleaning
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation/numbers
    return text

df["review"] = df["review"].apply(clean_text)

4. Tokenization & Vocabulary
from collections import Counter

def tokenize(text):
    return text.split()

counter = Counter()
for text in df["review"]:
    counter.update(tokenize(text))

vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(10000))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1
5. Convert Text to IDs & Padding
def encode(text, vocab, max_len=200):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    ids = ids[:max_len] + [vocab["<PAD>"]] * (max_len - len(ids))
    return ids
🏗 Model Implementation (model.py)

The Transformer model is implemented from scratch:

Embedding Layer: Converts token IDs to dense vectors.

Multi-Head Self-Attention: Captures relationships between tokens.

Feed-Forward Network (FFN): Processes attention outputs.

Residual Connections & Layer Normalization: Improves training stability.

Output Layer: Linear + Softmax for binary classification.

# Example structure
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, hidden_dim, num_layers, num_classes):
        super().__init__()
        # embedding, transformer layers, FFN
        # final linear layer to num_classes

All classes (Embedding, MultiHeadAttention, FFN, PositionalEncoding) are in model.py.

🏋️ Training & Hyperparameter Tuning
Hyperparameter Configurations
configs = [
    {"d_model":128, "num_heads":4, "hidden_dim":256, "lr":1e-4},
    {"d_model":256, "num_heads":4, "hidden_dim":512, "lr":1e-4},
    {"d_model":128, "num_heads":8, "hidden_dim":256, "lr":3e-4},
]
Training Script (train_model.py)

Train each configuration for a few epochs.

Evaluate using validation set.

Select the best configuration based on F1 Score.

Retrain the final model using the best parameters for more epochs.

Save Model & Vocab
torch.save(final_model.state_dict(), "model.pth")
torch.save(vocab, "vocab.pth")

📊 Evaluation
Metrics: Accuracy, Precision, Recall, F1 Score
Confusion matrix:
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
🤖 Inference

Use predict.py:
from predict import predict_sentiment
import torch
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

review = "This movie was fantastic!"
pred, prob = predict_sentiment(model, review, vocab, max_len=200, device=device)
print(f"Prediction: {pred}, Probabilities: {prob}")

Example Output

Review: "This movie was fantastic!"
Prediction: Positive
Probabilities: [0.05, 0.95]

🚀 Deployment with FastAPI & HTML Front-End
FastAPI backend (app.py)
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
import torch
from model import TransformerClassifier
from predict import predict_sentiment

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict_review(request: Request, review: str = Form(...)):
    pred, prob = predict_sentiment(model, review, vocab, max_len=200, device=device)
    return templates.TemplateResponse("index.html", {"request": request, "prediction": pred, "prob": prob})
HTML Front-End (templates/index.html)
<form method="post" action="/predict">
  <textarea name="review" placeholder="Enter your review here"></textarea>
  <button type="submit">Predict</button>
</form>
{% if prediction %}
<p>Prediction: {{ prediction }}</p>
<p>Probability: {{ prob }}</p>
{% endif %}
Run FastAPI Server
uvicorn app:app --reload

Access the app at: http://127.0.0.1:8000/.

📈 Results
Metric	Score
Accuracy	0.92
Precision	0.91
Recall	0.93
F1 Score	0.92
