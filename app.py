from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
from model_def import TransformerClassifier, tokenize
import os

app = FastAPI()

# ----------------------------
# Templates Setup
# ----------------------------
templates = Jinja2Templates(directory="templates")

# ----------------------------
# Device Setup
# ----------------------------
device = torch.device("cpu")

# Global variables (loaded at startup)
model = None
vocab = None

# ----------------------------
# Load Model on Startup
# ----------------------------
@app.on_event("startup")
def load_model():
    global model, vocab

    # Load vocab
    vocab = torch.load("vocab.pth", map_location=device)

    # IMPORTANT: Must match training configuration
    model_instance = TransformerClassifier(
        vocab_size=len(vocab),
        d_model=256,       # ✅ match training
        num_heads=4,
        hidden_dim=512,    # ✅ match training
        num_layers=2,
        num_classes=2
    ).to(device)

    # Load trained weights
    model_instance.load_state_dict(
        torch.load("model.pth", map_location=device)
    )

    model_instance.eval()

    model = model_instance
    print("✅ Model and vocab loaded successfully!")

# ----------------------------
# Helper Functions
# ----------------------------
def encode(text, max_len=200):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab.get("<UNK>", 0)) for token in tokens]
    ids = ids[:max_len]
    ids += [vocab.get("<PAD>", 0)] * (max_len - len(ids))
    return torch.tensor([ids]).to(device)

def get_sentiment_label(pred):
    return "Positive 😊" if pred == 1 else "Negative 😞"

# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": "",
            "input_text": ""
        }
    )

@app.post("/", response_class=HTMLResponse)
async def predict_sentiment(request: Request, user_input: str = Form(...)):
    input_tensor = encode(user_input)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()

    sentiment = get_sentiment_label(pred)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": sentiment,
            "input_text": user_input
        }
    )