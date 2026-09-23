import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

from model import CLSCrossAttentionFusion
from data import image_transform


CKPT_PATH = Path("checkpoints/cls_xattn.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = CLSCrossAttentionFusion(num_classes=len(ckpt["label_classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return model, tokenizer, ckpt["label_classes"]


@torch.no_grad()
def predict(image_path, text, model, tokenizer, classes):
    image = image_transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text_inputs = tokenizer([text], padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
    ids = text_inputs["input_ids"].to(device)
    mask = text_inputs["attention_mask"].to(device)

    logits = model(image, ids, mask)
    probs = torch.softmax(logits, dim=1)[0]
    pred = probs.argmax().item()
    return classes[pred], probs[pred].item()


if __name__ == "__main__":
    model, tokenizer, classes = load_model()

    image_path = "data/example.jpg"
    text = "пушистый домашний питомец"

    label, prob = predict(image_path, text, model, tokenizer, classes)
    print(f"Prediction: {label} ({prob:.3f})")