import os

from pathlib import Path
from typing import Any, List, Tuple

import torch
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from model import CLSCrossAttentionFusion
from data import image_transform, download_data_if_needed

CKPT_PATH: Path = Path("checkpoints/cls_xattn.pt")
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model() -> Tuple[CLSCrossAttentionFusion, PreTrainedTokenizerBase, List[str]]:
    ckpt: dict[str, Any] = torch.load(CKPT_PATH, map_location=device)
    model: CLSCrossAttentionFusion = CLSCrossAttentionFusion(num_classes=len(ckpt["label_classes"]), num_genders=3)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return model, tokenizer, ckpt["label_classes"]

@torch.no_grad()
def predict(image_path: str, text: str, model: CLSCrossAttentionFusion, tokenizer: PreTrainedTokenizerBase, classes: List[str]) -> Tuple[str, float, str, float]:
    image: torch.Tensor = image_transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    
    text_inputs: Any = tokenizer([text], padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
    ids: torch.Tensor = text_inputs["input_ids"].to(device)
    mask: torch.Tensor = text_inputs["attention_mask"].to(device)

    logits_cat: torch.Tensor
    logits_gen: torch.Tensor
    logits_cat, logits_gen = model(image, ids, mask)
    
    probs_cat: torch.Tensor = torch.softmax(logits_cat, dim=1)[0]
    pred_cat: int = probs_cat.argmax().item()
    cat_label: str = classes[pred_cat]
    cat_prob: float = probs_cat[pred_cat].item()
    
    probs_gen: torch.Tensor = torch.softmax(logits_gen, dim=1)[0]
    pred_gen: int = probs_gen.argmax().item()
    gen_labels: List[str] = ["Men", "Women", "Unisex"]
    gen_label: str = gen_labels[pred_gen]
    gen_prob: float = probs_gen[pred_gen].item()
    
    return cat_label, cat_prob, gen_label, gen_prob

if __name__ == "__main__":
    download_data_if_needed()

    model: CLSCrossAttentionFusion
    tokenizer: PreTrainedTokenizerBase
    classes: List[str]
    model, tokenizer, classes = load_model()

    image_path: str = "data/3238.jpg" 
    text: str = "Round toed, black sports shoes with red accents"

    cat: str
    cat_p: float
    gen: str
    gen_p: float
    cat, cat_p, gen, gen_p = predict(image_path, text, model, tokenizer, classes)
    
    print(f"Категория: {cat} ({cat_p:.3f})")
    print(f"Пол:       {gen} ({gen_p:.3f})")