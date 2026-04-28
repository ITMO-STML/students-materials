import os
import sys
from pathlib import Path
import numpy as np
import torch
import torchaudio
from torchaudio import functional as FA
import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from content_manager.vencoder.HubertSoft import HubertSoft
from Decoding_tools import viterbi_decode, viterbi_lookahead
from evaluation_tools import cer  # можно потом убрать, если не нужен
from DatasetsModels import CTCModel
from project_paths import DEFAULT_CTC_MODEL, DEFAULT_STRESS_MODEL

# ====== device ======
device = "cuda" if torch.cuda.is_available() else "cpu"

# ====== phoneme list ======
phoneme_list = ['a0', 'a4', 'b', "b'", 'c', 'ch', 'd', "d'", 'e0', 'f', "f'", 'g', 'h', 'i0','i4', 'j', 'k', "k'", 'l',
                "l'", 'm', "m'", 'n', "n'", 'o0', 'p', "p'", 'r', "r'", 's', "s'", 'sh', 't', "t'", 'u0', 'v', "v'",
                'y0', 'z', "z'", 'zh', 'sil', 'a1', 'i1','u1', 'y1', ]

# ====== model paths ======
TRIPLET_PATH = DEFAULT_STRESS_MODEL
CTC_PATH = DEFAULT_CTC_MODEL

# ====== model loading helper ======

def load_triplet_model(path):
    model = torch.load(path, map_location=device)
    model.eval().to(device)
    return model


def load_ctc_model(path, input_dim, hidden_dim, num_phonemes, num_layers=2):
    checkpoint = torch.load(path, map_location=device)
    model = CTCModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_phonemes=num_phonemes,
        num_layers=num_layers
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model



# ====== content encoder ======
content_encoder = HubertSoft(device=device)

# ====== utils ======

def extract_embeddings(wav_path):
    wave, sr = torchaudio.load(wav_path)
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)

    wave = FA.resample(wave, sr, 16000)

    with torch.no_grad():
        emb = content_encoder.encoder(wave[0].to(device))

    emb = emb[0]
    emb = emb / (emb ** 2).sum(0, keepdims=True).sqrt()
    return emb.T.cpu()

def ctc_greedy_decode(logits, input_lengths, idx2phoneme):
    """
    logits: [B, T, C]
    """
    preds = logits.argmax(-1)  # [B, T]
    results = []

    for seq, T in zip(preds, input_lengths):
        seq = seq[:T]

        decoded = []
        prev = 0  # blank

        for idx in seq.cpu().numpy():
            if idx != 0 and idx != prev:
                decoded.append(idx2phoneme[idx])
            prev = idx

        results.append(decoded)

    return results


def get_stresses(phonemes):
    stresses = []
    for phoneme in phonemes:
        if len(phoneme) == 2:
            if phoneme[1] in '124':
                stresses.append('-')
            elif phoneme[1] == '0':
                stresses.append('+')
    return stresses


def get_target_stresses(text):
    vowels = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
    stresses = []
    i = 0
    while i < len(text):
        if text[i] in vowels:
            if i + 1 < len(text) and text[i + 1] == "0":
                stresses.append("+")
                i += 2
            else:
                stresses.append("-")
                i += 1
        else:
            i += 1
    return stresses


# ====== inference ======

def run_inference(wav_path, model_type, target_text):
    if wav_path is None:
        return "No audio provided", "No audio provided"

    # Загрузка нужной модели
    if model_type == "triplet_model":
        model = load_triplet_model(TRIPLET_PATH)
        is_ctc = False
    elif model_type == "ctc_model":
        model = load_ctc_model(
            path=CTC_PATH,
            input_dim=256,          # подставь реальный input_dim
            hidden_dim=256,         # подставь hidden_dim
            num_phonemes=len(phoneme_list)
        )
        is_ctc = True
    else:
        return "Unknown model type", "Unknown model type"

    embs = extract_embeddings(wav_path)  # [T, D]

    probs = []
    preds = []
    phoneme2idx = {ph: i + 1 for i, ph in enumerate(phoneme_list)}
    idx2phoneme = {i + 1: ph for i, ph in enumerate(phoneme_list)}

    with torch.no_grad():
        x = embs.unsqueeze(0).to(device)  # [1, T, D]
        x_lens = torch.tensor([x.shape[1]], device=device)  # [1,]

        if is_ctc:
            logits = model(x)  # [1, T, V+1]

            pred_phonemes = ctc_greedy_decode(
                logits,
                x_lens,
                idx2phoneme
            )
            preds = pred_phonemes[0]  # список фонем для одного предложения
            phonemes = preds

        else:
            # Triplet-модель
            _, out_c, _ = model(x)
            pc = torch.softmax(out_c, dim=1)[0]  # [T, V]

            preds = []
            for t in range(pc.shape[0]):
                idx = pc[t].argmax().item()
                if idx < len(phoneme_list):
                    preds.append(phoneme_list[idx])
                else:
                    preds.append("sil")
                # Viterbi‑декодирование (под твою triplet‑логику, можно потом адаптировать под CTC)
                _, phonemes, boundaries = viterbi_lookahead({
                    "start": probs,
                    "center": probs,
                    "end": probs,
                })

    phrase = " ".join([p for i, p in enumerate(preds) if i == 0 or p != preds[i-1]])

    pred_stresses = get_stresses(phonemes)
    target_stresses = get_target_stresses(target_text)

    if len(pred_stresses) != len(target_stresses):
        stress_text = "Неправильное распознавание: количество ударений не совпадает"
    else:
        correct = sum(p == t for p, t in zip(pred_stresses, target_stresses))
        total = len(pred_stresses)
        stress_text = f"{correct} из {total} ударений на своих местах"

    return phrase, stress_text



# ====== UI ======

with gr.Blocks() as demo:
    gr.Markdown("## 🎧 Transcription Demo")

    with gr.Row():
        wav_input = gr.Audio(type="filepath", label="Input WAV")

    model_type = gr.Dropdown(
        label="Model type",
        choices=["triplet_model", "ctc_model"],
        value="triplet_model"
    )

    target_text = gr.Textbox(
        label="Target phrase (mark stresses with 0 after vowels)",
        placeholder="Например: пр0блема, м0локо",
    )

    run_btn = gr.Button("Run inference")

    phoneme_out = gr.Textbox(label="Predicted phoneme sequence")
    stress_out = gr.Textbox(label="Stress evaluation")

    run_btn.click(
        run_inference,
        inputs=[wav_input, model_type, target_text],
        outputs=[phoneme_out, stress_out]
    )

    demo.launch()
