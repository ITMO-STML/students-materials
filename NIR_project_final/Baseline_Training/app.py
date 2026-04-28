import os
import sys
from pathlib import Path
import numpy as np
import torch
import torchaudio
from torchaudio import functional as FA
import gradio as gr
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from content_manager.vencoder.HubertSoft import HubertSoft
from Decoding_tools import viterbi_decode
from evaluation_tools import cer
from project_paths import DEFAULT_BASELINE_MODEL

# ====== device ======
device = "cuda" if torch.cuda.is_available() else "cpu"

# ====== phoneme list ======
phoneme_list = [
    'a0', 'a4', 'b', "b'", 'c', 'ch', 'd', "d'", 'e0', 'f', "f'",
    'g', 'h', 'i0', 'i4', 'j', 'k', "k'", 'l', "l'", 'm', "m'",
    'n', "n'", 'o0', 'p', "p'", 'r', "r'", 's', "s'", 'sh',
    't', "t'", 'u0', 'v', "v'", 'y0', 'z', "z'", 'zh', 'sil'
]

# ====== model ======
MODEL_PATH = DEFAULT_BASELINE_MODEL

model = torch.load(MODEL_PATH, map_location=device)
model.eval().to(device)

content_encoder = HubertSoft(device=device)

# ====== utils ======

def plot_waveform_with_vlines_dual_labels(y, sr, labels1, labels2):
    y = np.asarray(y).squeeze()
    duration = len(y) / sr
    t = np.linspace(0, duration, len(y))

    def add_ends(labels):
        out = []
        for i, (start, lab) in enumerate(labels):
            end = labels[i + 1][0] if i + 1 < len(labels) else duration
            out.append((start, end, lab))
        return out

    labels1 = add_ends(labels1)
    labels2 = add_ends(labels2)

    step = max(1, len(y) // 100_000)
    y_vis = y[::step]
    t_vis = t[::step]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_vis,
        y=y_vis,
        mode="lines",
        name="Waveform",
        hoverinfo="skip"
    ))

    min_y = float(np.min(y_vis))
    max_y = float(np.max(y_vis))
    offset = 0.05 * (max_y - min_y + 1e-8)

    def add_vlines(labels, color, legend_name, y_offset):
        first = True
        for start, end, lab in labels:
            fig.add_shape(
                type="line",
                x0=start,
                x1=start,
                y0=min_y,
                y1=max_y,
                line=dict(color=color, width=2),
                opacity=0.6
            )

            if first:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=legend_name
                ))
                first = False

            fig.add_annotation(
                x=start,
                y=max_y - y_offset,
                text=lab,
                showarrow=True,
                ay=-10,
                font=dict(size=12, color=color)
            )

    add_vlines(labels1, "#3B82F6", "Ground truth", 0)
    add_vlines(labels2, "#EF4444", "Prediction", offset)

    fig.update_layout(
        height=420,
        title="Waveform with phoneme boundaries",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        hovermode="x",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


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


def load_seg_b2(path):
    with open(path) as f:
        lines = f.readlines()

    sf = float(lines[1].split("=")[1])
    labels, raw = [], []

    for l in lines[7:]:
        start, _, lab = l.split(",")
        labels.append((float(start) / (sf * 2), lab.strip()))
        raw.append(lab.strip() or "sil")

    return labels, raw


# ====== inference ======

def run_inference(wav_path, use_gt, seg_b2):
    embs = extract_embeddings(wav_path)

    probs = []
    preds = []

    with torch.no_grad():
        for e in embs:
            _, out_c, _ = model(e.unsqueeze(0).to(device))
            pc = torch.softmax(out_c, dim=1)[0]
            preds.append(phoneme_list[pc.argmax().item()])
            probs.append({p: float(v) for p, v in zip(phoneme_list, pc.cpu())})

    phrase = " ".join([p for i, p in enumerate(preds) if i == 0 or p != preds[i-1]])

    _, phonemes, boundaries = viterbi_decode({
        "start": probs,
        "center": probs,
        "end": probs,
    })

    wave, sr = torchaudio.load(wav_path)
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0)

    pred_lab = [(b * 0.02, p) for b, p in zip([0] + boundaries, phonemes)]

    gt_lab, cer_text = [], "—"
    if use_gt and seg_b2 is not None:
        gt_lab, gt_seq = load_seg_b2(seg_b2.name)
        _, score = cer(gt_seq, phonemes, ignore_stress=True)
        cer_text = f"{score:.2f}"

    fig = plot_waveform_with_vlines_dual_labels(
        wave.numpy(), sr, gt_lab, pred_lab
    )

    return phrase, fig, cer_text


# ====== UI ======

with gr.Blocks() as demo:
    gr.Markdown("## 🎧 Transcription Demo")

    with gr.Row():
        wav_input = gr.Audio(type="filepath", label="Input WAV")

    use_gt = gr.Checkbox(label="Groundtruth segmentation")
    seg_input = gr.File(
        label="Upload seg_B2",
        file_types=[".seg", ".seg_B2"],
        visible=False
    )

    use_gt.change(lambda x: gr.update(visible=x), use_gt, seg_input)

    run_btn = gr.Button("Run inference")

    phoneme_out = gr.Textbox(label="Predicted phoneme sequence")
    waveform_out = gr.Plot(label="Waveform & segmentation")
    cer_out = gr.Textbox(label="CER")

    run_btn.click(
        run_inference,
        inputs=[wav_input, use_gt, seg_input],
        outputs=[phoneme_out, waveform_out, cer_out]
    )

    demo.launch()
