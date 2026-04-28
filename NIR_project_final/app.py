import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torchaudio
from torchaudio import functional as FA

from CTC_models.CTC_tools import AttentionPhonemeModel, ctc_greedy_decode
from DatasetsModels import CTCModel, ClassificationModel
from Decoding_tools import viterbi_decode
from ToIPA import corpres_to_ipa_symbol, corpres2ipa, panphon_features
from content_manager.vencoder.HubertSoft import HubertSoft
from evaluation_tools import cer
from project_paths import (
    DEFAULT_BASELINE_MODEL,
    DEFAULT_CTC_ATTENTION_MODEL,
    DEFAULT_CTC_MODEL,
    EXAMPLE_PHONEMES,
    EXAMPLE_SEG_B2,
    EXAMPLE_WAV,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_HOP_SECONDS = 0.02
BASELINE_PHONEMES = [
    "a0", "a4", "b", "b'", "c", "ch", "d", "d'", "e0", "f", "f'",
    "g", "h", "i0", "i4", "j", "k", "k'", "l", "l'", "m", "m'",
    "n", "n'", "o0", "p", "p'", "r", "r'", "s", "s'", "sh",
    "t", "t'", "u0", "v", "v'", "y0", "z", "z'", "zh", "sil",
]
STRESS_PHONEMES = BASELINE_PHONEMES[:-1] + ["sil", "a1", "i1", "u1", "y1"]
FULL_CTC_PHONEMES = [
    "a0", "a1", "a2", "a4", "b", "b'", "c", "ch", "ch_", "d", "d'",
    "e0", "e1", "e2", "e4", "f", "f'", "g", "g'", "h", "h'",
    "i0", "i1", "i2", "i4", "j", "jr", "jl", "ji4", "k", "k'",
    "l", "l'", "m", "m'", "n", "n'", "o0", "o1", "o2", "o4",
    "p", "p'", "r", "r'", "s", "s'", "sc", "sh", "t", "t'",
    "u0", "u1", "u2", "u4", "v", "v'", "y0", "y1", "y2", "y4",
    "z", "z'", "zh", "zh'", "C", "CH", "H", "SC", "sil",
]
MODEL_CACHE = {}  # type: Dict[str, Any]


def example_target_text() -> str:
    if not EXAMPLE_PHONEMES.exists():
        return ""

    centers = []
    prev = None
    with EXAMPLE_PHONEMES.open(encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            parts = token.split("_")
            center = parts[1] if len(parts) == 3 else token
            if center != prev:
                centers.append(center)
            prev = center
    return " ".join(centers)


def pick_inventory(num_phonemes: int) -> List[str]:
    if num_phonemes == len(BASELINE_PHONEMES):
        return BASELINE_PHONEMES
    if num_phonemes == len(STRESS_PHONEMES):
        return STRESS_PHONEMES
    if num_phonemes == len(FULL_CTC_PHONEMES):
        return FULL_CTC_PHONEMES
    if num_phonemes < len(STRESS_PHONEMES):
        return STRESS_PHONEMES[:num_phonemes]
    return [f"ph_{idx}" for idx in range(num_phonemes)]


def parse_target_sequence(text: str) -> List[str]:
    if not text:
        return []
    return [token for token in re.split(r"[\s,;]+", text.strip()) if token]


def dedupe_consecutive(tokens: List[str]) -> List[str]:
    result = []
    prev = None
    for token in tokens:
        if token != prev:
            result.append(token)
        prev = token
    return result


def safe_corpres_to_ipa(token: str) -> str:
    try:
        return corpres_to_ipa_symbol(token)
    except Exception:
        return token


def sequence_to_ipa_text(tokens: List[str], kind: str = "corpres") -> str:
    if not tokens:
        return ""
    if kind == "ipa":
        return " ".join(tokens)
    return corpres2ipa(tokens)


def sequence_feature_matrix(tokens: List[str], kind: str = "corpres") -> np.ndarray:
    rows = []
    for token in tokens:
        ipa_symbol = token if kind == "ipa" else safe_corpres_to_ipa(token)
        features = panphon_features(ipa_symbol)
        if features:
            rows.append(features[0].numeric())
        else:
            rows.append([0] * 24)
    if not rows:
        return np.zeros((0, 24), dtype=int)
    return np.asarray(rows, dtype=int)


def placeholder_figure(title: str, text: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, showarrow=False, font=dict(size=16))
    fig.update_layout(title=title, xaxis=dict(visible=False), yaxis=dict(visible=False), height=360)
    return fig


def build_heatmap(
    z: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    colorscale: str = "RdBu",
) -> go.Figure:
    if z.size == 0 or not x_labels or not y_labels:
        return placeholder_figure(title, "No data")

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            colorbar=dict(title="value"),
        )
    )
    fig.update_layout(title=title, xaxis_title="Position", yaxis_title="Label", height=420)
    return fig


def build_feature_heatmap(tokens: List[str], kind: str, title: str) -> go.Figure:
    matrix = sequence_feature_matrix(tokens, kind=kind)
    labels = []
    for token in tokens:
        ipa = token if kind == "ipa" else safe_corpres_to_ipa(token)
        labels.append(f"{token} | {ipa}")
    return build_heatmap(
        matrix,
        [str(idx) for idx in range(matrix.shape[1])],
        labels,
        title,
        colorscale="Viridis",
    )


def build_posterior_heatmap(
    probabilities: np.ndarray,
    labels: List[str],
    title: str,
) -> go.Figure:
    if probabilities.size == 0:
        return placeholder_figure(title, "No posterior probabilities")
    x_labels = [str(idx) for idx in range(probabilities.shape[0])]
    return build_heatmap(
        probabilities.T,
        x_labels,
        labels,
        title,
        colorscale="Magma",
    )


def build_attention_heatmap(attention: np.ndarray, decoded_tokens: List[str], title: str) -> go.Figure:
    if attention.size == 0:
        return placeholder_figure(title, "No attention weights")
    x_labels = [str(idx) for idx in range(attention.shape[1])]
    y_labels = [f"{idx}: {token}" for idx, token in enumerate(decoded_tokens, start=1)]
    return build_heatmap(attention, x_labels, y_labels, title, colorscale="Blues")


def load_seg_b2(path: Any) -> Tuple[List[Tuple[float, str]], List[str]]:
    with Path(path).open(encoding="utf-8") as f:
        lines = f.readlines()

    sampling_freq = float(lines[1].split("=")[1])
    labels, raw = [], []
    for line in lines[7:]:
        start, _, label = line.split(",")
        labels.append((float(start) / (sampling_freq * 2), label.strip()))
        raw.append(label.strip() or "sil")
    return labels, raw


def resolve_file_path(file_value: Any) -> Optional[str]:
    if not file_value:
        return None
    if isinstance(file_value, (str, Path)):
        return str(file_value)
    if hasattr(file_value, "name"):
        return str(file_value.name)
    return None


def plot_waveform_with_labels(
    waveform: np.ndarray,
    sample_rate: int,
    target_labels: List[Tuple[float, str]],
    predicted_labels: List[Tuple[float, str]],
    title: str,
) -> go.Figure:
    waveform = np.asarray(waveform).squeeze()
    if waveform.size == 0:
        return placeholder_figure(title, "No waveform")

    duration = len(waveform) / sample_rate
    times = np.linspace(0, duration, len(waveform))
    step = max(1, len(waveform) // 100_000)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times[::step],
            y=waveform[::step],
            mode="lines",
            name="Waveform",
            hoverinfo="skip",
        )
    )

    min_y = float(np.min(waveform[::step]))
    max_y = float(np.max(waveform[::step]))
    offset = 0.06 * (max_y - min_y + 1e-8)

    def add_markers(labels: List[Tuple[float, str]], color: str, name: str, y_shift: float) -> None:
        for idx, (start, token) in enumerate(labels):
            fig.add_shape(
                type="line",
                x0=start,
                x1=start,
                y0=min_y,
                y1=max_y,
                line=dict(color=color, width=2),
                opacity=0.65,
            )
            fig.add_annotation(
                x=start,
                y=max_y - y_shift,
                text=token,
                showarrow=True,
                ay=-10,
                font=dict(size=11, color=color),
            )
            if idx == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(color=color, width=2),
                        name=name,
                    )
                )

    if target_labels:
        add_markers(target_labels, "#2563EB", "Ground truth", 0.0)
    if predicted_labels:
        add_markers(predicted_labels, "#DC2626", "Prediction", offset)

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        hovermode="x",
        template="plotly_dark",
        height=420,
        width=1000,
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig


def extract_embeddings(wav_path: str) -> torch.Tensor:
    cache_key = "content_encoder"
    if cache_key not in MODEL_CACHE:
        MODEL_CACHE[cache_key] = HubertSoft(device=DEVICE)

    content_encoder = MODEL_CACHE[cache_key]
    wave, sample_rate = torchaudio.load(wav_path)
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    wave = FA.resample(wave, sample_rate, 16000)

    with torch.no_grad():
        embeddings = content_encoder.encoder(wave[0].to(DEVICE))

    embeddings = embeddings[0]
    embeddings = embeddings / (embeddings ** 2).sum(0, keepdims=True).sqrt()
    return embeddings.T.cpu()


def load_baseline_model() -> Tuple[Any, List[str]]:
    if "baseline_model" in MODEL_CACHE:
        return MODEL_CACHE["baseline_model"], MODEL_CACHE["baseline_inventory"]

    loaded = torch.load(DEFAULT_BASELINE_MODEL, map_location=DEVICE)
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state = loaded["model_state_dict"]
        input_dim = state["linear1.weight"].shape[1]
        num_classes = state["final_center.weight"].shape[0]
        model = ClassificationModel(inputsize=input_dim, num_classes=num_classes, dropout=0.0)
        model.load_state_dict(state)
    else:
        model = loaded

    inventory = pick_inventory(model.state_dict()["final_center.weight"].shape[0])
    model.eval().to(DEVICE)
    MODEL_CACHE["baseline_model"] = model
    MODEL_CACHE["baseline_inventory"] = inventory
    return model, inventory


def infer_ctc_num_layers(state_dict: Dict[str, torch.Tensor], prefix: str = "lstm") -> int:
    layer_ids = set()
    pattern = re.compile(rf"^{prefix}\.weight_ih_l(\d+)(?:_reverse)?$")
    for key in state_dict:
        match = pattern.match(key)
        if match:
            layer_ids.add(int(match.group(1)))
    return max(layer_ids) + 1 if layer_ids else 2


def load_ctc_model() -> Tuple[CTCModel, List[str]]:
    if "ctc_model" in MODEL_CACHE:
        return MODEL_CACHE["ctc_model"], MODEL_CACHE["ctc_inventory"]

    checkpoint = torch.load(DEFAULT_CTC_MODEL, map_location=DEVICE)
    state = checkpoint["model_state_dict"]
    hidden_dim = state["lstm.weight_ih_l0"].shape[0] // 4
    input_dim = state["lstm.weight_ih_l0"].shape[1]
    num_phonemes = state["fc.weight"].shape[0] - 1
    num_layers = infer_ctc_num_layers(state, prefix="lstm")

    model = CTCModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_phonemes=num_phonemes,
        num_layers=num_layers,
    )
    model.load_state_dict(state)
    model.eval().to(DEVICE)

    inventory = pick_inventory(num_phonemes)
    MODEL_CACHE["ctc_model"] = model
    MODEL_CACHE["ctc_inventory"] = inventory
    return model, inventory


def load_attention_model() -> Tuple[AttentionPhonemeModel, List[str]]:
    if "attention_model" in MODEL_CACHE:
        return MODEL_CACHE["attention_model"], MODEL_CACHE["attention_inventory"]

    checkpoint = torch.load(DEFAULT_CTC_ATTENTION_MODEL, map_location=DEVICE)
    state = checkpoint["model_state_dict"]

    hidden_dim = state["encoder.weight_ih_l0"].shape[0] // 4
    input_dim = state["encoder.weight_ih_l0"].shape[1]
    phoneme_emb_dim = state["phoneme_embedding.weight"].shape[1]
    num_classes = state["phoneme_embedding.weight"].shape[0]
    num_phonemes = num_classes - 1
    decoder_dim = state["decoder_cell.weight_hh"].shape[1]
    attention_dim = state["encoder_proj.weight"].shape[0]
    num_layers = infer_ctc_num_layers(state, prefix="encoder")

    model = AttentionPhonemeModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_phonemes=num_phonemes,
        num_layers=num_layers,
        phoneme_emb_dim=phoneme_emb_dim,
        decoder_dim=decoder_dim,
        attention_dim=attention_dim,
        pad_idx=0,
    )
    model.load_state_dict(state)
    model.eval().to(DEVICE)

    inventory = pick_inventory(num_phonemes)
    MODEL_CACHE["attention_model"] = model
    MODEL_CACHE["attention_inventory"] = inventory
    return model, inventory


def collapse_ctc_path(frame_ids: List[int], inventory: List[str]) -> Tuple[List[str], List[int], List[str]]:
    collapsed = []
    boundaries = []
    frame_tokens = []
    prev_idx = None

    for frame_idx, idx in enumerate(frame_ids):
        token = "<blank>" if idx == 0 else inventory[idx - 1]
        frame_tokens.append(token)
        if idx != 0 and idx != prev_idx:
            collapsed.append(token)
            boundaries.append(frame_idx)
        prev_idx = idx

    return collapsed, boundaries, frame_tokens


def alignment_labels(boundaries: List[int], tokens: List[str]) -> List[Tuple[float, str]]:
    return [(boundary * FRAME_HOP_SECONDS, token) for boundary, token in zip(boundaries, tokens)]


def compute_cer(gt_sequence: List[str], pred_sequence: List[str]) -> str:
    if not gt_sequence or not pred_sequence:
        return "—"
    _, score = cer(gt_sequence, pred_sequence, ignore_stress=True)
    return f"{score:.2f}"


def compute_boundary_error(
    gt_labels: List[Tuple[float, str]],
    pred_labels: List[Tuple[float, str]],
) -> str:
    if not gt_labels or not pred_labels:
        return "—"

    gt_times = [start for start, _ in gt_labels]
    pred_times = [start for start, _ in pred_labels]
    usable = min(len(gt_times), len(pred_times))
    if usable <= 1:
        return "—"

    gt_boundaries = np.asarray(gt_times[1:usable], dtype=float)
    pred_boundaries = np.asarray(pred_times[1:usable], dtype=float)
    if gt_boundaries.size == 0 or pred_boundaries.size == 0:
        return "—"

    mae_ms = float(np.mean(np.abs(gt_boundaries - pred_boundaries)) * 1000.0)
    return f"{mae_ms:.1f} ms"


def run_baseline_inference(embeddings: torch.Tensor) -> Dict[str, Any]:
    model, inventory = load_baseline_model()
    probabilities = []
    frame_preds = []

    with torch.no_grad():
        for emb in embeddings:
            _, out_center, _ = model(emb.unsqueeze(0).to(DEVICE))
            probs = torch.softmax(out_center, dim=1)[0].cpu().numpy()
            probabilities.append(probs)
            frame_preds.append(inventory[int(np.argmax(probs))])

    probability_dist = {
        "start": [{label: float(prob[idx]) for idx, label in enumerate(inventory)} for prob in probabilities],
        "center": [{label: float(prob[idx]) for idx, label in enumerate(inventory)} for prob in probabilities],
        "end": [{label: float(prob[idx]) for idx, label in enumerate(inventory)} for prob in probabilities],
    }
    _, sequence, boundaries = viterbi_decode(probability_dist)

    return {
        "inventory": inventory,
        "probabilities": np.asarray(probabilities),
        "frame_tokens": frame_preds,
        "sequence": sequence,
        "boundaries": boundaries,
        "labels": alignment_labels([0] + boundaries, sequence),
        "ipa": sequence_to_ipa_text(sequence),
    }


def run_ctc_inference(embeddings: torch.Tensor) -> Dict[str, Any]:
    model, inventory = load_ctc_model()
    x = embeddings.unsqueeze(0).to(DEVICE)
    input_lengths = torch.tensor([x.shape[1]], device=DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    decoded = ctc_greedy_decode(logits, input_lengths, {idx + 1: ph for idx, ph in enumerate(inventory)})[0]
    frame_ids = logits.argmax(-1)[0].detach().cpu().tolist()
    collapsed, boundaries, frame_tokens = collapse_ctc_path(frame_ids, inventory)

    return {
        "inventory": ["<blank>"] + inventory,
        "probabilities": probs,
        "frame_tokens": frame_tokens,
        "sequence": decoded if decoded else collapsed,
        "boundaries": boundaries,
        "labels": alignment_labels([0] + boundaries, decoded if decoded else collapsed),
        "ipa": sequence_to_ipa_text(decoded if decoded else collapsed),
    }


def greedy_attention_decode(model: AttentionPhonemeModel, x: torch.Tensor, inventory: List[str]) -> Tuple[List[str], np.ndarray]:
    input_lengths = torch.tensor([x.shape[1]], device=DEVICE)
    decoder_inputs = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    tokens = []
    attention_rows = []
    max_steps = max(1, min(x.shape[1], 200))

    with torch.no_grad():
        for step in range(max_steps):
            logits, stop_logits, attention, _ = model(x, decoder_inputs, input_lengths)
            step_logits = logits[:, -1, :]
            step_attention = attention[:, -1, :]
            next_idx = int(step_logits.argmax(dim=-1).item())
            stop_prob = float(torch.sigmoid(stop_logits[:, -1]).item())

            if next_idx != 0:
                tokens.append(inventory[next_idx - 1])
                attention_rows.append(step_attention[0].detach().cpu().numpy())

            if stop_prob > 0.5 and step > 0:
                break

            next_token = torch.tensor([[next_idx]], dtype=torch.long, device=DEVICE)
            decoder_inputs = torch.cat([decoder_inputs, next_token], dim=1)

    if attention_rows:
        attention_matrix = np.stack(attention_rows, axis=0)
    else:
        attention_matrix = np.zeros((0, x.shape[1]), dtype=float)

    return tokens, attention_matrix


def run_attention_inference(embeddings: torch.Tensor) -> Dict[str, Any]:
    model, inventory = load_attention_model()
    x = embeddings.unsqueeze(0).to(DEVICE)
    decoded_tokens, attention = greedy_attention_decode(model, x, inventory)

    if attention.size:
        peak_frames = np.argmax(attention, axis=1).tolist()
        monotonic_frames = np.maximum.accumulate(peak_frames).tolist()
    else:
        monotonic_frames = []

    return {
        "sequence": decoded_tokens,
        "boundaries": monotonic_frames,
        "labels": alignment_labels([0] + monotonic_frames, decoded_tokens),
        "attention": attention,
        "ipa": sequence_to_ipa_text(decoded_tokens),
    }


def waveform_and_target(wav_path: str, seg_path: Optional[str]) -> Tuple[np.ndarray, int, List[Tuple[float, str]], List[str]]:
    wave, sample_rate = torchaudio.load(wav_path)
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0)
    else:
        wave = wave[0]

    gt_labels, gt_seq = [], []
    if seg_path:
        gt_labels, gt_seq = load_seg_b2(seg_path)

    return wave.numpy(), sample_rate, gt_labels, gt_seq


def make_model_outputs(
    name: str,
    result: Dict[str, Any],
    waveform: np.ndarray,
    sample_rate: int,
    gt_labels: List[Tuple[float, str]],
    gt_seq: List[str],
    alignment_mode: str,
    include_alignment: bool,
) -> Tuple[str, str, str, str, go.Figure, go.Figure]:
    predicted_seq = result.get("sequence", [])
    transcription = " ".join(predicted_seq)
    ipa_text = result["ipa"]
    cer_text = compute_cer(gt_seq, result.get("sequence", []))
    boundary_error_text = compute_boundary_error(gt_labels, result.get("labels", []))

    waveform_fig = plot_waveform_with_labels(
        waveform,
        sample_rate,
        gt_labels,
        result.get("labels", []),
        f"{name}: waveform and alignment",
    )

    if not include_alignment:
        alignment_fig = placeholder_figure(f"{name}: alignment", "Alignment map is shown only for CTC + Attention")
    elif alignment_mode == "attention":
        alignment_fig = build_attention_heatmap(
            result.get("attention", np.zeros((0, 0))),
            result.get("sequence", []),
            f"{name}: attention map",
        )
    else:
        alignment_fig = build_posterior_heatmap(
            result.get("probabilities", np.zeros((0, 0))),
            result.get("inventory", []),
            f"{name}: posterior alignment map",
        )

    return transcription, ipa_text, cer_text, boundary_error_text, waveform_fig, alignment_fig


def run_all_models(
    wav_path: str,
    use_gt: bool,
    seg_path: Any,
    target_transcription: str,
) -> Tuple[Any, ...]:
    if not wav_path:
        empty = placeholder_figure("No input", "Upload a WAV file")
        empty_summary = pd.DataFrame([{"model": "status", "transcription": "No file", "ipa": "", "cer": ""}])
        return (empty_summary, "", *("—", "—", "—", "—", empty, empty) * 3)

    embeddings = extract_embeddings(wav_path)
    seg_file_path = resolve_file_path(seg_path) if use_gt else None
    waveform, sample_rate, gt_labels, gt_seq = waveform_and_target(wav_path, seg_file_path)
    target_tokens = parse_target_sequence(target_transcription)
    target_ipa = sequence_to_ipa_text(target_tokens) if target_tokens else ""

    baseline_result = run_baseline_inference(embeddings)
    ctc_result = run_ctc_inference(embeddings)
    attention_result = run_attention_inference(embeddings)

    summary = pd.DataFrame(
        [
            {
                "model": "baseline",
                "transcription": " ".join(baseline_result["sequence"]),
                "ipa": baseline_result["ipa"],
                "cer": compute_cer(gt_seq, baseline_result["sequence"]) if use_gt else "—",
            },
            {
                "model": "ctc",
                "transcription": " ".join(ctc_result["sequence"]),
                "ipa": ctc_result["ipa"],
                "cer": compute_cer(gt_seq, ctc_result["sequence"]) if use_gt else "—",
            },
            {
                "model": "ctc_attention",
                "transcription": " ".join(attention_result["sequence"]),
                "ipa": attention_result["ipa"],
                "cer": compute_cer(gt_seq, attention_result["sequence"]) if use_gt else "—",
            },
        ]
    )

    baseline_outputs = make_model_outputs(
        "Baseline",
        baseline_result,
        waveform,
        sample_rate,
        gt_labels,
        gt_seq,
        alignment_mode="posterior",
        include_alignment=False,
    )
    ctc_outputs = make_model_outputs(
        "CTC",
        ctc_result,
        waveform,
        sample_rate,
        gt_labels,
        gt_seq,
        alignment_mode="posterior",
        include_alignment=False,
    )
    attention_outputs = make_model_outputs(
        "CTC + Attention",
        attention_result,
        waveform,
        sample_rate,
        gt_labels,
        gt_seq,
        alignment_mode="attention",
        include_alignment=True,
    )

    return (summary, target_ipa, *baseline_outputs, *ctc_outputs, *attention_outputs)


EXAMPLE_TARGET = example_target_text()


with gr.Blocks(title="Phoneme Transcription Workbench") as demo:
    gr.Markdown(
        """
        # Phoneme Transcription Workbench
        Upload a WAV file and compare baseline, CTC, and CTC with attention.
        You can also provide a target transcription in corpres notation to see its IPA conversion.
        """
    )

    with gr.Row():
        wav_input = gr.Audio(type="filepath", label="Input WAV")
        with gr.Column():
            use_gt = gr.Checkbox(label="Use ground-truth segmentation", value=False)
            seg_input = gr.File(
                label="seg_B2 file",
                type="file",
                file_types=[".seg", ".seg_B2"],
                visible=False,
            )
            target_text = gr.Textbox(
                label="Target transcription (corpres tokens, space-separated)",
                placeholder="Example: t a1 g d a0 d' e0 t a1 b j i1 s' n' i0 l sil",
                lines=3,
            )

    use_gt.change(lambda checked: gr.update(visible=checked), use_gt, seg_input)

    gr.Examples(
        examples=[[str(EXAMPLE_WAV), True, str(EXAMPLE_SEG_B2), EXAMPLE_TARGET]],
        inputs=[wav_input, use_gt, seg_input, target_text],
        label="Example",
    )

    run_btn = gr.Button("Run all models", variant="primary")

    summary_df = gr.Dataframe(label="Summary", interactive=False)
    target_ipa_box = gr.Textbox(label="Target IPA")

    def model_tab(title: str, show_alignment: bool) -> Tuple[Any, ...]:
        with gr.Tab(title):
            text = gr.Textbox(label="Predicted transcription")
            ipa = gr.Textbox(label="Predicted IPA")
            metric = gr.Textbox(label="CER")
            boundary_error = gr.Textbox(label="Boundary Error")
            waveform_fig = gr.Plot(label="Waveform and segmentation")
            alignment_fig = gr.Plot(
                label="Alignment map" if show_alignment else "Info",
                visible=True,
            )
        return text, ipa, metric, boundary_error, waveform_fig, alignment_fig

    with gr.Tabs():
        baseline_components = model_tab("Baseline", show_alignment=False)
        ctc_components = model_tab("CTC", show_alignment=False)
        attention_components = model_tab("CTC + Attention", show_alignment=True)

    run_btn.click(
        run_all_models,
        inputs=[wav_input, use_gt, seg_input, target_text],
        outputs=[
            summary_df,
            target_ipa_box,
            *baseline_components,
            *ctc_components,
            *attention_components,
        ],
    )


if __name__ == "__main__":
    demo.launch()
