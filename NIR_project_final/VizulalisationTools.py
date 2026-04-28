import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import plotly.graph_objects as go
import seaborn as sns

def chosen_phonemes_stats(embeddings, max_ph=40, max_trip=40):

    chosen_phonemes = []
    chosen_triplets = []
    for e in embeddings:
        chosen_phonemes.extend(e['label'])
        chosen_triplets.append("".join(e['label']))
    phonemes_stats = sorted((Counter(chosen_phonemes)).items(), key=lambda x: x[1], reverse=True)
    triplets_stats = sorted((Counter(chosen_triplets)).items(), key=lambda x: x[1], reverse=True)
    if len(phonemes_stats) > max_ph:
        phonemes_stats = phonemes_stats[:max_ph]
    if len(triplets_stats) > max_trip:
        triplets_stats = triplets_stats[:max_trip]

    plt.figure(figsize=(20, 30))
    plt.subplot(2, 1, 1)
    ph_labels, ph_values = zip(*phonemes_stats)
    plt.bar(ph_labels, ph_values)

    plt.subplot(2, 1, 2)
    tr_labels, tr_values = zip(*triplets_stats)
    plt.bar(tr_labels, tr_values)
    plt.xticks(rotation=90)

    plt.show()
    return phonemes_stats, triplets_stats

def plot_waveform_with_labels_plotly(y, sr, labels1, labels2):
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

    # немного прореживаем сигнал, чтобы plotly не тормозил
    step = max(1, len(y) // 100_000)
    y_vis = y[::step]
    t_vis = t[::step]

    fig = go.Figure()

    # --- осциллограмма ---
    fig.add_trace(go.Scatter(
        x=t_vis,
        y=y_vis,
        mode="lines",
        name="Waveform",
        line=dict(color="black"),
        hoverinfo="skip"
    ))

    amp = np.max(np.abs(y_vis))
    y1 = -1.3 * amp
    y2 = -1.6 * amp

    def add_tier(labels, y_level, color):
        for start, end, lab in labels:
            fig.add_shape(
                type="rect",
                x0=start,
                x1=end,
                y0=y_level - 0.05 * amp,
                y1=y_level + 0.05 * amp,
                fillcolor=color,
                opacity=0.7,
                line_width=0,
            )

            fig.add_annotation(
                x=(start + end) / 2,
                y=y_level,
                text=lab,
                showarrow=False,
                font=dict(size=12, color="white")
            )

    # две разметки
    add_tier(labels1, y1, "royalblue")
    add_tier(labels2, y2, "firebrick")

    fig.update_layout(
        height=450,
        title="Waveform with phoneme annotations",
        xaxis_title="Time (s)",
        yaxis=dict(visible=False),
        hovermode="x",
    )

    fig.show()



def plot_waveform_with_labels(y, sr, labels1, labels2):
    duration = len(y) / sr
    t = np.linspace(0, duration, len(y))

    def add_ends(labels):
        result = []
        for i, (start, lab) in enumerate(labels):
            end = labels[i + 1][0] if i + 1 < len(labels) else duration
            result.append((start, end, lab))
        return result

    labels1 = add_ends(labels1)
    labels2 = add_ends(labels2)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    # --- Осциллограмма ---
    axes[0].plot(t, y, color="black")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Waveform with phoneme annotations")

    # --- Первая разметка ---
    for start, end, lab in labels1:
        axes[1].axvspan(start, end, alpha=0.6)
        axes[1].text(
            (start + end) / 2,
            0.5,
            lab,
            ha="center",
            va="center"
        )

    axes[1].set_ylim(0, 1)
    axes[1].set_yticks([])
    axes[1].set_ylabel("Annot 1")

    # --- Вторая разметка ---
    for start, end, lab in labels2:
        axes[2].axvspan(start, end, alpha=0.6)
        axes[2].text(
            (start + end) / 2,
            0.5,
            lab,
            ha="center",
            va="center"
        )

    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    axes[2].set_ylabel("Annot 2")
    axes[2].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()
    return fig


def plot_waveform_with_vlines_dual_labels(y, sr, labels1, labels2):
    """
    y      : numpy array, моно аудио
    sr     : sample rate
    labels1: список (start_time, label)
    labels2: список (start_time, label)
    """
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

    # прореживание сигнала, чтобы не тормозило
    step = max(1, len(y) // 100_000)
    y_vis = y[::step]
    t_vis = t[::step]

    fig = go.Figure()

    # --- осциллограмма ---
    fig.add_trace(go.Scatter(
        x=t_vis,
        y=y_vis,
        mode="lines",
        name="Waveform",
        line=dict(color="white"),
        hoverinfo="skip"
    ))

    min_y = min(y_vis)
    max_y = max(y_vis)
    offset = 0.05 * (max_y - min_y)  # смещение для второго ряда

    # --- вертикальные линии + подписи ---
    def add_vlines(labels, color, legend_name, y_offset):
        # добавим линию один раз в легенду
        first_line = True
        for start, end, lab in labels:
            fig.add_shape(
                type="line",
                x0=start,
                x1=start,
                y0=min_y,
                y1=max_y,
                line=dict(color=color, width=2),
                opacity=0.6,
                name=legend_name if first_line else None  # только одна запись в легенду
            )
            first_line = False

            # подпись над линией
            fig.add_annotation(
                x=start,
                y=max_y - y_offset,
                text=lab,
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-10,
                font=dict(size=12, color=color)
            )

    # первая разметка — подписи сверху
    add_vlines(labels1, "royalblue", "Annot 1", y_offset=0)
    # вторая разметка — подписи чуть ниже
    add_vlines(labels2, "firebrick", "Annot 2", y_offset=offset)

    fig.update_layout(
        height=450,
        title="Waveform with phoneme boundaries",
        xaxis_title="Time (s)",
        yaxis=dict(title="Amplitude"),
        hovermode="x",
        legend=dict(y=0.95, x=0.95)
    )

    fig.show()
    return fig

def plot_triplet_distribution(prob_distributions, phoneme_list, threshold=0.01, max_examples=50):

    palette = sns.color_palette("tab20", len(phoneme_list))
    phoneme_colors = {phoneme: palette[i % len(palette)] for i, phoneme in enumerate(phoneme_list)}

    fig, ax = plt.subplots(figsize=(15, 6))

    bar_width = 0.2
    inner_gap = 0.05
    group_gap = 0.5

    x_ticks = []
    x_labels = []

    x_pos = 0

    for ex_idx in range(min(len(prob_distributions['start']), max_examples)):
        # for ex_idx in range(26, 39):
        for j, part in enumerate(['start', 'centre', 'end']):
            dist = prob_distributions[part][ex_idx]
            dist = [(p, v) for p, v in dist if v > threshold]
            total = sum(v for _, v in dist)
            dist = [(p, v / total) for p, v in dist if total > 0]

            bottom = 0
            for phoneme, prob in dist:
                ax.bar(x_pos, prob, width=bar_width, bottom=bottom,
                       color=phoneme_colors[phoneme], label=phoneme)
                bottom += prob

            x_ticks.append(x_pos)
            x_labels.append(f'{ex_idx}_{part[0]}')

            x_pos += bar_width + inner_gap

        x_pos += group_gap

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_ylabel("Normalized Probability")
    ax.set_title("Phoneme Distributions per Prediction (Start / Centre / End)")

    plt.tight_layout()
    plt.show()


def cer_detailed_viz(cers, subs_list, ins_list, dels_list):

    result = {"PER": cers,
              'Замены': subs_list,
              'Вставки': ins_list,
              'Удаления': dels_list,
              }
    labels = list(result.keys())
    data = list(result.values())

    plt.style.use('default')
    plt.figure(figsize=(10, 8), facecolor='white', edgecolor='k')
    plt.boxplot(data, labels=labels)

    plt.ylabel('Значения')

    plt.yscale('symlog', linthresh=80)
    plt.title('')
    plt.plot()
    return result