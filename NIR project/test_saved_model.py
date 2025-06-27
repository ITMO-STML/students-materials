import random
import torchaudio
from torchaudio import functional as FA
import torch
from content_manager.vencoder.HubertSoft import HubertSoft
import torch.nn as nn
from evaluation_utils import _accuracy_end, _accuracy_start, _accuracy_centre, _accuracy_ordered, _accuracy_unordered, _accuracy_centre_flex, _accuracy_ordered_flex_centre

def extract_embeddings_shifted_by_padding(file_path: str,
                                          model,
                                          device: torch.device,
                                          target_sr: int = 16000,
                                          base_step_ms: int = 20,
                                          desired_step_ms: int = 1) -> torch.Tensor:
    """
    Извлекает эмбеддинги с меньшим шагом (например, 1 мс), сдвигая сигнал через паддинг.
    """
    # 1. Загрузка и ресемплирование
    wave, sr = torchaudio.load(file_path)
    res_wave = FA.resample(wave, sr, target_sr)

    # 2. Подготовка смещённых копий
    shift_samples = int(target_sr * desired_step_ms / 1000)
    n_shifts = base_step_ms // desired_step_ms

    embs_per_shift = []

    for i in range(n_shifts):
        pad_left = torch.zeros((1, shift_samples * i), dtype=res_wave.dtype)
        padded_wave = torch.cat([pad_left, res_wave], dim=1)

        with torch.no_grad():
            ext_emb = model.encoder(padded_wave[0].to(device))
            emb = ext_emb[0]  # [T, D]

        # L2 нормализация каждого вектора
        emb = emb / (emb.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-9)
        emb = emb.T

        embs_per_shift.append(emb)

    # 3. Усечение всех эмбеддингов до min длины
    min_len = min(e.shape[0] for e in embs_per_shift)
    embs_per_shift = [e[:min_len] for e in embs_per_shift]

    # 4. Складываем эмбеддинги в нужном порядке: [n_shifts, T, D] -> [T * n_shifts, D]
    stacked = torch.stack(embs_per_shift)  # [n_shifts, T, D]
    result = stacked.permute(1, 0, 2).reshape(-1, stacked.shape[2])  # [T * n_shifts, D]

    return result

def build_y_true_from_labels(labels, total_ms, step_ms=1):
    """
    Строит y_true с триплетами start / centre / end для заданного шага эмбеддингов.

    :param labels: список словарей вида [{"phoneme": end_sec}, ...]
    :param total_ms: общая длительность аудио в миллисекундах
    :param step_ms: шаг, с которым извлекались эмбеддинги (в мс)
    :return: словарь с 'start', 'centre', 'end'
    """
    # Преобразуем [{'phoneme': time_sec}, ...] -> [(phoneme, end_ms)]
    parsed = []
    for entry in labels:
        for k, v in entry.items():
            parsed.append((k, int(round(v * 1000))))  # в мс

    # Формируем фонемную последовательность посекундно
    phoneme_seq = []
    cur_start = 0
    for phoneme, end in parsed:
        length = end - cur_start  # сколько мс длится фонема
        phoneme_seq.extend([phoneme] * length)
        cur_start = end

    # Дополняем или обрезаем до нужной длины
    if len(phoneme_seq) < total_ms:
        phoneme_seq.extend([parsed[-1][0]] * (total_ms - len(phoneme_seq)))
    elif len(phoneme_seq) > total_ms:
        phoneme_seq = phoneme_seq[:total_ms]

    # Строим список индексов по заданному шагу
    indices = list(range(0, total_ms, step_ms))

    # Формируем y_true
    y_true = {'start': [], 'centre': [], 'end': []}
    for i in indices:
        s = phoneme_seq[i - step_ms] if i - step_ms >= 0 else phoneme_seq[i]
        c = phoneme_seq[i]
        e = phoneme_seq[i + step_ms] if i + step_ms < len(phoneme_seq) else phoneme_seq[i]
        y_true['start'].append(s)
        y_true['centre'].append(c)
        y_true['end'].append(e)

    return y_true

class ClassificationModel(nn.Module):
    def __init__(self, inputsize, num_classes, dropout):
        super(ClassificationModel, self).__init__()
        self.inputsize = inputsize
        self.linear1 = nn.Linear(self.inputsize, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.final_start = nn.Linear(64, num_classes)
        self.final_center = nn.Linear(64, num_classes)
        self.final_end = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, element):
        a = self.relu(self.linear1(element))
        a = self.dropout(a)
        a = self.relu(self.linear2(a))
        a = self.dropout(a)
        a = self.relu(self.linear3(a))
        a = self.dropout(a)
        a = self.relu(self.linear4(a))
        a_full = self.dropout(a)
        a_start = self.final_start(a_full)
        a_center = self.final_center(a_full)
        a_end = self.final_end(a_full)
        return a_start, a_center, a_end

device = "cuda" if torch.cuda.is_available() else "cpu"
content_encoder = HubertSoft(device=device)

input_file = r'\\nid-tts-02\mnt\hot_store\trainee_common\ananeva\ila.wav'
label_file = r'\\nid-tts-02\mnt\hot_store\trainee_common\ananeva\ila.seg'

wave, sr = torchaudio.load(input_file)
res_wave = FA.resample(wave, sr, 16000)

# Get embeddings and normalize them
ext_emb = content_encoder.encoder(res_wave[0].to(device))
normed_emb = ext_emb[0] / (ext_emb[0] ** 2).sum(0, keepdims=True) ** 0.5

normed_emb = normed_emb.T

step = 20
#normed_emb = extract_embeddings_shifted_by_padding(input_file, content_encoder, device, desired_step_ms = step)
print(normed_emb.shape)

phoneme_list = ['a0', 'i0', 'o0', 'y0', 'u0', 'e0', 'l', "l'", 'm', "m'", 'n', "n'", ]
num_classes = len(phoneme_list)

model = torch.load(r'C:\Users\ext-ananeva\phon_clust\gitlab\ssl_phoneme_clusterizer\model_vow+sonants_var_borders.pth',
                    map_location=device)
model.eval()


# Initialize dictionaries to store probabilities
prob_distributions = {'start': [], 'centre': [], 'end': []}
y_pred = {'start': [], 'centre': [], 'end': []}
result = []

with torch.no_grad():
    start = []
    centre = []
    end = []
    for embs in normed_emb:
        embs = embs.unsqueeze(0).to(device)
        start_out, centre_out, end_out = model(embs)

        # Apply softmax to get probabilities
        start_probs = torch.softmax(start_out, dim=1)
        centre_probs = torch.softmax(centre_out, dim=1)
        end_probs = torch.softmax(end_out, dim=1)

        # Get the probability distributions
        start_dist = {phoneme: start_probs[0][i].item() for i, phoneme in enumerate(phoneme_list)}
        start_dist = sorted(start_dist.items(), key=lambda x: x[1], reverse=True)
        centre_dist = {phoneme: centre_probs[0][i].item() for i, phoneme in enumerate(phoneme_list)}
        centre_dist = sorted(centre_dist.items(), key=lambda x: x[1], reverse=True)
        end_dist = {phoneme: end_probs[0][i].item() for i, phoneme in enumerate(phoneme_list)}
        end_dist = sorted(end_dist.items(), key=lambda x: x[1], reverse=True)

        # Store the distributions
        prob_distributions['start'].append(start_dist)
        prob_distributions['centre'].append(centre_dist)
        prob_distributions['end'].append(end_dist)

        # Get the predicted classes (original functionality)
        pred_start = torch.argmax(start_out, dim=1)
        start.append(phoneme_list[pred_start[0]])
        pred_centre = torch.argmax(centre_out, dim=1)
        centre.append(phoneme_list[pred_centre[0]])
        pred_end = torch.argmax(end_out, dim=1)
        end.append(phoneme_list[pred_end[0]])

        preds = []
        preds.append(pred_start.cpu().numpy().tolist())
        preds.append(pred_centre.cpu().numpy().tolist())
        preds.append(pred_end.cpu().numpy().tolist())
        res = [phoneme_list[preds[i][0]] for i in range(len(preds))]
        res.append('')
        result += res

    y_pred['start'] = start
    y_pred['centre'] = centre
    y_pred['end'] = end

# Print results including probability distributions
print('Predictions as phoneme sequences:')
print('_'.join(result))
print("\nPredicted classes:")
print("Start:", y_pred['start'])
print("Centre:", y_pred['centre'])
print("End:", y_pred['end'])

print("\nProbability distributions for each frame:")
for i in range(len(prob_distributions['start'])):
    print(f"\nFrame {i + 1}:")
    print("Start probabilities:", prob_distributions['start'][i])
    print("Centre probabilities:", prob_distributions['centre'][i])
    print("End probabilities:", prob_distributions['end'][i])

# Your accuracy calculations (unchanged)
y_true = {'start': ['i0', 'i0', 'i0', 'i0', 'l', 'l', 'l', 'l', 'l', 'a0', 'a0', 'a0', 'a0', 'a0', 'a0', 'a0', 'a0', ],
          'centre': ['i0', 'i0', 'i0', 'l', 'l', 'l', 'l', 'l', 'l', 'a0', 'a0', 'a0', 'a0', 'a0', 'a0', 'a0', 'a0', ],
          'end': ['i0', 'i0', 'i0', 'l', 'l', 'l', 'l', 'l', 'a0', 'a0', 'a0', 'a0', 'a0', 'a0', 'a0', 'a0', 'a0', ]}

labels = [{"i0": 0.066}, {"l": 0.170}, {"a0": 0.353}]

#y_true = build_y_true_from_labels(labels, total_ms=353, step_ms=step)
print(y_true)
print(y_pred)


print("\nAccuracy metrics:")
print("accuracy start: " + str(_accuracy_start(y_true, y_pred)))
print("accuracy centre: " + str(_accuracy_centre(y_true, y_pred)))
print("accuracy end: " + str(_accuracy_end(y_true, y_pred)))
print("accuracy ordered: " + str(_accuracy_ordered(y_true, y_pred)))
print("accuracy unordered: " + str(_accuracy_unordered(y_true, y_pred)))
print("accuracy unordered flexible centre: " + str(_accuracy_centre_flex(y_true, y_pred)))
print("accuracy ordered flexible centre: " + str(_accuracy_ordered_flex_centre(y_true, y_pred)))

import matplotlib.pyplot as plt

# Порог вероятности — отфильтровываем слишком маленькие значения
threshold = 0.01

# Максимальное количество примеров, которые визуализируем (для удобства)
max_examples = 50

# Цвета для разных фонем (набор ограничен, для демонстрации)
import seaborn as sns
palette = sns.color_palette("tab20", len(phoneme_list))
phoneme_colors = {phoneme: palette[i % len(palette)] for i, phoneme in enumerate(phoneme_list)}

# График
fig, ax = plt.subplots(figsize=(15, 6))

bar_width = 0.2
inner_gap = 0.05
group_gap = 0.5

x_ticks = []
x_labels = []

x_pos = 0

for ex_idx in range(min(len(prob_distributions['start']), max_examples)):
#for ex_idx in range(26, 39):
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

        # Переходим к следующему столбику в группе
        x_pos += bar_width + inner_gap

        # После группы — добавляем расстояние перед следующей
    x_pos += group_gap

# Удаляем дублирующиеся легенды
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=45)
ax.set_ylabel("Normalized Probability")
ax.set_title("Phoneme Distributions per Prediction (Start / Centre / End)")

plt.tight_layout()
plt.show()