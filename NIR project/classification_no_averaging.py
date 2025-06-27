import torch.nn as nn
import os
import numpy as np
from collections import Counter

from torch.utils.data import DataLoader, ConcatDataset

import torch
import random
from typing import Dict, Any, List

import time

start = time.time()

device = "cpu"#'cuda' if torch.cuda.is_available() else 'cpu'

def load_selected_embeddings(root_dir, phoneme_list, samples_of_ph, speakers, triplet_variation='one border'):
    info = []
    selected_embeddings = []
    selected_phonemes = []

    for speaker in speakers:
        speaker_root_dir = os.path.join(root_dir, speaker)
        speaker_phonemes = []

        for root, _, files in os.walk(speaker_root_dir):
            if all(Counter(speaker_phonemes)[ph] >= samples_of_ph for ph in phoneme_list):
                break

            for file in files:
                if not file.endswith("_no_av_ph.txt"):
                    continue

                name = file.replace("_no_av_ph.txt", "")
                phoneme_file_path = os.path.join(root, file)
                emb_file_path = os.path.join(root, name + "_no_av_embs.npy")

                if not os.path.exists(emb_file_path):
                    print(f"Нет эмбеддингов для {phoneme_file_path}")
                    continue

                with open(phoneme_file_path, "r", encoding="utf-8") as f:
                    phonemes = [line.strip() for line in f.readlines()]

                emb_array = np.load(emb_file_path, allow_pickle=True)  # shape (N, D)

                if len(phonemes) != len(emb_array):
                    print(f"Несовпадение эмбеддингов и меток в {file}")
                    continue

                for phoneme, emb in zip(phonemes, emb_array):
                    base_ph = phoneme.split('_')[1]
                    start_ph = phoneme.split('_')[0]
                    end_ph = phoneme.split('_')[2]

                    def add_sample():
                        label = phoneme.split('_')
                        if len(label) != 3:
                            print(label)
                        selected_embeddings.append({
                            'label': label,
                            'embedding': torch.tensor(emb, dtype=torch.float32),
                            'place': file
                        })
                        selected_phonemes.append(phoneme)
                        speaker_phonemes.append(base_ph)

                    if base_ph in phoneme_list and start_ph in phoneme_list and end_ph in phoneme_list:
                        if Counter(speaker_phonemes)[base_ph] < samples_of_ph:
                            if triplet_variation == 'one border':
                                if start_ph != end_ph or start_ph != base_ph or end_ph != base_ph:
                                    add_sample()

                            elif triplet_variation == 'two borders':
                                if len({start_ph, base_ph, end_ph}) == 3:  # все разные
                                    add_sample()

                            elif triplet_variation == 'no borders':
                                if start_ph == end_ph == base_ph:
                                    add_sample()

                            elif triplet_variation == 'all':
                                add_sample()

                if all(Counter(speaker_phonemes)[ph] >= samples_of_ph for ph in phoneme_list):
                    break

        print(f"{speaker}: {Counter(speaker_phonemes)}")
        info.append({speaker: Counter(speaker_phonemes)})

    return selected_embeddings, selected_phonemes, info

def train(model, train_loader, loss_fn, optimizer, scheduler, num_epochs, device, loss_info=False):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for embs, labels, _ in train_loader:
            torch.cuda.empty_cache()
            embs = embs.to(device)
            labels = labels.to(device)

            start, centre, end = model(embs)

            loss_start = loss_fn(start, labels[:, 0])
            loss_centre = loss_fn(centre, labels[:, 1])
            loss_end = loss_fn(end, labels[:, 2])

            loss = loss_start + loss_centre + loss_end

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        if loss_info:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

from evaluation_utils import (
        _compute_predictions,
        _accuracy_ordered,
        _accuracy_unordered,
        _accuracy_centre_flexible,
        _accuracy_ordered_flex_centre,
        _accuracy_centre,
        _accuracy_start,
        _accuracy_centre_flex,
        _accuracy_end,
        _compute_map,
        _plot_pr_curves,
        _plot_confusion_matrices
)
def evaluate(model, data_loader, loss_fn, file_name, phoneme_list=None, device='cuda'):
    model.eval()

    results = _compute_predictions(model, data_loader, loss_fn, device)

    metrics = {
        'loss': results['avg_loss'],
        'accuracy_ordered': _accuracy_ordered(results['true_labels'], results['pred_labels']),
        'accuracy_unordered': _accuracy_unordered(results['true_labels'], results['pred_labels']),
        'accuracy_ordered_flex_centre': _accuracy_ordered_flex_centre(results['true_labels'], results['pred_labels']),
        'accuracy_centre_flexible': _accuracy_centre_flexible(results['true_labels'], results['pred_labels']),

        'accuracy_start': _accuracy_start(results['true_labels'], results['pred_labels']),
        'accuracy_centre': _accuracy_centre(results['true_labels'], results['pred_labels']),
        'accuracy_centre_flex': _accuracy_centre_flex(results['true_labels'], results['pred_labels']),
        'accuracy_end': _accuracy_end(results['true_labels'], results['pred_labels']),

        'mean_average_precision_centre': _compute_map(results['true_labels'], results['scores_centre'], phoneme_list),
        'confusion_matrices': _plot_confusion_matrices(
            results['true_labels'], results['pred_labels'], phoneme_list, file_name
        )
    }

    _plot_pr_curves(results['scores_centre'], results['true_labels']['centre'], phoneme_list, file_name)

    return metrics


def test(model, test_loader, loss_fn, file_name, phoneme_list=None, device='cuda'):
    result = evaluate(model, test_loader, loss_fn, file_name, phoneme_list, device)
    print(f"Test Loss: {result['loss']:.4f}")
    print(f"Ordered Accuracy: {result['accuracy_ordered']:.4f}")
    print(f"Unordered Accuracy: {result['accuracy_unordered']:.4f}")
    print("Confusion Matrix - START")
    print(result['confusion_matrices']['start'])
    print("Confusion Matrix - CENTRE")
    print(result['confusion_matrices']['centre'])
    print("Confusion Matrix - END")
    print(result['confusion_matrices']['end'])
    return result

class PhonemeEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, data, phoneme_list):
        self.phoneme_list = phoneme_list
        self.data = data
        self.phoneme2idx = {ph: i for i, ph in enumerate(self.phoneme_list)}
        self.num_phonemes = len(self.phoneme2idx)

        # Преобразуем метки в индексы
        self.labels = []
        for d in data:
            label = []
            for ph in d['label']:
                if ph in self.phoneme2idx:
                    ph_idxs = self.phoneme2idx[ph]
                else:
                    # phoneme_list.append(ph)
                    # self.phoneme2idx = {ph: i for i, ph in enumerate(phoneme_list)}
                    # self.num_phonemes = len(self.phoneme2idx)
                    # ph_idxs = self.phoneme2idx[ph]
                    ph_idxs = -1
                label.append(ph_idxs)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding = self.data[idx]['embedding']  # Tensor
        phonemes_original = self.data[idx]['label']
        ph_idx_list = self.labels[idx]  # Список из 3 индексов
        #
        # Возвращаем тензор Long для меток (3 позиции)
        phoneme_labels = torch.tensor(ph_idx_list, dtype=torch.long)

        return embedding, phoneme_labels, phonemes_original


def collate_fn(batch):
    embs = torch.stack([item[0] for item in batch])

    labels = torch.stack([item[1] for item in batch])
    original_labels = [item[2] for item in batch]

    return embs, labels, original_labels


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


def generate_random_dict(symbols) -> Dict[str, Any]:
    # Генерируем label: либо XXY, либо XYY (где X и Y - символы из symbols)
    first_char = random.choice(symbols)
    second_char = random.choice(symbols)

    # Выбираем, будет ли повтор в начале (XXY) или в конце (XYY)
    if random.random() > 0.5:
        # XXY (две одинаковые в начале)
        third_char = random.choice([c for c in symbols if c != first_char])
        label = [first_char, first_char, third_char]
    else:
        # XYY (две одинаковые в конце)
        third_char = second_char
        label = [first_char, second_char, third_char]

    # Генерируем embedding: список из 256 элементов вида [0.123]
    embedding = [random.random() for _ in range(256)]

    # Генерируем случайное имя файла
    file = f"file_{random.randint(1, 1000)}.txt"

    return {
        'label': label,
        'embedding': torch.tensor(embedding, dtype=torch.float32),
        'place': file
    }


def save_results(test_speakers_info, train_speakers_info, metrics_info, name):

    with open(f'results\\{name}.txt', 'a') as f:
        f.write(f'\n{name}')
        f.write(f'\n{test_speakers_info}')
        f.write(f'\n{train_speakers_info}')
        f.write(f'\n{metrics_info}')
    with open(f'results\\full.txt', 'a') as f:
        f.write(f'\nName:{name}')
        f.write(f'\naccuracy_ordered:{metrics_info["accuracy_ordered"]}')
        f.write(f'\naccuracy_unordered:{metrics_info["accuracy_unordered"]}')
        f.write(f'\naccuracy_ordered_flex_centre:{metrics_info["accuracy_ordered_flex_centre"]}')
        f.write(f'\naccuracy_centre_flexible:{metrics_info["accuracy_centre_flexible"]}')
        f.write(f'\naccuracy_centre_flexible:{metrics_info["accuracy_centre_flexible"]}')
        f.write(f'\naccuracy_start:{metrics_info["accuracy_start"]}')
        f.write(f'\naccuracy_centre:{metrics_info["accuracy_centre"]}')
        f.write(f'\naccuracy_centre_flex(only cent):{metrics_info["accuracy_centre_flex"]}')
        f.write(f'\naccuracy_end:{metrics_info["accuracy_end"]}')

    print('information saved')



root_dir = r"\\nid-tts-02\mnt\hot_store\trainee_common\ananeva\CORPRES_embs_no_averaging_test"
phoneme_list = ['a0', 'i0', 'p', "p'", 'k', "k'",  't', "t'",]  # интересующие фонемы (без позиций)
samples_of_ph = 100                   # сколько примеров на фонему
speakers = ["test_speaker"]

# Загрузка
selected_embeddings, selected_phonemes, _ = load_selected_embeddings(root_dir, phoneme_list, samples_of_ph, speakers)

# Пример: один элемент
if len(selected_embeddings) > 0:
    example = selected_embeddings[0]
    print("Label:", example['label'])       # например: b_start
    print("Tensor shape:", example['embedding'].shape)  # torch.Size([embedding_dim])
    print("From file:", example['place'])

    dataset = PhonemeEmbeddingDataset(selected_embeddings, phoneme_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for emb, phoneme_y, letters in dataloader:
        print(emb.shape)
        print(phoneme_y)
        print(letters)
        break


phoneme_list_full = ['a0', 'a1', 'a2', 'a4', 'b', "b'", 'c', 'ch', 'ch_', 'd', "d'", 'e0', 'e1', 'e2', 'e4', 'f', "f'",
                      'g', "g'", 'h', "h'", 'i0', 'i1', 'i2', 'i4', 'j', 'jr', 'jl', 'ji4', 'k', "k'", 'l', "l'", 'm',
                      "m'", 'n', "n'", 'o0', 'o1', 'o2', 'o4', 'p', "p'", 'r', "r'", 's', "s'", 'sc', 'sh', 't', "t'",
                      'u0', 'u1', 'u2', 'u4', 'v', "v'", 'y0', 'y1', 'y2', 'y4', 'z', "z'", 'zh', "zh'", 'C', 'CH', 'H',
                      'SC']

phoneme_list_short = ['a0', 'a1', 'a2', 'a4', 'b', "b'", 'd', "d'", 'e0', 'e1', 'e2', 'e4', 'f', "f'",
                       'i0', 'i1', 'i2', 'i4','l', "l'", 'm',
                      "m'", 'n', "n'", 'o0', 'o1', 'o2', 'o4', 'r', "r'", 's', "s'",
                      'u0', 'u1', 'u2', 'u4', 'v', "v'", 'y0', 'y1', 'y2', 'y4']


int_code = [random.randint(65, 90) for i in range(10)]

file_name = ''.join([chr(code) for code in int_code])

root_dir = r'\\nid-tts-02\mnt\hot_store\trainee_common\ananeva\CORPRES_embs_no_averaging'
train_speakers = ['Vladimir', 'Maria', 'Mikhail', 'Anna']
test_speakers = ['Galina', 'Victoria', 'Petr', 'Alexander']

#phoneme_list = ['a0', 'i0', 'o0', 'y0', 'u0', 'e0', 'l', "l'", 'm', "m'", 'n', "n'",]
#phoneme_list = ['a0', 'i0', 'o0', 'y0', 'u0', 'e0', 'p', "p'", 't', "t'", 'k', "k'",]
#phoneme_list = ['a0', 'i0', 'o0', 'y0', 'u0', 'e0', 's', "s'", 'f', "f'", 'h', "h'",]
#phoneme_list = ['a0', 'i0', 'o0', 'y0', 'u0', 'e0', 'b', "b'", 'd', "d'", 'g', "g'",]
phoneme_list = ['a0',  'o0', 'u0', 'e0', 's', "c", 't',]
#phoneme_list = ['a0', 'e0', 'i0', 'i1', 'i4', "l'", "n'", "m'",]
#phoneme_list = ['a0', 'e0', 'i0', 'i1', 'i4', "s'", "sc", "ch"]
# phoneme_list = ['a0', 'i0', 'o0', 'y0', 'u0', 'e0', 'l', "l'", 'm', "m'", 'n', "n'", 'p', "p'", 't', "t'", 'k', "k'",
#                  's', "s'", 'f', "f'", 'h', "h'", 'b', "b'", 'd', "d'", 'g', "g'", 's', "c", 't', "ch", 'sh', 'v', "v'"]
#phoneme_list = ['a0', 'a1', 'a2', 'a4', 'i0', 'i1', 'i4', 'o0', 'y0','y1', 'y2', 'u0', 'u1', 'u2', 'e0']

#phoneme_list = ['a0', 'o0', 'y0', 'u0', 'e0', 'l', 'm', 'n', 'p',  't',  'k', "s", 'f', 'h', 'b', "d", 'g', 's', "c", 't', 'sh',]
#phoneme_list = ['a0', 'i0', 'o0', 'u0', 'e0', "l'", "m'", "n'", "p'", "t'", "k'", "s'", "f'", "h'", "b'", "d'", "g'", "ch", 'sc',]
#phoneme_list = ['a0','i1','i4', "l'", "m'", "n'", "p'", "t'", "s'"]
num_classes = len(phoneme_list)

samples_per_phoneme = 1000
# train_data = [generate_random_dict(phoneme_list) for i in range(1024)]
# test_data = [generate_random_dict(phoneme_list) for i in range(1024)]
# train_data_1bord, _, test_speakers_info = load_selected_embeddings(root_dir, phoneme_list, samples_per_phoneme, train_speakers, triplet_variation='one border')
# test_data_1bord, _, train_speakers_info = load_selected_embeddings(root_dir, phoneme_list, samples_per_phoneme, test_speakers, triplet_variation='one border')

# train_data_nobord, _, test_speakers_info = load_selected_embeddings(root_dir, phoneme_list, samples_per_phoneme, train_speakers, triplet_variation='no borders')
# test_data_nobord, _, train_speakers_info = load_selected_embeddings(root_dir, phoneme_list, samples_per_phoneme, test_speakers, triplet_variation='no borders')

# triplet_variation =  ['all', 'one border', 'no borders', 'two borders']

# train_dataset_1bord = PhonemeEmbeddingDataset(train_data_1bord, phoneme_list)
# test_dataset_1bord = PhonemeEmbeddingDataset(test_data_1bord, phoneme_list)

# train_dataset_nobord = PhonemeEmbeddingDataset(train_data_nobord, phoneme_list)
# test_data_nobord = PhonemeEmbeddingDataset(test_data_nobord, phoneme_list)

# train_dataset = ConcatDataset([train_dataset_1bord, train_dataset_nobord])
# test_dataset = ConcatDataset([test_dataset_1bord, test_data_nobord])

train_data, _, test_speakers_info = load_selected_embeddings(root_dir, phoneme_list, samples_per_phoneme, train_speakers, triplet_variation='one border')
test_data, _, train_speakers_info = load_selected_embeddings(root_dir, phoneme_list, samples_per_phoneme, test_speakers, triplet_variation='one border')

train_dataset = PhonemeEmbeddingDataset(train_data, phoneme_list)
test_dataset = PhonemeEmbeddingDataset(test_data, phoneme_list)

#TODO: ADD SAMPLER
# weights = get_sample_weights(train_dataset, selected_outputs)
# sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)# sampler=sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


model = ClassificationModel(256, num_classes, dropout=0.1).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

num_epochs = 10

train(model, train_loader,loss_fn, optimizer, scheduler, num_epochs, device)

#torch.save(model, 'model_vow+sonants_var_borders.pth')

result = test(model, test_loader, loss_fn, file_name, phoneme_list=phoneme_list, device=device)

print(result)
file_name = file_name + ''.join(phoneme_list)
save_results(test_speakers_info, train_speakers_info, result, file_name)

end = time.time()

print(f'{(end - start) / 60} minutes')