import torch
import torch.nn as nn

from torch.utils.data import Dataset



class PhonemeEmbeddingDataset(Dataset):

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


class ClassificationModelCTChead(nn.Module):
    def __init__(self, inputsize, num_classes, dropout):
        super(ClassificationModelCTChead, self).__init__()
        self.inputsize = inputsize
        self.linear1 = nn.Linear(self.inputsize, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        # self.final_start = nn.Linear(64, num_classes)
        self.final_center = nn.Linear(64, num_classes)
        # self.final_end = nn.Linear(64, num_classes)
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
        # a_start = self.final_start(a_full)
        a_center = self.final_center(a_full)
        # a_end = self.final_end(a_full)
        return a_center


class CTCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_phonemes, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, num_phonemes + 1)  # + blank

    def forward(self, x):
        # x: [B, T, D]
        out, _ = self.lstm(x)          # [B, T, 2H]
        logits = self.fc(out)          # [B, T, V+1]
        return logits
