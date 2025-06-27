from torchvision import datasets as dts
import torchvision.transforms as transforms
import torch as trch
import torch.nn as nn
from collections import Counter
import os
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import WeightedRandomSampler
from itertools import product

from torch.nn import functional as F
from collections import defaultdict

# Функция для загрузки эмбеддингов
def load_selected_embeddings(root_dir, phoneme_list, samples_of_ph, speakers):
    selected_embeddings = []
    selected_phonemes = []

    for speaker in speakers:
        speaker_root_dir = root_dir + '\\' + speaker
        speaker_phonemes = []

        # Обход всех файлов в директории и поддиректориях
        for root, _, files in os.walk(speaker_root_dir):
            if all(Counter(speaker_phonemes)[phoneme] >= samples_of_ph for phoneme in phoneme_list):
                break

            for file in files:
                if file.endswith("_phonemes.txt"):  # Ищем файлы со списками фонем
                    name = file.replace("_phonemes.txt", "")  # Базовое имя файла
                    phoneme_file_path = os.path.join(root, file)
                    emb_file_path = os.path.join(root, name + "_embs.npy")  # Соответствующий файл эмбеддингов

                    # Проверяем, существует ли файл с эмбеддингами
                    if not os.path.exists(emb_file_path):
                        print(f"Нет эмбеддингов для {phoneme_file_path}")
                        continue

                    # Читаем список фонем
                    with open(phoneme_file_path, "r", encoding="utf-8") as f:
                        phonemes = [line.strip() for line in f.readlines()]

                    # Загружаем эмбеддинги
                    emb_dict = np.load(emb_file_path, allow_pickle=True).item()

                    # Выбираем только нужные фонемы
                    for phoneme in phoneme_list:
                        if phoneme.rstrip('_') in phonemes and phoneme in emb_dict:
                            if Counter(speaker_phonemes)[phoneme] < samples_of_ph:
                                tensor = torch.tensor(emb_dict[phoneme], dtype=torch.float32)
                                label = define_label(phoneme, phoneme_list, parameter='noize')

                                # TODO: передавать не индекс а фонему (исправить даталодер)

                                selected_embeddings.append({'label': label,
                                                            'embedding': torch.squeeze(tensor, -1),
                                                            'place': file})         #torch.squeeze(tensor, -1)
                                selected_phonemes.append(phoneme)

                                selected_phonemes.append(phoneme)
                                speaker_phonemes.append(phoneme)

                    if all(Counter(speaker_phonemes)[phoneme] >= samples_of_ph for phoneme in phoneme_list):
                        break
        print(Counter(speaker_phonemes))
    return selected_embeddings, selected_phonemes

def define_label(phoneme, phoneme_list, parameter='all'):
    label = -1
    if parameter == 'all':
        label = phoneme_list.index(phoneme)

    elif parameter == 'place':
        pass

    elif parameter == 'manner':
        if phoneme in ['h', "h'", 'f', "f'", 'l', "l'", 's', "s'", 'sc', 'sh', 'v', "v'", 'z', "z'", 'zh', "zh'"]:
            label = 0
        elif phoneme in ['b', "b'", 'd', "d'", 'g', "g'", 'k', "k'", 'm', "m'", 'n', "n'", 'p', "p'", 't', "t'"]:
            label = 1
        elif phoneme in ['c', 'ch']:
            label = 2
        elif phoneme in ['r', "r'"]:
            label = 3

    elif parameter == 'palatalization':
        label = 1 if ("'" in phoneme
                          or phoneme == 'sc'
                          or phoneme == 'j'
                          or phoneme == 'ch') \
               else 0

    elif parameter == 'noize':
        if phoneme in ['a0', 'a1', 'a2', 'a4', 'e0', 'e1', 'e2', 'e4', 'i0', 'i1', 'i2', 'i4',
                       'o0', 'o1', 'o2', 'o4', 'u0', 'u1', 'u2', 'u4', 'y0', 'y1', 'y2', 'y4', ]:
            label = 0
        elif phoneme in ['j', 'jr', 'jl', 'ji4', 'l', "l'", 'm', "m'", 'n', "n'", 'r', "r'",]:
            label = 1
        elif phoneme in ['b', "b'",  'd', "d'",'g', "g'",  'v', "v'", 'z', "z'", 'zh', "zh'", 'C', 'CH', 'H', 'SC']:
            label = 2
        elif phoneme in ['c', 'ch', 'ch_', 'f', "f'", 'h', "h'", 'k', "k'", 'p', "p'",  's', "s'", 'sc', 'sh', 't', "t'"]:
            label = 3

    return label

def get_sample_weights(train_ds):
    """
    Рассчитывает веса выборки для взвешенного сэмплера, учитывая несбалансированные классы.

    :param train_ds: Датасет, где каждый элемент — (данные, label1, label2, ..., labelN)
    :return: torch.Tensor с весами для каждого примера
    """
    # Извлекаем все метки
    labels_list = list(zip(*[sample[1:] for sample in train_ds]))  # Пропускаем данные (sample[0])

    # Считаем количество примеров для каждого класса в каждой метке
    class_counts_list = [Counter(labels) for labels in labels_list]

    # Рассчитываем веса для каждого класса (обратная частота)
    class_weights_list = [{cls: 1.0 / count for cls, count in class_counts.items()} for class_counts in
                          class_counts_list]

    # Рассчитываем общий вес для каждого примера (усредняем по всем меткам)
    weights = torch.tensor([
        sum(class_weights_list[i][labels[i]] for i in range(len(labels_list))) / len(labels_list)
        for labels in zip(*labels_list)
    ], dtype=torch.float)

    return weights

def prepare_evaluation(label, outputs, files):
    all_preds, all_labels, all_files, mistakes = [], [], [], []
    # Получаем предсказанные классы
    _, predicted = torch.max(outputs, 1)

    # Сохраняем предсказания и истинные метки для вычисления метрик
    all_preds.extend(predicted.cpu().numpy())
    all_labels.extend(label.cpu().numpy())
    all_files.extend(files)

    for i in range(len(files)):
        if predicted[i] != label[i]:
            mistakes.append({'file': files[i],
                             'true_label': phoneme_list[label[i]],
                             'predicted_label': phoneme_list[predicted[i]]})

    return all_preds, all_labels, all_files, mistakes

def evaluate(all_labels, all_preds, mistakes, label_name, phoneme_list, pr_rec_inf=False, conf_matrix=False):

    file_name = label_name + '_mist.txt'
    with open(file_name, 'w') as f:
        for m in mistakes:
            for key, value in m.items():
                f.write(f'\n{key}: {value}')

    # Вычисляем precision, recall и F1-score для каждого класса
    all_labels_reduced, all_preds_reduced = [], []
    for i, l in enumerate(all_labels): #если класса не было, его нужно убрать
        if l != -1:
            all_labels_reduced.append(l)
            all_preds_reduced.append(all_preds[i])

    precision, recall, _, _ = precision_recall_fscore_support(all_labels_reduced, all_preds_reduced, average=None)
    bal_acc = balanced_accuracy_score(all_labels_reduced, all_preds_reduced)

    # Выводим precision и recall для каждого класса
    if pr_rec_inf:
        print(label_name)
        for i, (p, r) in enumerate(zip(precision, recall)):
            print(f'Class {phoneme_list[i]}: Precision = {p:.4f}, Recall = {r:.4f}')

    if conf_matrix:
        num_classes = len(set(all_labels_reduced))
        if num_classes > 15:
            fig, ax = plt.subplots(figsize=(12, 12))
            fontsize = 6
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            fontsize = 10


        if len(phoneme_list) == num_classes:
            cm = confusion_matrix(all_labels_reduced, all_preds_reduced, labels=range(num_classes),
                                  normalize='true')
            disp = ConfusionMatrixDisplay(cm, display_labels=phoneme_list)
        else:
            cm = confusion_matrix(all_labels_reduced, all_preds_reduced, labels=range(num_classes),
                                  normalize='true')
            disp = ConfusionMatrixDisplay(cm, display_labels=list(range(0, num_classes)))
        disp.plot(ax=ax, colorbar=False)

        for text in ax.texts:
            text.set_fontsize(fontsize)

        plt.show()


    print(f'Accuracy on test set {label_name}: {bal_acc:}')
    # if prev_acc < bal_acc:
    # torch.save(cmodel.state_dict(), f'models/{phoneme_list[0]}.pt')
    return bal_acc

def train(numepch, train_dataloader, test_dataloader, phoneme_list, loss_coefs=[1,1,1],  loss_inf=False, device='cuda', batch_distr_hist=False):
    cmodel.train()

    for epoch in range(numepch):
        running_loss = 0.0

        for i, (embs, label, label_2, label_3,  file) in enumerate(train_dataloader):

            dict_hist = {0: 0, 1: 0}

            ax = embs.to(device)
            ay = label.to(device)
            ay_2 = label_2.to(device)
            ay_3 = label_3.to(device)

            mask2 = ay_2 >= 0
            ay_2[ay_2 < 0] = 0

            mask3 = ay_3 >= 0
            ay_3[ay_3 < 0] = 0

            outputs, outputs_2, outputs_3 = cmodel(ax)

            loss_1 = F.cross_entropy(outputs, ay) #TODO: put loss to params

            dict_hist[0] += ay_2[(ay_2 == 0)*mask2].numel()
            dict_hist[1] += ay_2[(ay_2 == 1) * mask2].numel()

            loss_2 = (F.cross_entropy(outputs_2, ay_2, reduction='none')*mask2).sum()/(mask2.sum() + 1e-6)

            loss_3 = (F.cross_entropy(outputs_3, ay_3, reduction='none') * mask3).sum() / (mask3.sum() + 1e-6)
            loss = loss_coefs[0]*loss_1 + loss_2 * loss_coefs[1] + loss_3 * loss_coefs[1]

            #losses = F.cross_entropy(outputs, ay, reduction='none') # *[1, 0]
            if batch_distr_hist:
                plt.figure(figsize=(5, 3))
                plt.bar(dict_hist.keys(), dict_hist.values(), color=['blue', 'orange'])
                plt.xticks([0, 1], ['Class 0', 'Class 1'])
                plt.ylabel("Количество примеров")
                plt.xlabel("Классы")
                plt.title("Распределение классов в батче")
                plt.show()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        if loss_inf is True:
            print(f"Epoch [{epoch + 1}/{numepch}], Loss: {running_loss / len(train_dataloader):.10f}")
            print(f'New LR: {scheduler.get_last_lr()[0]:.6f}')
    test(test_dataloader, phoneme_list, )


def test(test_dataloader, phoneme_list, pr_rec_inf=False, conf_matrix=False, device='cuda'):
    cmodel.eval()
    all_preds, all_labels, all_files, mistakes = [], [], [], []
    all_preds2, all_labels2, all_files2, mistakes2 = [], [], [], []
    all_preds3, all_labels3, all_files3, mistakes3 = [], [], [], []
    with (torch.no_grad()):
        for embs, label, label_2, label_3, files in test_dataloader:

            embs = embs.to(device)
            embs = embs.view(-1, 256)
            label = label.to(device)
            label2 = label_2.to(device)
            label3 = label_3.to(device)

            # Получаем предсказания от модели
            outputs, outputs2, outputs3 = cmodel(embs)

            all_preds_batch, all_labels_batch, all_files_batch, mistakes_batch = prepare_evaluation(label, outputs, files)
            all_preds.extend(all_preds_batch)
            all_labels.extend(all_labels_batch)
            all_files.extend(all_files_batch)
            mistakes.extend(mistakes_batch)

            all_preds_batch2, all_labels_batch2, all_files_batch2, mistakes_batch2 = prepare_evaluation(label2, outputs2, files)
            all_preds2.extend(all_preds_batch2)
            all_labels2.extend(all_labels_batch2)
            all_files2.extend(all_files_batch2)
            mistakes2.extend(mistakes_batch2)

            all_preds_batch3, all_labels_batch3, all_files_batch3, mistakes_batch3 = prepare_evaluation(label3,
                                                                                                        outputs3, files)
            all_preds3.extend(all_preds_batch3)
            all_labels3.extend(all_labels_batch3)
            all_files3.extend(all_files_batch3)
            mistakes3.extend(mistakes_batch3)


    acc = evaluate(all_labels, all_preds, mistakes, 'all', phoneme_list, pr_rec_inf=pr_rec_inf, conf_matrix=conf_matrix)
    acc2 = evaluate(all_labels2, all_preds2, mistakes2, 'palatalization', phoneme_list, pr_rec_inf=pr_rec_inf, conf_matrix=conf_matrix)
    acc3 = evaluate(all_labels3, all_preds3, mistakes3, 'type', phoneme_list, pr_rec_inf=pr_rec_inf, conf_matrix=conf_matrix)

    #if prev_acc < bal_acc:
        #torch.save(cmodel.state_dict(), f'models/{phoneme_list[0]}.pt')
    return acc, acc2, acc3



class Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        # self.map = {'a0': 1,
        #               //
        self.map_palatal = {'b': 0, "b'": 1, 'c': 0, 'ch': 1, 'd': 0, "d'": 1, 'f': 0, "f'": 1, 'g': 0, "g'": 1,
                    'h': 0, "h'": 1, 'j': 1, 'k': 0, "k'": 1, 'l': 0, "l'": 1, 'm': 0, "m'": 1,
                    'n': 0, "n'": 1, 'p': 0, "p'": 1, 'r': 0, "r'": 1, 's': 0, "s'": 1, 'sc': 1,
                    'sh': 0, 't': 0, "t'": 1, 'v': 0, "v'": 1, 'z': 0, "z'": 1, 'zh': 0, "zh'": 1}
        # 0 щелевые, 1 смычные, 2 аффрикаты 3 дрожащий
        self.map_type = {'b': 1, "b'": 1, 'c': 2, 'ch': 2, 'd': 1, "d'": 1, 'f': 0, "f'": 0, 'g': 1, "g'": 1,
                            'h': 0, "h'": 0, 'j': 0, 'k': 1, "k'": 1, 'l': 0, "l'": 0, 'm': 1, "m'": 1,
                            'n': 1, "n'": 1, 'p': 1, "p'": 1, 'r': 3, "r'": 3, 's': 0, "s'": 0, 'sc': 0,
                            'sh': 0, 't': 1, "t'": 1, 'v': 0, "v'": 0, 'z': 0, "z'": 0, 'zh': 0, "zh'": 0}
        # self.map_type = {'b': 1, "b'": 1, 'c': 2, 'ch': 2, 'd': 1, "d'": 1, 'f': 0, "f'": 0, 'g': 1, "g'": 1,
        #                  'h': 0, "h'": 0, 'j': 0, 'k': 1, "k'": 1, 'l': 0, "l'": 0, 'm': 1, "m'": 1,
        #                  'n': 1, "n'": 1, 'p': 1, "p'": 1, 'r': 3, "r'": 0, 's': 0, "s'": 0, 'sc': 0,
        #                  'sh': 0, 't': 1, "t'": 2, 'v': 0, "v'": 0, 'z': 0, "z'": 0, 'zh': 0, "zh'": 0}
        self.map_openess = {'i0': 0, 'i1': 0, 'i2': 0, 'i4': 0,'y0': 0, 'y1': 0, 'y2': 0, 'y4': 0,'u0': 0, 'u1': 0, 'u2': 0, 'u4': 0,
                            'e0': 1, 'e1': 1, 'e2': 1, 'e4': 1, 'o0': 1, 'o1': 1, 'o2': 1, 'o4': 1,
                            'a0': 2, 'a1': 2, 'a2': 2, 'a4': 2,}
        self.map_frontness = {'i0': 0, 'i1': 0, 'i2': 0, 'i4': 0,'e0': 0, 'e1': 0, 'e2': 0, 'e4': 0,
                              'u0': 1, 'u1': 1, 'u2': 1, 'u4': 1,'o0': 1, 'o1': 1, 'o2': 1, 'o4': 1,'a0': 1, 'a1': 1, 'a2': 1, 'a4':1,
                              'y0': 2, 'y1': 2, 'y2': 2, 'y4': 2,}



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        features = sample['embedding'].clone().detach().to(torch.float32)
        #features = torch.tensor(sample['embedding'], dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.long)
        file = sample['place']
        label_2 = torch.tensor(self.map_palatal.get(phoneme_list[label], -1), dtype=torch.long)
        label_3 = torch.tensor(self.map_type.get(phoneme_list[label], -1), dtype=torch.long)
        return features, label, label_2, label_3, file

class ClassificationModel(nn.Module):
    def __init__(self, inputsize, num_classes, dropout):
        super(ClassificationModel, self).__init__()
        self.inputsize = inputsize

        self.linear1 = nn.Linear(self.inputsize, 500)
        self.linear2 = nn.Linear(500, 200)
        self.linear3 = nn.Linear(200, 100)
        self.linear4 = nn.Linear(100, 50)
        self.final = nn.Linear(50, num_classes)
        self.final_2 = nn.Linear(50, 2)
        self.final_3 = nn.Linear(50, 4)
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
        a_0 = self.dropout(a)
        a = self.final(a_0)
        a_2 = self.final_2(a_0)
        a_3 = self.final_3(a_0)
        return a, a_2, a_3

output_directory = r'\\nid-tts-02\mnt\hot_store\trainee_common\ananeva\CORPRES_embs'
speakers = ['Vladimir', 'Maria', 'Mikhail', 'Anna']

stressed_vowels = ['a0', 'y0', 'e0', 'i0', 'u0', 'o0']
vowels = [['a0', 'a1', 'a2', 'a4', 'e0', 'o0', 'o1', 'u0', 'u1', 'u2', 'u4','y0', 'y1', 'y2', 'y4',]]

ru_phoneme = ['a0', 'a1', 'a2', 'a4', 'b', "b'", 'c', 'ch', 'ch_', 'd', "d'", 'e0', 'e1', 'e2', 'e4', 'f', "f'",
              'g', "g'", 'h', "h'", 'i0', 'i1', 'i2', 'i4', 'j', 'jr', 'jl', 'ji4', 'k', "k'", 'l', "l'", 'm',
              "m'", 'n', "n'", 'o0', 'o1', 'o2', 'o4', 'p', "p'", 'r', "r'", 's', "s'", 'sc', 'sh', 't', "t'",
              'u0', 'u1', 'u2', 'u4', 'v', "v'", 'y0', 'y1', 'y2', 'y4', 'z', "z'", 'zh', "zh'", 'C', 'CH', 'H',
              'SC']

ru_phoneme_reduced = ['a0', 'a1', 'a2', 'a4', 'b', "b'", 'c', 'ch', 'd', "d'", 'e0', 'f', "f'",
              'g', "g'", 'h', "h'", 'i0', 'i1', 'i2', 'i4', 'j', 'k', "k'", 'l', "l'", 'm',
              "m'", 'n', "n'", 'o0', 'p', "p'", 'r', "r'", 's', "s'", 'sc', 'sh', 't', "t'",
              'u0', 'u1', 'u2', 'u4', 'v', "v'", 'y0', 'y1', 'y2', 'y4', 'z', "z'", 'zh']



fricatives = ['s', "s'", 'sh', 'sc', 'z', "z'", 'zh', 'f', "f'", 'h', "h'", 'l', "l'", 'v', "v'", 'j' ] #fricatives
plosives = ['b', "b'",  'd', "d'", 'g', "g'",'p', "p'", 'k', "k'",  't', "t'",'m',"m'", 'n', "n'"]
#phoneme_list = ['d', "d'", 'g', "g'",'p', "p'", 'k', "k'", 't', "t'",  'b', "b'", 'd', "d'",'m', "m'", 'n', "n'"]
voiced_cons = ['b', "b'",  'd', "d'", 'g', "g'", 'z', "z'", 'zh', "zh'", ] # no 'v', "v'",
devoiced_cons = ['f', "f'", 'h', "h'",'k', "k'", 'p', "p'",'s', "s'", 'sc', 'sh', 't', "t'", ]
sonants = ['l', "l'", 'm',"m'", 'n', "n'",'r', "r'", 'j']

consonants = ['b', "b'", 'c', 'ch', 'd', "d'", 'f', "f'", 'g', "g'", 'h', "h'", 'j', 'k', "k'", 'l', "l'", 'm', "m'",
              'n', "n'", 'p', "p'", 'r', "r'", 's', "s'", 'sc', 'sh', 't', "t'", 'v', "v'",  'z', "z'", 'zh', "zh'"]

#phoneme_list = ['s', "s'", 'sh', 'sc', 'z', "z'", 'zh']
#phoneme_list = ['s', 'sh', 'z',  'zh', 'a0']
#phoneme_list = ['f', "f'", 'h', "h'", 'l', "l'", 'r', "r'", 's', "s'",'v', "v'", 'z', "z'",
                #'p', "p'", 'k', "k'", 't', "t'",  'b', "b'", 'd', "d'",'m', "m'", 'n', "n'"]
#phoneme_list = ['a0', 'i0']
#phoneme_list = ['a0', 'u0']
#phoneme_list = ['a0', 'o0']
#phoneme_list = ['a0', 'y0']
#phoneme_list = ['p', "p'", 'k', "k'",  't', "t'",]
pairs = [['p', "p'",], ['k', "k'"], ['t', "t'"], ['f', "f'"], ['h', "h'",], ['l', "l'"],['r', "r'"],
         ['s', "s'"], ['v', "v'"], ['z', "z'",], ['b', "b'",],['m', "m'"],['n', "n'"],['g', "g'"],
         ['sh', 'sc'], ['f', "f'",]]



param_grid = {
    'lr': [0.001],
    'batch_size': [int(16),],# int(32), int(64), int(128)],
    'dropout': [0.1],
    'weight_decay': [0.0001,], #0.0002, 0.0003, 0.0004, 0.0005 ],
    'loss_function': [nn.CrossEntropyLoss()],
    'num_epoch': [10],
    'phonemes': [sonants]#, fricatives, plosives, voiced_cons, devoiced_cons, sonants,]
}


best_acc = 0
best_params = None
param_dicts = []
for lr, batch_size, dropout, weight_decay, loss_function, num_epoch, phon_list in product(
        param_grid['lr'], param_grid['batch_size'], param_grid['dropout'],
        param_grid['weight_decay'], param_grid['loss_function'], param_grid['num_epoch'], param_grid['phonemes']):

    print(lr, batch_size, dropout, weight_decay, loss_function, num_epoch)

    phoneme_list = ru_phoneme_reduced
    numclases = len(phoneme_list)
    samples_of_ph = 50000
    test_samples_of_ph = samples_of_ph

    train_data, train_lables = load_selected_embeddings(output_directory, phoneme_list, samples_of_ph,
                                                        ['Vladimir', 'Maria', 'Mikhail', 'Anna'])
    test_data, test_lables = load_selected_embeddings(output_directory, phoneme_list, test_samples_of_ph,
                                                      ['Galina', 'Victoria', 'Petr', 'Alexander'])

    train_ds = Dataset(train_data)
    test_ds = Dataset(test_data)

    device = 'cuda'

    print(f'Classifying {", ".join(phoneme_list)}, {samples_of_ph} samples of each into {numclases} classes')

    cmodel = ClassificationModel(256, numclases, dropout).to(device)
    lossfn = loss_function
    optimizer = trch.optim.AdamW(cmodel.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    weights = get_sample_weights(train_ds)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_ds))
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, sampler=sampler)

    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    for coef1, coef2, coef3 in product([1], [0], [0]):
    #for coef1, coef2, coef3 in product([0.0001, 0.5, 1, 1.5], [0.0001, 0.5, 1, 1.5], [0.0001, 0.5, 1, 1.5]):
        print(f'{coef1}, {coef2}, {coef3}')
        train(num_epoch, train_dataloader, test_dataloader, phoneme_list,
              loss_coefs=[coef1, coef2, coef3],
              loss_inf=False,
              batch_distr_hist=False)

        acc, acc2, acc3 = test(test_dataloader, phoneme_list, conf_matrix=True)
        param_dicts.append({'acc': acc,
                            'params': f'lr ={lr}, dropout = {dropout}, weight_decay = {weight_decay}, '
                                      f'loss_function = {loss_function}, num_epoch = {num_epoch},'
                                      f' batch_size = {batch_size}'})
        if acc > best_acc:
            best_acc = acc
            best_params = (lr, batch_size, dropout, weight_decay, loss_function, num_epoch)

print(f'Best acc: {best_acc} with params: {best_params}')
print(sorted(param_dicts, key=lambda x: x['acc']))

#для учета размеров классов
#labels_tr = np.array([sample['label'] for sample in train_ds.data])
#class_count_tr = np.bincount(labels_tr)
#weights_tr = 1.0 / class_count_tr[labels_tr]
#sampler_tr = WeightedRandomSampler(weights_tr, num_samples=samples_of_ph, replacement=True)

#labels = np.array([sample['label'] for sample in test_ds.data])
#class_count = np.bincount(labels)
#weights = 1.0 / class_count[labels]
#sampler = WeightedRandomSampler(weights, num_samples=samples_of_ph, replacement=True)


#optimizer = trch.optim.Adam(cmodel.parameters(), lr=0.001)

#for wdk in [0.1, 0.01, 0.001, 0.0001]:
 #   for lr in [0.1, 0.01, 0.001, 0.0001]:
  #      cmodel = ClassificationModel(256, numclases)
   #     lossfn = nn.CrossEntropyLoss()
    #    print(f'wldk: {wdk}, lr: {lr}')
     #   optimizer = trch.optim.AdamW(cmodel.parameters(), lr=lr, weight_decay=wdk)
      #  scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        #train(numepch, train_dataloader, test_dataloader, phoneme_list)

print('done')

