import os
import torch
import pickle
import numpy as np
import torchaudio
from torchaudio import functional as FA
from collections import defaultdict
from content_manager.vencoder.HubertSoft import HubertSoft
from tqdm import tqdm



def read_segmentation(seg_path, n_frames=None, framerate=None):
    if isinstance(seg_path, str):
        with open(seg_path, "rb") as f:
            bin_lines = f.readlines()
            if b'[History]\r\n' in bin_lines:
                idx_history = bin_lines.index(b'[History]\r\n')
                bin_lines = bin_lines[:idx_history]
            try:
                lines = [line.decode("utf-8").strip() for line in bin_lines]
            except UnicodeDecodeError:
                lines = [line.decode("cp1251").strip() for line in bin_lines]

            # workaround for some strange files
            lines[0] = lines[0].lstrip("п»ї").lstrip(chr(65279))  # Это BOM
    else:
        bin_lines = seg_path.readlines()
        lines = [line.strip() for line in bin_lines]

    parameters_index = lines.index("[PARAMETERS]")
    labels_index = lines.index("[LABELS]")

    parameters = {}
    labels = []
    for line in lines[(parameters_index + 1):labels_index]:
        parameter, value = line.split("=")
        parameters[parameter] = int(value)

    rate = 1
    if framerate is not None and parameters.get("SAMPLING_FREQ", None) is not None:
        if parameters["SAMPLING_FREQ"] != framerate:
            rate = framerate / parameters["SAMPLING_FREQ"]
            parameters["SAMPLING_FREQ"] = framerate

    if parameters["N_CHANNEL"] > 1:
        rate /= parameters["N_CHANNEL"]

    timestamps = []
    for line in lines[(labels_index + 1):]:
        if '[History]' in line:
            break
        if line == '':
            continue
        timestamp, _, label = line.split(",", maxsplit=2)
        if label == "" and labels[-1:] == [label]:
            continue

        timestamps.append(int(float(timestamp) / parameters["BYTE_PER_SAMPLE"] * rate))
        labels.append(label)

    if timestamps and timestamps[0] != 0:
        timestamps.insert(0, 0)
        labels.insert(0, "")

    if n_frames is not None:
        if not timestamps or timestamps[-1] != n_frames:
            timestamps.append(n_frames)

    segmentation = list(zip(labels, zip(timestamps, timestamps[1:])))

    return segmentation, parameters


def process_audio_and_extract_embs(file_name, content_encoder, device):
    file = file_name
    mean_embs = defaultdict(list)
    segmentation, params = read_segmentation(file_name + '.seg_B2')
    # Кусочек из getmetrics
    try:
        ## Считываем аудио и ресемплим к 16000 Гц:
        wave, sr = torchaudio.load(file + '.wav')
        res_wave = FA.resample(wave, sr, 16000)

        ## Получаем эмбеддинги моделью content_encoder и нормируем их:
        ext_emb = content_encoder.encoder(res_wave[0].to(device))
        normed_emb = ext_emb[0] / (ext_emb[0] ** 2).sum(0, keepdims=True) ** 0.5

        ## Находим масштаб:
        scale_sec_per_emb = res_wave.size(1) / 16000 / ext_emb.size(2)

        ## Переводим фонемную разметку в секунды:
        ph_sec = [[b[0], [b[1][0] / 22050, b[1][1] / 22050]] for b in segmentation]

        ## Собираем эмбеддинги, соответствующие целевой фонеме (их может быть несколько в предложении):
        for t1, t2, ph in [(p[1][0], p[1][1], p[0]) for p in ph_sec]:
            count = 1
            if ph in mean_embs.keys(): # Если фонемы повторяются, сохраняем их с нижними подчеркиваниями
                ph = ph + '_' * count
                count += 1

            ph_embs = []
            if normed_emb[:, int(t1 / scale_sec_per_emb):int(t2 / scale_sec_per_emb)].shape[1] == 0:
                continue

            ph_embs.append(normed_emb[:, int(t1 / scale_sec_per_emb):int(t2 / scale_sec_per_emb)])

            ## Считаем среднее по одной фонеме
            phonemes_tensor = torch.cat([t.mean(dim=1).unsqueeze(1) for t in ph_embs], dim=-1)
            phonemes_tensor = phonemes_tensor / (phonemes_tensor ** 2).sum(0, keepdims=True) ** 0.5

            mean_embs[ph] = phonemes_tensor

    except Exception as E:
        print(f'Exception during get_embs: {E}')
    return mean_embs # Возвращает словарь {'phoneme':'embb'}


def process_directory(input_dir, output_dir, content_encoder, device):
    """
    Проходит по всем файлам в папках внутри input_dir, создает аналогичную структуру
    в output_dir и сохраняет файлы с эмбеддингами фонем.
    """
    for root, dirs, files in tqdm(os.walk(input_dir)):
        for file in files:

            if file.endswith(".seg_B2"):  # Ищем только WAV-файлы
                name = file.split(".")[0]
                file_name = os.path.splitext(os.path.join(root, file))[0]  # Убираем расширение

                # Генерируем путь для сохранения эмбеддингов
                relative_path = os.path.relpath(root, input_dir)  # Относительный путь
                new_dir = os.path.join(output_dir, relative_path)  # Создаем аналогичную структуру

                os.makedirs(new_dir, exist_ok=True)  # Создаем папку, если её нет

                # Извлекаем эмбеддинги
                ph_embs = process_audio_and_extract_embs(file_name, content_encoder, device)
                new_dict = {}
                for phoneme, tensor in ph_embs.items():
                    new_dict[phoneme] = tensor.cpu().numpy()
                np.save(os.path.join(new_dir, name + "_embs.npy"), new_dict)

                with open(os.path.join(new_dir, name + "_phonemes.txt"), "w", encoding='utf-8') as f:
                    for phoneme in ph_embs.keys():
                        f.write(f"{phoneme.rstrip('_')}\n")


# Пример использования
if __name__ == "__main__":
    input_directory = r'\\nid-tts-02\mnt\hot_store\tts_data\CORPRES\Victoria'  # Путь к исходной директории

    output_directory = r'\\nid-tts-02\mnt\hot_store\trainee_common\ananeva\CORPRES_embs\Victoria'  # Путь к новой директории


    device = "cuda" if torch.cuda.is_available() else "cpu"
    content_encoder = HubertSoft(device=device)

    process_directory(input_directory, output_directory, content_encoder, device)