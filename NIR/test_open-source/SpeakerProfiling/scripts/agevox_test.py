from argparse import ArgumentParser
import os
import sys
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from TIMIT.lightning_model_uncertainty_loss import LightningModel
from sklearn.metrics import mean_absolute_error
import pytorch_lightning as pl

import torch
import torch.utils.data as data
from torch.utils.data import Dataset

from tqdm import tqdm 
import pandas as pd
import numpy as np
import glob  
import logging
import librosa
import torch.nn.utils.rnn as rnn_utils

from sklearn.metrics import mean_absolute_error, mean_squared_error

class VoxCelebConfig(object):
    
    with open("config.json", "r") as jsonfile:
        config = json.load(jsonfile)
    
    dir = config['dataDir']['dir']
    
    data_path = "./voxceleb2/"
    metadata_path = "./vox-profile-release/agevoxceleb/utt2age.test"
    speaker_csv_path = os.path.join(project_root, 'Dataset', 'data_info_height_age.csv')
    
    # Параметры модели
    batch_size = int(config['model_parameters']['batch_size'])
    epochs = int(config['model_parameters']['epochs'])
    model_type = config['model_parameters']['model_type']
    upstream_model = config['model_parameters']['upstream_model']
    feature_dim = int(config['model_parameters']['feature_dim'])
    lr = float(config['model_parameters']['lr'])
    narrow_band = config['model_parameters']['narrow_band']
    
    num_layers = 6
    gpu = int(config['gpu'])
    n_workers = int(config['n_workers'])
    
    loss = "UncertaintyLoss"
    
    model_checkpoint = "./Model/pretrained_model.ckpt"
    
    run_name = 'agevoxceleb_test_' + model_type

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

class AgeVoxCelebDataset(Dataset):
    def __init__(self, wav_folder, hparams, is_train=False):
        self.hparams = hparams
        self.voxceleb2_base_path = wav_folder
        self.target_sr = 16000
        self.max_samples = 240000
        
        csv_path = hparams.speaker_csv_path
        df = pd.read_csv(csv_path)
        
        self.age_mean = df[df['Use'] == 'TRN']['age'].mean()
        self.age_std = df[df['Use'] == 'TRN']['age'].std()
        
        self.height_mean = df[df['Use'] == 'TRN']['height'].mean()
        self.height_std = df[df['Use'] == 'TRN']['height'].std()
        
        logging.info(f"AgeVoxCelebDataset: Using TIMIT stats - age_mean={self.age_mean:.1f}, age_std={self.age_std:.1f}")
        logging.info(f"AgeVoxCelebDataset: Using TIMIT stats - height_mean={self.height_mean:.1f}, height_std={self.height_std:.1f}")
        
        metadata_path = hparams.metadata_path
        self.df = pd.read_csv(metadata_path, sep='\s+', header=None, names=['utterance_id', 'age_raw'])
        
        self.valid_items = []
        self.missing_files = 0
        
        for _, row in self.df.iterrows():
            audio_path = self._find_audio_file(row['utterance_id'])
            if audio_path:
                self.valid_items.append({
                    'utterance_id': row['utterance_id'],
                    'audio_path': audio_path,
                    'age_raw': row['age_raw']  
                })
            else:
                self.missing_files += 1
        
        logging.info(f"AgeVoxCelebDataset: загружено {len(self.valid_items)} записей, пропущено {self.missing_files}")
    
    def _find_audio_file(self, utterance_id):
        parts = utterance_id.split('/')
        if len(parts) >= 3:
            speaker, video, segment = parts[0], parts[1], parts[2]
            
            wav_path = os.path.join(self.voxceleb2_base_path, speaker, video, f"{segment}.wav")
            if os.path.exists(wav_path):
                return wav_path
            
            m4a_path = os.path.join(self.voxceleb2_base_path, speaker, video, f"{segment}.m4a")
            if os.path.exists(m4a_path):
                return m4a_path
        return None
    
    def _load_audio(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            elif len(audio) < 48000:
                audio = np.pad(audio, (0, max(0, 48000 - len(audio))), 'constant')
            
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            return torch.FloatTensor(audio)
        except Exception as e:
            logging.error(f"Ошибка загрузки {file_path}: {e}")
            return None
    
    def __getitem__(self, index):
        item = self.valid_items[index]
        
        audio = self._load_audio(item['audio_path'])
        if audio is None:
            audio = torch.zeros(48000)
        
        age_normalized = (item['age_raw'] - self.age_mean) / self.age_std
        age = torch.FloatTensor([age_normalized])

        height_raw = 175.0  
        height_normalized = (height_raw - self.height_mean) / self.height_std
        height = torch.FloatTensor([height_normalized])
        
        gender = torch.FloatTensor([0])
        
        return audio, height, age, gender
    
    def __len__(self):
        return len(self.valid_items)
    
def collate_fn(batch):
    """ТОЧНО КАК В TIMIT"""
    (seq, height, age, gender) = zip(*batch)
    seql = [x.reshape(-1,) for x in seq]
    seq_length = [x.shape[0] for x in seql]
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    return data, height, age, gender, seq_length

if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--data_path', type=str, default=VoxCelebConfig.data_path)
    parser.add_argument('--metadata_path', type=str, default=VoxCelebConfig.metadata_path)
    parser.add_argument('--speaker_csv_path', type=str, default=VoxCelebConfig.speaker_csv_path)
    parser.add_argument('--batch_size', type=int, default=VoxCelebConfig.batch_size)
    parser.add_argument('--epochs', type=int, default=VoxCelebConfig.epochs)
    parser.add_argument('--num_layers', type=int, default=VoxCelebConfig.num_layers)
    parser.add_argument('--feature_dim', type=int, default=VoxCelebConfig.feature_dim)
    parser.add_argument('--lr', type=float, default=VoxCelebConfig.lr)
    parser.add_argument('--gpu', type=int, default=VoxCelebConfig.gpu)
    parser.add_argument('--n_workers', type=int, default=VoxCelebConfig.n_workers)
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=VoxCelebConfig.model_checkpoint)
    parser.add_argument('--upstream_model', type=str, default=VoxCelebConfig.upstream_model)
    parser.add_argument('--model_type', type=str, default=VoxCelebConfig.model_type)
    parser.add_argument('--narrow_band', type=str, default=VoxCelebConfig.narrow_band)
    
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    if not torch.cuda.is_available():
        device = 'cpu'
        hparams.gpu = 0
    else:        
        device = 'cuda'
        print(f'Testing Model on AgeVoxCeleb Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

    csv_path = hparams.speaker_csv_path
    print(f"Loading TIMIT stats from: {csv_path}")
    df = pd.read_csv(csv_path)
    a_mean = df[df['Use'] == 'TRN']['age'].mean()
    a_std = df[df['Use'] == 'TRN']['age'].std()
    h_mean = df[df['Use'] == 'TRN']['height'].mean()
    h_std = df[df['Use'] == 'TRN']['height'].std()
    print(f"Using TIMIT normalization: age_mean={a_mean:.1f}, age_std={a_std:.1f}")
    print(f"Using TIMIT normalization: height_mean={h_mean:.1f}, height_std={h_std:.1f}")


    test_set = AgeVoxCelebDataset(
        wav_folder=hparams.data_path,
        hparams=hparams,
        is_train=False
    )

    testloader = data.DataLoader(
        test_set, 
        batch_size=1, 
        shuffle=False, 
        num_workers=hparams.n_workers,
        collate_fn=collate_fn,
    )

    # Модель
    if hparams.model_checkpoint and os.path.exists(hparams.model_checkpoint):
        print(f"Loading model from {hparams.model_checkpoint}...")
        model = LightningModel.load_from_checkpoint(hparams.model_checkpoint, HPARAMS=vars(hparams))
        model.to(device)
        model.eval()
        
        age_pred = []
        age_true_raw = [] 
        age_true_normalized = []
        file_ids = []
        
        utterance_ids = [item['utterance_id'] for item in test_set.valid_items]
        print(f"Found {len(utterance_ids)} utterances")

        for idx, batch in enumerate(tqdm(testloader, desc="Processing")):
            x, y_h, y_a, y_g, x_len = batch
            x = x.to(device)
            
            y_a_normalized = torch.stack(y_a).reshape(-1,)
            
            with torch.no_grad():
                y_hat_h, y_hat_a, y_hat_g = model(x, x_len)
            
            y_hat_a = y_hat_a.to('cpu')
            
            age_pred_val = y_hat_a.item() * a_std + a_mean
            age_true_val = test_set.valid_items[idx]['age_raw']
            age_true_norm_val = y_a_normalized.item()
            
            age_pred.append(age_pred_val)
            age_true_raw.append(age_true_val)
            age_true_normalized.append(age_true_norm_val)
            
            if idx < len(utterance_ids):
                file_ids.append(utterance_ids[idx])
            else:
                file_ids.append(f"unknown_{idx}")

        # Метрики
        age_true = np.array(age_true_raw)
        age_pred = np.array(age_pred)
        
        # Проверка нормализации
        print(f"Normalized true age - mean: {np.mean(age_true_normalized):.3f}, std: {np.std(age_true_normalized):.3f}")
        print(f"Normalized pred age - mean: {np.mean((age_pred - a_mean)/a_std):.3f}, std: {np.std((age_pred - a_mean)/a_std):.3f}")
        print(f"{'='*60}")
        
        amae = mean_absolute_error(age_true, age_pred)
        armse = np.sqrt(mean_squared_error(age_true, age_pred))
        
        abs_errors = np.abs(age_true - age_pred)
        
        print("\nAGEVOXCELEB TEST RESULTS")
        print(f"Age MAE  = {amae:.3f} years")
        print(f"Age RMSE = {armse:.3f} years")
        print(f"Std of Error = {np.std(abs_errors):.3f} years")
        print(f"\nError distribution:")
        print(f"  ≤ 1 year:  {np.mean(abs_errors <= 1)*100:.1f}%")
        print(f"  ≤ 3 years: {np.mean(abs_errors <= 3)*100:.1f}%")
        print(f"  ≤ 5 years: {np.mean(abs_errors <= 5)*100:.1f}%")
        print(f"  ≤ 10 years: {np.mean(abs_errors <= 10)*100:.1f}%")
        print(f"\nAbsolute error stats:")
        print(f"  Min: {np.min(abs_errors):.3f} years")
        print(f"  Median: {np.median(abs_errors):.3f} years")
        print(f"  Max: {np.max(abs_errors):.3f} years")
        print(f"{'='*60}")
        print(f"Total samples: {len(age_true)}")

        # Сохраняем результаты
        csv_data = []
        for i in range(len(age_true)):
            csv_data.append({
                "id": file_ids[i],
                "age_true": age_true[i],
                "age_pred": age_pred[i],
                "abs_error": abs(age_true[i] - age_pred[i])
            })
        
        csv_file = "agevoxceleb_predictions.csv"
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        print(f"Predictions saved to: {csv_file}")
        
        with open("agevoxceleb_age_predictions.txt", 'w') as f:
            for i in range(len(age_true)):
                f.write(f"{file_ids[i]} {age_true[i]:.0f} {age_pred[i]:.2f}\n")
        
        print(f"Predictions also saved to: agevoxceleb_age_predictions.txt")
        
    else:
        print(f'Model checkpoint not found: {hparams.model_checkpoint}')