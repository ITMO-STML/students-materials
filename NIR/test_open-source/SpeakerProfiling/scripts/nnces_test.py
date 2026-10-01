import torch
import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import logging
from tqdm import tqdm
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from TIMIT.lightning_model_uncertainty_loss import LightningModel

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

class NNCESConfig(object):

    with open("config.json", "r") as jsonfile:
        config = json.load(jsonfile)
    
    data_path = "./nonnative-children-english-speech-nnces-corpus/versions/1/Read_Speech_Data/Read_Speech_Data/"
    metadata_csv = "./nnces_full_metadata.csv"
    
    speaker_csv_path = os.path.join(project_root, 'Dataset', 'data_info_height_age.csv')
    
    batch_size = 1 
    epochs = int(config['model_parameters']['epochs'])
    model_type = config['model_parameters']['model_type']
    upstream_model = config['model_parameters']['upstream_model']
    feature_dim = int(config['model_parameters']['feature_dim'])
    lr = float(config['model_parameters']['lr'])
    narrow_band = config['model_parameters']['narrow_band']
    
    num_layers = 6
    
    gpu = int(config['gpu']) if config['gpu'] != '-1' else 0
    n_workers = int(config['n_workers']) if config['n_workers'] != '0' else 4
    loss = "UncertaintyLoss"
    
    model_checkpoint = "./Model/pretrained_model.ckpt"
    
    run_name = 'nnces_test_' + model_type

class NNCESDataset(Dataset):
    def __init__(self, metadata_csv, wav_root, hparams):
        self.hparams = hparams
        self.wav_root = Path(wav_root)
        self.target_sr = 16000
        self.max_samples = 240000
        
        self.df = pd.read_csv(metadata_csv)
        logging.info(f"Loaded {len(self.df)} files from {metadata_csv}")
        
        df_stats = pd.read_csv(hparams.speaker_csv_path)
        self.age_mean = df_stats[df_stats['Use'] == 'TRN']['age'].mean()
        self.age_std = df_stats[df_stats['Use'] == 'TRN']['age'].std()
        self.height_mean = df_stats[df_stats['Use'] == 'TRN']['height'].mean()
        self.height_std = df_stats[df_stats['Use'] == 'TRN']['height'].std()
        
        logging.info(f"TIMIT stats - age: mean={self.age_mean:.1f}, std={self.age_std:.1f}")
        
        # Проверяем существование файлов
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            audio_path = self.wav_root / row['filename']
            if audio_path.exists():
                self.valid_indices.append(idx)
        
        logging.info(f"Valid files: {len(self.valid_indices)} / {len(self.df)}")
    
    def _load_audio(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            elif len(audio) < 48000:
                audio = np.pad(audio, (0, max(0, 48000 - len(audio))), 'constant')
            if np.max(np.abs(audio)) > 0:
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
            return torch.FloatTensor(audio)
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            return None
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        
        audio_path = self.wav_root / row['filename']
        audio = self._load_audio(audio_path)
        if audio is None:
            audio = torch.zeros(48000)
        
        # Нормализация возраста
        age_raw = float(row['age'])
        age_normalized = (age_raw - self.age_mean) / self.age_std
        age = torch.FloatTensor([age_normalized])
        
        height_raw = 160.0 if row['gender'] == 'female' else 165.0
        height_normalized = (height_raw - self.height_mean) / self.height_std
        height = torch.FloatTensor([height_normalized])
        
        gender_value = 1 if row['gender'] == 'female' else 0
        gender = torch.FloatTensor([gender_value])
        
        return audio, height, age, gender, row['filename']
    
    def __len__(self):
        return len(self.valid_indices)

def collate_fn(batch):
    seq, height, age, gender, filename = zip(*batch)
    seql = [x.reshape(-1,) for x in seq]
    seq_length = [x.shape[0] for x in seql]
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    return data, height, age, gender, seq_length, filename

if __name__ == "__main__":
    
    hparams = NNCESConfig()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        hparams.gpu = 0
    
    print(f"\n{'='*60}")
    print(f"INFERENCE ON NNCES DATASET")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model checkpoint: {hparams.model_checkpoint}")
    print(f"Model type: {hparams.model_type}")
    print(f"Upstream model: {hparams.upstream_model}")
    print(f"Feature dim: {hparams.feature_dim}")
    
    if not os.path.exists(hparams.model_checkpoint):
        print(f"\nModel not found: {hparams.model_checkpoint}")
        print("\nSearching for .ckpt files...")
        os.system(f"find . -name '*.ckpt' 2>/dev/null | head -5")
        exit(1)
    
    df_stats = pd.read_csv(hparams.speaker_csv_path)
    age_mean = df_stats[df_stats['Use'] == 'TRN']['age'].mean()
    age_std = df_stats[df_stats['Use'] == 'TRN']['age'].std()
    
    print(f"\nTIMIT Normalization Stats:")
    print(f"  Age mean: {age_mean:.1f}, std: {age_std:.1f}")
    
    dataset = NNCESDataset(
        metadata_csv=hparams.metadata_csv,
        wav_root=hparams.data_path,
        hparams=hparams
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=hparams.n_workers,
        collate_fn=collate_fn
    )
    
    hparams_dict = {
        'model_type': hparams.model_type,
        'upstream_model': hparams.upstream_model,
        'num_layers': hparams.num_layers,
        'feature_dim': hparams.feature_dim,
        'lr': hparams.lr,
        'batch_size': hparams.batch_size,
        'epochs': hparams.epochs,
        'gpu': hparams.gpu,
        'n_workers': hparams.n_workers,
        'narrow_band': hparams.narrow_band,
        'loss': hparams.loss,
        'speaker_csv_path': hparams.speaker_csv_path
    }
    
    print(f"\nLoading model...")
    model = LightningModel.load_from_checkpoint(
        hparams.model_checkpoint, 
        HPARAMS=hparams_dict
    )
    model.to(device)
    model.eval()
    
    print(f"\nRunning inference on {len(dataset)} files...")
    results = []
    
    for batch in tqdm(dataloader, desc="Processing"):
        x, y_h, y_a, y_g, x_len, filename = batch
        x = x.to(device)
        
        with torch.no_grad():
            _, y_hat_a, y_hat_g = model(x, x_len)
        
        age_pred = y_hat_a.item() * age_std + age_mean
        gender_pred = 'female' if y_hat_g.item() > 0.5 else 'male'
        
        results.append({
            'filename': filename[0],
            'age_predicted': age_pred,
            'gender_predicted': gender_pred
        })
    
    # Сохраняем результаты
    results_df = pd.DataFrame(results)
    
    # Добавляем истинные значения
    results_df['age_true'] = [dataset.df.iloc[dataset.valid_indices[i]]['age'] for i in range(len(results))]
    results_df['gender_true'] = [dataset.df.iloc[dataset.valid_indices[i]]['gender'] for i in range(len(results))]
    
    # Вычисляем метрики
    age_mae = mean_absolute_error(results_df['age_true'], results_df['age_predicted'])
    age_rmse = np.sqrt(mean_squared_error(results_df['age_true'], results_df['age_predicted']))
    gender_acc = (results_df['gender_true'] == results_df['gender_predicted']).mean()
    
    print("\nNNCES RESULTS")
    print(f"Age Prediction:")
    print(f"  MAE:  {age_mae:.2f} years")
    print(f"  RMSE: {age_rmse:.2f} years")
    print(f"\nGender Prediction:")
    print(f"  Accuracy: {gender_acc:.2%}")
    print(f"\nTotal samples: {len(results_df)}")
    print(f"{'='*60}")
    
    # Сохраняем CSV
    output_file = "nnces_predictions.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to: {output_file}")
    
    # Пример
    print(f"\nSample predictions (first 10):")
    print(results_df.head(10).to_string())