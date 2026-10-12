import torch
import sys, os
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

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
from TIMIT.lightning_model_uncertainty_loss import LightningModel

logging.basicConfig(format='%(asctime)s %(levelname)-3s ==> %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

class VoiceBioConfig(object):
    
    with open("config.json", "r") as jsonfile:
        config = json.load(jsonfile)
    
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
    
    speaker_csv_path = "./SpeakerProfiling/Dataset/data_info_height_age.csv"

class VoiceBiometricsDataset(Dataset):
    def __init__(self, dataset_path, df_persons, hparams):
        self.hparams = hparams
        self.dataset_path = dataset_path
        self.df_persons = df_persons
        self.target_sr = 16000
        self.max_samples = 240000
        
        csv_path = hparams.speaker_csv_path
        df = pd.read_csv(csv_path)
        self.age_mean = df[df['Use'] == 'TRN']['age'].mean()
        self.age_std = df[df['Use'] == 'TRN']['age'].std()
        self.height_mean = df[df['Use'] == 'TRN']['height'].mean()
        self.height_std = df[df['Use'] == 'TRN']['height'].std()
        
        logging.info(f"VoiceBiometricsDataset: Using TIMIT stats - age_mean={self.age_mean:.1f}, age_std={self.age_std:.1f}")
        
        self.valid_items = []
        self._collect_files()
        
        logging.info(f"VoiceBiometricsDataset: загружено {len(self.valid_items)} записей")
    
    def _collect_files(self):
        id_to_birth = {row['id']: row['date of birth'].year 
                      for _, row in self.df_persons.iterrows() 
                      if pd.notna(row['date of birth'])}
        
        for person_dir in Path(self.dataset_path).iterdir():
            if not person_dir.is_dir() or not person_dir.name.startswith('id'):
                continue
            
            person_id = person_dir.name
            if person_id not in id_to_birth:
                continue
            
            birth_year = id_to_birth[person_id]
            
            for year_dir in person_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                
                try:
                    record_year = int(year_dir.name)
                    age = record_year - birth_year
                    
                    if age < 0 or age > 100:
                        continue
                    
                    for audio_file in year_dir.glob('*.wav'):
                        self.valid_items.append({
                            'id': person_id,
                            'path': str(audio_file),
                            'year': record_year,
                            'age_raw': age
                        })
                except ValueError:
                    continue
    
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
        
        audio = self._load_audio(item['path'])
        if audio is None:
            audio = torch.zeros(48000)
        
        age_normalized = (item['age_raw'] - self.age_mean) / self.age_std
        height_normalized = (175.0 - self.height_mean) / self.height_std
        
        return audio, torch.FloatTensor([height_normalized]), torch.FloatTensor([age_normalized]), torch.FloatTensor([0])
    
    def __len__(self):
        return len(self.valid_items)

def collate_fn(batch):
    seq, height, age, gender = zip(*batch)
    seql = [x.reshape(-1,) for x in seq]
    seq_length = [x.shape[0] for x in seql]
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    return data, height, age, gender, seq_length

if __name__ == "__main__":
    dataset_path = "./voice_biometrics_age/voice_biometrics_age"
    excel_path = "./voice_biometrics_age/metadata.xlsx"
    output_csv = "voicebio_predictions.csv"
    
    df_persons = pd.read_excel(excel_path)
    df_persons['date of birth'] = pd.to_datetime(df_persons['date of birth'], errors='coerce')
    
    hparams = VoiceBioConfig()
    
    df_stats = pd.read_csv(hparams.speaker_csv_path)
    a_mean = df_stats[df_stats['Use'] == 'TRN']['age'].mean()
    a_std = df_stats[df_stats['Use'] == 'TRN']['age'].std()
    
    dataset = VoiceBiometricsDataset(dataset_path, df_persons, hparams)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Использую {device}')
    
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
    
    model = LightningModel.load_from_checkpoint(
        hparams.model_checkpoint, 
        HPARAMS=hparams_dict
    )
    model.to(device)
    model.eval()
    
    results = []
    
    for idx, batch in enumerate(tqdm(dataloader, desc="Обработка")):
        x, y_h, y_a, y_g, x_len = batch
        x = x.to(device)
        
        with torch.no_grad():
            y_hat_h, y_hat_a, y_hat_g = model(x, x_len)
        
        pred_age = y_hat_a.item() * a_std + a_mean
        true_age = dataset.valid_items[idx]['age_raw']
        
        results.append({
            'id': dataset.valid_items[idx]['id'],
            'year': dataset.valid_items[idx]['year'],
            'true_age': true_age,
            'pred_age': round(pred_age, 2),
            'abs_error': round(abs(true_age - pred_age), 2)
        })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    
    age_true = df_results['true_age'].values
    age_pred = df_results['pred_age'].values
    
    amae = mean_absolute_error(age_true, age_pred)
    armse = np.sqrt(mean_squared_error(age_true, age_pred))
    abs_errors = np.abs(age_true - age_pred)
    
    print("\nVOICE BIOMETRICS AGE - TIMIT MODEL RESULTS")
    print(f"Всего записей: {len(df_results)}")
    print(f"Всего говорящих: {df_results['id'].nunique()}")
    print(f"\nAge MAE  = {amae:.2f} years")
    print(f"Age RMSE = {armse:.2f} years")
    print(f"Std of Error = {np.std(abs_errors):.2f} years")
    print(f"\nError distribution:")
    print(f"  ≤ 1 year:  {np.mean(abs_errors <= 1)*100:.1f}%")
    print(f"  ≤ 3 years: {np.mean(abs_errors <= 3)*100:.1f}%")
    print(f"  ≤ 5 years: {np.mean(abs_errors <= 5)*100:.1f}%")
    print(f"  ≤ 10 years: {np.mean(abs_errors <= 10)*100:.1f}%")
    print(f"\nРезультаты сохранены в: {output_csv}")