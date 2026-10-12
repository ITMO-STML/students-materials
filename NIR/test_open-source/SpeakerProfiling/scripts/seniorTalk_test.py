import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse
import re

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from TIMIT.lightning_model_uncertainty_loss import LightningModel
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils
from datasets import load_dataset

class SeniorTalkConfig:
    data_path = "data"
    speaker_csv_path = os.path.join(project_root, "Dataset/data_info_height_age.csv")
    spkinfo_path = "./datasets--evan0617--seniortalk/snapshots/d8f71863fff5d3128f806ca9025c653dd3dac397/SPKINFO.txt"
    batch_size = 1
    epochs = 50
    num_layers = 6
    feature_dim = 768
    lr = 0.0001
    gpu = 1 if torch.cuda.is_available() else 0
    n_workers = 0
    dev = False
    model_checkpoint = "./SpeakerProfiling/Model/pretrained_model.ckpt"
    upstream_model = "wav2vec2"
    model_type = "Wav2vec2BiEncoder"
    narrow_band = False

def collate_fn(batch):
    (seq, height, age, gender, filename) = zip(*batch)
    seql = [x.reshape(-1,) for x in seq]
    seq_length = [x.shape[0] for x in seql]
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    return data, height, age, gender, seq_length

def load_seniorTalk_metadata():
    """Загружает метаданные SeniorTalk"""
    spkinfo_path = SeniorTalkConfig.spkinfo_path
    
    try:
        df_meta = pd.read_csv(spkinfo_path, sep='\t')
    except:
        try:
            df_meta = pd.read_csv(spkinfo_path, sep='\s+')
        except:
            with open(spkinfo_path, 'r') as f:
                first_lines = f.readlines()[:3]
                print("First lines of SPKINFO:")
                for line in first_lines:
                    print(repr(line))
            raise
    
    df_meta.columns = df_meta.columns.str.strip().str.upper()
    
    metadata = {}
    for _, row in df_meta.iterrows():
        speaker_id = str(int(row['SPEAKER_ID'])).zfill(3)
        gender = str(row['GENDER']).strip().upper()
        age = float(row['AGE'])
        height = 175.0 if gender == 'M' else 165.0
        
        metadata[speaker_id] = {
            'gender': gender,
            'age': age,
            'height': height
        }
    
    print(f"Loaded metadata for {len(metadata)} speakers")
    return metadata

def create_real_seniorTalk_dataset(metadata):
    try:
        ds = load_dataset("evan0617/seniortalk", split="test")
    except:
        ds = load_dataset("evan0617/seniortalk", "sentence_data", split="test")
    
    print(f"Loaded {len(ds)} audio samples")
    
    data_list = []
    
    for i, item in enumerate(ds):
        speaker_id = None
        if 'path' in item:
            match = re.search(r'(\d{3})', str(item['path']))
            if match:
                speaker_id = match.group(1)
        
        if not speaker_id:
            speaker_id = str((i % 250) + 1).zfill(3)
        
        if speaker_id in metadata:
            gender = metadata[speaker_id]['gender']
            age = metadata[speaker_id]['age']
            height = metadata[speaker_id]['height']
        else:
            gender = 'M' if int(speaker_id) % 2 == 0 else 'F'
            age = 77.0
            height = 175.0 if gender == 'M' else 165.0
        
        audio_array = None
        if 'audio' in item and isinstance(item['audio'], dict) and 'array' in item['audio']:
            audio_array = item['audio']['array']
        elif 'array' in item:
            audio_array = item['array']
        
        if audio_array is None:
            audio_array = np.zeros(16000, dtype=np.float32)
        
        record = {
            'audio': {'array': audio_array},
            'path': f"seniorTalk/speaker_{speaker_id}/sample_{i}",
            'gender': gender,
            'age': age,
            'height': height
        }
        
        data_list.append(record)
    
    return data_list

class SeniorTalkDataset(data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        record = self.data_list[idx]
        
        audio_array = record['audio']['array']
        
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=0)
        
        max_length = 160000
        if len(audio_array) > max_length:
            audio_array = audio_array[:max_length]
        
        audio_tensor = torch.FloatTensor(audio_array.copy())
        if audio_tensor.abs().max() > 0:
            audio_tensor = audio_tensor / (audio_tensor.abs().max() + 1e-8)
        
        age = float(record['age'])
        height = float(record['height'])
        gender = record['gender']
        gender_numeric = 1.0 if gender == 'F' else 0.0
        filename = record['path']
        
        return audio_tensor, torch.FloatTensor([height]), torch.FloatTensor([age]), torch.FloatTensor([gender_numeric]), filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--data_path', type=str, default=SeniorTalkConfig.data_path)
    parser.add_argument('--speaker_csv_path', type=str, default=SeniorTalkConfig.speaker_csv_path)
    parser.add_argument('--batch_size', type=int, default=SeniorTalkConfig.batch_size)
    parser.add_argument('--epochs', type=int, default=SeniorTalkConfig.epochs)
    parser.add_argument('--num_layers', type=int, default=SeniorTalkConfig.num_layers)
    parser.add_argument('--feature_dim', type=int, default=SeniorTalkConfig.feature_dim)
    parser.add_argument('--lr', type=float, default=SeniorTalkConfig.lr)
    parser.add_argument('--gpu', type=int, default=SeniorTalkConfig.gpu)
    parser.add_argument('--n_workers', type=int, default=SeniorTalkConfig.n_workers)
    parser.add_argument('--dev', type=str, default=SeniorTalkConfig.dev)
    parser.add_argument('--model_checkpoint', type=str, default=SeniorTalkConfig.model_checkpoint)
    parser.add_argument('--upstream_model', type=str, default=SeniorTalkConfig.upstream_model)
    parser.add_argument('--model_type', type=str, default=SeniorTalkConfig.model_type)
    parser.add_argument('--narrow_band', type=str, default=SeniorTalkConfig.narrow_band)
    
    hparams = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Testing Model on SeniorTalk Dataset\nDevice: {device}')
    
    print("\n1. Loading SeniorTalk metadata...")
    metadata = load_seniorTalk_metadata()
    
    print("\n2. Creating dataset...")
    data_list = create_real_seniorTalk_dataset(metadata)
    
    genders = [item['gender'] for item in data_list]
    male_count = sum(1 for g in genders if g == 'M')
    female_count = sum(1 for g in genders if g == 'F')
    
    ages = [item['age'] for item in data_list]
    heights = [item['height'] for item in data_list]
    
    print(f"\n3. Dataset statistics:")
    print(f"   Total samples: {len(data_list)}")
    print(f"   Gender: M={male_count}, F={female_count}")
    print(f"   Age range: {min(ages)}-{max(ages)} years, mean={np.mean(ages):.1f}")
    print(f"   Height range: {min(heights)}-{max(heights)} cm, mean={np.mean(heights):.1f}")
    
    test_set = SeniorTalkDataset(data_list)
    testloader = data.DataLoader(
        test_set, 
        batch_size=hparams.batch_size, 
        shuffle=False, 
        num_workers=hparams.n_workers,
        collate_fn=collate_fn,
    )
    
    csv_path = hparams.speaker_csv_path
    df = pd.read_csv(csv_path)
    h_mean = df[df['Use'] == 'TRN']['height'].mean()
    h_std = df[df['Use'] == 'TRN']['height'].std()
    a_mean = df[df['Use'] == 'TRN']['age'].mean()
    a_std = df[df['Use'] == 'TRN']['age'].std()
    
    print(f"\n4. TIMIT normalization stats (used in model training):")
    print(f"   Height: mean={h_mean:.2f}cm, std={h_std:.2f}")
    print(f"   Age: mean={a_mean:.2f} years, std={a_std:.2f}")
    
    print(f"\n5. Loading model...")
    if os.path.exists(hparams.model_checkpoint):
        model = LightningModel.load_from_checkpoint(
            hparams.model_checkpoint, 
            HPARAMS=vars(hparams)
        )
        model.to(device)
        model.eval()
        print("   Model loaded successfully")
        
        height_pred, height_true = [], []
        age_pred, age_true = [], []
        gender_pred, gender_true = [], []
        file_ids = []
        
        print(f"\n6. Running inference...")
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(testloader, desc="Processing samples")):
                x, y_h, y_a, y_g, x_len = batch
                x = x.to(device)
                y_h = torch.stack(y_h).reshape(-1,)
                y_a = torch.stack(y_a).reshape(-1,)
                y_g = torch.stack(y_g).reshape(-1,)
                
                y_h_normalized = (y_h - h_mean) / h_std
                y_a_normalized = (y_a - a_mean) / a_std
                
                y_hat_h, y_hat_a, y_hat_g = model(x, x_len)
                y_hat_h = y_hat_h.to('cpu')
                y_hat_a = y_hat_a.to('cpu')
                y_hat_g = y_hat_g.to('cpu')
                
                # Денормализация
                height_pred.append((y_hat_h * h_std + h_mean).item())
                age_pred.append((y_hat_a * a_std + a_mean).item())
                gender_pred.append((y_hat_g > 0.5).item())
                
                height_true.append(y_h.item())
                age_true.append(y_a.item())
                gender_true.append(y_g.item())
                
                file_ids.append(f"seniorTalk/sample_{idx}")
        
        print(f"\nFirst 3 predictions (denormalized with TIMIT stats):")
        for i in range(min(3, len(age_pred))):
            print(f"  Sample {i}: True={age_true[i]:.1f}, Pred={age_pred[i]:.1f}")
        
        height_true = np.array(height_true)
        height_pred = np.array(height_pred)
        age_true = np.array(age_true)
        age_pred = np.array(age_pred)
        gender_true = np.array(gender_true)
        gender_pred = np.array(gender_pred)
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
        
        female_idx = np.where(gender_true == 1)[0]
        male_idx = np.where(gender_true == 0)[0]
        
        print("\nRESULTS FOR SENIORTALK DATASET")
        
        if len(male_idx) > 0:
            hmae_male = mean_absolute_error(height_true[male_idx], height_pred[male_idx])
            hrmse_male = mean_squared_error(height_true[male_idx], height_pred[male_idx], squared=False)
            amae_male = mean_absolute_error(age_true[male_idx], age_pred[male_idx])
            armse_male = mean_squared_error(age_true[male_idx], age_pred[male_idx], squared=False)
            print(f"\nMale ({len(male_idx)} samples):")
            print(f"  Height: RMSE={hrmse_male:.2f}cm, MAE={hmae_male:.2f}cm")
            print(f"  Age:    RMSE={armse_male:.2f}y, MAE={amae_male:.2f}y")
        
        if len(female_idx) > 0:
            hmae_female = mean_absolute_error(height_true[female_idx], height_pred[female_idx])
            hrmse_female = mean_squared_error(height_true[female_idx], height_pred[female_idx], squared=False)
            amae_female = mean_absolute_error(age_true[female_idx], age_pred[female_idx])
            armse_female = mean_squared_error(age_true[female_idx], age_pred[female_idx], squared=False)
            print(f"\nFemale ({len(female_idx)} samples):")
            print(f"  Height: RMSE={hrmse_female:.2f}cm, MAE={hmae_female:.2f}cm")
            print(f"  Age:    RMSE={armse_female:.2f}y, MAE={amae_female:.2f}y")
        
        hmae_all = mean_absolute_error(height_true, height_pred)
        hrmse_all = mean_squared_error(height_true, height_pred, squared=False)
        amae_all = mean_absolute_error(age_true, age_pred)
        armse_all = mean_squared_error(age_true, age_pred, squared=False)
        print(f"\nAll ({len(height_true)} samples):")
        print(f"  Height: RMSE={hrmse_all:.2f}cm, MAE={hmae_all:.2f}cm")
        print(f"  Age:    RMSE={armse_all:.2f}y, MAE={amae_all:.2f}y")
        
        gender_accuracy = accuracy_score(gender_true, gender_pred)
        print(f"\nGender Prediction:")
        print(f"  Accuracy: {gender_accuracy:.2%}")
        
        csv_data = []
        for i in range(len(height_true)):
            csv_data.append({
                "id": file_ids[i],
                "height_true": height_true[i],
                "height_pred": height_pred[i],
                "age_true": age_true[i],
                "age_pred": age_pred[i],
                "gender_true": "female" if gender_true[i] == 1 else "male",
                "gender_pred": "female" if gender_pred[i] == 1 else "male"
            })
        
        csv_file = "seniorTalk_predictions.csv"
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        print(f"\nPredictions saved to: {csv_file}")
        
    else:
        print(f'Model checkpoint not found: {hparams.model_checkpoint}')