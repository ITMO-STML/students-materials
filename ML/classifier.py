from dataclasses import dataclass
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchvision
from torchvision import transforms
import cv2
from PIL import Image
import os
from typing import Tuple, List
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

load_data_percent = 1 # 0.01

class Config:
    def __init__(self, path):
        self.path_to_dataset_folder = path
        self.path_to_Video_folder = self.path_to_dataset_folder + 'Video/Combined/'
        self.path_to_Audio_folder = self.path_to_dataset_folder + 'Audio/WAV_16000/'
        self.path_to_Train_csv = self.path_to_dataset_folder + '/Data_Train_modified.csv'
        self.path_to_Test_csv = self.path_to_dataset_folder + '/Data_Test_original.csv'
        self.path_to_Val_csv = self.path_to_dataset_folder + '/Data_Val_modified.csv'

class CMUMOSEIDataset(Dataset):
    def __init__(self, csv_path, video_dir, audio_dir, transform=None, max_length=3):
        full_data = pd.read_csv(csv_path)

        self.data = full_data.sample(frac=load_data_percent, random_state=42)
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.transform = transform
        self.max_length = max_length
        self.target_sample_rate = 16000
        self.target_audio_length = self.target_sample_rate * self.max_length
        
        # Convert sentiment to 3 classes: negative (0), neutral (1), positive (2)
        self.data['sentiment_class'] = self.data['sentiment'].apply(self._sentiment_to_class)
        
        print(f"Loaded {len(self.data)} samples from {csv_path}")
    
    def _sentiment_to_class(self, sentiment):
        if sentiment < -0.5:
            return 0  # negative
        elif sentiment > 0.5:
            return 2  # positive
        else:
            return 1  # neutral
    
    def __len__(self):
        return len(self.data)
    
    def _extract_audio_features(self, audio_path, start_time, end_time):
        if not os.path.exists(audio_path):
            return torch.zeros(40)
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        end_sample = min(end_sample, waveform.shape[1])
        
        if start_sample >= end_sample:
            return torch.zeros(40)
        
        audio_segment = waveform[:, start_sample:end_sample]
        
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_segment = resampler(audio_segment)
        
        if audio_segment.shape[0] > 1:
            audio_segment = torch.mean(audio_segment, dim=0, keepdim=True)
        
        current_length = audio_segment.shape[1]
        if current_length < self.target_audio_length:
            padding = self.target_audio_length - current_length
            audio_segment = torch.nn.functional.pad(audio_segment, (0, padding))
        else:
            audio_segment = audio_segment[:, :self.target_audio_length]
        
        # Используем Mel-спектры
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=40
        )
        
        mel_features = mel_transform(audio_segment)
        log_mel = torch.log(mel_features + 1e-6)
        mean_features = log_mel.mean(dim=2)
        
        return mean_features.squeeze()
            
    
    def _extract_video_features(self, video_path, start_time, end_time):
        if not os.path.exists(video_path):
            return torch.zeros(5, 3, 64, 64)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return torch.zeros(5, 3, 64, 64)
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps is None:
            fps = 25
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return torch.zeros(5, 3, 64, 64)
        
        # Sample frames
        frames_to_sample = 5
        frame_indices = []
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames - 1))
        
        if end_frame <= start_frame:
            # Если временной интервал некорректный, берем первые кадры
            if total_frames > 0:
                frame_indices = np.linspace(0, min(total_frames-1, 24), frames_to_sample, dtype=int)
            else:
                frame_indices = [0] * frames_to_sample
        else:
            frame_indices = np.linspace(start_frame, end_frame, frames_to_sample, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            
            # Правильная проверка на успешное чтение кадра
            if ret and frame is not None and isinstance(frame, np.ndarray):
                # Проверяем размеры кадра
                if frame.size > 0 and len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    if self.transform:
                        frame = self.transform(frame)
                    frames.append(frame)
                else:
                    # Невалидный кадр - добавляем черный
                    black_frame = torch.zeros(3, 64, 64)
                    frames.append(black_frame)
            else:
                black_frame = torch.zeros(3, 64, 64)
                frames.append(black_frame)
        
        cap.release()
        
        if len(frames) == 0:
            return torch.zeros(5, 3, 64, 64)
        
        # Убеждаемся, что все кадры имеют правильную форму
        processed_frames = []
        for frame in frames:
            if isinstance(frame, torch.Tensor) and frame.shape == (3, 64, 64):
                processed_frames.append(frame)
            else:
                processed_frames.append(torch.zeros(3, 64, 64))
        
        video_tensor = torch.stack(processed_frames)
        return video_tensor
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        video_id = row['video']
        start_time = float(row['start_time'])
        end_time = float(row['end_time'])
        sentiment_class = row['sentiment_class']
        
        # Load audio features
        audio_path = os.path.join(self.audio_dir, f"{video_id}.wav")
        if not os.path.exists(audio_path):
            audio_path = os.path.join(self.audio_dir, video_id)
            
        audio_features = self._extract_audio_features(audio_path, start_time, end_time)
        
        # Load video features
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(self.video_dir, video_id)
            
        video_features = self._extract_video_features(video_path, start_time, end_time)
        
        return {
            'audio_features': audio_features.float(),
            'video_features': video_features.float(),
            'labels': torch.tensor(sentiment_class, dtype=torch.long)
        }

def collate_fn(batch):
    audio_features = [item['audio_features'] for item in batch]
    video_features = [item['video_features'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    audio_features = torch.stack(audio_features)
    video_features = torch.stack(video_features)
    labels = torch.stack(labels)
    
    return {
        'audio_features': audio_features,
        'video_features': video_features,
        'labels': labels
    }

class SimpleAudioEncoder(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=64):
        super(SimpleAudioEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        return self.encoder(x)

class SimpleVideoEncoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super(SimpleVideoEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.feature_size = 64
        self.proj = nn.Sequential(
            nn.Linear(self.feature_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(batch_size * num_frames, C, H, W)
        
        features = self.cnn(x)
        features = features.view(batch_size, num_frames, -1)
        features = torch.mean(features, dim=1)
        return self.proj(features)

class AttentionFusion(nn.Module):
    def __init__(self, audio_dim, video_dim, hidden_dim=32):
        super(AttentionFusion, self).__init__()
        self.audio_attention = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.video_attention = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, audio_features, video_features):
        audio_weights = self.audio_attention(audio_features)
        video_weights = self.video_attention(video_features)
        
        attention_weights = torch.cat([audio_weights, video_weights], dim=1)
        attention_weights = self.softmax(attention_weights)
        
        audio_weighted = audio_features * attention_weights[:, 0:1]
        video_weighted = video_features * attention_weights[:, 1:2]
        
        fused_features = audio_weighted + video_weighted
        return fused_features, attention_weights

class MultimodalSentimentClassifier(nn.Module):
    def __init__(self, num_classes=3, audio_hidden_dim=64, video_hidden_dim=64):
        super(MultimodalSentimentClassifier, self).__init__()
        self.audio_encoder = SimpleAudioEncoder(hidden_dim=audio_hidden_dim)
        self.video_encoder = SimpleVideoEncoder(hidden_dim=video_hidden_dim)
        self.attention_fusion = AttentionFusion(audio_hidden_dim, video_hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(audio_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, audio_input, video_input):
        audio_features = self.audio_encoder(audio_input)
        video_features = self.video_encoder(video_input)
        
        fused_features, attention_weights = self.attention_fusion(audio_features, video_features)
        
        logits = self.classifier(fused_features)
        return logits, attention_weights

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        audio_features = batch['audio_features'].to(device, dtype=torch.float32)
        video_features = batch['video_features'].to(device, dtype=torch.float32)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits, _ = model(audio_features, video_features)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        current_loss = total_loss / (batch_idx + 1)
        current_acc = 100 * correct / total
        pbar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Acc': f'{current_acc:.2f}%'
        })
    
    if total == 0:
        return 0, 0
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            audio_features = batch['audio_features'].to(device, dtype=torch.float32)
            video_features = batch['video_features'].to(device, dtype=torch.float32)
            labels = batch['labels'].to(device)
            
            logits, _ = model(audio_features, video_features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100 * correct / total
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
    
    if total == 0:
        return 0, 0
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='Audio and Video sentiment classifier')
    parser.add_argument('-P', '--path', help='Path to dataset dir', default='/home/danya/datasets/CMU-MOSEI/')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     device = 'cpu'
    print(f"Using device: {device}")
    
    config = Config(args.path)
    
    video_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading datasets...")
    
    train_dataset = CMUMOSEIDataset(
        config.path_to_Train_csv,
        config.path_to_Video_folder,
        config.path_to_Audio_folder,
        transform=video_transform,
        max_length=3
    )
    
    val_dataset = CMUMOSEIDataset(
        config.path_to_Val_csv,
        config.path_to_Video_folder,
        config.path_to_Audio_folder,
        transform=video_transform,
        max_length=3
    )
    
    test_dataset = CMUMOSEIDataset(
        config.path_to_Test_csv,
        config.path_to_Video_folder,
        config.path_to_Audio_folder,
        transform=video_transform,
        max_length=3
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn
    )
    
    model = MultimodalSentimentClassifier(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print("="*70)
    
    best_val_accuracy = 0
    
    epoch_pbar = tqdm(range(args.epochs), desc="Epochs")
    for epoch in epoch_pbar:
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Train Acc': f'{train_acc:.2f}%',
            'Val Acc': f'{val_acc:.2f}%'
        })
        
        print(f'\nEpoch {epoch+1}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'  🎯 New best model saved! Validation accuracy: {val_acc:.2f}%')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    main()
