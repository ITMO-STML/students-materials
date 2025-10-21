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

# Fix for NumPy compatibility
np.int = int
np.float = float
np.bool = bool

load_data_percent = 1

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

        # Convert sentiment to 3 classes
        self.data['sentiment_class'] = self.data['sentiment'].apply(self._sentiment_to_class)
        
        class_counts = self.data['sentiment_class'].value_counts()
        print(f"Class distribution: {dict(class_counts)}")

        print(f"Loaded {len(self.data)} samples from {csv_path}")

    def _sentiment_to_class(self, sentiment):
        if sentiment < -0.2:
            return 0  # negative
        elif sentiment > 0.2:
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

        frames_to_sample = 5
        frame_indices = []

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames - 1))

        if end_frame <= start_frame:
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

            if ret and frame is not None and frame.size > 0 and len(frame.shape) == 3:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            else:
                black_frame = torch.zeros(3, 64, 64)
                frames.append(black_frame)

        cap.release()

        if len(frames) == 0:
            return torch.zeros(5, 3, 64, 64)

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

        # Audio features
        audio_path = os.path.join(self.audio_dir, f"{video_id}.wav")
        if not os.path.exists(audio_path):
            audio_path = os.path.join(self.audio_dir, video_id)
        audio_features = self._extract_audio_features(audio_path, start_time, end_time)

        # Video features
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

class ImprovedAudioEncoder(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128):
        super(ImprovedAudioEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.encoder(x)

class ImprovedVideoEncoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super(ImprovedVideoEncoder, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)
        )
        
        self.feature_size = 128 * 4 * 4
        
        self.frame_proj = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.temporal_agg = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        
        # Process each frame
        x = x.view(batch_size * num_frames, C, H, W)
        features = self.cnn(x)
        features = features.view(batch_size * num_frames, -1)
        frame_features = self.frame_proj(features)
        
        # Reshape back to (batch_size, num_frames, feature_dim)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        
        # Aggregate temporal information (mean pooling)
        aggregated = torch.mean(frame_features, dim=1)
        
        return self.temporal_agg(aggregated)

class CrossModalFusion(nn.Module):
    def __init__(self, audio_dim, video_dim, fusion_dim=256):
        super(CrossModalFusion, self).__init__()
        
        # Project both modalities to common space
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion mechanism
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, audio_features, video_features):
        audio_proj = self.audio_proj(audio_features)
        video_proj = self.video_proj(video_features)
        
        # Concatenate features
        fused = torch.cat([audio_proj, video_proj], dim=1)
        fused = self.fusion(fused)
        
        return fused

class WorkingMultimodalClassifier(nn.Module):
    def __init__(self, num_classes=3, audio_hidden_dim=128, video_hidden_dim=128):
        super(WorkingMultimodalClassifier, self).__init__()
        
        self.audio_encoder = ImprovedAudioEncoder(hidden_dim=audio_hidden_dim)
        self.video_encoder = ImprovedVideoEncoder(hidden_dim=video_hidden_dim)
        self.fusion = CrossModalFusion(audio_hidden_dim, video_hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, audio_input, video_input):
        audio_features = self.audio_encoder(audio_input)
        video_features = self.video_encoder(video_input)
        
        fused_features = self.fusion(audio_features, video_features)
        logits = self.classifier(fused_features)
        
        return logits, None

def calculate_class_weights(dataset):
    labels = dataset.data['sentiment_class'].values
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    weights = total_samples / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32)

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

    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0

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

    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0

    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='Working Audio and Video sentiment classifier')
    parser.add_argument('-P', '--path', help='Path to dataset dir', default='/home/danya/datasets/CMU-MOSEI/')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')  # Reduced learning rate

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = Config(args.path)

    video_transform = transforms.Compose([
        transforms.Resize((512, 512)),
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

    class_weights = calculate_class_weights(train_dataset)
    print(f"Class weights: {class_weights}")

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

    model = WorkingMultimodalClassifier(num_classes=3).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    print("="*70)
    print("STARTING TRAINING WITH WORKING MODEL")
    print("="*70)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Using weighted loss: {class_weights.tolist()}")
    print("="*70)

    best_val_accuracy = 0
    patience = 7
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}:')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Current LR: {optimizer.param_groups[0]["lr"]:.2e}')

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'  🎯 New best model saved! Validation accuracy: {val_acc:.2f}%')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'  ⏳ No improvement for {patience_counter} epochs')

        if patience_counter >= patience:
            print(f'  🛑 Early stopping after {epoch+1} epochs')
            break

    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")

    # Test the best model
    print("\nTesting best model...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
