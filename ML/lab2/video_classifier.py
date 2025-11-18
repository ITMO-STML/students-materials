import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

class LabelEncoderSimple:
    def __init__(self):
        self.classes_ = []
        self.class_to_idx = {}
        
    def fit(self, labels):
        self.classes_ = sorted(list(set(labels)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self
        
    def transform(self, labels):
        if isinstance(labels, str):
            return self.class_to_idx[labels]
        return [self.class_to_idx[label] for label in labels]
        
    def inverse_transform(self, indices):
        if isinstance(indices, int):
            return self.classes_[indices]
        return [self.classes_[idx] for idx in indices]

def cv2_to_tensor_fast(frame, target_size=(112, 112)):
    frame_resized = cv2.resize(frame, target_size)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    tensor = torch.zeros(3, target_size[1], target_size[0], dtype=torch.float32)
    for c in range(3):
        for h in range(target_size[1]):
            for w in range(target_size[0]):
                tensor[c, h, w] = float(frame_rgb[h, w, c]) / 255.0
    return tensor

class VideoDataset(Dataset):
    def __init__(self, dataset, video_path, max_frames=8, clip_duration_sec=0.5):
        self.dataset = dataset
        self.video_path = video_path
        self.max_frames = max_frames
        self.clip_duration_sec = clip_duration_sec
        self.clip_entries = []
        
        age_groups = []
        genders = []
        languages = []
        
        for idx, item in enumerate(dataset):
            info = item['info']
            video_id = item['id']
            video_file = os.path.join(self.video_path, video_id + '.mp4')
            
            if (not info['Age Group'] or not info['Gender'] or not info['Language'] or 
                not os.path.exists(video_file)):
                continue
            
            age_str = str(info['Age Group'])
            gender_str = str(info['Gender'])
            language_str = str(info['Language'])
            
            age_groups.append(age_str)
            genders.append(gender_str)
            languages.append(language_str)
            
            cap = cv2.VideoCapture(video_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 1
            cap.release()
            
            if total_frames <= 0 or fps <= 0:
                continue
                
            clip_len = max(int(fps * self.clip_duration_sec), 1)
            num_clips = min(2, total_frames // clip_len)
            
            for i in range(num_clips):
                start = i * clip_len
                end = min(start + clip_len, total_frames)
                if end - start >= 1:
                    self.clip_entries.append((video_file, start, end, age_str, gender_str, language_str))
        
        self.age_encoder = LabelEncoderSimple()
        self.gender_encoder = LabelEncoderSimple()
        self.language_encoder = LabelEncoderSimple()
        
        self.age_encoder.fit(age_groups)
        self.gender_encoder.fit(genders)
        self.language_encoder.fit(languages)
    
    def __len__(self):
        return len(self.clip_entries)
    
    def __getitem__(self, idx):
        video_file, start_frame, end_frame, age_str, gender_str, language_str = self.clip_entries[idx]
        
        frames = self.load_video_clip(video_file, start_frame, end_frame)
        
        age = self.age_encoder.transform(age_str)
        gender = self.gender_encoder.transform(gender_str)
        language = self.language_encoder.transform(language_str)
        
        return frames, age, gender, language
    
    def load_video_clip(self, video_path, start_frame, end_frame):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0 or start_frame >= total_frames:
            cap.release()
            return torch.zeros((self.max_frames, 3, 112, 112), dtype=torch.float32)
        
        end_frame = min(end_frame, total_frames)
        window_len = end_frame - start_frame
        
        frame_indices = []
        if self.max_frames == 1:
            frame_indices = [start_frame]
        else:
            for i in range(self.max_frames):
                rel_pos = int(i * (window_len - 1) / (self.max_frames - 1)) if self.max_frames > 1 else 0
                frame_indices.append(start_frame + rel_pos)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                tensor = cv2_to_tensor_fast(frame, (112, 112))
                frames.append(tensor)
            else:
                frames.append(torch.zeros((3, 112, 112), dtype=torch.float32))
        
        cap.release()
        
        if not frames:
            return torch.zeros((self.max_frames, 3, 112, 112), dtype=torch.float32)
        
        frames_tensor = torch.stack(frames)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames_tensor = (frames_tensor - mean) / std
        
        return frames_tensor

class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
    
    def forward(self, x):
        return self.conv3d(x)

class TaskSpecificHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TaskSpecificHead, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class MTLVideoModel(nn.Module):
    def __init__(self, num_age_classes, num_gender_classes, num_language_classes):
        super(MTLVideoModel, self).__init__()
        self.shared_encoder = SharedEncoder()
        self.age_head = TaskSpecificHead(16, num_age_classes)
        self.gender_head = TaskSpecificHead(16, num_gender_classes)
        self.language_head = TaskSpecificHead(16, num_language_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        shared_features = self.shared_encoder(x)
        shared_features = shared_features.view(shared_features.size(0), -1)
        
        age_out = self.age_head(shared_features)
        gender_out = self.gender_head(shared_features)
        language_out = self.language_head(shared_features)
        
        return age_out, gender_out, language_out

class SingleTaskModel(nn.Module):
    def __init__(self, num_classes):
        super(SingleTaskModel, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = nn.Linear(16, num_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        features = self.conv3d(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

def train_mtl_epoch(model, dataloader, device, optimizer, criteria, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    age_correct = 0
    gender_correct = 0
    language_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs}', leave=False)
    
    for batch_idx, (frames, age_target, gender_target, language_target) in enumerate(pbar):
        frames = frames.to(device)
        age_target = age_target.to(device)
        gender_target = gender_target.to(device)
        language_target = language_target.to(device)
        
        optimizer.zero_grad()
        
        outputs_age, outputs_gender, outputs_language = model(frames)
        
        loss_age = criteria[0](outputs_age, age_target)
        loss_gender = criteria[1](outputs_gender, gender_target)
        loss_language = criteria[2](outputs_language, language_target)
        
        total_batch_loss = loss_age + loss_gender + loss_language
        total_batch_loss.backward()
        optimizer.step()
        
        total_loss += total_batch_loss.item()
        
        age_pred = outputs_age.argmax(dim=1)
        gender_pred = outputs_gender.argmax(dim=1)
        language_pred = outputs_language.argmax(dim=1)
        
        age_correct += (age_pred == age_target).sum().item()
        gender_correct += (gender_pred == gender_target).sum().item()
        language_correct += (language_pred == language_target).sum().item()
        total_samples += age_target.size(0)
        
        # Update progress bar
        current_loss = total_loss / (batch_idx + 1)
        age_acc = age_correct / total_samples
        gender_acc = gender_correct / total_samples
        language_acc = language_correct / total_samples
        
        pbar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Age': f'{age_acc:.4f}',
            'Gender': f'{gender_acc:.4f}',
            'Lang': f'{language_acc:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    age_acc = age_correct / total_samples
    gender_acc = gender_correct / total_samples
    language_acc = language_correct / total_samples
    
    return avg_loss, age_acc, gender_acc, language_acc

def train_single_epoch(model, dataloader, device, optimizer, criterion, task_idx, epoch, total_epochs, task_name):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f'{task_name} Epoch {epoch}/{total_epochs}', leave=False)
    
    for batch_idx, (frames, age_target, gender_target, language_target) in enumerate(pbar):
        if task_idx == 0:
            targets = age_target
        elif task_idx == 1:
            targets = gender_target
        else:
            targets = language_target
            
        frames = frames.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total_samples += targets.size(0)
        
        # Update progress bar
        current_loss = total_loss / (batch_idx + 1)
        accuracy = correct / total_samples
        
        pbar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Acc': f'{accuracy:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_samples
    
    return avg_loss, accuracy

def evaluate_mtl(model, dataloader, device, criteria):
    model.eval()
    total_loss = 0.0
    age_correct = 0
    gender_correct = 0
    language_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc='Validating', leave=False)
    
    with torch.no_grad():
        for frames, age_target, gender_target, language_target in pbar:
            frames = frames.to(device)
            age_target = age_target.to(device)
            gender_target = gender_target.to(device)
            language_target = language_target.to(device)
            
            outputs_age, outputs_gender, outputs_language = model(frames)
            
            loss_age = criteria[0](outputs_age, age_target)
            loss_gender = criteria[1](outputs_gender, gender_target)
            loss_language = criteria[2](outputs_language, language_target)
            
            total_batch_loss = loss_age + loss_gender + loss_language
            total_loss += total_batch_loss.item()
            
            age_pred = outputs_age.argmax(dim=1)
            gender_pred = outputs_gender.argmax(dim=1)
            language_pred = outputs_language.argmax(dim=1)
            
            age_correct += (age_pred == age_target).sum().item()
            gender_correct += (gender_pred == gender_target).sum().item()
            language_correct += (language_pred == language_target).sum().item()
            total_samples += age_target.size(0)
            
            # Update progress bar
            current_loss = total_loss / len(dataloader)
            age_acc = age_correct / total_samples
            gender_acc = gender_correct / total_samples
            language_acc = language_correct / total_samples
            
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Age': f'{age_acc:.4f}',
                'Gender': f'{gender_acc:.4f}',
                'Lang': f'{language_acc:.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    age_acc = age_correct / total_samples
    gender_acc = gender_correct / total_samples
    language_acc = language_correct / total_samples
    
    return avg_loss, age_acc, gender_acc, language_acc

def evaluate_single(model, dataloader, device, criterion, task_idx, task_name):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f'Validating {task_name}', leave=False)
    
    with torch.no_grad():
        for frames, age_target, gender_target, language_target in pbar:
            if task_idx == 0:
                targets = age_target
            elif task_idx == 1:
                targets = gender_target
            else:
                targets = language_target
                
            frames = frames.to(device)
            targets = targets.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total_samples += targets.size(0)
            
            # Update progress bar
            current_loss = total_loss / len(dataloader)
            accuracy = correct / total_samples
            
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{accuracy:.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_samples
    
    return avg_loss, accuracy

def run_ablation_study(train_loader, val_loader, num_age_classes, num_gender_classes, num_language_classes, device, ablation_epochs):
    print("Running Ablation Study...")
    
    criterion_age = nn.CrossEntropyLoss()
    criterion_gender = nn.CrossEntropyLoss()
    criterion_language = nn.CrossEntropyLoss()
    
    mtl_model = MTLVideoModel(num_age_classes, num_gender_classes, num_language_classes).to(device)
    age_model = SingleTaskModel(num_age_classes).to(device)
    gender_model = SingleTaskModel(num_gender_classes).to(device)
    language_model = SingleTaskModel(num_language_classes).to(device)
    
    mtl_optimizer = optim.Adam(mtl_model.parameters(), lr=0.001)
    age_optimizer = optim.Adam(age_model.parameters(), lr=0.001)
    gender_optimizer = optim.Adam(gender_model.parameters(), lr=0.001)
    language_optimizer = optim.Adam(language_model.parameters(), lr=0.001)
    
    print(f"\nTraining MTL model ({ablation_epochs} epochs)...")
    for epoch in range(ablation_epochs):
        mtl_loss, mtl_age_acc, mtl_gender_acc, mtl_lang_acc = train_mtl_epoch(
            mtl_model, train_loader, device, mtl_optimizer, 
            [criterion_age, criterion_gender, criterion_language], epoch + 1, ablation_epochs
        )
        print(f"MTL Epoch {epoch + 1}/{ablation_epochs} - Loss: {mtl_loss:.4f}, Age: {mtl_age_acc:.4f}, Gender: {mtl_gender_acc:.4f}, Lang: {mtl_lang_acc:.4f}")
    
    mtl_val_loss, mtl_val_age, mtl_val_gender, mtl_val_lang = evaluate_mtl(
        mtl_model, val_loader, device, [criterion_age, criterion_gender, criterion_language]
    )
    
    print(f"\nTraining single-task age model ({ablation_epochs} epochs)...")
    for epoch in range(ablation_epochs):
        age_loss, age_acc = train_single_epoch(age_model, train_loader, device, age_optimizer, criterion_age, 0, epoch + 1, ablation_epochs, "Age")
        print(f"Age Epoch {epoch + 1}/{ablation_epochs} - Loss: {age_loss:.4f}, Acc: {age_acc:.4f}")
    
    age_val_loss, age_val_acc = evaluate_single(age_model, val_loader, device, criterion_age, 0, "Age")
    
    print(f"\nTraining single-task gender model ({ablation_epochs} epochs)...")
    for epoch in range(ablation_epochs):
        gender_loss, gender_acc = train_single_epoch(gender_model, train_loader, device, gender_optimizer, criterion_gender, 1, epoch + 1, ablation_epochs, "Gender")
        print(f"Gender Epoch {epoch + 1}/{ablation_epochs} - Loss: {gender_loss:.4f}, Acc: {gender_acc:.4f}")
    
    gender_val_loss, gender_val_acc = evaluate_single(gender_model, val_loader, device, criterion_gender, 1, "Gender")
    
    print(f"\nTraining single-task language model ({ablation_epochs} epochs)...")
    for epoch in range(ablation_epochs):
        lang_loss, lang_acc = train_single_epoch(language_model, train_loader, device, language_optimizer, criterion_language, 2, epoch + 1, ablation_epochs, "Language")
        print(f"Language Epoch {epoch + 1}/{ablation_epochs} - Loss: {lang_loss:.4f}, Acc: {lang_acc:.4f}")
    
    lang_val_loss, lang_val_acc = evaluate_single(language_model, val_loader, device, criterion_language, 2, "Language")
    
    print("\n" + "="*50)
    print("Ablation Study Results")
    print("="*50)
    print(f"MTL Model      - Age: {mtl_val_age:.4f}, Gender: {mtl_val_gender:.4f}, Language: {mtl_val_lang:.4f}")
    print(f"Single Task    - Age: {age_val_acc:.4f}, Gender: {gender_val_acc:.4f}, Language: {lang_val_acc:.4f}")
    
    improvement_age = mtl_val_age - age_val_acc
    improvement_gender = mtl_val_gender - gender_val_acc
    improvement_language = mtl_val_lang - lang_val_acc
    
    print(f"\nMTL Improvement - Age: {improvement_age:+.4f}, Gender: {improvement_gender:+.4f}, Language: {improvement_language:+.4f}")
    
    results = {
        'mtl': {'age': mtl_val_age, 'gender': mtl_val_gender, 'language': mtl_val_lang},
        'single_age': age_val_acc,
        'single_gender': gender_val_acc,
        'single_language': lang_val_acc
    }
    
    return results, mtl_model

if __name__ == "__main__":
    # Hyperparameters
    MAX_FRAMES = 8
    BATCH_SIZE = 2
    LIMIT_VIDEOS = 5
    ABLATION_EPOCHS = 2  # Количество эпох для ablation study
    FINAL_EPOCHS = 3     # Количество эпох для финального обучения

    print("Loading dataset...")
    ds = load_dataset("FreedomIntelligence/TalkVid")
    path_to_videos = os.getenv('VIDEO_PATH', '/home/danya/datasets/VidTalk/dataset_videos/')

    base = ds['test']
    if LIMIT_VIDEOS is not None:
        base = base.select(range(min(LIMIT_VIDEOS, len(base))))

    dataset = VideoDataset(base, path_to_videos, max_frames=MAX_FRAMES, clip_duration_sec=0.5)

    all_indices = list(range(len(dataset)))
    if len(all_indices) > 1:
        train_idx, val_idx = train_test_split(all_indices, test_size=0.2, random_state=42)
    else:
        train_idx, val_idx = [0], [0]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset size: {len(dataset)} | train={len(train_dataset)}, val={len(val_dataset)}")
    print(f"Age classes: {dataset.age_encoder.classes_}")
    print(f"Gender classes: {dataset.gender_encoder.classes_}")
    print(f"Language classes: {dataset.language_encoder.classes_}")
    print(f"Ablation epochs: {ABLATION_EPOCHS}, Final epochs: {FINAL_EPOCHS}")

    num_age_classes = len(dataset.age_encoder.classes_)
    num_gender_classes = len(dataset.gender_encoder.classes_)
    num_language_classes = len(dataset.language_encoder.classes_)

    device = torch.device('cpu')
    print(f"Using device: {device}")

    if len(dataset) > 0:
        # Ablation study
        ablation_results, model = run_ablation_study(
            train_loader, val_loader, num_age_classes, num_gender_classes, num_language_classes, device, ABLATION_EPOCHS
        )

        # Финальное обучение лучшей модели (MTL)
        print("\n" + "="*50)
        print(f"Final MTL Training ({FINAL_EPOCHS} epochs)")
        print("="*50)
        
        criterion_age = nn.CrossEntropyLoss()
        criterion_gender = nn.CrossEntropyLoss()
        criterion_language = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(FINAL_EPOCHS):
            train_loss, train_age_acc, train_gender_acc, train_lang_acc = train_mtl_epoch(
                model, train_loader, device, optimizer, 
                [criterion_age, criterion_gender, criterion_language], epoch + 1, FINAL_EPOCHS
            )
            val_loss, val_age_acc, val_gender_acc, val_lang_acc = evaluate_mtl(
                model, val_loader, device, [criterion_age, criterion_gender, criterion_language]
            )
            
            print(f"\nEpoch {epoch+1}/{FINAL_EPOCHS} Summary:")
            print(f"  Train - Loss: {train_loss:.4f} | Age: {train_age_acc:.4f} | Gender: {train_gender_acc:.4f} | Lang: {train_lang_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f} | Age: {val_age_acc:.4f} | Gender: {val_gender_acc:.4f} | Lang: {val_lang_acc:.4f}")

        torch.save({
            'model_state_dict': model.state_dict(),
            'age_encoder': dataset.age_encoder.classes_,
            'gender_encoder': dataset.gender_encoder.classes_,
            'language_encoder': dataset.language_encoder.classes_,
        }, 'mtl_video_model.pth')

        print("\n✅ MTL Model saved successfully as 'mtl_video_model.pth'")
    else:
        print("❌ No data available for training")
    
    print("\n🎯 Script execution completed!")