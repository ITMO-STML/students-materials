import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import video
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

def cv2_to_tensor(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    tensor = torch.zeros(3, frame_rgb.shape[0], frame_rgb.shape[1], dtype=torch.float32)
    for c in range(3):
        for h in range(frame_rgb.shape[0]):
            for w in range(frame_rgb.shape[1]):
                tensor[c, h, w] = float(frame_rgb[h, w, c]) / 255.0
    return tensor

def cv2_to_tensor_fast(frame, target_size=(112, 112)):
    frame_resized = cv2.resize(frame, target_size)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
    return tensor

class VideoDataset(Dataset):
    def __init__(self, dataset, video_path, max_frames=8, clip_duration_sec=0.5):
        self.dataset = dataset
        self.video_path = video_path
        self.max_frames = max_frames
        self.clip_duration_sec = clip_duration_sec
        
        self.age_groups = []
        self.genders = []
        self.languages = []
        self.valid_indices = []
        self.video_files = []
        
        self.clip_entries = []
        
        missing_files = 0
        for idx, item in enumerate(dataset):
            info = item['info']
            video_id = item['id']
            video_file = os.path.join(self.video_path, video_id + '.mp4')
            if (
                info['Age Group'] is None or
                info['Gender'] is None or
                info['Language'] is None or
                not os.path.exists(video_file)
            ):
                if not os.path.exists(video_file):
                    missing_files += 1
                continue
            
            age_str = info['Age Group']
            gender_str = info['Gender']
            language_str = info['Language']
            self.age_groups.append(age_str)
            self.genders.append(gender_str)
            self.languages.append(language_str)
            self.valid_indices.append(idx)
            self.video_files.append(video_file)
            
            cap = cv2.VideoCapture(video_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            readable = True
            if total_frames > 0 and fps > 0:
                test_positions = [0, max(total_frames // 2, 0), max(total_frames - 1, 0)]
                for pos in test_positions:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    ok, frm = cap.read()
                    if not ok or frm is None:
                        readable = False
                        break
            cap.release()
            if total_frames <= 0 or fps <= 0 or not readable:
                continue
            clip_len = max(int(round(fps * self.clip_duration_sec)), 1)
            end_limit = total_frames
            start = 0
            while start + clip_len <= end_limit:
                end = start + clip_len
                self.clip_entries.append((video_file, start, end, age_str, gender_str, language_str))
                start = end
        
        print(f"Valid videos: {len(self.video_files)}/{len(dataset)} (excluded missing files: {missing_files})")
        print(f"Total clips: {len(self.clip_entries)}")
        
        self.age_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()
        self.language_encoder = LabelEncoder()
        
        self.age_encoder.fit(self.age_groups)
        self.gender_encoder.fit(self.genders)
        self.language_encoder.fit(self.languages)
    
    def __len__(self):
        return len(self.clip_entries)
    
    def __getitem__(self, idx):
        video_file, start_frame, end_frame, age_str, gender_str, language_str = self.clip_entries[idx]
        
        frames = self.load_video_clip(video_file, start_frame, end_frame)
        
        age = self.age_encoder.transform([age_str])[0]
        gender = self.gender_encoder.transform([gender_str])[0]
        language = self.language_encoder.transform([language_str])[0]
        
        return frames, torch.tensor(age, dtype=torch.long), torch.tensor(gender, dtype=torch.long), torch.tensor(language, dtype=torch.long)
    
    def load_video_clip(self, video_path, start_frame, end_frame):
        if not os.path.exists(video_path):
            return torch.zeros((self.max_frames, 3, 112, 112), dtype=torch.float32)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0 or start_frame >= total_frames:
            cap.release()
            return torch.zeros((self.max_frames, 3, 112, 112), dtype=torch.float32)
        end_frame = min(end_frame, total_frames)
        
        window_len = max(end_frame - start_frame, 1)
        frame_indices = []
        if self.max_frames == 1:
            frame_indices = [start_frame]
        else:
            for i in range(self.max_frames):
                rel = int(i * (window_len - 1) / (self.max_frames - 1)) if self.max_frames > 1 else 0
                frame_indices.append(start_frame + rel)
        
        frames = []
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                try:
                    tensor = cv2_to_tensor_fast(frame, (112, 112))
                    frames.append(tensor)
                except:
                    frames.append(torch.zeros((3, 112, 112), dtype=torch.float32))
            else:
                frames.append(torch.zeros((3, 112, 112), dtype=torch.float32))
        cap.release()
        
        frames_tensor = torch.stack(frames) if frames else torch.zeros((self.max_frames, 3, 112, 112), dtype=torch.float32)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames_tensor = (frames_tensor - mean) / std
        return frames_tensor

class SimpleVideoModel(nn.Module):
    def __init__(self, num_age_classes, num_gender_classes, num_language_classes):
        super(SimpleVideoModel, self).__init__()
        
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.age_classifier = nn.Linear(64, num_age_classes)
        self.gender_classifier = nn.Linear(64, num_gender_classes)
        self.language_classifier = nn.Linear(64, num_language_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        
        features = self.conv3d(x)
        features = features.view(features.size(0), -1)
        
        return self.age_classifier(features), self.gender_classifier(features), self.language_classifier(features)

def train_one_epoch(model, dataloader, device, epoch: int, total_epochs: int):
    model.train()
    running_loss = 0.0
    age_acc_sum = 0.0
    gender_acc_sum = 0.0
    language_acc_sum = 0.0
    num_batches = 0

    for frames, age_target, gender_target, language_target in tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", leave=False):
        frames = frames.to(device)
        age_target = age_target.to(device)
        gender_target = gender_target.to(device)
        language_target = language_target.to(device)

        optimizer.zero_grad()

        outputs_age, outputs_gender, outputs_language = model(frames)

        loss_age = criterion_age(outputs_age, age_target)
        loss_gender = criterion_gender(outputs_gender, gender_target)
        loss_language = criterion_language(outputs_language, language_target)

        total_loss = loss_age + loss_gender + loss_language
        total_loss.backward()
        optimizer.step()

        age_pred = torch.argmax(outputs_age, 1)
        gender_pred = torch.argmax(outputs_gender, 1)
        language_pred = torch.argmax(outputs_language, 1)

        age_acc = (age_pred == age_target).float().mean().item()
        gender_acc = (gender_pred == gender_target).float().mean().item()
        language_acc = (language_pred == language_target).float().mean().item()

        running_loss += total_loss.item()
        age_acc_sum += age_acc
        gender_acc_sum += gender_acc
        language_acc_sum += language_acc
        num_batches += 1

    epoch_loss = running_loss / max(1, num_batches)
    epoch_age_acc = age_acc_sum / max(1, num_batches)
    epoch_gender_acc = gender_acc_sum / max(1, num_batches)
    epoch_language_acc = language_acc_sum / max(1, num_batches)

    return epoch_loss, epoch_age_acc, epoch_gender_acc, epoch_language_acc

def predict_video_simple(model, video_path, dataset, max_frames=8):
    model.eval()
    if not os.path.exists(video_path):
        return {"error": f"Video file not found: {video_path}"}
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return {"error": "Empty video file"}
    frame_indices = []
    for i in range(max_frames):
        frame_idx = int(i * (total_frames - 1) / (max_frames - 1)) if max_frames > 1 else 0
        frame_indices.append(frame_idx)
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            try:
                frame_tensor = cv2_to_tensor_fast(frame, (112, 112))
                frames.append(frame_tensor)
            except:
                frames.append(torch.zeros((3, 112, 112), dtype=torch.float32))
        else:
            frames.append(torch.zeros((3, 112, 112), dtype=torch.float32))
    cap.release()
    if frames:
        frames_tensor = torch.stack(frames)
    else:
        frames_tensor = torch.zeros((max_frames, 3, 112, 112), dtype=torch.float32)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames_tensor = (frames_tensor - mean) / std
    frames_tensor = frames_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs_age, outputs_gender, outputs_language = model(frames_tensor)
    age_pred = torch.argmax(outputs_age, 1).item()
    gender_pred = torch.argmax(outputs_gender, 1).item()
    language_pred = torch.argmax(outputs_language, 1).item()
    return {
        'age': dataset.age_encoder.inverse_transform([age_pred])[0],
        'gender': dataset.gender_encoder.inverse_transform([gender_pred])[0],
        'language': dataset.language_encoder.inverse_transform([language_pred])[0]
    }


if __name__ == "__main__":
    print("Loading dataset...")
    ds = load_dataset("FreedomIntelligence/TalkVid")
    path_to_videos = os.getenv('VIDEO_PATH', '/home/danya/datasets/VidTalk/dataset_videos/')
    print('='*50)
    print('path to videos', path_to_videos)
    print('='*50)

    MAX_FRAMES = 25
    BATCH_SIZE = 32
    LIMIT_VIDEOS = 5

    EPOCHS = 10
    NUM_TEST_SAMPLES = 5

    base = ds['test']

    if LIMIT_VIDEOS is not None:
        base = base.select(range(min(LIMIT_VIDEOS, len(base))))
    small_ds = {'test': base}

    dataset = VideoDataset(small_ds['test'], path_to_videos, max_frames=MAX_FRAMES, clip_duration_sec=0.5)

    all_indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(all_indices, test_size=0.3, random_state=42, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, shuffle=True)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Dataset size (clips): {len(dataset)} | train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    print(f"Age classes: {list(dataset.age_encoder.classes_)}")
    print(f"Gender classes: {list(dataset.gender_encoder.classes_)}")
    print(f"Language classes: {list(dataset.language_encoder.classes_)}")

    num_age_classes = len(dataset.age_encoder.classes_)
    num_gender_classes = len(dataset.gender_encoder.classes_)
    num_language_classes = len(dataset.language_encoder.classes_)

    print(f"Number of classes - Age: {num_age_classes}, Gender: {num_gender_classes}, Language: {num_language_classes}")

    model = SimpleVideoModel(num_age_classes, num_gender_classes, num_language_classes)

    device = torch.device('cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    criterion_age = nn.CrossEntropyLoss()
    criterion_gender = nn.CrossEntropyLoss()
    criterion_language = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        loss, age_a, gender_a, lang_a = train_one_epoch(model, train_loader, device, epoch, EPOCHS)
        print(f"Epoch {epoch:02d}/{EPOCHS} - Loss: {loss:.4f} | Age Acc: {age_a:.4f} | Gender Acc: {gender_a:.4f} | Lang Acc: {lang_a:.4f}")

    print("\nTesting prediction on multiple samples...")

    samples = []
    for i in range(min(NUM_TEST_SAMPLES, len(test_idx))):
        clip_global_idx = test_idx[i]
        video_file, start_f, end_f, age_str, gender_str, lang_str = dataset.clip_entries[clip_global_idx]
        gt = {
            'age': age_str,
            'gender': gender_str,
            'language': lang_str
        }
        samples.append((video_file, gt))

    if not samples:
        print("No valid samples found for testing")
    else:
        y_true_age, y_pred_age = [], []
        y_true_gender, y_pred_gender = [], []
        y_true_lang, y_pred_lang = [], []

        for idx, (video_path, gt) in enumerate(samples, 1):
            if not os.path.exists(video_path):
                print(f"[{idx}] Missing: {os.path.basename(video_path)}")
                continue
            print(f"[{idx}] {os.path.basename(video_path)}")
            pred = predict_video_simple(model, video_path, dataset)
            if 'error' in pred:
                print(f"    Error: {pred['error']}")
                continue
            print(f"    Actual    -> Age={gt['age']}, Gender={gt['gender']}, Language={gt['language']}")
            print(f"    Predicted -> Age={pred['age']}, Gender={pred['gender']}, Language={pred['language']}")

            y_true_age.append(dataset.age_encoder.transform([gt['age']])[0])
            y_pred_age.append(list(dataset.age_encoder.classes_).index(pred['age']))
            y_true_gender.append(dataset.gender_encoder.transform([gt['gender']])[0])
            y_pred_gender.append(list(dataset.gender_encoder.classes_).index(pred['gender']))
            y_true_lang.append(dataset.language_encoder.transform([gt['language']])[0])
            y_pred_lang.append(list(dataset.language_encoder.classes_).index(pred['language']))

        from sklearn.metrics import classification_report, confusion_matrix

        if y_true_age:
            print("\n=== Metrics: Age ===")
            age_labels = list(range(len(dataset.age_encoder.classes_)))
            print(classification_report(y_true_age, y_pred_age, labels=age_labels, target_names=list(dataset.age_encoder.classes_), zero_division=0))
            print("Confusion Matrix (Age):")
            print(confusion_matrix(y_true_age, y_pred_age, labels=age_labels))

        if y_true_gender:
            print("\n=== Metrics: Gender ===")
            gender_labels = list(range(len(dataset.gender_encoder.classes_)))
            print(classification_report(y_true_gender, y_pred_gender, labels=gender_labels, target_names=list(dataset.gender_encoder.classes_), zero_division=0))
            print("Confusion Matrix (Gender):")
            print(confusion_matrix(y_true_gender, y_pred_gender, labels=gender_labels))

        if y_true_lang:
            print("\n=== Metrics: Language ===")
            lang_labels = list(range(len(dataset.language_encoder.classes_)))
            print(classification_report(y_true_lang, y_pred_lang, labels=lang_labels, target_names=list(dataset.language_encoder.classes_), zero_division=0))
            print("Confusion Matrix (Language):")
            print(confusion_matrix(y_true_lang, y_pred_lang, labels=lang_labels))

    torch.save({
        'model_state_dict': model.state_dict(),
        'age_encoder': dataset.age_encoder,
        'gender_encoder': dataset.gender_encoder,
        'language_encoder': dataset.language_encoder,
    }, 'video_model.pth')
    print("\nModel saved successfully as 'simple_video_model.pth'")

    print("Script execution completed!")