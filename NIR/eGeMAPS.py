import os
import numpy as np
import pandas as pd
import opensmile
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = "/mnt/vas/open_data/InTheWild/release_in_the_wild"
RANDOM_SEED = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
SAMPLE_SIZE = None  # Set to e.g., 5000 for quick test, None for full dataset

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(data_path):
    """Load metadata and create file paths."""
    meta_path = os.path.join(data_path, "meta.csv")
    df = pd.read_csv(meta_path)
    df['file_path'] = df['file'].apply(lambda x: os.path.join(data_path, x))
    df['label_num'] = df['label'].map({'bona-fide': 0, 'spoof': 1})
    
    # Filter existing files
    files_exist = df['file_path'].apply(os.path.exists)
    df = df[files_exist].reset_index(drop=True)
    
    print(f"Loaded {len(df)} files")
    print(f"Real (0): {(df['label_num']==0).sum()}, Fake (1): {(df['label_num']==1).sum()}")
    print(f"Unique speakers: {df['speaker'].nunique()}")
    
    return df

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def init_opensmile():
    """Initialize openSMILE with eGeMAPS feature set."""
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    print("openSMILE initialized (88 eGeMAPS features)")
    return smile

def extract_features(df, smile, sample_size=None):
    """Extract eGeMAPS features for all audio files."""
    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"Using sample of {sample_size} files")
    
    features_list = []
    labels = []
    speakers = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Progress: {idx}/{len(df)}")
        
        try:
            feats = smile.process_file(row['file_path'])
            features_list.append(feats.values.flatten())
            labels.append(row['label_num'])
            speakers.append(row['speaker'])
        except Exception as e:
            print(f"Error: {row['file']} - {e}")
            continue
    
    X = np.array(features_list)
    y = np.array(labels)
    speakers_arr = np.array(speakers)
    
    print(f"Extracted features from {len(X)} files")
    print(f"Feature vector size: {X.shape[1]}")
    
    return X, y, speakers_arr

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def random_cv(X, y):
    """Random train-test split validation."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nRandom CV Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    return acc, rf, y_test, y_pred

def speaker_cv(X, y, speakers):
    """Leave-One-Group-Out cross-validation by speaker."""
    logo = LeaveOneGroupOut()
    scores = []
    
    print(f"\nRunning speaker-wise CV on {len(np.unique(speakers))} speakers...")
    
    for train_idx, test_idx in logo.split(X, y, groups=speakers):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    
    acc = np.mean(scores)
    print(f"\nSpeaker-wise CV Accuracy: {acc*100:.2f}%")
    
    return acc, scores

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_accuracy_comparison(random_acc, speaker_acc, save_path='accuracy_comparison.png'):
    """Plot comparison between random and speaker CV."""
    plt.figure(figsize=(6, 5))
    methods = ['Random CV', 'Speaker CV']
    accuracies = [random_acc*100, speaker_acc*100]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = plt.bar(methods, accuracies, color=colors, edgecolor='black')
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    plt.ylabel('Accuracy (%)')
    plt.ylim(60, 100)
    plt.title('eGeMAPS Generalization Ability\nDeepfake Detection')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")

def plot_confusion_matrix(y_test, y_pred, save_path='confusion_matrix.png'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("="*60)
    print("DEEPFAKE DETECTION WITH eGeMAPS")
    print("="*60)
    
    # 1. Load data
    print("\n[1/4] Loading data...")
    df = load_data(DATA_PATH)
    
    # 2. Initialize openSMILE and extract features
    print("\n[2/4] Extracting eGeMAPS features...")
    smile = init_opensmile()
    X, y, speakers = extract_features(df, smile, sample_size=SAMPLE_SIZE)
    
    # 3. Random CV
    print("\n[3/4] Random cross-validation...")
    random_acc, rf, y_test, y_pred = random_cv(X, y)
    
    # 4. Speaker CV
    print("\n[4/4] Speaker-wise cross-validation...")
    speaker_acc, speaker_scores = speaker_cv(X, y, speakers)
    
    # Results summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Random CV accuracy:     {random_acc*100:.1f}%")
    print(f"Speaker CV accuracy:    {speaker_acc*100:.1f}%")
    print(f"Drop:                   {(random_acc - speaker_acc)*100:.1f}%")
    print("="*60)
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_accuracy_comparison(random_acc, speaker_acc)
    plot_confusion_matrix(y_test, y_pred)
    
    # Save results
    results = {
        'random_accuracy': random_acc,
        'speaker_accuracy': speaker_acc,
        'drop': random_acc - speaker_acc,
        'n_files': len(X),
        'n_speakers': len(np.unique(speakers))
    }
    
    # Save features for later use
    joblib.dump(X, 'X_features.pkl')
    joblib.dump(y, 'y_labels.pkl')
    joblib.dump(speakers, 'speakers.pkl')
    
    print("\nSaved: X_features.pkl, y_labels.pkl, speakers.pkl")
    
    return results

if __name__ == "__main__":
    results = main()
