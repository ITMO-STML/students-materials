import numpy as np
import pandas as pd
import os
import random  # ДОБАВИТЬ!
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras import Sequential, layers, callbacks

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # отключить GPU
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

WAV2VEC_PATH = "./feature_extraction/wav2vec_emb/wav2vec_original_embeddings"
WAVLM_PATH = "./feature_extraction/wavlm_emb/wavlm_original_embeddings"
METADATA_PATH = "./feature_extraction/timit_features_classic/features_metadata.csv"

WAV2VEC_DIM = 1024
WAVLM_DIM = 1024

RANDOM_SEED = 42


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass  # Для старых версий TF

def find_embedding_file(emb_path, speaker_id, filename):
    possible_names = [
        f"{speaker_id}_{filename}.npy",       
        f"{speaker_id}_{filename}.npy".lower(), 
        f"{speaker_id}_{filename}.npy".upper(), 
        f"{filename}.npy",
        f"{speaker_id}{filename}.npy",
        f"{speaker_id}_{filename}.npy".replace('_', '')
    ]
    
    for prefix in ['M', 'F']:
        possible_names.append(f"{prefix}{speaker_id}_{filename}.npy")
        possible_names.append(f"{prefix}{speaker_id}{filename}.npy")
        possible_names.append(f"{prefix}{speaker_id}_{filename}.npy".lower())
    
    possible_names = list(dict.fromkeys(possible_names))
    
    for name in possible_names:
        full_path = os.path.join(emb_path, name)
        if os.path.exists(full_path):
            return full_path
    return None

# Загрузка данных
def load_embeddings(emb_path, expected_dim):
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Загружено {len(metadata)} записей из метаданных")

    X = []
    missing = 0
    found = 0
    
    for idx, row in metadata.iterrows():
        speaker_id = row['speaker_id']
        filename = row['file_name']
        
        emb_path_full = find_embedding_file(emb_path, speaker_id, filename)
        
        if emb_path_full:
            emb = np.load(emb_path_full)
            if len(emb.shape) == 2:
                emb = np.mean(emb, axis=0)
            X.append(emb)
            found += 1
        else:
            missing += 1
            if missing <= 5:
                print(f"Файл не найден: speaker={speaker_id}, file={filename}")
            X.append(np.zeros(expected_dim))
    
    print(f"Найдено файлов: {found}, пропущено: {missing}")
    
    if missing > 0:
        print(f"Всего пропущено файлов: {missing}")
        print("Проверьте формат имен файлов в папке и в metadata")
    
    X = np.array(X)
    
    if X.shape[1] != expected_dim:
        print(f"Ожидалась размерность {expected_dim}, получено {X.shape[1]}")
    
    ages = metadata['age'].values
    y = np.array([0 if age <= 25 else (1 if age <= 50 else 2) for age in ages])
    
    return X, y

def build_neural_embedding_model(input_dim, dropout_rate=0.5):
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED)),
        layers.Dropout(dropout_rate),
        layers.Dense(3, activation='softmax',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED))
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, 
                 loss='sparse_categorical_crossentropy', 
                 metrics=['accuracy'])
    return model

def train_and_evaluate(X, y, model_name, input_dim, dropout_rate=0.5):
    print(f"МОДЕЛЬ: {model_name}")
    print(f"Размерность: {input_dim}, Dropout: {dropout_rate}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Распределение классов: {np.bincount(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_SEED
    )
    
    model = build_neural_embedding_model(input_dim, dropout_rate)
    print(f"Параметров модели: {model.count_params()}")
    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
        shuffle=True
    )
    
    # Оценка
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nРезультаты {model_name}:")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test F1 (weighted): {f1:.4f}")
    
    return {'accuracy': acc, 'f1': f1, 'model': model, 'scaler': scaler}

if __name__ == "__main__":
    results = {}
    
    if os.path.exists(WAV2VEC_PATH):
        print("Загрузка wav2vec эмбеддингов...")
        X_w2v, y_w2v = load_embeddings(WAV2VEC_PATH, WAV2VEC_DIM)
        results['wav2vec'] = train_and_evaluate(X_w2v, y_w2v, 'wav2vec', WAV2VEC_DIM, dropout_rate=0.5)
    else:
        print(f"Путь wav2vec не найден: {WAV2VEC_PATH}")
    
    if os.path.exists(WAVLM_PATH):
        print("\nЗагрузка wavLM эмбеддингов...")
        X_wlm, y_wlm = load_embeddings(WAVLM_PATH, WAVLM_DIM)
        results['wavlm'] = train_and_evaluate(X_wlm, y_wlm, 'wavLM', WAVLM_DIM, dropout_rate=0.5)
    else:
        print(f"Путь wavLM не найден: {WAVLM_PATH}")
    
    if results:
        print("Результаты по моделям:")
        for name, res in results.items():
            print(f"{name}: Accuracy = {res['accuracy']:.4f}, F1 = {res['f1']:.4f}")