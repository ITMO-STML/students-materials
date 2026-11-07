import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, regularizers
import warnings
warnings.filterwarnings('ignore')

# Конфиг
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass

# Пути к данным
MFCC_PATH = "./feature_extraction/timit_features_classic/mfcc"
FBANK_PATH = "./feature_extraction/timit_features_classic/fbank"
WAV2VEC_PATH = "./feature_extraction/wav2vec_emb/wav2vec_original_embeddings"
WAVLM_PATH = "./feature_extraction/wavlm_emb/wavlm_original_embeddings"
METADATA_PATH = "./feature_extraction/timit_features_classic/features_metadata.csv"

DIM_MFCC = 40
DIM_FBANK = 40
DIM_W2V = 1024
DIM_WLM = 1024

print(f"Random seed: {RANDOM_SEED}")


# Загрузка данных
def find_embedding_file(emb_path, speaker_id, filename):
    possible_names = [
        f"{speaker_id}_{filename}.npy",
        f"{filename}.npy",
        f"{speaker_id}{filename}.npy",
    ]
    for prefix in ['M', 'F']:
        possible_names.append(f"{prefix}{speaker_id}_{filename}.npy")
    
    possible_names = list(dict.fromkeys(possible_names))
    
    for name in possible_names:
        full_path = os.path.join(emb_path, name)
        if os.path.exists(full_path):
            return full_path
    return None


def load_classical_features(data_path, expected_dim):
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Загружено {len(metadata)} записей")
    
    X = []
    found = 0
    
    for idx, row in metadata.iterrows():
        speaker_id = row['speaker_id']
        filename = row['file_name']
        
        file_name = f"{speaker_id}_{filename}.npy"
        file_path = os.path.join(data_path, file_name)
        
        if os.path.exists(file_path):
            data = np.load(file_path)
            if len(data.shape) == 2:
                data = np.mean(data, axis=0)
            X.append(data)
            found += 1
        else:
            X.append(np.zeros(expected_dim))
    
    X = np.array(X)
    print(f"Найдено {found}/{len(metadata)} файлов")
    return X


def load_neural_embeddings(emb_path, expected_dim):
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Загружено {len(metadata)} записей")
    
    X = []
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
            X.append(np.zeros(expected_dim))
    
    X = np.array(X)
    print(f"Найдено {found}/{len(metadata)} файлов")
    return X


def load_labels():
    metadata = pd.read_csv(METADATA_PATH)
    ages = metadata['age'].values
    y = np.array([0 if age <= 25 else (1 if age <= 50 else 2) for age in ages])
    print(f"Распределение классов: {np.bincount(y)}")
    return y


def load_all_modalities():
    """Загрузка и объединение всех 4 признаков"""
    print("\nЗагрузка всех признаков")
    
    print("\n1. Загрузка MFCC...")
    X_mfcc = load_classical_features(MFCC_PATH, DIM_MFCC)
    
    print("\n2. Загрузка Filterbanks...")
    X_fbank = load_classical_features(FBANK_PATH, DIM_FBANK)
    
    print("\n3. Загрузка wav2vec...")
    X_w2v = load_neural_embeddings(WAV2VEC_PATH, DIM_W2V)
    
    print("\n4. Загрузка wavLM...")
    X_wlm = load_neural_embeddings(WAVLM_PATH, DIM_WLM)
    
    print("\n5. Загрузка меток...")
    y = load_labels()
    
    X_combined = np.concatenate([X_mfcc, X_fbank, X_w2v, X_wlm], axis=1)
    print(f"\nИтоговая размерность: {X_combined.shape[1]}")
    print(f"X shape: {X_combined.shape}, y shape: {y.shape}")
    
    return X_combined, y


# Residual блок
def residual_block(x, units, dropout_rate=0.3, kernel_regularizer=None):
    shortcut = x
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(units, kernel_regularizer=kernel_regularizer)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(units, kernel_regularizer=kernel_regularizer)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units, kernel_regularizer=kernel_regularizer)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    return x


# ReSNet
def build_resnet(input_dim, dropout_rate=0.4, l2_reg=0.0003):
    reg = regularizers.l2(l2_reg)
    
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(1024, kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = residual_block(x, 512, dropout_rate, reg)
    x = residual_block(x, 256, dropout_rate, reg)
    x = residual_block(x, 128, dropout_rate, reg)
    x = residual_block(x, 64, dropout_rate, reg)
    
    se = layers.GlobalAveragePooling1D()(layers.Reshape((1, 64))(x))
    se = layers.Dense(32, activation='relu')(se)
    se = layers.Dense(64, activation='sigmoid')(se)
    se = layers.Reshape((1, 64))(se)
    x = layers.Multiply()([x, se])
    
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(3, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=0.005)
    except:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_resnet(X_train, y_train, X_val, y_val, input_dim):
    model = build_resnet(input_dim)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=150, batch_size=32, callbacks=[early_stop, reduce_lr], verbose=0)
    
    return model


# FCNN
def build_improved_mlp(input_dim):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])
    
    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=0.01)
    except:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_improved_mlp(X_train, y_train, X_val, y_val, input_dim):
    model = build_improved_mlp(input_dim)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=150, batch_size=32, callbacks=[early_stop, reduce_lr], verbose=0)
    
    return model


# stacking ensemble (ResNet + FCNN)
def train_stacking_ensemble(X, y):
    print("\nStacking ensemble (ResNet + Improved MLP)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_SEED
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    input_dim = X.shape[1]
    
    print("\n1. Обучение ResNet...")
    model_resnet = train_resnet(X_train, y_train, X_val, y_val, input_dim)
    
    print("\n2. Обучение Improved MLP...")
    model_improved_mlp = train_improved_mlp(X_train, y_train, X_val, y_val, input_dim)
    
    print("\nПолучение предсказаний от базовых моделей...")
    
    pred_resnet_val = model_resnet.predict(X_val)
    pred_improved_mlp_val = model_improved_mlp.predict(X_val)
    
    pred_resnet_test = model_resnet.predict(X_test)
    pred_improved_mlp_test = model_improved_mlp.predict(X_test)
    
    X_meta_train = np.concatenate([
        pred_resnet_val, pred_improved_mlp_val
    ], axis=1)
    
    X_meta_test = np.concatenate([
        pred_resnet_test, pred_improved_mlp_test
    ], axis=1)
    
    print(f"Размерность мета-признаков: {X_meta_train.shape[1]} (2 модели x 3 класса = 6)")
    
    print("\n3. Обучение Meta-классификатора (MLP)...")
    
    meta_model = tf.keras.Sequential([
        layers.Input(shape=(6,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(3, activation='softmax')
    ])
    
    meta_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    X_meta_train_split, X_meta_val, y_meta_train, y_meta_val = train_test_split(
        X_meta_train, y_val, test_size=0.2, stratify=y_val, random_state=RANDOM_SEED
    )
    
    meta_model.fit(X_meta_train_split, y_meta_train, validation_data=(X_meta_val, y_meta_val),
                   epochs=100, batch_size=16, callbacks=[early_stop], verbose=0)
    
    y_pred = np.argmax(meta_model.predict(X_meta_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\nРезультаты stacking ensemble")

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 (weighted): {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['0-25 лет', '26-50 лет', '50+ лет']))
    
    return {'accuracy': acc, 'f1': f1}


# Основной запуск
if __name__ == "__main__":
    print("\nStacking ensemble для объединенных признаков")

    
    X, y = load_all_modalities()
    
    result = train_stacking_ensemble(X, y)
    
    print("\nИтоговый результат")
    print(f"Stacking Ensemble (ResNet + Improved MLP): F1 = {result['f1']:.4f}, Acc = {result['accuracy']:.4f}")
