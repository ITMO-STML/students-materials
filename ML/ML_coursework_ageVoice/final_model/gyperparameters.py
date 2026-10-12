import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, regularizers
import warnings
warnings.filterwarnings('ignore')

# Конфиг
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

os.environ['TF_DETERMINISTIC_OPS'] = '1' # для воспроизводимости результата
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass

# Оптимизация памяти GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU настроен: {len(gpus)} устройство(а)")
    except RuntimeError as e:
        print(e)

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
    """Загрузка и объединение всех 4 модальностей"""
    print("\n" + "="*60)
    print("ЗАГРУЗКА ВСЕХ МОДАЛЬНОСТЕЙ")
    print("="*60)
    
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


# ReSNet (с возможностью передачи параметров)
def build_resnet(input_dim, dropout_rate=0.4, l2_reg=0.0003, learning_rate=0.0005):
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
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.005)
    except:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_resnet_with_params(X_train, y_train, X_val, y_val, input_dim, dropout_rate, l2_reg, batch_size, verbose=0):
    """Обучение ResNet с заданными параметрами"""
    model = build_resnet(input_dim, dropout_rate=dropout_rate, l2_reg=l2_reg)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=150, batch_size=batch_size, callbacks=[early_stop, reduce_lr], verbose=verbose)
    
    return model


# Подбор параметров для ReSNet
def hyperparameter_search_resnet(X_train, X_val, y_train, y_val, input_dim):
    """
    Перебор гиперпараметров для ResNet
    Оптимизация по F1-score (weighted)
    """
    
    param_grid = {
        'dropout': [0.3, 0.4],
        'l2_reg': [0.0002, 0.0003, 0.0004],
        'batch_size': [16, 32, 64]
    }
    
    results = []
    best_score = 0
    best_params = None
    
    total = len(param_grid['dropout']) * len(param_grid['l2_reg']) * len(param_grid['batch_size'])
    print("\nПодбор гиперпараметров для ResNet (по f1-score, {total} комбинаций)")
    
    combo_num = 1
    
    for dropout in param_grid['dropout']:
        for l2_reg in param_grid['l2_reg']:
            for batch_size in param_grid['batch_size']:
                
                print(f"\n[{combo_num}/{total}] Тестирование ResNet:")
                print(f"  dropout={dropout}, l2_reg={l2_reg}, batch_size={batch_size}")
                
                model = build_resnet(input_dim=input_dim, dropout_rate=dropout, l2_reg=l2_reg)
                
                class F1Callback(callbacks.Callback):
                    def __init__(self, validation_data):
                        self.validation_data = validation_data
                        self.val_f1_scores = []
                    
                    def on_epoch_end(self, epoch, logs=None):
                        val_pred = np.argmax(self.model.predict(self.validation_data[0], verbose=0), axis=1)
                        val_f1 = f1_score(self.validation_data[1], val_pred, average='weighted')
                        self.val_f1_scores.append(val_f1)
                        logs['val_f1'] = val_f1
                
                f1_callback = F1Callback(validation_data=(X_val, y_val))
                early_stop = callbacks.EarlyStopping(monitor='val_f1', mode='max', patience=8, restore_best_weights=True, verbose=0)
                reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_f1', mode='max', factor=0.5, patience=4, min_lr=1e-6, verbose=0)
                
                model.fit(X_train, y_train, validation_data=(X_val, y_val),
                          epochs=50, batch_size=batch_size, callbacks=[early_stop, reduce_lr, f1_callback], verbose=0)
                
                best_val_f1 = max(f1_callback.val_f1_scores) if f1_callback.val_f1_scores else 0
                print(f"  → Лучший val_f1: {best_val_f1:.4f}")
                
                results.append({'dropout': dropout, 'l2_reg': l2_reg, 'batch_size': batch_size, 'best_val_f1': best_val_f1})
                
                if best_val_f1 > best_score:
                    best_score = best_val_f1
                    best_params = {'dropout': dropout, 'l2_reg': l2_reg, 'batch_size': batch_size}
                    print(f"  НОВЫЙ ЛУЧШИЙ! ")
                
                combo_num += 1
    
    results_df = pd.DataFrame(results).sort_values('best_val_f1', ascending=False)
    print(f"\nЛучшие параметры для ResNet: {best_params}, val_f1 = {best_score:.4f}")
    
    return best_params, results_df


# Подбор параметров для FCNN
def hyperparameter_search_mlp(X_train, X_val, y_train, y_val, input_dim):
    """
    Перебор гиперпараметров для Improved MLP
    """
    param_grid = {
        'learning_rate': [0.0003, 0.0005, 0.0008],
        'dropout': [0.3, 0.4, 0.5],
        'batch_size': [16, 32, 64]
    }
    
    results = []
    best_score = 0
    best_params = None
    
    total = len(param_grid['learning_rate']) * len(param_grid['dropout']) * len(param_grid['batch_size'])
    print("\nПодбор гиперпараметров для Improved MLP ({total} комбинаций)")
    
    combo_num = 1
    
    for lr in param_grid['learning_rate']:
        for dropout in param_grid['dropout']:
            for batch_size in param_grid['batch_size']:
                
                print(f"\n[{combo_num}/{total}] Тестирование Improved MLP:")
                print(f"  learning_rate={lr}, dropout={dropout}, batch_size={batch_size}")
                
                model = tf.keras.Sequential([
                    layers.Input(shape=(input_dim,)),
                    layers.Dense(1024, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(dropout),
                    layers.Dense(512, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(dropout),
                    layers.Dense(256, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(max(0.2, dropout - 0.1)),
                    layers.Dense(128, activation='relu'),
                    layers.Dropout(max(0.2, dropout - 0.1)),
                    layers.Dense(3, activation='softmax')
                ])
                
                try:
                    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.01)
                except:
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                
                model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                class F1Callback(callbacks.Callback):
                    def __init__(self, validation_data):
                        self.validation_data = validation_data
                        self.val_f1_scores = []
                    
                    def on_epoch_end(self, epoch, logs=None):
                        val_pred = np.argmax(self.model.predict(self.validation_data[0], verbose=0), axis=1)
                        val_f1 = f1_score(self.validation_data[1], val_pred, average='weighted')
                        self.val_f1_scores.append(val_f1)
                        logs['val_f1'] = val_f1
                
                f1_callback = F1Callback(validation_data=(X_val, y_val))
                early_stop = callbacks.EarlyStopping(monitor='val_f1', mode='max', patience=8, restore_best_weights=True, verbose=0)
                
                model.fit(X_train, y_train, validation_data=(X_val, y_val),
                          epochs=50, batch_size=batch_size, callbacks=[early_stop, f1_callback], verbose=0)
                
                best_val_f1 = max(f1_callback.val_f1_scores) if f1_callback.val_f1_scores else 0
                print(f"  → Лучший val_f1: {best_val_f1:.4f}")
                
                results.append({'learning_rate': lr, 'dropout': dropout, 'batch_size': batch_size, 'best_val_f1': best_val_f1})
                
                if best_val_f1 > best_score:
                    best_score = best_val_f1
                    best_params = {'learning_rate': lr, 'dropout': dropout, 'batch_size': batch_size}
                    print(f"  НОВЫЙ ЛУЧШИЙ! ")
                
                combo_num += 1
    
    results_df = pd.DataFrame(results).sort_values('best_val_f1', ascending=False)
    print(f"\nЛучшие параметры для Improved MLP: {best_params}, val_f1 = {best_score:.4f}")
    
    return best_params, results_df


# Подбор параметров длоя мета-классификатора
def hyperparameter_search_meta(X_meta_train, y_train, X_meta_val, y_val):
    """
    Перебор гиперпараметров для мета-классификатора
    Вход: 6 признаков (вероятности от двух базовых моделей)
    """
    param_grid = {
        'units1': [16, 32, 64],
        'units2': [8, 16, 32],
        'dropout1': [0.2, 0.3, 0.4],
        'dropout2': [0.1, 0.2, 0.3],
        'batch_size': [8, 16, 32],
        'learning_rate': [0.001, 0.0005, 0.0001]
    }
    
    results = []
    best_score = 0
    best_params = None
    
    total = 3**6  # 729 комбинаций слишком много
    print("\nПодбор гиперпараметров для мета-классификатора (выборочно, 20 комбинаций)")
    
    # Для ускорения берём случайные комбинации
    np.random.seed(RANDOM_SEED)
    
    for combo_num in range(1, 21):
        units1 = np.random.choice(param_grid['units1'])
        units2 = np.random.choice(param_grid['units2'])
        dropout1 = np.random.choice(param_grid['dropout1'])
        dropout2 = np.random.choice(param_grid['dropout2'])
        batch_size = np.random.choice(param_grid['batch_size'])
        lr = np.random.choice(param_grid['learning_rate'])
        
        print(f"\n[{combo_num}/20] Тестирование мета-классификатора:")
        print(f"  units1={units1}, units2={units2}, dropout1={dropout1}, dropout2={dropout2}, batch_size={batch_size}, lr={lr}")
        
        meta_model = tf.keras.Sequential([
            layers.Input(shape=(6,)),
            layers.Dense(units1, activation='relu'),
            layers.Dropout(dropout1),
            layers.Dense(units2, activation='relu'),
            layers.Dropout(dropout2),
            layers.Dense(3, activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        meta_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        
        history = meta_model.fit(X_meta_train, y_train, validation_data=(X_meta_val, y_val),
                                 epochs=100, batch_size=batch_size, callbacks=[early_stop], verbose=0)
        
        best_val_acc = max(history.history['val_accuracy'])
        print(f"  → Лучший val_accuracy: {best_val_acc:.4f}")
        
        results.append({
            'units1': units1, 'units2': units2, 'dropout1': dropout1,
            'dropout2': dropout2, 'batch_size': batch_size, 'learning_rate': lr,
            'best_val_acc': best_val_acc
        })
        
        if best_val_acc > best_score:
            best_score = best_val_acc
            best_params = {'units1': units1, 'units2': units2, 'dropout1': dropout1,
                          'dropout2': dropout2, 'batch_size': batch_size, 'learning_rate': lr}
            print(f"  НОВЫЙ ЛУЧШИЙ! ")
    
    results_df = pd.DataFrame(results).sort_values('best_val_acc', ascending=False)
    print(f"\nЛучшие параметры для мета-классификатора: {best_params}, val_acc = {best_score:.4f}")
    
    return best_params, results_df


# FCNN (с параметрами)
def build_improved_mlp_with_params(input_dim, learning_rate=0.0005, dropout=0.4):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(max(0.2, dropout - 0.1)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(max(0.2, dropout - 0.1)),
        layers.Dense(3, activation='softmax')
    ])
    
    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    except:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_improved_mlp_with_params(X_train, y_train, X_val, y_val, input_dim, learning_rate, dropout, batch_size, verbose=0):
    model = build_improved_mlp_with_params(input_dim, learning_rate=learning_rate, dropout=dropout)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=150, batch_size=batch_size, callbacks=[early_stop, reduce_lr], verbose=verbose)
    
    return model


# Stacking ensemble с подбором для всех моделей
def train_stacking_ensemble_full(X, y):
    print("\nStacking ensemble с подбором гиперпараметров для всех моделей")
    
    # Разделение данных
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
    
    # Подбор параметров для RESNET
    best_params_resnet, _ = hyperparameter_search_resnet(X_train, X_val, y_train, y_val, input_dim)
    
    # Подбор параметров для FCNN
    best_params_mlp, _ = hyperparameter_search_mlp(X_train, X_val, y_train, y_val, input_dim)
    
    # Финальное обучение
    print("\nФинальное обучение с лучшими параметрами")
    
    print(f"\nПараметры ResNet: dropout={best_params_resnet['dropout']}, l2_reg={best_params_resnet['l2_reg']}, batch_size={best_params_resnet['batch_size']}")
    print(f"Параметры Improved MLP: lr={best_params_mlp['learning_rate']}, dropout={best_params_mlp['dropout']}, batch_size={best_params_mlp['batch_size']}")
    
    print("\n1. Обучение ResNet...")
    model_resnet = train_resnet_with_params(
        X_train, y_train, X_val, y_val, input_dim,
        dropout_rate=best_params_resnet['dropout'],
        l2_reg=best_params_resnet['l2_reg'],
        batch_size=best_params_resnet['batch_size'],
        verbose=1
    )
    
    print("\n2. Обучение Improved MLP...")
    model_improved_mlp = train_improved_mlp_with_params(
        X_train, y_train, X_val, y_val, input_dim,
        learning_rate=best_params_mlp['learning_rate'],
        dropout=best_params_mlp['dropout'],
        batch_size=best_params_mlp['batch_size'],
        verbose=1
    )
    
    # Получение предсказаний
    print("\nПолучение предсказаний от базовых моделей...")
    
    pred_resnet_val = model_resnet.predict(X_val)
    pred_improved_mlp_val = model_improved_mlp.predict(X_val)
    pred_resnet_test = model_resnet.predict(X_test)
    pred_improved_mlp_test = model_improved_mlp.predict(X_test)
    
    X_meta_train = np.concatenate([pred_resnet_val, pred_improved_mlp_val], axis=1)
    X_meta_test = np.concatenate([pred_resnet_test, pred_improved_mlp_test], axis=1)
    
    # Разделение мета-данных для валидации
    X_meta_train_split, X_meta_val, y_meta_train, y_meta_val = train_test_split(
        X_meta_train, y_val, test_size=0.2, stratify=y_val, random_state=RANDOM_SEED
    )
    
    # Подбор параметров для мета-классификатора
    best_params_meta, _ = hyperparameter_search_meta(X_meta_train_split, y_meta_train, X_meta_val, y_meta_val)
    
    # Финальное обучение
    print("\nФинальное обучение мета-классификатора с лучшими параметрами...")
    print(f"  units1={best_params_meta['units1']}, units2={best_params_meta['units2']}, dropout1={best_params_meta['dropout1']}, dropout2={best_params_meta['dropout2']}")
    
    meta_model = tf.keras.Sequential([
        layers.Input(shape=(6,)),
        layers.Dense(best_params_meta['units1'], activation='relu'),
        layers.Dropout(best_params_meta['dropout1']),
        layers.Dense(best_params_meta['units2'], activation='relu'),
        layers.Dropout(best_params_meta['dropout2']),
        layers.Dense(3, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=best_params_meta['learning_rate'])
    meta_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    meta_model.fit(X_meta_train_split, y_meta_train, validation_data=(X_meta_val, y_meta_val),
                   epochs=100, batch_size=best_params_meta['batch_size'], callbacks=[early_stop], verbose=1)
    
    # Оценка результата
    y_pred = np.argmax(meta_model.predict(X_meta_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    f1_per_class = f1_score(y_test, y_pred, average=None)
    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    
    print("\nРезультаты stacking ensemble")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 (weighted): {f1:.4f}")
    
    print(f"\nМетрики по классам:")
    print(f"{'Класс':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*51)
    class_names = ['0-25 лет', '26-50 лет', '50+ лет']
    for i, name in enumerate(class_names):
        print(f"{name:<15} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} {f1_per_class[i]:<12.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return {
        'accuracy': acc,
        'f1': f1,
        'best_params_resnet': best_params_resnet,
        'best_params_mlp': best_params_mlp,
        'best_params_meta': best_params_meta,
        'f1_per_class': f1_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class
    }


# Основной запуск
if __name__ == "__main__":
    print("\nStacking ensemble для объединенных признаков")

    X, y = load_all_modalities()
    
    # Запуск с полным подбором гиперпараметров
    result = train_stacking_ensemble_full(X, y)
    
    print("\nИтоговый результат")
    print(f"\nЛучшие гиперпараметры:")
    print(f"  ResNet: dropout={result['best_params_resnet']['dropout']}, l2_reg={result['best_params_resnet']['l2_reg']}, batch_size={result['best_params_resnet']['batch_size']}")
    print(f"  Improved MLP: lr={result['best_params_mlp']['learning_rate']}, dropout={result['best_params_mlp']['dropout']}, batch_size={result['best_params_mlp']['batch_size']}")
    print(f"  Meta-классификатор: units1={result['best_params_meta']['units1']}, units2={result['best_params_meta']['units2']}, dropout1={result['best_params_meta']['dropout1']}, dropout2={result['best_params_meta']['dropout2']}, batch_size={result['best_params_meta']['batch_size']}")
    print(f"\nStacking Ensemble (ResNet + Improved MLP):")
    print(f"  Weighted F1: {result['f1']:.4f}")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"\nF1 по классам:")
    print(f"  0-25 лет:  {result['f1_per_class'][0]:.4f}")
    print(f"  26-50 лет: {result['f1_per_class'][1]:.4f}")
    print(f"  50+ лет:   {result['f1_per_class'][2]:.4f}")