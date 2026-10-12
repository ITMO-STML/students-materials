import numpy as np
import pandas as pd
import os
import random  # ДОБАВИТЬ!
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras import Sequential, layers, callbacks

MFCC_PATH = "./feature_extraction/timit_features_classic/mfcc"
METADATA_PATH = "./feature_extraction/timit_features_classic/features_metadata.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Переменные окружения для детерминированности на GPU
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Ограничение потоков CPU (для воспроизводимости на CPU)
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass

# Для старых версий TensorFlow
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


# Загрузка данных
def load_mfcc_data():
    """Загрузка MFCC признаков и меток"""
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Загружено {len(metadata)} записей")
    
    X = []
    missing = 0
    
    for idx, row in metadata.iterrows():
        speaker_id = row['speaker_id']
        filename = row['file_name']
        
        mfcc_filename = f"{speaker_id}_{filename}.npy"
        mfcc_path = os.path.join(MFCC_PATH, mfcc_filename)
        
        if os.path.exists(mfcc_path):
            mfcc = np.load(mfcc_path)
            X.append(np.mean(mfcc, axis=0))
        else:
            missing += 1
            if missing <= 5:
                print(f"Файл не найден: {mfcc_path}")
            X.append(np.zeros(40))
    
    if missing > 0:
        print(f"Всего пропущено файлов: {missing}")
    
    X = np.array(X)
    ages = metadata['age'].values
    y = np.array([0 if age <= 25 else (1 if age <= 50 else 2) for age in ages])
    
    return X, y

# Построение модели с явными инициализаторами
def build_mlp_model(input_dim=40, dropout_rate=0.3):
    """MLP с детерминированными инициализаторами весов"""
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED)),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='relu',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED)),
        layers.Dropout(dropout_rate),
        layers.Dense(3, activation='softmax',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED))
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Основной блок
if __name__ == "__main__":
    # Загрузка
    X, y = load_mfcc_data()
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Распределение классов: {np.bincount(y)}")
    
    # Разделение на train + test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    # Нормализация
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Разделение train на train + val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_SEED
    )
    
    # Построение модели
    model = build_mlp_model(input_dim=40, dropout_rate=0.3)
    print(f"Параметров модели: {model.count_params():,}")
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True,
        verbose=1
    )
    
    # Обучение
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        shuffle=True,
        verbose=1
    )
    
    # Оценка
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n Результаты")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 (weighted): {f1:.4f}")