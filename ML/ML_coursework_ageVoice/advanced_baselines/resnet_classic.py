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

# Пути для классических признаков
DATA_PATH = "./feature_extraction/timit_features_classic/fbank"
METADATA_PATH = "./feature_extraction/timit_features_classic/features_metadata.csv"
INPUT_DIM = 40  # размерность

print(f"Random seed: {RANDOM_SEED}")


# Загрузка данных
def load_data(data_path, expected_dim):
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
            # Усредняем по времени
            if len(data.shape) == 2:
                data = np.mean(data, axis=0)
            X.append(data)
            found += 1
        else:
            X.append(np.zeros(expected_dim))
    
    X = np.array(X)
    ages = metadata['age'].values
    y = np.array([0 if age <= 25 else (1 if age <= 50 else 2) for age in ages])
    
    print(f"Найдено {found}/{len(metadata)} файлов")
    return X, y


# Улучшенный Residual блок
def improved_residual_block(x, units, dropout_rate=0.3, kernel_regularizer=None):
    """Улучшенный остаточный блок с pre-activation"""
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


# Baseline
def build_baseline(input_dim=40):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_baseline(X, y):
    print("\nBaseline MLP")
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
    
    model = build_baseline(input_dim=INPUT_DIM)
    print(f"Параметров модели: {model.count_params():,}")
    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\nРезультаты Baseline MLP")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 (weighted): {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['0-25 лет', '26-50 лет', '50+ лет']))
    
    return {'accuracy': acc, 'f1': f1}


# Improved ResNet
def build_improved_resnet(input_dim=40, dropout_rate=0.4, l2_reg=0.0005):
    """Improved ResNet для (адаптированная версия)"""
    reg = regularizers.l2(l2_reg)
    
    inputs = layers.Input(shape=(input_dim,))
    
    # Расширение до 256
    x = layers.Dense(256, kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Residual блоки
    x = improved_residual_block(x, 128, dropout_rate, reg)
    x = improved_residual_block(x, 64, dropout_rate, reg)
    x = improved_residual_block(x, 32, dropout_rate, reg)
    
    # Squeeze-and-Excitation блок
    se = layers.GlobalAveragePooling1D()(layers.Reshape((1, 32))(x))
    se = layers.Dense(16, activation='relu')(se)
    se = layers.Dense(32, activation='sigmoid')(se)
    se = layers.Reshape((1, 32))(se)
    x = layers.Multiply()([x, se])
    
    # Финальная классификация
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(3, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=0.005)
    except:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


def train_improved_resnet(X, y):
    print("\nImproved ResNet")
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
    
    class_weight_dict = {
        0: 1.5,
        1: 0.7,
        2: 5.0
    }
    print(f"Class weights: {class_weight_dict}")
    
    model = build_improved_resnet(input_dim=INPUT_DIM, dropout_rate=0.4, l2_reg=0.0005)
    print(f"Параметров модели: {model.count_params():,}")
    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\nРезультаты Improved ResNet")

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 (weighted): {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['0-25 лет', '26-50 лет', '50+ лет']))
    
    return {'accuracy': acc, 'f1': f1}

# Optimized ResNet
def build_optimized_resnet(input_dim=40, dropout_rate=0.4, l2_reg=0.0003):
    """Optimized ResNet"""
    reg = regularizers.l2(l2_reg)
    
    inputs = layers.Input(shape=(input_dim,))
    
    # Расширение до 256
    x = layers.Dense(256, kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Residual блоки с плавным уменьшением
    for units in [128, 64, 32]:
        shortcut = x
        
        x = layers.Dense(units, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(units, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        
        if shortcut.shape[-1] != units:
            shortcut = layers.Dense(units, kernel_regularizer=reg)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Squeeze-and-Excitation блок
    se = layers.GlobalAveragePooling1D()(layers.Reshape((1, 32))(x))
    se = layers.Dense(16, activation='relu')(se)
    se = layers.Dense(32, activation='sigmoid')(se)
    se = layers.Reshape((1, 32))(se)
    x = layers.Multiply()([x, se])
    
    # Финальная классификация
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


def train_optimized_resnet(X, y):
    print("\nOptimized ResNet")
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
    
    class_weight_dict = {
        0: 1.5,
        1: 0.7,
        2: 5.0
    }
    print(f"Class weights: {class_weight_dict}")
    
    model = build_optimized_resnet(input_dim=INPUT_DIM, dropout_rate=0.4, l2_reg=0.0003)
    print(f"Параметров модели: {model.count_params():,}")
    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\nРезультаты Optimized ResNet")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 (weighted): {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['0-25 лет', '26-50 лет', '50+ лет']))
    
    return {'accuracy': acc, 'f1': f1}


# Основной запуск
if __name__ == "__main__":
    print("Классификация возраста")
 
    X, y = load_data(DATA_PATH, INPUT_DIM)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    results = {}
    
    # Baseline MLP
    results['Baseline MLP'] = train_baseline(X, y)
    
    # Improved ResNet
    results['Improved ResNet'] = train_improved_resnet(X, y)
    
    # Optimized ResNet
    results['Optimized ResNet'] = train_optimized_resnet(X, y)
    
    # Сравнение
    print("\nСравнение всех моделей")
    print(f"\n{'Модель':<25} {'Accuracy':<12} {'F1 Score':<12}")
    
    for name, res in results.items():
        print(f"{name:<25} {res['accuracy']:<12.4f} {res['f1']:<12.4f}")
    
    best_name = max(results, key=lambda x: results[x]['f1'])
    best_result = results[best_name]
    
    print(f"Лучшая модель: {best_name}")
    print(f"   F1 Score: {best_result['f1']:.4f}")
    print(f"   Accuracy: {best_result['accuracy']:.4f}")