import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, regularizers
import optuna
from optuna.integration import TFKerasPruningCallback
import warnings
warnings.filterwarnings('ignore')

# Конфиг
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

WAVLM_PATH = "./feature_extraction/wav2vec_emb/wav2vec_original_embeddings"
METADATA_PATH = "./feature_extraction/timit_features_classic/features_metadata.csv"
WAVLM_DIM = 1024

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


def load_embeddings(emb_path, expected_dim):
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
    ages = metadata['age'].values
    y = np.array([0 if age <= 25 else (1 if age <= 50 else 2) for age in ages])
    
    print(f"Найдено {found}/{len(metadata)} файлов")
    return X, y


# Лучшая архитекрура (Optimized ReSNet)
def build_model(input_dim, learning_rate, dropout_rate, l2_reg, units_config):
    """
    Построение модели с заданными гиперпараметрами
    units_config: список [units1, units2, units3, units4] для residual блоков
    """
    reg = regularizers.l2(l2_reg)
    
    inputs = layers.Input(shape=(input_dim,))
    
    # Расширение
    x = layers.Dense(1024, kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Residual блоки с настраиваемыми размерами
    for units in units_config:
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
    se = layers.GlobalAveragePooling1D()(layers.Reshape((1, units_config[-1]))(x))
    se = layers.Dense(units_config[-1] // 2, activation='relu')(se)
    se = layers.Dense(units_config[-1], activation='sigmoid')(se)
    se = layers.Reshape((1, units_config[-1]))(se)
    x = layers.Multiply()([x, se])
    
    # Финальная классификация
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(3, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.005)
    except:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# Функция цели для Optuna
def objective(trial, X_train, X_val, y_train, y_val):
    """
    Optuna оптимизирует гиперпараметры для максимизации F1 на валидации
    """
    # Гиперпараметры
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.6)
    l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Количество нейронов в residual блоках
    units1 = trial.suggest_categorical('units1', [256, 512, 1024])
    units2 = trial.suggest_categorical('units2', [128, 256, 512])
    units3 = trial.suggest_categorical('units3', [64, 128, 256])
    units4 = trial.suggest_categorical('units4', [32, 64, 128])
    units_config = [units1, units2, units3, units4]
    
    # Веса классов
    weight_young = trial.suggest_float('weight_young', 0.5, 2.0)
    weight_middle = trial.suggest_float('weight_middle', 0.5, 1.5)
    weight_old = trial.suggest_float('weight_old', 2.0, 8.0)
    class_weight = {0: weight_young, 1: weight_middle, 2: weight_old}
    
    # Модель
    model = build_model(WAVLM_DIM, learning_rate, dropout_rate, l2_reg, units_config)
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    pruning_callback = TFKerasPruningCallback(trial, 'val_f1_score')
    
    # Добавим метрику F1
    class F1ScoreCallback(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            val_pred = np.argmax(self.model.predict(X_val, verbose=1), axis=1)
            val_f1 = f1_score(y_val, val_pred, average='weighted')
            logs['val_f1_score'] = val_f1
    
    # Обучение
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr, pruning_callback, F1ScoreCallback()],
        verbose=1
    )
    
    # Лучшее значение F1 на валидации
    best_val_f1 = max(history.history.get('val_f1_score', [0]))
    
    return best_val_f1


# Обучение с лучшими параметрами
def train_with_best_params(X, y, best_params):
    """
    Обучаем модель на всех тренировочных данных с лучшими параметрами
    и тестируем на отложенной выборке
    """
    print("\nОбучение с лучшими гиперпараметрами")
    print(f"Лучшие параметры:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Разделяем на train+val и test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)
    
    # Разделяем train на train и val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=RANDOM_SEED
    )
    
    # Извлекаем параметры
    learning_rate = best_params['learning_rate']
    dropout_rate = best_params['dropout_rate']
    l2_reg = best_params['l2_reg']
    batch_size = best_params['batch_size']
    units_config = [
        best_params['units1'],
        best_params['units2'],
        best_params['units3'],
        best_params['units4']
    ]
    class_weight = {
        0: best_params['weight_young'],
        1: best_params['weight_middle'],
        2: best_params['weight_old']
    }
    
    # Строим модель
    model = build_model(WAVLM_DIM, learning_rate, dropout_rate, l2_reg, units_config)
    print(f"\nПараметров модели: {model.count_params():,}")
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
    
    # Обучение
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Оценка на тесте
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\nРезультаты на тестовой выборке")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 (weighted): {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['0-25 лет', '26-50 лет', '50+ лет']))
    
    return model, acc, f1


# Основной запуск
if __name__ == "__main__":
    print("\nКлассификация возраста (WavLM эмбеддинги)")
    print("\nОптимизация гиперпараметров через Optuna")
    
    # Загружаем данные
    X, y = load_embeddings(WAVLM_PATH, WAVLM_DIM)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Распределение классов: {np.bincount(y)}")
    
    # Разделяем данные для Optuna
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)
    
    # Разделяем train на train и val для Optuna
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=RANDOM_SEED
    )
    
    print(f"\nРазмеры выборок:")
    print(f"  Train: {X_train.shape[0]}")
    print(f"  Val:   {X_val.shape[0]}")
    print(f"  Test:  {X_test.shape[0]}")
    
    # Запуск Optuna
    print("\nЗапуск Optuna (поиск лучших гиперпараметров)")

    # Создаём исследование
    study = optuna.create_study(
        direction='maximize',  # максимизируем F1
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Функция для передачи данных
    def objective_wrapper(trial):
        return objective(trial, X_train, X_val, y_train, y_val)
    
    # Запускаем оптимизацию
    n_trials = 30
    print(f"Запускаем {n_trials} экспериментов...")
    study.optimize(objective_wrapper, n_trials=n_trials, show_progress_bar=True)
    
    # Результаты
    print("\nРезультаты Optuna")
    print(f"Лучшее значение F1 на валидации: {study.best_value:.4f}")
    print(f"\nЛучшие гиперпараметры:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Сохраняем результаты в таблицу
    results_df = study.trials_dataframe()
    results_df.to_csv("optuna_results.csv", index=False)
    print(f"\nРезультаты всех экспериментов сохранены в optuna_results.csv")
    
    # Обучение с лучшими параметрами
    best_model, test_acc, test_f1 = train_with_best_params(X, y, study.best_params)
    
    # Сохраняем лучшую модель
    best_model.save("best_age_model_wav2vec.h5")
    print(f"\nЛучшая модель сохранена в best_age_model.h5")
    
    # Итоговый вывод
    print("\nИтоговый результат")
    print(f"Лучшие гиперпараметры:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nИтоговый F1 Score на тесте: {test_f1:.4f}")
    print(f"Итоговый Accuracy на тесте: {test_acc:.4f}")
