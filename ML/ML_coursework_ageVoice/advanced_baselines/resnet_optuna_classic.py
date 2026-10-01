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

DATA_PATH = "./feature_extraction/timit_features_classic/fbank"
METADATA_PATH = "./feature_extraction/timit_features_classic/features_metadata.csv"
INPUT_DIM = 40

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


# Improved Residual блок
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


# Построение модели Improved ResNet с настраиваемыми параметрами
def build_improved_resnet(input_dim=40, learning_rate=0.0005, dropout_rate=0.4, 
                                l2_reg=0.0005, units_config=None, dense_units=None):
    """
    Improved ResNet с настраиваемыми гиперпараметрами
    units_config: список [units1, units2, units3, units4] для residual блоков
    dense_units: список [dense1, dense2] для финальных полносвязных слоев
    """
    if units_config is None:
        units_config = [128, 64, 32, 16]
    if dense_units is None:
        dense_units = [64, 32]
    
    reg = regularizers.l2(l2_reg)
    
    inputs = layers.Input(shape=(input_dim,))
    
    # Расширение
    x = layers.Dense(256, kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Residual блоки
    for units in units_config:
        x = improved_residual_block(x, units, dropout_rate, reg)
    
    # Squeeze-and-Excitation блок
    last_units = units_config[-1]
    se = layers.GlobalAveragePooling1D()(layers.Reshape((1, last_units))(x))
    se = layers.Dense(last_units // 2, activation='relu')(se)
    se = layers.Dense(last_units, activation='sigmoid')(se)
    se = layers.Reshape((1, last_units))(se)
    x = layers.Multiply()([x, se])
    
    # Финальная классификация
    x = layers.Flatten()(x)
    for du in dense_units:
        x = layers.Dense(du, activation='relu', kernel_regularizer=reg)(x)
        x = layers.Dropout(dropout_rate)(x)
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
    units1 = trial.suggest_categorical('units1', [256, 128, 64])
    units2 = trial.suggest_categorical('units2', [128, 64, 32])
    units3 = trial.suggest_categorical('units3', [64, 32, 16])
    units4 = trial.suggest_categorical('units4', [32, 16, 8])
    units_config = [units1, units2, units3, units4]
    
    # Количество нейронов в финальных dense слоях
    dense1 = trial.suggest_categorical('dense1', [128, 64, 32])
    dense2 = trial.suggest_categorical('dense2', [64, 32, 16])
    dense_units = [dense1, dense2]
    
    # Веса классов
    weight_young = trial.suggest_float('weight_young', 0.5, 2.5)
    weight_middle = trial.suggest_float('weight_middle', 0.5, 1.5)
    weight_old = trial.suggest_float('weight_old', 3.0, 10.0)
    class_weight = {0: weight_young, 1: weight_middle, 2: weight_old}
    
    # Количество эпох
    epochs = trial.suggest_categorical('epochs', [50, 75, 100])
    
    # Модель
    model = build_improved_resnet(
        input_dim=INPUT_DIM,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        units_config=units_config,
        dense_units=dense_units
    )
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=0
    )
    
    pruning_callback = TFKerasPruningCallback(trial, 'val_f1_score')
    
    # Callback для вычисления F1
    class F1ScoreCallback(callbacks.Callback):
        def __init__(self, validation_data):
            super().__init__()
            self.validation_data = validation_data
            
        def on_epoch_end(self, epoch, logs=None):
            val_pred = np.argmax(self.model.predict(self.validation_data[0], verbose=0), axis=1)
            val_f1 = f1_score(self.validation_data[1], val_pred, average='weighted')
            logs['val_f1_score'] = val_f1
    
    f1_callback = F1ScoreCallback((X_val, y_val))
    
    # Обучение
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr, pruning_callback, f1_callback],
        verbose=0
    )
    
    # Лучшее значение F1 на валидации
    best_val_f1 = max(history.history.get('val_f1_score', [0]))
    
    return best_val_f1


# Обучение с лучшими параметрами
def train_improved_resnet_with_best_params(X, y, best_params):
    """
    Обучает Improved ResNet на всех тренировочных данных с лучшими параметрами
    и тестирует на отложенной выборке
    """
    print("\nОбучение Improved ResNet с лучшими гиперпараметрами")
    print("Лучшие параметры:")
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
    epochs = best_params['epochs']
    units_config = [
        best_params['units1'],
        best_params['units2'],
        best_params['units3'],
        best_params['units4']
    ]
    dense_units = [best_params['dense1'], best_params['dense2']]
    class_weight = {
        0: best_params['weight_young'],
        1: best_params['weight_middle'],
        2: best_params['weight_old']
    }
    
    # Строим модель
    model = build_improved_resnet(
        input_dim=INPUT_DIM,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        units_config=units_config,
        dense_units=dense_units
    )
    print(f"\nПараметров модели: {model.count_params():,}")
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=6,
        min_lr=1e-6,
        verbose=1
    )
    
    # Обучение
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Оценка на тесте
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\nРезультаты Improved ResNet с оптимизированными параметрами")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 (weighted): {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['0-25 лет', '26-50 лет', '50+ лет']))
    
    return model, acc, f1


# Основной запуск
if __name__ == "__main__":
    print("\nКлассификация возраста")
    print("Оптимизация гиперпараметров для Improved ResNet через Optuna")
    
    # Загружаем данные
    X, y = load_data(DATA_PATH, INPUT_DIM)
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
        direction='maximize',
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
    print("\nЛучшие гиперпараметры:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Сохраняем результаты в таблицу
    results_df = study.trials_dataframe()
    results_df.to_csv("optuna_results_improved_resnet_mfcc.csv", index=False)
    print("\nРезультаты всех экспериментов сохранены в optuna_results_improved_resnet_mfcc.csv")
    
    # Обучение с лучшими параметрами
    best_model, test_acc, test_f1 = train_improved_resnet_with_best_params(X, y, study.best_params)
    
    # Сохраняем лучшую модель
    best_model.save("best_improved_resnet_mfcc.h5")
    print("\nЛучшая модель сохранена в best_improved_resnet_mfcc.h5")
    
    # Итоговый вывод
    print("\nИтоговый результат для Improved ResNet")
    print("Лучшие гиперпараметры:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nИтоговый F1 Score на тесте: {test_f1:.4f}")
    print(f"Итоговый Accuracy на тесте: {test_acc:.4f}")