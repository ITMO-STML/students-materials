# %% [markdown]
# # Определение просодических характеристик

# %% [markdown]
# Топ-15 самых важных признаков:
# * Само слово: 23.3131
# * has_stress: 18.8089
# * next_word_pos: 12.3302
# * word_length: 8.6399
# * word_form: 5.9292
# * words_after: 5.6466
# * part_of_speech: 5.2393
# * prev_word_pos: 4.8262
# * semantics2: 3.9640
# * genesys: 2.6516
# * words_before: 2.1488
# * sentence_length: 1.9985
# * semantics1: 1.9372
# * position_in_sentence: 1.6702
# * starts_with_capital: 0.6663

# %%
# !pip3 install torch

# %%
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error
import xgboost as xgb
from tqdm.auto import tqdm
import json
import html

# %%
class XMLParser:
    """Базовый парсер XML структуры"""
    
    def __init__(self):
        self.sentences_data = []
    
    def parse_xml(self, xml_content):
        """Парсит XML и возвращает структурированные данные"""
        soup = BeautifulSoup(xml_content, 'xml')
        sentences = soup.find_all('sentence')
        
        print("Парсинг XML...")
        for sentence in tqdm(sentences, desc="Обработка предложений"):
            sentence_data = self._parse_sentence(sentence)
            if sentence_data['words']:  # Добавляем только предложения со словами
                self.sentences_data.append(sentence_data)
        
        return self.sentences_data
    
    def _parse_sentence(self, sentence):
        """Парсит одно предложение"""
        sentence_data = {
            'words': [],
            'elements': [],
            'pauses': []
        }
        
        for element in sentence.children:
            if element.name:
                element_data = self._parse_element(element)
                if element_data:
                    sentence_data['elements'].append(element_data)
                    if element_data['type'] == 'word':
                        sentence_data['words'].append(element_data)
                    elif element_data['type'] == 'pause':
                        sentence_data['pauses'].append(element_data)
        
        return sentence_data
    
    def _parse_element(self, element):
        """Парсит отдельный элемент предложения"""
        if not element.name:
            return None
            
        element_data = {
            'type': element.name,
            'attributes': dict(element.attrs)
        }
        
        if element.name == 'word':
            element_data.update(self._parse_word(element))
        elif element.name == 'pause':
            element_data['time'] = int(element.get('time', 0))
            element_data['pause_type'] = element.get('type', 'none')
        elif element.name == 'intonation':
            element_data['intonation_type'] = element.get('type', -1)
        elif element.name == 'content':
            element_data['punkt_end'] = bool(element.get('PunktEnd'))
            element_data['link_type'] = element.get('LinkType', '0')
        
        return element_data
    
    def _parse_word(self, word):
        """Парсит слово с аллофонами"""
        original = word.get('original', '')
        content = html.unescape(original) if original else ''
        
        # Фразовое ударение
        nucleus = word.get('nucleus', '0')
        phrasal_stress = nucleus == '2'
        
        # Буквы
        letters = word.find_all('letter')
        stressed_letter = any(letter.get('stress') == '1' for letter in letters)
        
        # Аллофоны
        allophones = []
        allophone_elements = word.find_all('allophone')
        for allo in allophone_elements:
            ph_value = allo.get('ph', '')
            if ph_value:
                allophones.append(ph_value)
        
        # Словарная информация
        dictitem = word.find('dictitem')
        if dictitem:
            dict_data = {
                'subpart_of_speech': dictitem.get('subpart_of_speech', '0'),
                'form': dictitem.get('form', '0'),
                'genesys': dictitem.get('genesys', '0'),
                'stress_dict': dictitem.get('stress_dict', '0'),
                'semantics2': dictitem.get('semantics2', '0')
            }
        else:
            dict_data = {
                'subpart_of_speech': '0',
                'form': '0',
                'genesys': '0',
                'stress_dict': '0',
                'semantics2': '0'
            }
        
        return {
            'content': content,
            'word_length': len(content),
            'has_comma': ',' in content,
            'has_dot': '.' in content,
            'has_dash': '-' in content,
            'has_exclamation': '!' in content,
            'has_question': '?' in content,
            'nucleus': nucleus,
            'phrasal_stress': phrasal_stress,
            'stressed_letter': stressed_letter,
            'letter_count': len(letters),
            'allophones': allophones,
            'allophone_count': len(allophones),
            **dict_data
        }

# %%
class TrainingDataExtractor:
    """Извлекает данные для обучения из распарсенного XML"""
    
    def __init__(self):
        self.data = []
    
    def extract_from_parsed_data(self, sentences_data):
        """Извлекает данные для обучения из распарсенных предложений"""
        for sentence_data in sentences_data:
            self._process_sentence(sentence_data)
        
        return pd.DataFrame(self.data)
    
    def _process_sentence(self, sentence_data):
        """Обрабатывает одно предложение"""
        elements = sentence_data['elements']
        
        for i, element in enumerate(elements):
            if element['type'] == 'word':
                word_features = self._extract_word_features(element, elements, i)
                self.data.append(word_features)
    
    def _extract_word_features(self, word_element, elements, word_index):
        """Извлекает признаки для слова"""
        features = word_element.copy()
        
        # Удаляем служебные поля
        features.pop('type', None)
        features.pop('attributes', None)
        
        # Инициализируем признаки паузы
        features['pause_len'] = -1
        features['pause_type'] = 'none'
        features['intonation_type'] = -1
        features['has_punkt_end'] = False
        features['link_type'] = '0'
        
        # Анализируем последующие элементы для определения паузы
        for next_element in elements[word_index + 1:]:
            if next_element['type'] == 'pause':
                pause_time = next_element.get('time', 0)
                pause_type = next_element.get('pause_type', 'none')
                
                if pause_time > 0:
                    features['pause_len'] = pause_time
                    features['pause_type'] = pause_type
                break  # Первая пауза после слова
            
            elif next_element['type'] == 'intonation':
                features['intonation_type'] = next_element.get('intonation_type', -1)
            
            elif next_element['type'] == 'content':
                if next_element.get('punkt_end'):
                    features['has_punkt_end'] = True
                if next_element.get('link_type'):
                    features['link_type'] = next_element['link_type']
            
            elif next_element['type'] == 'word':
                break  # Следующее слово - прекращаем поиск
        
        return features

# %%
class PausePredictor:
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
    
    def prepare_features(self, df):
        """Подготавливает признаки для модели"""
        # Бинарные признаки
        binary_features = ['has_comma', 'has_dot', 'has_exclamation', 'has_question', 
                        'phrasal_stress', 'stressed_letter', 'has_punkt_end', 'has_dash']
        
        # Числовые признаки
        numeric_features = ['word_length', 'letter_count', 'allophone_count', 'intonation_type']
        
        # Категориальные признаки для кодирования
        categorical_features = ['subpart_of_speech', 'form', 'genesys', 'stress_dict', 'semantics2', 'link_type', 'pause_type']
        
        # Создаем копию данных и добавляем отсутствующие признаки со значениями по умолчанию
        features_df = df.copy()
        
        # Добавляем отсутствующие признаки со значениями по умолчанию
        for feature in binary_features + numeric_features + categorical_features:
            if feature not in features_df.columns:
                if feature in binary_features:
                    features_df[feature] = False
                elif feature in numeric_features:
                    features_df[feature] = -1 if feature == 'intonation_type' else 0
                else:  # categorical
                    features_df[feature] = '0'
        
        # Теперь безопасно выбираем нужные колонки
        features_df = features_df[binary_features + numeric_features + categorical_features].copy()
        
        # Заполняем пропуски
        features_df = features_df.fillna({
            'intonation_type': -1,
            'link_type': '0',
            'pause_type': 'none',
            'has_punkt_end': False,
            'has_dash': False
        })
        
        # Преобразуем типы данных
        for col in binary_features:
            features_df[col] = features_df[col].astype(bool)
        
        for col in numeric_features:
            features_df[col] = features_df[col].astype(int)
        
        for col in categorical_features:
            features_df[col] = features_df[col].astype(str)
        
        # Кодируем категориальные признаки
        for col in categorical_features:
            if col not in self.label_encoders:
                # Создаем новый encoder и обучаем на всех возможных значениях
                self.label_encoders[col] = LabelEncoder()
                # Собираем все уникальные значения для обучения
                all_values = features_df[col].unique().tolist()
                if '0' not in all_values:
                    all_values.append('0')
                self.label_encoders[col].fit(all_values)
            
            # Преобразуем значения
            try:
                features_df[col] = self.label_encoders[col].transform(features_df[col])
            except ValueError as e:
                # Если встретилось новое значение, используем значение по умолчанию
                print(f"Предупреждение: новое значение в признаке {col}, используем '0'")
                features_df[col] = 0  # значение по умолчанию
        
        # Сохраняем список признаков
        self.feature_columns = binary_features + numeric_features + categorical_features
        
        return features_df
    
    def train(self, df):
        """Обучает модели"""
        # Подготавливаем признаки
        X = self.prepare_features(df)
        y_pause = df['has_pause']
        y_duration = df['pause_len']
        
        # Проверяем баланс классов
        class_counts = y_pause.value_counts()
        print(f"Баланс классов: {class_counts.to_dict()}")
        
        # Если только один класс, добавляем синтетические данные или используем другую стратегию
        if len(class_counts) == 1:
            print("Предупреждение: только один класс в данных! Добавьте больше разнообразных данных.")
            # Временно создадим искусственный дисбаланс для тестирования
            # На практике нужно исправить парсер
        
        # Масштабируем признаки
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Разделяем данные
        X_train, X_test, y_pause_train, y_pause_test, y_dur_train, y_dur_test = train_test_split(
            X_scaled, y_pause, y_duration, test_size=0.2, random_state=42, stratify=y_pause
        )
        
        # Обучаем классификатор (есть пауза или нет)
        print("Обучение классификатора...")
        self.classifier = xgb.XGBClassifier(
            n_estimators=500,
            subsample=0.8,
            max_depth=7,
            learning_rate=0.05,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=len(y_pause_train[y_pause_train==0]) / len(y_pause_train[y_pause_train==1]) if sum(y_pause_train) > 0 else 1
        )
        self.classifier.fit(X_train, y_pause_train)
        
        # Обучаем регрессор (длительность паузы)
        print("Обучение регрессора...")
        pause_indices = y_dur_train > 0
        if sum(pause_indices) > 0:
            self.regressor = RandomForestRegressor(
                n_estimators=100,
                min_samples_split=2,
                min_samples_leaf=4,
                max_depth=10,
                random_state=42
            )
            self.regressor.fit(X_train[pause_indices], y_dur_train[pause_indices])
        
        # Оценка качества
        y_pause_pred = self.classifier.predict(X_test)
        print("Классификация (пауза есть/нет):")
        print(classification_report(y_pause_test, y_pause_pred))
        
        if self.regressor and sum(y_dur_test > 0) > 0:
            pause_test_indices = y_dur_test > 0
            y_dur_pred = self.regressor.predict(X_test[pause_test_indices])
            mae = mean_absolute_error(y_dur_test[pause_test_indices], y_dur_pred)
            print(f"Регрессия (длительность паузы) MAE: {mae:.2f}")
    
    def predict(self, text_data):
        """Предсказывает паузы для новых данных"""
        if not self.classifier:
            raise ValueError("Модель не обучена!")
        
        # Создаем DataFrame из входных данных
        input_df = pd.DataFrame(text_data)
        
        # Подготавливаем признаки
        X = self.prepare_features(input_df)
        
        # Проверяем, что все необходимые признаки присутствуют
        missing_features = set(self.feature_columns) - set(X.columns)
        if missing_features:
            print(f"Предупреждение: отсутствуют признаки: {missing_features}")
            for feature in missing_features:
                if feature in ['has_comma', 'has_dot', 'has_exclamation', 'has_question', 
                            'phrasal_stress', 'stressed_letter', 'has_punkt_end', 'has_dash']:
                    X[feature] = False
                elif feature in ['word_length', 'letter_count', 'allophone_count', 'intonation_type']:
                    X[feature] = -1 if feature == 'intonation_type' else 0
                else:
                    X[feature] = '0'
        
        # Убедимся, что порядок признаков соответствует обучению
        X = X[self.feature_columns]
        
        X_scaled = self.scaler.transform(X)
        
        # Предсказываем наличие паузы
        pause_pred = self.classifier.predict(X_scaled)
        pause_proba = self.classifier.predict_proba(X_scaled)[:, 1]
        
        # Предсказываем длительность паузы
        duration_pred = np.zeros(len(text_data))
        if self.regressor:
            # Предсказываем длительность только для слов с паузами
            pause_indices = pause_pred == 1
            if sum(pause_indices) > 0:
                duration_pred[pause_indices] = self.regressor.predict(X_scaled[pause_indices])
        
        # Форматируем результат
        result = {
            "words": []
        }
        
        for i in range(len(text_data)):
            word_result = {
                "content": text_data[i]['content'],
                "phrasal_stress": bool(text_data[i].get('phrasal_stress', False)),
                "pause_len": int(duration_pred[i]) if pause_pred[i] == 1 else -1,
                "pause_probability": float(pause_proba[i])
            }
            result["words"].append(word_result)
        
        return result

# %%
def train():
    # Загрузка XML файла
    with open('/home/danya/datasets/text_to_speech/crime_and_punishment.Result.xml', 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    # Парсинг XML
    print("Парсинг XML...")
    xml_parser = XMLParser()
    sentences_data = xml_parser.parse_xml(xml_content)
    print(f"Загружено {len(sentences_data)} предложений")
    
    # Извлечение данных для обучения
    print("Извлечение данных для обучения...")
    extractor = TrainingDataExtractor()
    df = extractor.extract_from_parsed_data(sentences_data)
    
    print(f"Загружено {len(df)} слов")
    print(f"С паузами: {sum(df['pause_len'] > 0)}")
    print(f"Фразовые ударения: {sum(df['phrasal_stress'])}")
    
    # Создаем целевую переменную
    df['has_pause'] = (df['pause_len'] > 0).astype(int)
    
    # Обучение модели
    print("\nОбучение модели...")
    predictor = PausePredictor()
    predictor.train(df)
    
    return predictor, xml_parser, extractor

# %%
import json
import pandas as pd
from bs4 import BeautifulSoup

def predict_pauses_from_xml(predictor, xml_parser, xml_file_path):
    """Предсказывает паузы для XML файла используя обученную модель"""
    
    # Парсим XML файл
    with open(xml_file_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    sentences_data = xml_parser.parse_xml(xml_content)
    
    # Извлекаем данные для предсказания
    extractor = TrainingDataExtractor()
    df_for_prediction = extractor.extract_from_parsed_data(sentences_data)
    
    # Преобразуем в формат для предсказания
    prediction_data = df_for_prediction.to_dict('records')
    
    # Предсказываем паузы
    prediction_result = predictor.predict(prediction_data)
    
    # Форматируем результат
    formatted_result = {
        "words": [
            {
                "content": word_pred["content"],
                "phrasal_stress": word_pred["phrasal_stress"],
                "pause_len": word_pred["pause_len"]
            }
            for word_pred in prediction_result["words"]
        ]
    }
    
    return formatted_result

def _parse_word_for_prediction(word):
    """Парсит слово из XML для предсказания"""
    import html
    
    original = word.get('original', '')
    content = html.unescape(original) if original else ''
    
    # Фразовое ударение определяется атрибутом nucleus="2"
    nucleus = word.get('nucleus', '0')
    phrasal_stress = nucleus == '2'
    
    # Признаки из букв
    letters = word.find_all('letter')
    stressed_letter = any(letter.get('stress') for letter in letters)
    
    # Признаки из аллофонов
    allophones = word.find_all('allophone')
    
    # Признаки из dictitem
    dictitem = word.find('dictitem')
    if dictitem:
        subpart_of_speech = dictitem.get('subpart_of_speech', '0')
        form = dictitem.get('form', '0')
        genesys = dictitem.get('genesys', '0')
        stress_dict = dictitem.get('stress_dict', '0')
        semantics2 = dictitem.get('semantics2', '0')
    else:
        subpart_of_speech = '0'
        form = '0'
        genesys = '0'
        stress_dict = '0'
        semantics2 = '0'
    
    # Анализируем content и другие элементы после слова
    has_punkt_end = False
    link_type = '0'
    intonation_type = -1
    pause_type = 'none'
    
    parent = word.parent
    if parent:
        elements = list(parent.children)
        word_index = elements.index(word)
        
        # Проверяем следующие элементы
        for next_elem in elements[word_index+1:]:
            if hasattr(next_elem, 'name'):
                if next_elem.name == 'content':
                    if next_elem.get('PunktEnd'):
                        has_punkt_end = True
                    if next_elem.get('LinkType'):
                        link_type = next_elem.get('LinkType', '0')
                elif next_elem.name == 'intonation':
                    intonation_type = next_elem.get('type', -1)
                elif next_elem.name == 'pause':
                    pause_type = next_elem.get('type', 'none')
                    break  # Пауза после слова - конец синтагмы
                elif next_elem.name == 'word':
                    # Прекращаем поиск при встрече следующего слова
                    break
    
    return {
        'content': content,
        'word_length': len(content),
        'has_comma': ',' in content,
        'has_dot': '.' in content,
        'has_dash': '-' in content,
        'has_exclamation': '!' in content,
        'has_question': '?' in content,
        'phrasal_stress': phrasal_stress,
        'stressed_letter': stressed_letter,
        'letter_count': len(letters),
        'allophone_count': len(allophones),
        'subpart_of_speech': subpart_of_speech,
        'form': form,
        'genesys': genesys,
        'stress_dict': stress_dict,
        'semantics2': semantics2,
        'has_punkt_end': has_punkt_end,
        'link_type': link_type,
        'intonation_type': intonation_type,
        'pause_type': pause_type,
        'nucleus': nucleus
    }

# %%
# Обучение
predictor, xml_parser, extractor = train()

# Предсказание
result = predict_pauses_from_xml(predictor, xml_parser, '/home/danya/datasets/text_to_speech/Test_Procody.xml')

# Сохранение результата
with open('Test_Procody.json', 'w') as f:
    json.dump([result], f, ensure_ascii=False, indent=4)


