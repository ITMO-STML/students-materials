from typing import Dict, List
from gigachat.models import Chat, Messages, MessagesRole, Function, FunctionParameters
from gigachat.models.function_parameters_property import FunctionParametersProperty
from importlib import import_module

all_actual_tools = [
    'Inf2Irregular',
    'snouns',
    'CompoundSubjectHelper',
    'CollectiveNounHelper',
    'CountableUncountableHelper',
    'ArticleUsageHelper',
    'KnownUnknownHelper',
    'GeneralSpecificHelper',
    'NoArticleHelper',
    'DefArticleHelper',
    'IndefArticleHelper',
    'PrepositionHelper',
    'AdjectiveOrderHelper',
    'PastObligationHelper',
    'ThereExpressionsHelper',
    'ItExpressionsHelper',
    'CausativeHelper',
    'AdjCollocationHelper',
    'VerbNounPrepHelper',
    'MakeDoCollocationHelper',
    'HaveTakeCollocationHelper',
]


class Inf2Irregular:
    # mapping: Dict[int, Dict[str, str]]

    def __init__(self, path_to_data='irregular-verbs-de.csv'):
        self.mapping = {2: {}, 3: {}}
        self.create_map(path_to_data)

    def description(self):
        description = Function(name ="check_regularity",
                               description="Проверка правильности глагола. Полезна, когда нужно поискать "
                                           "в таблице неправльных глаголов.",
                               parameters=FunctionParameters(type="object",
                                                             properties={'infinitive':{"type": "string",
                                                                                 "description":
                                                                                     "Инфинитив глагола для проверки"},
                                                              },
                                                             required=["infinitive"]))

        return description

    def check_regularity(self, infinitive: str, form: int | None = None):
        inf = infinitive.lower()
        if form is None:  # только True / False
            return inf in self.mapping[2]
        assert form in (2, 3), "form must be 2 or 3"
        return self.mapping[form].get(inf, inf)

    def create_map(self, path_to_data: str) -> None:
        with open(path_to_data, 'r') as f:
            file = f.readlines()
        for verbs in file:
            inf, snd_f, thrd_f, de = verbs.replace('\n', '').split('","')
            self.mapping[2][inf.replace('"', '')] = snd_f.replace('"', '')
            self.mapping[3][inf.replace('"', '')] = thrd_f.replace('"', '')
        g = 1

class snouns:
    singular_nouns_with_s = {
        "measles", "mumps", "aerobics", "gymnastics", "darts",
        "mathematics", "politics", "news", "thanks", "happiness"
    }

    plural_measurements = {"metres", "hours", "miles", "minutes", "seconds", "pounds"}

    always_plural_nouns = {
        "goods", "whereabouts", "remains", "stairs", "proceeds"
    }

    two_part_objects = {
        "glasses", "jeans", "pyjamas", "scales",
        "scissors", "spectacles", "trousers"
    }

    def description(self):

        description = Function(name ="explain_noun",
                               description="Объясняет правила согласования глагольной формы для данного существительного",
                               parameters=FunctionParameters(type="object",
                                                             properties= {'noun':{"type":"string",
                                                                                  "description":
                                                                                      "Существительное для анализа"},
                                                              },
                                                             required=["noun"]),)




        return description

    def get_verb_number(self, noun: str) -> str:
        noun = noun.lower()
        if noun in self.singular_nouns_with_s:
            return "singular"
        elif noun in self.always_plural_nouns or noun in self.two_part_objects:
            return "plural"
        elif any(noun.endswith(m) for m in self.plural_measurements):
            return "singular"
        return "unknown"

    def explain_noun(self, noun: str) -> str:
        number = self.get_verb_number(noun)
        if number == "singular":
            return f"The noun '{noun}' ends in -s but takes a singular verb due to its category (illness, abstract, etc.)."
        elif number == "plural":
            return f"The noun '{noun}' is treated as plural and takes a plural verb."
        else:
            return f"No special grammar rule found for '{noun}'. Context is needed."

class CompoundSubjectHelper:
    """Определяет согласование глагола с составным подлежащим"""
    def description(self):
        return {
            "name": "check_compound_subject",
            "description": "Определяет, должна ли использоваться множественная или единственная форма глагола с составным подлежащим",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_phrase": {"type": "string", "description": "Фраза с двумя подлежащими, соединёнными 'and' или 'both ... and'"}
                },
                "required": ["subject_phrase"]
            }
        }

    def analyze_subject(self, subject_phrase: str) -> str:
        phrase = subject_phrase.lower().strip()
        # Единые концепции
        if any(phrase == item for item in ["fish and chips", "bread and butter", "macaroni and cheese"]):
            return "singular"
        # Названия произведений
        if '"' in phrase or '“' in phrase or any(keyword in phrase for keyword in ['film', 'book', 'novel', 'movie']):
            return "singular"
        # Составные подлежащие
        if "both" in phrase and "and" in phrase:
            return "plural"
        if " and " in phrase:
            return "plural"
        return "unknown"

    def check_compound_subject(self, subject_phrase: str) -> str:
        """Обёртка для совместимости с function_call"""
        return self.analyze_subject(subject_phrase)


class CollectiveNounHelper:
    """Согласование глагола с собирательными существительными"""
    always_plural = {"police", "people", "cattle"}
    collectives = {"family", "government", "group", "staff", "team", "band", "class",
                   "united nations", "british airways", "microsoft corporation"}

    def description(self):
        return {"name": "check_collective_noun",
            "description": "Определяет число глагола для собирательного существительного",
            "parameters": {"type": "object",
                "properties": {"noun_phrase": {"type": "string", "description": "Собирательное существительное или фраза"}},
                "required": ["noun_phrase"]
            }
        }

    def analyze_collective(self, phrase: str) -> str:
        p = phrase.lower().strip()
        if p in self.always_plural:
            return "plural"
        if p.startswith("a ") and " of " in p:
            return "singular"
        if p in self.collectives:
            return "either"
        if p.startswith("the majority of") or p.startswith("a number of") or p.startswith("a couple of"):
            return "plural"
        return "either"

    def check_collective_noun(self, noun_phrase: str) -> str:
        return self.analyze_collective(noun_phrase)

class CountableUncountableHelper:
    """Классификация существительных как исчисляемых или неисчисляемых"""
    countable_examples = {"a coffee", "a chicken", "a drawing", "a stone"}
    uncountable_examples = {"coffee", "some coffee", "chicken", "some chicken", "drawing", "stone"}

    def description(self):
        return {"name": "check_countability",
                "description": "Определяет, употребляется ли существительное как исчисляемое или неисчисляемое",
                "parameters": {"type": "object",
                               "properties": {"noun_phrase": {"type": "string", "description": "Существительное или фраза"}},
                               "required": ["noun_phrase"]
                               }
                }

    def analyze_countability(self, phrase: str) -> str:
        p = phrase.lower().strip()
        # Прямые примеры из правила
        if any(p == example for example in self.countable_examples):
            return "countable"
        if any(p == example for example in self.uncountable_examples):
            return "uncountable"
        # Общие случаи
        if p.startswith("a ") or p.startswith("an "):
            return "countable"
        if p.startswith("some ") or not p.split()[0] in ['a', 'an', 'the']:
            return "uncountable"
        return "unknown"

    def check_countability(self, noun_phrase: str) -> str:
        return self.analyze_countability(noun_phrase)

class ArticleUsageHelper:
    """Правила использования артиклей и форм при именовании, описании и классификации"""
    def description(self):
        return {
            "name": "check_article_usage",
            "description": "Определяет корректное использование 'a/an', 'the' или формы множественного числа при именовании/описании/классификации",
            "parameters": {
                "type": "object",
                "properties": {
                    "phrase": {"type": "string", "description": "Фраза для анализа использования артиклей или формы"}
                },
                "required": ["phrase"]
            }
        }

    def analyze_usage(self, phrase: str) -> str:
        p = phrase.strip()
        lower = p.lower()
        # 1) 'a/an' для именования примера
        if lower.startswith(('a ', 'an ')) and 'one ' not in lower:
            return "a/an используется для именования или описания одного экземпляра"
        # 2) 'one' подчеркивает единственность
        if lower.startswith('one '):
            return "one используется, чтобы указать ровно один экземпляр"
        # 3) без артикля, во множественном числе — общее утверждение о группе
        words = lower.split()
        if words[0] != 'the' and words[-1].endswith('s'):
            return "множественное число без артикля — обобщение о всей группе"
        # 4) 'the' + единственное число — формальное обобщение о виде или группе
        if lower.startswith('the ') and not lower.split()[1].endswith('s'):
            return "the + единственное число — формальное обобщение о виде или группе"
        # 5) 'an' перед гласным звуком
        if lower.startswith('an '):
            return "an подходит перед словом, начинающимся на гласный звук"
        return "Неопределённый контекст для анализа фразы"

    def check_article_usage(self, phrase: str) -> str:
        return self.analyze_usage(phrase)

class KnownUnknownHelper:
    """
    Определяет, следует ли использовать 'a/an' (новая информация) или 'the' (известная информация)
    """
    def description(self):
        return {
            "name": "check_known_unknown",
            "description": "Определяет, вводится ли новая информация (a/an) или это уже известная информация (the)",
            "parameters": {
                "type": "object",
                "properties": {
                    "phrase": {
                        "type": "string",
                        "description": "Полное предложение или фраза для анализа"
                    }
                },
                "required": ["phrase"]
            }
        }

    def analyze_known_unknown(self, phrase: str) -> str:
        p = phrase.strip()
        lower = p.lower()

        # 1. Если перед существительным стоит "the", считаем информацию известной
        if lower.startswith("the "):
            return "Используется 'the' — информация известна слушателю/читателю."

        # 2. Уникальные объекты и организации
        unique_markers = [" only one", " the beginning", " the first", " the last"]
        if any(marker in lower for marker in unique_markers):
            return "Используется 'the' — объект уникален или упоминается впервые как единственный."

        # 3. Суперлативы
        if "greatest" in lower or "best" in lower or "highest" in lower:
            return "Используется 'the' — присутствует суперлатив."

        # 4. Определяющие придаточные:
        if " who " in lower or " which " in lower or " that " in lower:
            return "Используется 'the' — определяющее придаточное делает информацию известной."

        # 5. Препозитивные фразы местоположения ("in the café next to...")
        loc_markers = [" in the ", " at the ", " next to ", " near "]
        if any(marker in lower for marker in loc_markers):
            return "Используется 'the' — контекст/местоположение делает информацию известной."

        # 6. Иначе — вводим первую раз упоминаемую информацию
        if lower.startswith(("a ", "an ")):
            return "Используется 'a/an' — вводится новая (неизвестная) информация."
        else:
            # Если нет явного артикля — считаем новую
            return "Используется 'a/an' (опущено) — вводится новая информация."

    def check_known_unknown(self, phrase: str) -> str:
        return self.analyze_known_unknown(phrase)

class GeneralSpecificHelper:
    """
    Определяет использование артикля или его отсутствие для обобщённого и конкретного смысла
    """
    def description(self):
        return {
            "name": "check_general_specific",
            "description": "Проверяет, требуется ли артикль для выражения общего или конкретного значения",
            "parameters": {
                "type": "object",
                "properties": {
                    "phrase": {
                        "type": "string",
                        "description": "Фраза с существительным во множественном или неисчисляемом числе"
                    }
                },
                "required": ["phrase"]
            }
        }

    def analyze_general_specific(self, phrase: str) -> str:
        p = phrase.strip()
        lower = p.lower()

        # 1) Обобщённый смысл: множественное или неисчисляемое без артикля
        # если фраза начинается с существительного во мн. числе или без артикля + uncountable
        if not lower.startswith("the ") and (lower.endswith("s") or
           any(unc in lower for unc in ["bread", "hope", "prison", "school", "hospital"])):
            return "Без артикля — обобщённое значение (весь класс или функция)"

        # 2) Конкретный смысл: с "the" перед множественным или неисчисляемым
        if lower.startswith("the "):
            return "С 'the' — конкретное значение или фокус на определённый объект/тип"

        # 3) Специальные функции здания vs функция
        # если слова school/prison/hospital употреблены с артиклем,
        # вероятно, имеется в виду здание, а не функция
        keywords = ["school", "prison", "hospital"]
        for kw in keywords:
            if lower.startswith(f"the {kw}"):
                return "С 'the' — конкретное здание или место"

        return "Контекст неясен: требуется дополнительная информация"

    def check_general_specific(self, phrase: str) -> str:
        return self.analyze_general_specific(phrase)

class NoArticleHelper:
    """
    Определяет случаи, когда артикль не используется
    """
    def description(self):
        return {
            "name": "check_no_article",
            "description": "Проверяет, нужно ли опустить артикль перед данным словом или фразой",
            "parameters": {
                "type": "object",
                "properties": {
                    "phrase": {"type": "string", "description": "Слово или фраза для проверки"}
                },
                "required": ["phrase"]
            }
        }

    def analyze_no_article(self, phrase: str) -> str:
        p = phrase.strip()
        lower = p.lower()

        # 1) Имена и титулы
        if any(p.startswith(prefix) for prefix in ["Mr ", "Mrs ", "Dr ", "President ", "Queen "]):
            return "Без артикля — имя или титул"

        # 2) Континенты, страны (не мн.ч.), города, озёра, горы, улицы, парки
        geo = {"europe", "africa", "asia", "japan", "argentina", "slovenia"}
        if lower in geo or lower.startswith(("mount ", "lake ", "street", "square", "park")):
            return "Без артикля — географическое название"

        # 3) Материалы, жидкости, газы
        materials = {"silk", "olive oil", "pure oxygen", "water"}
        if lower in materials or lower.startswith("made of "):
            return "Без артикля — материал, жидкость или газ"

        # 4) Приёмы пищи, виды спорта, языки, школьные предметы, болезни
        foods = {"breakfast", "lunch", "dinner"}
        sports = {"tennis", "squash", "football", "basketball"}
        langs = {"swahili", "physics", "biology", "mathematics"}
        illnesses = {"cancer", "measles", "mumps", "aerobics"}  # пример
        if lower in foods | sports | langs or any(ill in lower for ill in illnesses):
            return "Без артикля — приём пищи, спорт, язык, предмет или болезнь"

        # 5) Названия компаний, магазинов, журналов
        if p.istitle() and len(p.split()) == 1:
            return "Без артикля — название организации или издания"

        # 6) Noun + number
        if any(tok.isdigit() for tok in p.split()):
            return "Без артикля — конструкция с номером"

        return "Артикль может потребоваться — не подпадает под исключение без артикля"

    def check_no_article(self, phrase: str) -> str:
        return self.analyze_no_article(phrase)


class DefArticleHelper:
    """
    Определяет случаи, когда нужен определённый артикль 'the'
    """
    def description(self):
        return {
            "name": "check_definite_article",
            "description": "Проверяет, нужен ли определённый артикль 'the' перед данным словом или фразой",
            "parameters": {
                "type": "object",
                "properties": {
                    "phrase": {"type": "string", "description": "Слово или фраза для проверки"}
                },
                "required": ["phrase"]
            }
        }

    def analyze_definite(self, phrase: str) -> str:
        lower = phrase.strip().lower()
        # 1) Plurals обозначают всю группу: always 'the'
        if lower.startswith("the ") and lower.split()[1].endswith("s"):
            return "С 'the' — речь о всей совокупности (множественное число)"

        # 2) Geographical regions, моря, реки, mountain ranges, deserts
        if any(lower.startswith(prefix) for prefix in ["the west", "the pacific", "the black sea", "the rhone", "the pyrenees"]):
            return "С 'the' — географический регион или природный объект"

        # 3) Institutions, СМИ, здания, музеи
        if any(lower.startswith(f"the {word}") for word in ["united nations", "world health organisation", "media", "theatre", "radio", "cinema"]):
            return "С 'the' — название организации, СМИ или вид искусства"

        # 4) Measurements and instruments
        if " by the " in lower or lower.startswith("the violin") or lower.startswith("the gram"):
            return "С 'the' — измерение или музыкальный инструмент"

        # 5) Superlatives, ordinal numbers, unique
        if any(tok in lower for tok in ["most", "last", "only", "first", "tenth", "greatest"]):
            return "С 'the' — суперлатив, порядковое числительное или уникальный объект"

        # 6) Noun + of (университет, залив, река и т.д.)
        if " of " in lower:
            return "С 'the' — конструкция 'noun of noun' (университет, залив и т.п.)"

        return "Определённый артикль может не требоваться в этом контексте"

    def check_definite_article(self, phrase: str) -> str:
        return self.analyze_definite(phrase)


class IndefArticleHelper:
    """
    Определяет случаи, когда нужен неопределённый артикль 'a/an'
    """
    def description(self):
        return {
            "name": "check_indefinite_article",
            "description": "Проверяет, нужен ли неопределённый артикль 'a/an' перед данным словом",
            "parameters": {
                "type": "object",
                "properties": {
                    "phrase": {"type": "string", "description": "Слово или фраза для проверки"}
                },
                "required": ["phrase"]
            }
        }

    def analyze_indefinite(self, phrase: str) -> str:
        p = phrase.strip()
        lower = p.lower()

        # 1) Jobs, nationalities, beliefs
        if lower.startswith(("a ", "an ")) and any(tok in lower for tok in [" engineer", " doctor", " italian", " structural engineer"]):
            return "С 'a/an' — профессия, национальность или убеждение"

        # 2) Большие числа и дроби
        if any(lower.startswith(prefix) for prefix in ["a million", "a hundred", "a fifth", "a hundredth"]):
            return "С 'a' — большая числовая величина или дробь"

        # 3) Цены, скорости, частота
        if any(lower.endswith(suffix) for suffix in ["per kilo", "km an hour", "an hour", "once a day"]) or "two dollars a" in lower:
            return "С 'a/an' — цена, скорость или частота"

        return "Неопределённый артикль может не требоваться в этом контексте"

    def check_indefinite_article(self, phrase: str) -> str:
        return self.analyze_indefinite(phrase)

class PrepositionHelper:
    """
    Анализирует использование предлогов at, in и on для выражения места и времени
    """
    def description(self):
        return {
            "name": "check_preposition_usage",
            "description": "Определяет корректный предлог (at/in/on) для данного контекста места или времени",
            "parameters": {
                "type": "object",
                "properties": {
                    "phrase": {
                        "type": "string",
                        "description": "Фраза с предлогом для анализа"
                    }
                },
                "required": ["phrase"]
            }
        }

    def analyze_preposition(self, phrase: str) -> str:
        p = phrase.strip()
        lower = p.lower()

        # PLACE: точка в пространстве
        if lower.startswith("at "):
            return "at — специфическая точка в пространстве"
        # PLACE: внутри области или комнаты
        if lower.startswith("in "):
            return "in — внутри области, помещения или окружения"
        # PLACE: на поверхности или линии
        if lower.startswith("on "):
            return "on — на поверхности или по линии"

        # TIME: особые моменты и периоды
        if lower.startswith("at ") and any(tok in lower for tok in [
            "five to", "new year", "night", "weekend", "time of"
        ]):
            return "at — время (конкретный момент, праздник, часть суток, уикенд)"
        # TIME: месяцы, сезоны, года, столетия
        if lower.startswith("in ") and any(tok in lower for tok in [
            "morning", "afternoon", "december", "winter", "1889", "twentieth century"
        ]):
            return "in — внутри периода (часть дня, месяц, сезон, год, столетие)"
        # TIME: дни, даты, специальные дни
        if lower.startswith("on ") and any(tok in lower for tok in [
            "thursday", "labour day", "31st", "1st", "monday"
        ]):
            return "on — конкретный день или дата"

        return "Контекст неясен — проверьте использование at/in/on для места или времени"

    def check_preposition_usage(self, phrase: str) -> str:
        return self.analyze_preposition(phrase)

class AdjectiveOrderHelper:
    """
    Проверяет порядок прилагательных. Вместо фиксированных словарей
    мы даём модели «инструкцию с примерами», а она сама распределяет
    незнакомые прилагательные по категориям.
    """
    ORDER = ["opinion", "size", "quality/character", "age/shape",
             "colour", "origin", "material", "purpose/type"]

    def description(self):
        return {
            "name": "check_adjective_order",
            "description": (
                "Определи категорию (opinion, size, age …) для каждого прилагательного "
                "и проверь, что они идут в правильном порядке. "
                "Если порядок нарушен – верни правильный вариант."
            ),
            "parameters": {
                "type": "object",
                "properties": {"phrase": {"type": "string"}},
                "required": ["phrase"]
            },
            # примеры
            "examples": [
                # opinion перед size
                {"phrase": "a fantastic new phone"},
                # size перед colour
                {"phrase": "a big red ball"},
                # age перед colour  перед origin
                {"phrase": "an old black Italian car"},
            ]
        }

    def check_adjective_order(self, phrase: str) -> str:
        return (
            "Пример для LLM: проверь порядок прилагательных, "
            "следуя инструкции и примерам из description."
        )

class PastObligationHelper:
    """
    Инструмент для выбора корректной английской конструкции,
    выражающей прошедшее обязательство / необходимость.
    ──────────────────────────────────────────────────────────
        obligation | performed | результат
        ───────────┼───────────┼──────────────────────────────────────────────
           True     |   True    | had to + V₁
           True     |   False   | should / ought to have + V₃
           False    |   False   | didn’t have / need to + V₁
           False    |   True    | needn’t have + V₃
    """

    def description(self):
        return Function(
            name="past_obligation_form",
            description="Определяет, какую форму использовать для прошедшего "
                        "обязательства (had to / should have / didn’t have to / needn’t have) "
                        "на основе двух признаков.",
            parameters=FunctionParameters(
                type="object",
                properties={
                    "obligation": {
                        "type": "boolean",
                        "description": "Была ли необходимость / обязанность?"
                    },
                    "performed": {
                        "type": "boolean",
                        "description": "Было ли действие фактически выполнено?"
                    }
                },
                required=["obligation", "performed"]
            )
        )

    def past_obligation_form(self, obligation: bool, performed: bool) -> str:
        if obligation and performed:
            return "had to + base verb  (обязан был и выполнил)"
        if obligation and not performed:
            return "should / ought to have + past participle  (обязан был, но НЕ выполнил)"
        if not obligation and not performed:
            return "didn’t have to / didn’t need to + base verb  (обязанности не было, действие не совершалось)"
        if not obligation and performed:
            return "needn’t have + past participle  (обязанности не было, но действие совершили зря)"
        # сюда не попадём, т.к. оба параметра обязательны
        return "Недостаточно данных для рекомендации"

class ThereExpressionsHelper:
    def description(self):
        return Function(
            name="check_there_expression",
            description="Определяет, какая грамматическая форма идёт после выражения "
                        "с there + be (to-infinitive / -ing / that-clause).",
            parameters=FunctionParameters(
                type="object",
                properties={
                    "phrase": {
                        "type": "string",
                        "description": "Фраза, начинающаяся после 'there (be)…'"
                    }
                },
                required=["phrase"]
            )
        )

    CERTAINTY = {"sure", "certain", "expected", "likely", "bound", "supposed"}
    NEG_TO    = {"alternative", "choice", "need", "reason"}
    NEG_ING   = {"point", "trouble", "difficulty", "chance", "question", "hope"}

    def check_there_expression(self, phrase: str) -> str:
        p = phrase.lower().strip()

        # 0) Должно начинаться с 'there ...'
        triggers = (
            list(self.CERTAINTY) +
            [f"no {w}" for w in self.NEG_TO | self.NEG_ING] +
            ["no knowing", "no denying", "no doubt", "any doubt"]
        )
        if any(trg in p for trg in triggers) and not p.startswith(("there ", "there's", "there is", "there was", "there were")):
            return ("✖ Такие выражения следует начинать с 'There …'. "
                    "Например: “There’s no point in arguing …”")

        # 1) certainty / expectation → to be
        for key in self.CERTAINTY:
            if key in p and p.endswith("to be"):
                return "✔ certainty/expectation → 'to be' корректно."

        # 2) negative + to-infinitive
        if any(f"no {kw}" in p for kw in self.NEG_TO) and " to " in p:
            return "✔ 'no …' + to-infinitive корректно."

        # 3) negative + -ing / of-ing
        if any(f"no {kw}" in p for kw in self.NEG_ING):
            if "ing" in p:
                return "✔ 'no …' + -ing корректно."
            return "✖ После выражения 'no …' ожидается форма -ing."

        # 4) no knowing + wh / if / whether
        if p.startswith("no knowing"):
            return "✔ 'no knowing' + wh/if/whether корректно."

        # 5) no denying / doubt + that-clause
        if ("no denying" in p or "no doubt" in p or "any doubt" in p) and "that" in p:
            return "✔ 'no denying/doubt that …' корректно."

        return "Не похоже на типичное выражение с 'there'."

class ItExpressionsHelper:
    def description(self):
        return Function(
            name="check_it_expression",
            description="Определяет тип выражения после 'it' и проверяет корректность "
                        "следующей грамматической формы.",
            parameters=FunctionParameters(
                type="object",
                properties={
                    "phrase": {
                        "type": "string",
                        "description": (
                            "Часть предложения, начинающаяся после 'it' "
                            "(например: 'seems as if …', 'no secret that …')."
                        )
                    }
                },
                required=["phrase"]
            )
        )

    NEG_THAT = {"no secret", "no surprise", "no wonder",
                "no coincidence", "no accident"}

    def check_it_expression(self, phrase: str) -> str:
        p = phrase.lower().strip()

        # 0) «no secret» должно начинаться с it
        if "no secret" in p and not p.startswith(("it's", "it is", "it isn't")):
            return ("✖ Выражения с 'no secret' следует начинать с 'It’s / It isn’t'. "
                    "Правильно: “It isn’t any secret (that) …”")

        # 1) it seems / looks  +  as if / though
        if p.startswith(("seems as", "seems like")):
            return "✔ 'it seems as if/though …' — впечатление, конструкция корректна."
        if p.startswith(("looks as", "looks like")):
            return "✔ 'it looks as if/though …' — вероятность, конструкция корректна."

        # 2) negative phrase + that-clause
        if any(p.startswith(expr) for expr in self.NEG_THAT):
            if "that" in p:
                return "✔ Отрицательная фраза + that-clause корректна."
            return "✖ После выражения 'no …' рекомендуется добавить 'that'-clause."

        # 3) it's no good / no use + V-ing
        if p.startswith(("no good", "no use")):
            if "ing" in p:
                return "✔ 'it's no good/use + V-ing' корректно."
            return "✖ После 'no good/use' должна быть форма -ing."

        # 4) it's no longer + adjective + to-infinitive
        if p.startswith("no longer"):
            if " to " in p:
                return "✔ 'it's no longer + adj + to-inf.' корректно."
            return "✖ После 'no longer + прил.' должен идти to-infinitive."

        return "Не относится к типовым выражениям после 'it'. Проверьте форму."

class CausativeHelper:
    """
    Подсказывает правильную форму «causative»-конструкции:
      • have + object + V₃
      • get  + object + V₃
      • have + object + infinitive
      • get  + object + to + infinitive
    Выбор зависит от трёх признаков:
        arranged   – субъект САМ организовал действие?
        unexpected – действие случилось БЕЗ его просьбы / неожиданно?
        force      – нужно подчеркнуть «уговорил / заставил» исполнителя?
    """

    # ---------------------- metadata ---------------------- #
    def description(self):
        return Function(
            name="causative_form",
            description=(
                "Выбирает подходящую causative-конструкцию "
                "(have/get something done; have/get someone do/to do) "
                "по трём логическим признакам."
            ),
            parameters=FunctionParameters(
                type="object",
                properties={
                    "arranged": {
                        "type": "boolean",
                        "description": "Субъект сам заказал/организовал действие?"
                    },
                    "unexpected": {
                        "type": "boolean",
                        "description": "Действие произошло неожиданно, без просьбы субъекта?"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Нужно подчеркнуть, что субъект уговорил/заставил исполнителя?"
                    }
                },
                required=["arranged", "unexpected", "force"]
            )
        )

    def causative_form(self, arranged: bool, unexpected: bool, force: bool) -> str:
        # 1. Неожиданное, нерегулярное (паспорт украли) → have + obj + V₃
        if unexpected:
            return ("have + object + past participle  "
                    "(e.g. *Liz had her passport stolen*)")

        # 2. «Уговорили / заставили» исполнителя → get + obj + to + infinitive
        if force:
            return ("get + object + to + infinitive  "
                    "(e.g. *We finally got them to give us a refund*)")

        # 3. Субъект сам заказал услугу
        if arranged:
            return ("have/get + object + past participle  "
                    "(have … done – формальнее, get … done – разговорнее)")

        # 4. Активный вариант (американский стиль) «заставил выполнить работу»
        return ("have + object + bare infinitive  "
                "(e.g. *I had the mechanic repair my washing machine*)")

class AdjCollocationHelper:
    """
    Проверяет соответствие пары «adjective + noun» частотным коллокациям
    (light/heavy/weak/strong/faint/poor/good/bad/little/great) и
    предлагает исправленный вариант, если пара звучит неестественно.
    """

    # частичный словарь частотных сочетаний
    ADJCOLLO = {
        "light":  {"rain", "wind", "sleeper", "smoker", "meal", "punishment"},
        "heavy":  {"rain", "traffic", "work", "punishment", "meal", "industry"},
        "weak":   {"argument", "coffee", "signal", "taste", "ruler"},
        "strong": {"argument", "coffee", "wind", "influence", "smell", "leader"},
        "faint":  {"smell", "chance", "hope", "possibility"},
        "poor":   {"health", "memory", "performance", "relation"},
        "good":   {"behaviour", "news", "luck", "time"},
        "bad":    {"behaviour", "news", "luck", "time"},
        "little": {"difficulty", "interest", "pleasure", "time"},
        "great":  {"difficulty", "interest", "pleasure", "success", "time"}
    }

    def description(self):
        return Function(
            name="check_collocation",
            description="Проверяет, естественно ли сочетается прилагательное с существительным "
                        "и предлагает корректную коллокацию при ошибке.",
            parameters=FunctionParameters(
                type="object",
                properties={
                    "adjective": {"type": "string", "description": "Прилагательное"},
                    "noun":      {"type": "string", "description": "Существительное"}
                },
                required=["adjective", "noun"]
            )
        )

    # -------- core logic -------- #
    def check_collocation(self, adjective: str, noun: str) -> str:
        adj = adjective.lower().strip()
        n   = noun.lower().strip()

        # правильная пара
        if adj in self.ADJCOLLO and n in self.ADJCOLLO[adj]:
            return f"✔ '{adjective} {noun}' — устойчивая коллокация."

        # неправильная: ищем подходящее прилагательное
        for good_adj, nouns in self.ADJCOLLO.items():
            if n in nouns:
                return (f"✖ '{adjective} {noun}' звучит неестественно. "
                        f"Правильнее: **{good_adj} {noun}**.")
        # не нашли существительное в таблице
        return ("Не удалось проверить коллокацию: этого существительного "
                "нет в базе частых сочетаний.")

class VerbNounPrepHelper:
    """
    Проверяет устойчивые коллокации вида «verb + noun + preposition»
    и предлагает правильный предлог (или верб) при ошибке.
    ──────────────────────────────────────────────────────────────
    Словарь (по учебнику):

      have   · faith **in**   · confidence **in**
      keep   · account **of** · an eye **on** · faith **with**
      make   · a success **of** · the most **of** · a fool **of**
      run    · the risk **of** · rings **round**
      take   · account **of** · a dislike **to** · pity **on**
             · pleasure **in** · trust **in**
    """

    VNPCOLLO = {
        "have": {
            "faith": "in",
            "confidence": "in"
        },
        "keep": {
            "account": "of",
            "eye": "on",
            "faith": "with"
        },
        "make": {
            "success": "of",
            "most": "of",
            "fool": "of"
        },
        "run": {
            "risk": "of",
            "rings": "round"
        },
        "take": {
            "account": "of",
            "dislike": "to",
            "pity": "on",
            "pleasure": "in",
            "trust": "in"
        }
    }

    def description(self):
        return Function(
            name="check_verb_noun_prep",
            description="Проверяет сочетание 'verb + noun + preposition' "
                        "и предлагает правильный вариант, если предлог/глагол неверен.",
            parameters=FunctionParameters(
                type="object",
                properties={
                    "verb": {"type": "string", "description": "Глагол"},
                    "noun": {"type": "string", "description": "Существительное без артикля"},
                    "preposition": {"type": "string", "description": "Использованный предлог"}
                },
                required=["verb", "noun", "preposition"]
            )
        )

    def check_verb_noun_prep(self, verb: str, noun: str, preposition: str) -> str:
        v, n, p = verb.lower(), noun.lower(), preposition.lower()

        if v not in self.VNPCOLLO:
            return f"Глагол '{verb}' не входит в список частых коллокаций — проверьте в словаре."

        if n not in self.VNPCOLLO[v]:
            # Неподходящее существительное для этого глагола
            # Поищем, с каким глаголом оно сочетается
            for good_v, mapping in self.COLLO.items():
                if n in mapping:
                    good_p = mapping[n]
                    return (f"✖ '{verb} {noun} {preposition}' не звучит естественно. "
                            f"Используйте: **{good_v} {noun} {good_p}**.")
            return f"Существительное '{noun}' не встречается в частых коллокациях."

        # Нужное существительное найдено — проверяем предлог
        correct_prep = self.VNPCOLLO[v][n]
        if p == correct_prep:
            return f"✔ Корректная коллокация: '{verb} {noun} {preposition}'."
        return (f"✖ Неверный предлог. Правильно: **{verb} {noun} {correct_prep}**.")


class MakeDoCollocationHelper:
    """
    Проверяет, следует ли использовать MAKE или DO c данным существительным/
    словосочетанием, и предлагает корректную замену при ошибке.
    """

    MAKE_SET = {
        "appearance", "appointment", "arrangements", "attempt", "bed",
        "better", "worse", "call", "change", "charge", "choice", "comment",
        "confession", "contribution", "decision", "difference", "discovery",
        "effort", "enemy", "enquiry", "exception", "excuse", "fire", "fortune",
        "friends", "fuss", "gesture", "job of", "habit", "journey", "list",
        "living", "mess", "mistake", "money", "noise", "offer", "plan", "point",
        "profit", "progress", "promise", "remark", "sound", "speech", "start",
        "suggestion", "time", "trouble", "oneself understood", "war", "will"
    }

    DO_SET = {
        "best", "business", "cleaning", "cooking", "ironing", "washing-up",
        "course", "damage", "dishes", "duty", "exam", "test", "exercise",
        "experiment", "favour", "good", "evil", "some good", "hair", "face",
        "nails", "harm", "homework", "housework", "injury", "job", "justice",
        "kindness", "laundry", "service", "operation", "research", "right",
        "wrong", "shopping", "sport", "teeth", "well", "badly"
    }

    def description(self):
        return Function(
            name="check_make_do",
            description="Проверяет, корректно ли выбрано make/do с указанным существительным "
                        "и предлагает исправление.",
            parameters=FunctionParameters(
                type="object",
                properties={
                    "verb": {"type": "string", "description": "'make' или 'do'"},
                    "noun_phrase": {
                        "type": "string",
                        "description": "Существительное/словосочетание без артикля (например 'progress')"
                    }
                },
                required=["verb", "noun_phrase"]
            )
        )

    def check_make_do(self, verb: str, noun_phrase: str) -> str:
        v = verb.lower().strip()
        n = noun_phrase.lower().strip()

        if v not in {"make", "do"}:
            return "Глагол должен быть 'make' или 'do'."

        # Попытаемся найти ключевое слово в словаре
        def match(target_set):
            return any(n == item or n.startswith(item + " ") for item in target_set)

        make_ok = match(self.MAKE_SET)
        do_ok   = match(self.DO_SET)

        # проверяет корректность
        if (v == "make" and make_ok) or (v == "do" and do_ok):
            return f"✔ Коллокация '{verb} {noun_phrase}' корректна."

        # подобрать правильный глагол
        correct = "make" if make_ok else "do" if do_ok else None
        if correct:
            return (f"✖ Лучше сказать **{correct} {noun_phrase}** "
                    f"(а не '{verb} {noun_phrase}').")
        return "Не удалось найти слово в списке частых коллокаций."


class HaveTakeCollocationHelper:
    """
    Проверяет, следует ли употребить HAVE, TAKE или допустимы оба глагола с данным существительным.
    Если выбрано неверно, предлагает корректную коллокацию.
    """

    HAVE_ONLY = {
        "appointment", "argument", "baby", "care", "chance to", "chat", "dance", "drink",
        "effect", "fall", "fit", "go", "idea", "lunch", "dinner", "meal", "problem",
        "quarrel", "race", "right", "row", "say", "something to eat", "talk", "think",
        "time", "wash"
    }

    TAKE_ONLY = {
        "action", "advantage", "breath", "care of", "chance on", "control", "decision",
        "effect", "exception", "medicine", "message", "offence", "part", "photo",
        "place", "power", "precedence", "responsibility", "risk", "root", "sides",
        "step", "steps", "turns", "trouble", "years", "months", "weeks", "days",
        "hours"
    }

    BOTH = {
        "bath", "shower", "break", "exam", "test", "guess", "holiday", "vacation",
        "look", "nap", "rest", "seat", "sip", "stroll", "swim"
    }

    def description(self):
        return Function(
            name="check_have_take",
            description="Определяет, корректно ли использован глагол have/take с указанным существительным "
                        "и даёт рекомендацию при ошибке.",
            parameters=FunctionParameters(
                type="object",
                properties={
                    "verb": {"type": "string", "description": "'have' или 'take'"},
                    "noun_phrase": {"type": "string", "description": "Существительное/словосочетание без артикля"}
                },
                required=["verb", "noun_phrase"]
            )
        )

    def check_have_take(self, verb: str, noun_phrase: str) -> str:
        v = verb.lower().strip()
        n = noun_phrase.lower().strip()

        if v not in {"have", "take"}:
            return "Глагол должен быть 'have' или 'take'."

        if n in self.BOTH:
            return f"✔ Допустимы оба глагола: 'have/take {noun_phrase}'."

        if v == "have" and n in self.HAVE_ONLY:
            return f"✔ Правильно: 'have {noun_phrase}'."
        if v == "take" and n in self.TAKE_ONLY:
            return f"✔ Правильно: 'take {noun_phrase}'."

        # найдём правильный глагол
        if n in self.HAVE_ONLY:
            return f"✖ Лучше сказать **have {noun_phrase}**."
        if n in self.TAKE_ONLY:
            return f"✖ Лучше сказать **take {noun_phrase}**."

        return "Не удалось найти существительное в списке частых коллокаций."

if __name__ == '__main__':
    i2i = Inf2Irregular('/Users/tatianakhaidukova/Documents/GitHub/STC_NLU_internship/irregular-verbs-de.csv')
    print(i2i.check_regularity('arise', 2))
    g = 1

    gn = snouns()
    print(gn.get_verb_number("scissors"))  # plural
    print(gn.explain_noun("politics"))  # explanation

    csh = CompoundSubjectHelper()
    print("CompoundSubjectHelper.analyze_subject('Mum and Dad') ->", csh.analyze_subject('Mum and Dad'))  # plural
    print("CompoundSubjectHelper.analyze_subject('Fish and chips') ->", csh.analyze_subject('Fish and chips'))  # singular

    col = CollectiveNounHelper()
    print("CollectiveNounHelper: family ->", col.analyze_collective('family'))
    print("CollectiveNounHelper: police ->", col.analyze_collective('police'))
    print("CollectiveNounHelper: a team of inspectors ->", col.analyze_collective('a team of inspectors'))
    print("CollectiveNounHelper: United Nations ->", col.analyze_collective('United Nations'))
    print("CollectiveNounHelper: the majority of people ->", col.analyze_collective('the majority of people'))

    cuh = CountableUncountableHelper()  # countable/uncountable helper
    print("CountableUncountableHelper: a coffee ->", cuh.analyze_countability('a coffee'))
    print("CountableUncountableHelper: coffee ->", cuh.analyze_countability('coffee'))
    print("CountableUncountableHelper: some chicken ->", cuh.analyze_countability('some chicken'))
    print("CountableUncountableHelper: a stone ->", cuh.analyze_countability('a stone'))

    auh = ArticleUsageHelper()
    phrases = [
        "That's a scarab beetle.",
        "There's a room available at the Marriott on Friday night.",
        "There's one room available at the Marriott on Friday night.",
        "An African elephant has larger ears than an Indian elephant.",
        "African elephants have larger ears than Indian elephants.",
        "The African elephant has larger ears than the Indian elephant.",
        "An elephant walked right past our hut yesterday evening.",
        "The homeless will be removed from the streets and placed in hostels."
    ]
    for ph in phrases:
        print(f"ArticleUsageHelper: {ph!r} ->", auh.analyze_usage(ph))

    kuh = KnownUnknownHelper()
    print(kuh.analyze_known_unknown(
        "In 1907 an English soldier set up an organisation to educate boys."))
    # → «a/an используется для ввода новой информации»
    print(kuh.analyze_known_unknown(
        "The organisation was the beginning of the World Scout Movement."))
    # → «the используется, объект уникален…»
    print(kuh.analyze_known_unknown(
        "The BBC's funding is under threat again."))
    # → «the используется, объект уникален»
    print(kuh.analyze_known_unknown(
        "Is Michael Schumacher the greatest motor racing driver ever?"))
    # → «the используется, присутствует суперлатив»
    print(kuh.analyze_known_unknown(
        "Has the last candidate arrived yet?"))
    # → «the используется, последний…»
    print(kuh.analyze_known_unknown(
        "Meet me in the café next to the bus stop."))
    # → «the используется, контекст местоположения»
    print(kuh.analyze_known_unknown(
        "An elephant walked right past our hut yesterday evening."))
    # → «a/an используется для именования одного экземпляра»
    print(kuh.analyze_known_unknown(
        "There's a room available at the Marriott on Friday night."))
    # → «a/an используется для ввода новой информации»

    gsh = GeneralSpecificHelper()
    print(gsh.analyze_general_specific("Tourists are often blamed for changing the character of a town."))
    # → Без артикля — обобщённое значение (весь класс или функция)

    print(gsh.analyze_general_specific("Did you notice what the tourists in the castle were doing?"))
    # → С 'the' — конкретное значение или фокус на определённый объект/тип

    print(gsh.analyze_general_specific("It is commonly accepted today that brown bread is good for you."))
    # → Без артикля — обобщённое значение

    print(gsh.analyze_general_specific("Did you remember to get the brown bread out of the freezer?"))
    # → С 'the' — конкретное значение

    no_h = NoArticleHelper()
    def_h = DefArticleHelper()
    ind_h = IndefArticleHelper()

    examples = [
        ("James", no_h.analyze_no_article),
        ("the United States", def_h.analyze_definite),
        ("Europe", no_h.analyze_no_article),
        ("the West", def_h.analyze_definite),
        ("breakfast", no_h.analyze_no_article),
        ("the Black Sea", def_h.analyze_definite),
        ("a structural engineer", ind_h.analyze_indefinite),
        ("a hundred thousand", ind_h.analyze_indefinite),
        ("two dollars a kilo", ind_h.analyze_indefinite),
        ("cosmopolitan", no_h.analyze_no_article),
        ("the Times", def_h.analyze_definite),
        ("Mount Everest", no_h.analyze_no_article),
        ("tennis", no_h.analyze_no_article),
        ("the most dangerous profession", def_h.analyze_definite),
        ("an Italian", ind_h.analyze_indefinite),
    ]

    for phrase, func in examples:
        print(f"{phrase!r} -> {func(phrase)}")

    prh = PrepositionHelper()
    examples = [
        "at the bus stop",
        "in the wood",
        "on the table",
        "at 8 Baker Street",
        "in the room",
        "on the Champs de Mars",
        "at five to seven",
        "at New Year",
        "at night",
        "at the weekend",
        "in the morning",
        "in December",
        "in the twentieth century",
        "on Thursday",
        "on the 31st of May",
        "on Labour Day"
    ]
    for ex in examples:
        print(f"{ex!r} -> {prh.analyze_preposition(ex)}")

    aoh = AdjectiveOrderHelper()
    tests = [
        "fantastic new MP3 player",  # correct
        "new fantastic MP3 player",  # incorrect
        "old gas heating system",  # correct
        "gas heating old system",  # incorrect
        "beautiful well-preserved eighteenth-century French stone farmhouse"  # correct
    ]
    for t in tests:
        print(f"{t!r} -> {aoh.check_adjective_order(t)}")

    poh = PastObligationHelper()
    print(poh.past_obligation_form(obligation=True,  performed=True))   # had to …
    print(poh.past_obligation_form(obligation=True,  performed=False))  # should/ought to have …
    print(poh.past_obligation_form(obligation=False, performed=False))  # didn't have/need to …
    print(poh.past_obligation_form(obligation=False, performed=True))   # needn't have …

    teh = ThereExpressionsHelper()
    print(teh.check_there_expression("no point in arguing about it"))
    # ✖ совет начать с There …
    print(teh.check_there_expression("there's no point in arguing about it"))
    # ✔ корректно

    ieh = ItExpressionsHelper()
    print(ieh.check_it_expression("no secret that she’s leaving"))  # ✖ совет начать с It’s
    print(ieh.check_it_expression("it's no secret that she’s leaving"))  # ✔ корректно

    ch = CausativeHelper()
    scenarios = [
        # arranged service
        (True, False, False),
        # unexpected/unpleasant
        (False, True, False),
        # force/persuade
        (False, False, True),
        # classic active causative
        (False, False, False)
    ]
    for a, u, f in scenarios:
        print(f"arranged={a}, unexpected={u}, force={f}  ->  {ch.causative_form(a, u, f)}")

    ach = AdjCollocationHelper()
    tests = [
        ("strong", "rain"),  # wrong
        ("heavy", "rain"),  # correct
        ("weak", "argument"),  # correct
        ("light", "traffic")  # wrong
    ]
    for adj, noun in tests:
        print(f"{adj} {noun}  ->  {ach.check_collocation(adj, noun)}")

    vnp = VerbNounPrepHelper()

    tests = [
        ("keep", "eye", "on"),  # correct
        ("keep", "eye", "in"),  # wrong prep
        ("run", "risk", "of"),  # correct
        ("run", "risk", "to"),  # wrong prep
        ("make", "success", "of"),  # correct
        ("take", "pity", "of"),  # wrong prep (should be on)
        ("have", "confidence", "to"),  # wrong prep (should be in)
        ("run", "rings", "round"),  # correct
        ("run", "rings", "around")  # wrong prep
    ]
    for v, n, p in tests:
        print(f"{v} {n} {p}  ->  {vnp.check_verb_noun_prep(v, n, p)}")

    mdh = MakeDoCollocationHelper()
    tests = [
        ("make", "progress"),  # correct
        ("do", "progress"),  # wrong
        ("make", "homework"),  # wrong
        ("do", "homework"),  # correct
        ("make", "a suggestion"),  # correct ('suggestion' → make)
        ("do", "an experiment")  # correct ('experiment' → do)
    ]
    for v, n in tests:
        print(f"{v} {n}  ->  {mdh.check_make_do(v, n)}")


    hth = HaveTakeCollocationHelper()
    tests = [
        ("have", "appointment"),  # correct
        ("take", "appointment"),  # wrong
        ("take", "photo"),  # correct
        ("have", "photo"),  # wrong
        ("have", "bath"),  # both OK
        ("take", "bath"),  # both OK
        ("have", "risk"),  # wrong
        ("take", "risk")  # correct
    ]
    for v, n in tests:
        print(f"{v} {n} -> {hth.check_have_take(v, n)}")


# 1. список имён
TOOL_NAMES = [
    'Inf2Irregular', 'snouns',
    'CompoundSubjectHelper', 'CollectiveNounHelper',
    'CountableUncountableHelper', 'ArticleUsageHelper',
    'KnownUnknownHelper', 'GeneralSpecificHelper', 'NoArticleHelper',
    'DefArticleHelper', 'IndefArticleHelper',
    'PrepositionHelper', 'AdjectiveOrderHelper', 'PastObligationHelper',
    'ThereExpressionsHelper', 'ItExpressionsHelper', 'CausativeHelper',
    'AdjCollocationHelper', 'VerbNounPrepHelper',
    'MakeDoCollocationHelper', 'HaveTakeCollocationHelper',
]

# 2. строим JSON-схемы и mapping
def _get_cls(name):
    if name in globals():
        return globals()[name]
    mod = import_module('tools')          # все классы объявлены тут же
    return getattr(mod, name)

def _prepare():
    schemas = []          # self.functions
    mapping = {}          # name → python-callable

    for cls_name in TOOL_NAMES:
        cls = _get_cls(cls_name)
        obj = cls()

        if not hasattr(obj, 'description'):
            print(f'[skip] {cls_name}: no description()')
            continue

        # JSON-описание
        desc = obj.description()
        if not isinstance(desc, Function):
            desc = Function(**desc)       # dict → Function
        schemas.append(desc)
        tool_name = desc.name            # например, "check_regularity"

        # ищем одноимённый метод
        if hasattr(obj, tool_name):
            mapping[tool_name] = getattr(obj, tool_name)
        else:                             # для is_/analyze_
            for alt in dir(obj):
                if alt.endswith(tool_name.split('_', 1)[-1]):
                    mapping[tool_name] = getattr(obj, alt)
                    break
            else:
                print(f'[warn] {cls_name}: no callable for {tool_name}')

    return schemas, mapping

_SCHEMAS, _TOOLS = _prepare()

# отдаём наружу то, что ждёт main.py
def all_actual_tools():
    return _TOOLS

def tool_schemas():
    return _SCHEMAS
