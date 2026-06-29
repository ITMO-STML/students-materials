"""
Текстовые утилиты, специфичные для русского.

Задача: грубый матчинг "extracted_fact ↔ expected_fact" с учётом морфологии.
В sanity-наборе ожидаемый факт записан в одной форме ("Льва Николаевича Толстого"),
а LLM может извлечь в другой ("Лев Николаевич Толстой"). Простой substring-match
эти варианты теряет, а это создаёт ложно-низкий recall.

Решение: лемматизируем оба текста через pymorphy3 и сравниваем по подпоследовательности
лемм. pymorphy3 — опциональная зависимость; если её нет, откатываемся к старому
поведению (substring без учёта морфологии).

Логику пайплайна (где fact_text должен быть verbatim в d⁺, чтобы LLM смогла его
изменить) НЕ трогаем — фуззи только для evaluation/recall.
"""
from __future__ import annotations

import re
from typing import Optional


# Токен = непустая последовательность букв-цифр-дефисов (расширяемый Unicode-\w).
# Не захватывает пунктуацию, тире и em-dash, что важно для дат вида "1805—1812".
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# ──────────────────────────────────────────────────────────────────────
# Lazy-инициализация морфо-анализатора
# ──────────────────────────────────────────────────────────────────────

_MORPH = None
_MORPH_AVAILABLE: Optional[bool] = None  # None = ещё не пробовали; True/False = известно


def _get_morph():
    """Возвращает pymorphy3.MorphAnalyzer или None, если pymorphy3 не установлен."""
    global _MORPH, _MORPH_AVAILABLE
    if _MORPH_AVAILABLE is False:
        return None
    if _MORPH is not None:
        return _MORPH
    try:
        import pymorphy3  # type: ignore
        _MORPH = pymorphy3.MorphAnalyzer()
        _MORPH_AVAILABLE = True
        return _MORPH
    except ImportError:
        _MORPH_AVAILABLE = False
        return None


def morph_available() -> bool:
    """Проверить, доступна ли лемматизация (для логирования)."""
    return _get_morph() is not None


# ──────────────────────────────────────────────────────────────────────
# Лемматизация
# ──────────────────────────────────────────────────────────────────────

def lemmatize_tokens(text: str) -> list[str]:
    """
    Разбивает text на токены и возвращает леммы.

    Числа сохраняются как есть (parser pymorphy3 их норм. форму не меняет, но
    мы всё равно явно пропускаем — это и быстрее, и надёжнее).
    Если pymorphy3 недоступен — возвращает сырые токены в нижнем регистре.
    """
    morph = _get_morph()
    tokens = _TOKEN_RE.findall(text.lower())
    if morph is None:
        return tokens
    lemmas: list[str] = []
    for tok in tokens:
        if tok.isdigit():
            lemmas.append(tok)
            continue
        parses = morph.parse(tok)
        if parses:
            lemmas.append(parses[0].normal_form)
        else:
            lemmas.append(tok)
    return lemmas


# ──────────────────────────────────────────────────────────────────────
# Fuzzy matching: needle ↔ haystack
# ──────────────────────────────────────────────────────────────────────

def fuzzy_contains(needle: str, haystack: str) -> bool:
    """
    Возвращает True, если needle содержится в haystack ИЛИ наоборот.

    Стратегия (от быстрого к медленному):
      1. Точный substring (с lower) — самый частый кейс.
      2. Подпоследовательность лемм: лемматизировать обе строки и проверить,
         что список лемм needle встречается подряд внутри лемм haystack
         (или наоборот для коротких needle).
      3. Set-inclusion для needle длиной ≥ 3 лемм: все леммы needle есть
         в haystack (порядок неважен). Это покрывает случаи переставленных
         компонентов имени, например "Толстой Лев Николаевич" ↔ "Льва Николаевича Толстого".

    Логика "или наоборот" нужна, потому что в нашей задаче needle (ожидаемый
    факт) и haystack (извлечённый факт) могут быть кусками разной длины:
      expected="Льва Николаевича Толстого", extracted="Толстой"  → match
      expected="1380",                      extracted="8 сентября 1380 года" → match
    """
    if not needle or not haystack:
        return False

    n_low = needle.strip().lower()
    h_low = haystack.strip().lower()
    if not n_low or not h_low:
        return False

    # 1. Быстрый путь: точная подстрока
    if n_low in h_low or h_low in n_low:
        return True

    # 2. Лемматизация
    n_lemmas = lemmatize_tokens(n_low)
    h_lemmas = lemmatize_tokens(h_low)
    if not n_lemmas or not h_lemmas:
        return False

    # 2a. Подстрока на леммах: с любой из сторон
    if _is_lemma_subseq(n_lemmas, h_lemmas) or _is_lemma_subseq(h_lemmas, n_lemmas):
        return True

    # 3. Множественное включение — только для needle ≥ 3 токенов
    # (для коротких "1380" или "Толстого" set-inclusion слишком слабый сигнал
    # и даст false positives на длинных haystack).
    if len(n_lemmas) >= 3 and set(n_lemmas) <= set(h_lemmas):
        return True
    # Симметрично — если haystack короткий (≥3 лемм) и весь сидит внутри needle
    if len(h_lemmas) >= 3 and set(h_lemmas) <= set(n_lemmas):
        return True

    return False


def _is_lemma_subseq(short: list[str], long: list[str]) -> bool:
    """short — целиком подряд внутри long?"""
    n, H = len(short), len(long)
    if n == 0 or n > H:
        return False
    for i in range(H - n + 1):
        if long[i : i + n] == short:
            return True
    return False


# Алиас под имя, которое исторически использовалось в sanity-скрипте
def fuzzy_in(needle: str, haystack: str) -> bool:
    """Backward-совместимое имя для fuzzy_contains."""
    return fuzzy_contains(needle, haystack)
