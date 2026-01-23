from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


lecture_prompt = ChatPromptTemplate.from_template(
    """
Ты преподаватель университета.

Используй ТОЛЬКО информацию из контекста.
Если информации нет — напиши "Нет данных в материалах".

Контекст:
{context}

Тема:
{question}

Составь структурированную лекцию.
"""
)


enrich_prompt = ChatPromptTemplate.from_template(
    """
Ты научный редактор.

Дополнить лекцию:
- современными практиками
- примерами из индустрии
- аккуратно расширить материал

Исходная лекция:
{lecture}
"""
)
