import streamlit as st
from MessagePassing import AutoEssayManager

from docx import Document
from io import BytesIO

st.set_page_config(page_title="📘 Мультиагентный генератор текста", layout="wide")

st.title("📘 StudyHelper - Генератор академических текстов")
st.markdown("Создание структуры, генерация и форматирование с помощью LLM-агентов.")

# Ввод темы и API-ключа
topic = st.text_input("📝 Введите тему работы")
serpapi_key = st.text_input("🔐 Введите SerpAPI API-ключ", type="password")

# Выбор моделей для агентов
with st.expander("⚙️ Настройка моделей агентов"):
    models = {
        "analyst": st.selectbox("🧠 Analyst", ["phi4:14b", "deepseek-r1:8b", "llama3:8b", "mistral-nemo:latest"], index=0),
        "writer": st.selectbox("✍️ Writer", ["phi4:14b", "deepseek-r1:8b", "llama3:8b", "mistral-nemo:latest"], index=0),
        "reviewer": st.selectbox("🧪 Reviewer", ["deepseek-r1:8b", "phi4:14b", "llama3:8b", "mistral-nemo:latest"], index=0),
        "citation": st.selectbox("🔗 CitationAgent", ["deepseek-r1:8b", "phi4:14b", "llama3:8b", "mistral-nemo:latest"], index=0),
        "editor": st.selectbox("📚 Editor", ["deepseek-r1:8b", "phi4:14b", "llama3:8b", "mistral-nemo:latest"], index=0),
        "formatter": st.selectbox("🎨 Formatter", ["deepseek-r1:8b", "phi4:14b", "llama3:8b", "mistral-nemo:latest"], index=0)
    }

# Запуск
if st.button("🚀 Сгенерировать текст"):
    if not topic or not serpapi_key:
        st.error("❗ Введите тему и API-ключ.")
    else:
        with st.spinner("🧠 Работа агентов..."):
            manager = AutoEssayManager(topic=topic, models=models, serpapi_key=serpapi_key)
            results = manager.run_streamlit()

        st.success("✅ Готово!")

        st.subheader("📋 Требования")
        st.text(results["requirements"])

        st.subheader("📚 Источники")
        st.text(results["sources"])

        st.subheader("✍️ Сгенерированные черновики")
        for i, draft in enumerate(results["drafts"]):
            with st.expander(f"Черновик {i + 1}"):
                st.text(draft[:4000])

        st.subheader("🧪 Рецензии")
        st.text(results["review_summary"])

        st.subheader("✅ Лучший черновик")
        st.text(results["best_draft"][:4000])

        st.subheader("🔗 Ссылки и библиография (ГОСТ)")
        st.text(results["with_citations"][:4000])

        st.subheader("📚 Отредактированный текст")
        st.text(results["edited"][:4000])

        st.subheader("🎨 Отформатированный текст (ГОСТ)")
        st.text(results["formatted"][:4000])

        st.subheader("📑 Финальная рецензия на оформленный текст")
        st.text(results["final_review"])

        # 📥 Кнопка скачивания DOCX
        doc = Document()
        doc.add_paragraph(results["formatted"])

        doc_io = BytesIO()
        doc.save(doc_io)
        doc_io.seek(0)

        st.download_button(
            label="📥 Скачать итоговый текст (DOCX)",
            data=doc_io,
            file_name=f"AutoEssay_{topic.replace(' ', '_')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
