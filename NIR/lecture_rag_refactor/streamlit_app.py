# streamlit_app.py
from __future__ import annotations

from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Lecture RAG", page_icon="📚", layout="wide")
st.title("📚 Lecture RAG — генерация лекций из ваших материалов")

# ---- Настройки ----
st.sidebar.header("Настройки")
topic_default = "Введение в deep learning"
topic = st.sidebar.text_input("Тема лекции", value=topic_default)

with st.sidebar.expander("Переменные окружения"):
    st.write("Убедись, что задан `API_KEY` (и, если нужно, `BASE_URL`).")
    st.code('setx API_KEY "..."  # Windows\nexport API_KEY=... # Linux/Mac')

# ---- Путь к resources ----
resources_dir_candidates = [
    PROJECT_ROOT / "lecture_rag" / "resources",
    PROJECT_ROOT / "resources",
]
resources_dir = next((p for p in resources_dir_candidates if p.exists()), resources_dir_candidates[0])
resources_dir.mkdir(parents=True, exist_ok=True)

st.sidebar.caption(f"📁 Папка ресурсов: {resources_dir}")

# ---- Загрузка файлов ----
st.subheader("1) Загрузите материалы")
uploaded_files = st.file_uploader(
    "Поддерживаются: PDF, DOCX, PPTX, TXT, MD",
    type=["pdf", "docx", "pptx", "txt", "md"],
    accept_multiple_files=True,
)

col_a, col_b = st.columns([1, 1], gap="large")

with col_a:
    if st.button("⬇️ Сохранить загруженные файлы в resources", disabled=not uploaded_files):
        saved = 0
        for uf in uploaded_files:
            out_path = resources_dir / uf.name
            out_path.write_bytes(uf.getbuffer())
            saved += 1
        st.success(f"Сохранено файлов: {saved} в {resources_dir}")

with col_b:
    if st.button("🧹 Очистить resources"):
        deleted = 0
        for p in resources_dir.glob("*"):
            if p.is_file():
                p.unlink()
                deleted += 1
        st.warning(f"Удалено файлов: {deleted}")

st.divider()

# ---- Генерация ----
st.subheader("2) Сгенерировать лекцию")

@st.cache_resource
def get_pipeline(resources_dir_str: str):
    """
    Собираем пайплайн точно так же, как в lecture_rag/cli.py
    """
    from lecture_rag.adapters.loaders_langchain import LangChainMultiFormatLoader
    from lecture_rag.app.pipeline import LectureRAGPipeline
    from lecture_rag.config import load_settings

    settings = load_settings()

    settings = settings.__class__(
        api_key=settings.api_key,
        base_url=settings.base_url,
        resources_dir=resources_dir_str,
        lector_model=settings.lector_model,
        enricher_model=settings.enricher_model,
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
    )

    loader = LangChainMultiFormatLoader(recursive=True)
    return LectureRAGPipeline(loader=loader, settings=settings)

rebuild = st.checkbox("♻️ Пересобрать пайплайн/индекс (сбросить кэш)", value=False)
if rebuild:
    get_pipeline.clear()

generate_btn = st.button("🚀 Сгенерировать", type="primary")

if generate_btn:
    has_files = any(resources_dir.glob("*"))
    if not has_files:
        st.error(f"В папке {resources_dir} нет документов. Загрузите файлы выше.")
    else:
        try:
            with st.spinner("Генерация лекции..."):
                pipeline = get_pipeline(str(resources_dir))
                result = pipeline.run(topic)

            st.success("Готово!")

            # Покажем удобно: черновик и финал
            tab1, tab2 = st.tabs(["📝 Черновик (draft)", "✅ Финальная версия (final)"])
            with tab1:
                st.write(result.draft)
            with tab2:
                st.write(result.final)

            st.download_button(
                "💾 Скачать финальную лекцию как .md",
                data=result.final,
                file_name="lecture.md",
                mime="text/markdown",
            )

        except Exception as e:
            st.exception(e)

st.divider()
st.caption("Подсказка: если меняешь файлы в resources, включи «Пересобрать пайплайн/индекс» и сгенерируй заново.")
