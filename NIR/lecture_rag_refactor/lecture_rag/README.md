# Lecture RAG (2-agent)

## Run

1) Create `.env` with:

```
API_KEY=...your key...
# optional
# BASE_URL=https://foundation-models.api.cloud.ru/v1
# RESOURCES_DIR=resources
```

2) Put your source files into `resources/` (supports: pdf, docx, pptx, txt, md).

3) Run:

```
python -m lecture_rag.cli --topic "Введение в deep learning"
```

Or:

```
python main_refactored.py --topic "..."
```

## Notes about DOCX/PPTX

DOCX/PPTX loaders use LangChain `Unstructured*` loaders. If you don't have them, install:

```
pip install unstructured
```
