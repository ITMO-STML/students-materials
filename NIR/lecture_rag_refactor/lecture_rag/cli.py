from __future__ import annotations

import argparse

from lecture_rag.adapters.loaders_langchain import LangChainMultiFormatLoader
from lecture_rag.app.pipeline import LectureRAGPipeline
from lecture_rag.config import load_settings


def main():
    parser = argparse.ArgumentParser(description="RAG lecture generator (2-agent)")
    parser.add_argument("--topic", type=str, default="Введение в deep learning", help="Lecture topic")
    parser.add_argument("--resources", type=str, default=None, help="Path to resources directory (overrides RESOURCES_DIR)")
    args = parser.parse_args()

    settings = load_settings()
    if args.resources:
        # settings is frozen dataclass, so create a new instance
        settings = settings.__class__(
            api_key=settings.api_key,
            base_url=settings.base_url,
            resources_dir=args.resources,
            lector_model=settings.lector_model,
            enricher_model=settings.enricher_model,
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            top_k=settings.top_k,
        )

    loader = LangChainMultiFormatLoader(recursive=True)
    pipeline = LectureRAGPipeline(loader=loader, settings=settings)

    print("🔹 Генерация лекции из материалов...")
    result = pipeline.run(args.topic)

    print("\n===== РЕЗУЛЬТАТ =====\n")
    print(result.final)


if __name__ == "__main__":
    main()
