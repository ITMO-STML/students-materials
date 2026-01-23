from __future__ import annotations

from dataclasses import dataclass

from lecture_rag.adapters.llm_cloudru import make_enricher_llm, make_lector_llm
from lecture_rag.agents.enricher import EnricherAgent
from lecture_rag.agents.lector import LectorAgent
from lecture_rag.rag.chains import build_enrich_chain, build_lecture_chain
from lecture_rag.rag.index import build_retriever


@dataclass
class LecturePipelineResult:
    topic: str
    draft: str
    final: str


class LectureRAGPipeline:
    """Central orchestrator ("hexagon center")."""

    def __init__(self, *, loader, settings):
        self.loader = loader
        self.settings = settings

    def run(self, topic: str) -> LecturePipelineResult:
        # 1) Load files
        documents = self.loader.load_dir(self.settings.resources_dir)
        if not documents:
            raise RuntimeError(
                f"No documents loaded from '{self.settings.resources_dir}'. "
                f"Put your files there (pdf/docx/pptx/txt/md)."
            )

        # 2) Build retriever
        retriever = build_retriever(
            documents,
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            embedding_model=self.settings.embedding_model,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            top_k=self.settings.top_k,
        )

        # 3) Build LLMs
        lector_llm = make_lector_llm(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            model=self.settings.lector_model,
        )
        enricher_llm = make_enricher_llm(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            model=self.settings.enricher_model,
        )

        # 4) Build chains + agents
        lecture_chain = build_lecture_chain(lector_llm, retriever)
        enrich_chain = build_enrich_chain(enricher_llm)

        lector = LectorAgent(lecture_chain)
        enricher = EnricherAgent(enrich_chain)

        # 5) Execute
        draft = lector.run(topic)
        final = enricher.run(draft)

        return LecturePipelineResult(topic=topic, draft=draft, final=final)
