from __future__ import annotations


class EnricherAgent:
    """Agent #2: reviews/enriches the lecture.

    In the future you can plug a WebSearchPort here to:
      - extract claims / TODOs from the draft
      - search/verify
      - rewrite with citations
    """

    def __init__(self, enrich_chain, websearch=None):
        self.enrich_chain = enrich_chain
        self.websearch = websearch

    def run(self, lecture: str) -> str:
        return self.enrich_chain.invoke({"lecture": lecture})
