from __future__ import annotations


class LectorAgent:
    """Agent #1: generates the lecture draft using RAG context."""

    def __init__(self, lecture_chain):
        self.lecture_chain = lecture_chain

    def run(self, topic: str) -> str:
        return self.lecture_chain.invoke(topic)
