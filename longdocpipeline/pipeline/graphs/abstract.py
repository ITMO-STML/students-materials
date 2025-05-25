from abc import ABC, abstractmethod

from langgraph.graph.state import CompiledStateGraph


class GraphWrapper(ABC):

    def __init__(self, app: CompiledStateGraph):
        self.app = app

    @abstractmethod
    def execute(self): ...
