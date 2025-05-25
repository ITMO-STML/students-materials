# todo: RAG-886
# pylint: disable-all
import operator
from typing import Annotated, Callable, List, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from longdocpipeline.config.application_properties import application_properties
from longdocpipeline.pipeline.gigachat_provider import gigachat
from longdocpipeline.pipeline.graphs.abstract import GraphWrapper


class ConcatGraphWrapper(GraphWrapper):
    def __init__(
            self,
            map_prompt: ChatPromptTemplate,
            count_tokens: Callable,
    ):
        self.llm = gigachat
        self.token_limit = application_properties.token_limit
        self.count_tokens = count_tokens
        self.map_prompt = map_prompt

        self.map_chain = self.map_prompt | self.llm | StrOutputParser()

        # Общее состояние графа
        # Содержит входные документы, соответствующие им саммари и финальное саммари
        class OverallState(TypedDict):
            # operator.add используется, чтобы объединить все сгенерированные саммари
            # из отдельных узлов обратно в один список – это соответствует reduce части
            contents: List[str]
            results: Annotated[list, operator.add]
            final_result: str

        # Состояние узла (state of the node) – передаётся между узлами, когда они выполняются,
        # и каждый узел обновляет внутреннее состояние с его выходом
        class ContentState(TypedDict):
            content: str

        # Генерация саммари для документа
        def generate_result(state: ContentState):
            response = self.map_chain.invoke(state["content"])
            return {"results": [response]}

        # Используется как ребро графа
        def map_results(state: OverallState):
            # Возвращается список `Send` объектов
            # Каждый `Send` объект состоит из имени узла в графе и состояния,
            # которое отправляется на этот узел
            return [
                Send("generate_result", {"content": content})
                for content in state["contents"]
            ]

        # Генерация финального саммари
        def concat_final_result(state: OverallState):
            response = "\n".join(state["results"])
            return {"final_result": response}

        # Сбор графа
        graph = StateGraph(OverallState)
        graph.add_node("generate_result", generate_result)
        graph.add_node("generate_final_result", concat_final_result)

        graph.add_conditional_edges(START, map_results, ["generate_result"])
        graph.add_edge("generate_result", "generate_final_result")
        graph.add_edge("generate_final_result", END)

        app = graph.compile()
        super().__init__(app)

    def execute(self, docs: List[Document]):
        steps = []
        for step in self.app.stream(
                {"contents": [doc.page_content for doc in docs]},
                stream_mode="values",
        ):
            steps.append(step)
        # return steps
        intermediate_results = steps[-1]["results"]
        final_result = str(steps[-1]["final_result"])
        return {
            "final_result": final_result,
            "intermediate_results": intermediate_results,
        }
