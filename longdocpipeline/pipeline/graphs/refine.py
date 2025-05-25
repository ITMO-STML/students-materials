# todo: RAG-886
# pylint: disable-all
from typing import List, Literal, TypedDict, Callable

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from longdocpipeline.config.application_properties import application_properties
from longdocpipeline.pipeline.gigachat_provider import gigachat
from longdocpipeline.pipeline.graphs.abstract import GraphWrapper


class RefineGraphWrapper(GraphWrapper):
    def __init__(
            self,
            map_prompt: ChatPromptTemplate,
            iter_prompt: ChatPromptTemplate,
            count_tokens: Callable,
            map_parser: PydanticOutputParser = None,
            iter_parser: PydanticOutputParser = None
    ):
        self.llm = gigachat

        self.count_tokens = count_tokens
        self.map_prompt = map_prompt
        self.iter_prompt = iter_prompt
        self.map_parser = map_parser if map_parser else StrOutputParser()
        self.iter_parser = iter_parser if iter_parser else StrOutputParser()

        self.map_chain = self.map_prompt | self.llm | self.map_parser
        self.reduce_chain = self.iter_prompt | self.llm | self.iter_parser

        self.config = RunnableConfig(recursion_limit=application_properties.runnable.refine_recursion_limit)

        # Общее состояние графа
        # Содержит входные документы и соответствующие им саммари
        # Также есть индекс, чтобы отслеживать, где мы находимся в последовательности документов
        class State(TypedDict):
            contents: List[str]
            index: int
            summary: str

        # Определяем функции для каждого узла
        # Узел, генерирующий изначальное саммари:
        def generate_initial_summary(state: State, config: RunnableConfig):
            summary = self.map_chain.invoke(
                state["contents"][0],
                config,
            )
            return {"summary": summary, "index": 1}

        # Узел, который дополняет саммари на основании нового документа
        def refine_summary(state: State, config: RunnableConfig):
            content = state["contents"][state["index"]]
            summary = self.refine_chain.invoke(
                {"existing_result": state["summary"], "context": content},
                config,
            )

            return {"summary": summary, "index": state["index"] + 1}

        def should_refine(state: State) -> Literal["refine_summary", END]:
            if state["index"] >= len(state["contents"]):
                return END
            else:
                return "refine_summary"

        graph = StateGraph(State)
        # Узлы
        graph.add_node("generate_initial_summary", generate_initial_summary)
        graph.add_node("refine_summary", refine_summary)

        # Рёбра
        graph.add_edge(START, "generate_initial_summary")
        graph.add_conditional_edges("generate_initial_summary", should_refine)
        graph.add_conditional_edges("refine_summary", should_refine)
        app = graph.compile()
        super().__init__(app)

    def execute(self, docs: List[Document]):
        intermediate_summaries = []
        for step in self.app.stream(
                {"contents": [doc.page_content for doc in docs]},
                stream_mode="values",
                config=self.config
        ):
            if summary := step.get("summary"):
                intermediate_summaries.append(summary)

        final_summary = intermediate_summaries[-1]
        intermediate_summaries = intermediate_summaries[:-1]
        return {
            "final_result": final_summary,
            "intermediate_results": intermediate_summaries,
        }
