# todo: RAG-886
# pylint: disable-all
import operator
from typing import Annotated, Callable, List, Literal, TypedDict

from langchain.chains.combine_documents.reduce import split_list_of_docs, collapse_docs
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from longdocpipeline.config.application_properties import application_properties
from longdocpipeline.pipeline.gigachat_provider import gigachat
from longdocpipeline.pipeline.graphs.abstract import GraphWrapper


class MapReduceGraphWrapper(GraphWrapper):
    def __init__(
            self,
            map_prompt: ChatPromptTemplate,
            iter_prompt: ChatPromptTemplate,
            count_tokens: Callable,
            map_parser: PydanticOutputParser = None,
            iter_parser: PydanticOutputParser = None
    ):
        self.llm = gigachat
        self.token_limit = application_properties.token_limit
        self.count_tokens = count_tokens
        self.map_prompt = map_prompt
        self.iter_prompt = iter_prompt
        self.map_parser = map_parser if map_parser else StrOutputParser()
        self.iter_parser = iter_parser if iter_parser else StrOutputParser()

        self.map_chain = self.map_prompt | self.llm | self.map_parser
        self.reduce_chain = self.iter_prompt | self.llm | self.iter_parser

        # Общее состояние графа
        # Содержит входные документы, соответствующие им саммари и финальное саммари
        class OverallState(TypedDict):
            # operator.add используется, чтобы объединить все сгенерированные саммари
            # из отдельных узлов обратно в один список – это соответствует reduce части
            contents: List[str]
            summaries: Annotated[list, operator.add]
            collapsed_summaries: List[Document]
            final_summary: str

        # Состояние узла (state of the node) – передаётся между узлами, когда они выполняются,
        # и каждый узел обновляет внутреннее состояние с его выходом
        class SummaryState(TypedDict):
            content: str

        # Генерация саммари для документа
        def generate_summary(state: SummaryState):
            response = self.map_chain.invoke(state["content"])
            return {"summaries": [response]}

        # Используется как ребро графа
        def map_summaries(state: OverallState):
            # Возвращается список `Send` объектов
            # Каждый `Send` объект состоит из имени узла в графе и состояния,
            # которое отправляется на этот узел
            return [
                Send("generate_summary", {"content": content})
                for content in state["contents"]
            ]

        def collect_summaries(state: OverallState):
            return {
                "collapsed_summaries": [
                    Document(str(summary)) if self.map_parser else Document(summary) 
                    for summary in state["summaries"]
                ]
            }

        # Узел, который разбивает саммари на части, длина которых не превышает
        # контекстное окно модели (token_max)
        def collapse_summaries(state: OverallState):
            doc_lists = split_list_of_docs(
                state["collapsed_summaries"], self.count_tokens, self.token_limit
            )
            results = []
            for doc_list in doc_lists:
                results.append(collapse_docs(doc_list, self.map_chain.invoke))

            return {"collapsed_summaries": results}

        # Условный узел в графе, который определяет, нужно ли разбивать саммари на части
        def should_collapse(
                state: OverallState,
        ) -> Literal["collapse_summaries", "generate_final_summary"]:
            num_tokens = self.count_tokens(state["collapsed_summaries"])
            if num_tokens > self.token_limit:
                return "collapse_summaries"
            else:
                return "generate_final_summary"

        # Генерация финального саммари
        def generate_final_summary(state: OverallState):
            messages =  [st.page_content for st in state["collapsed_summaries"]]
            message = '\n'.join(messages)
            response = self.reduce_chain.invoke(message)
            return {"final_summary": response}

        # Сбор графа
        graph = StateGraph(OverallState)
        graph.add_node("generate_summary", generate_summary)
        graph.add_node("collect_summaries", collect_summaries)
        graph.add_node("collapse_summaries", collapse_summaries)
        graph.add_node("generate_final_summary", generate_final_summary)

        graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
        graph.add_edge("generate_summary", "collect_summaries")
        graph.add_conditional_edges("collect_summaries", should_collapse)
        graph.add_conditional_edges("collapse_summaries", should_collapse)
        graph.add_edge("generate_final_summary", END)

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
        intermediate_summaries = steps[-1]["summaries"]
        final_summary = steps[-1]["final_summary"]
        return {
            "final_result": final_summary,
            "intermediate_results": intermediate_summaries,
        }
