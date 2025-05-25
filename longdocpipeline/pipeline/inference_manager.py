import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from longdocpipeline.config.application_properties import application_properties
from longdocpipeline.pipeline.constants import (
    TASK_ALGORITHM_SUPPORT,
    AI_ROLE_NAME,
    USER_ROLE_NAME,
    AlgoType
)
from longdocpipeline.pipeline.gigachat_provider import gigachat
from longdocpipeline.pipeline.graphs.abstract import GraphWrapper
from longdocpipeline.pipeline.graphs.concat import ConcatGraphWrapper
from longdocpipeline.pipeline.graphs.map_reduce import MapReduceGraphWrapper
from longdocpipeline.pipeline.graphs.refine import RefineGraphWrapper
from longdocpipeline.pipeline.postprocessing import Postprocessor
from longdocpipeline.pipeline.preprocessing import Preprocessor
from longdocpipeline.pipeline.prompt_setup import build_prompts


logger = logging.getLogger(__name__)


class ProcessDocResult(BaseModel):
    result_dict: dict
    history: list


class LongDocInferenceManager:

    def __init__(self):
        self.llm = gigachat

    def _count_tokens(self, string: str) -> int:
        return self.llm.get_num_tokens(string)

    def _count_tokens_in_docs(self, documents: List[Document]) -> int:
        return sum(self._count_tokens(doc.page_content) for doc in documents)

    def _init_graph(
            self,
            algorithm: AlgoType,
            map_prompt: ChatPromptTemplate,
            iter_prompt: ChatPromptTemplate,
            oneshot: bool,
            map_parser: PydanticOutputParser = None,
            iter_parser: PydanticOutputParser = None
    ) -> GraphWrapper:
        """
        Сейчас граф пересоздаётся при каждом запросе потому что это быстро и проще
        """
        if oneshot:
            algorithm = AlgoType.CONCAT
            logger.info(
                msg=f"Overriding algorithm with 'concat' because there's only one chunk."
            )

        if algorithm == AlgoType.CONCAT:
            graph = ConcatGraphWrapper(
                map_prompt=map_prompt,
                count_tokens=self._count_tokens_in_docs,
            )
        elif algorithm == AlgoType.MAP_REDUCE:
            graph = MapReduceGraphWrapper(
                map_prompt=map_prompt,
                iter_prompt=iter_prompt,
                count_tokens=self._count_tokens_in_docs,
                map_parser=map_parser,
                iter_parser=iter_parser
            )
        elif algorithm == AlgoType.REFINE:
            graph = RefineGraphWrapper(
                map_prompt=map_prompt,
                iter_prompt=iter_prompt,
                count_tokens=self._count_tokens_in_docs,
                map_parser=map_parser,
                iter_parser=iter_parser
            )
        else:
            raise NotImplementedError

        logger.info(msg=f"Using algorithm='{algorithm}'")
        return graph

    def init_preprocessor(
            self,
            task,
            split_mode=None,
            custom_chunk_size=None,
            custom_chunk_overlap=None,
            custom_separator=None,
    ) -> Preprocessor:
        """
        Много сущностей берёт из LDIM, поэтому отсюда его инициализировать проще
        """
        preprocessor = Preprocessor(
            task=task,
            token_limit=application_properties.token_limit,
            count_tokens=self._count_tokens,
            split_mode=split_mode or application_properties.preprocessing.split_mode,
            custom_chunk_size=custom_chunk_size or application_properties.preprocessing.custom_chunk_size,
            custom_chunk_overlap=custom_chunk_overlap or application_properties.preprocessing.custom_chunk_overlap,
            custom_separator=custom_separator or application_properties.preprocessing.custom_separator,
        )
        return preprocessor

    def process_doc(
            self,
            preprocessed_docs: List[Document],
            task,
            algorithm,
            oneshot=None,
            query=None,
            history=None,
            map_parser: PydanticOutputParser=None,
            iter_parser: PydanticOutputParser=None,
            custom_system_prompt=None,
            custom_map_prompt=None,
            custom_iter_prompt=None,
    ) -> ProcessDocResult:
        supported_algorithms = TASK_ALGORITHM_SUPPORT[task]
        if algorithm not in supported_algorithms:
            raise ValueError(
                f"Algorithm '{algorithm}' not supported for task '{task}'; "
                f"supported are: '{supported_algorithms}'"
            )

        if query:
            query = Preprocessor.sanitize(query)
        for d in preprocessed_docs:
            d.page_content = Preprocessor.sanitize(d.page_content)

        if custom_system_prompt:
            custom_system_prompt = Preprocessor.sanitize(custom_system_prompt)
        if custom_map_prompt:
            custom_map_prompt = Preprocessor.sanitize(
                custom_map_prompt, task=task, prompt_type="general"
            )
        if custom_iter_prompt:
            custom_iter_prompt = Preprocessor.sanitize(
                custom_iter_prompt,
                task=task,
                prompt_type=algorithm if algorithm == "refine" else "general",
            )

        map_prompt, iter_prompt = build_prompts(
            task=task,
            algorithm=algorithm,
            oneshot=oneshot,
            query=query,
            history=history,
            map_parser=map_parser,
            iter_parser=iter_parser,
            custom_system_prompt=custom_system_prompt,
            custom_map_prompt=custom_map_prompt,
            custom_iter_prompt=custom_iter_prompt,
        )

        graph = self._init_graph(algorithm, map_prompt, iter_prompt, oneshot, map_parser, iter_parser)
        result_dict = graph.execute(preprocessed_docs)

        postprocessor = Postprocessor(task)
        result_dict["final_result"] = postprocessor(result_dict["final_result"])

        # составить текущий тёрн для исходящей истории
        # user_prompt -- это дефолтный oneshot-промпт для текущего таска или кастомный map-промпт
        # поле {context} остаётся незаполненным
        user_prompt, _ = build_prompts(
            task=task,
            algorithm=algorithm,
            oneshot=True,
            query=query,
            history=None,
            map_parser=map_parser,
            iter_parser=iter_parser,
            custom_system_prompt=custom_system_prompt,
        )
        user_prompt = user_prompt[-1].prompt.template

        reply = result_dict["final_result"]

        # system-промпт в истории не передаём
        updated_history = (history or []) + [
            (USER_ROLE_NAME, user_prompt),
            (AI_ROLE_NAME, reply),
        ]

        return ProcessDocResult(result_dict=result_dict, history=updated_history)
