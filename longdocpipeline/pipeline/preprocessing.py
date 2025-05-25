import logging
from typing import Callable, List, Literal

from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document

from longdocpipeline.pipeline.constants import (
    TASKS_REQUIRING_QUERY,
    TASK_SPECIFIC_PREPROC_PARAMS,
    SplitModeType,
    TaskType,
)

logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(
            self,
            task: TaskType,
            token_limit: int,
            count_tokens: Callable,
            split_mode: SplitModeType,
            custom_chunk_size: int,
            custom_chunk_overlap: int,
            custom_separator: str,
    ):
        default_params = TASK_SPECIFIC_PREPROC_PARAMS[task]
        chunk_size = default_params.chunk_size
        chunk_overlap = default_params.chunk_overlap
        chunk_size = custom_chunk_size or chunk_size
        if custom_chunk_overlap is not None:
            chunk_overlap = custom_chunk_overlap

        assert (
                token_limit >= chunk_size
        ), "'chunk_size' can't be greater than 'token_limit'"

        if split_mode == SplitModeType.PARAGRAPH:
            separator = "\n\n"
            keep_separator = None
        elif split_mode == SplitModeType.SENTENCE:
            separator = ". "
            keep_separator = "end"
        elif split_mode == SplitModeType.MARKDOWN_HEADER:
            separator = "# "
            keep_separator = "start"
        else:
            raise NotImplementedError

        if custom_separator:
            separator = custom_separator
            keep_separator = "start"

        self.splitter = CharacterTextSplitter(
            separator=separator,
            keep_separator=keep_separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=count_tokens,
        )
        logger.info(
            f"Splitting text using mode='{split_mode}', separator='{separator}', chunk_size={chunk_size}, chunk_overlap={chunk_overlap}..."
        )

        self.oneshot = None

    def __call__(self, text: str) -> List[Document]:
        if not text:
            raise ValueError("'text' can't be empty!")

        doc = Document(page_content=text, metadata={})
        split_doc = self.splitter.split_documents([doc])  # needs a list
        logger.info(f"Split text into {len(split_doc)} part(s)")

        self.oneshot = True if len(split_doc) == 1 else False

        return split_doc

    @staticmethod
    def sanitize(
            text: str,
            task: TaskType = None,
            prompt_type: Literal["general", "refine"] = None,
    ) -> str:
        """
        заменить '{' и '}' на '{{' и '}}' кроме тех подстрок, которые действительно будут заменяться
        """
        mandatory_fields = []
        if prompt_type:
            assert task, "task required for prompt sanitization"

            mandatory_fields.append("context")
            if task in TASKS_REQUIRING_QUERY:
                mandatory_fields.append("query")
            if prompt_type == "refine":
                mandatory_fields.append("existing_result")

        skip_positions = []
        for field in mandatory_fields:
            substring = "{" + field + "}"
            start_position = text.find(substring)
            for i in range(len(substring)):
                skip_positions.append(start_position + i)
        text = list(text)
        for i, sym in reversed(list(enumerate(text))):
            if i not in skip_positions and sym in ["{", "}"]:
                text.insert(i, sym)
        text = "".join(text)
        return text
