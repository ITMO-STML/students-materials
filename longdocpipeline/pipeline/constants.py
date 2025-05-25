from enum import StrEnum

from pydantic import BaseModel


class TaskType(StrEnum):
    GENERAL = "general"
    SUM = "sum"
    QA = "qa"
    TRANSLATION = "translation"
    NER_UNIQUE = "ner_unique"
    SUM_CONFERENCE = "sum_conference"


class AlgoType(StrEnum):
    MAP_REDUCE = "map_reduce"
    REFINE = "refine"
    CONCAT = "concat"


class SplitModeType(StrEnum):
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    MARKDOWN_HEADER = "markdown_header"


TASK_ALGORITHM_SUPPORT = {  # первое значение в списке будет дефолтным во фронте
    TaskType.SUM: [AlgoType.MAP_REDUCE, AlgoType.REFINE],
    TaskType.QA: [AlgoType.MAP_REDUCE, AlgoType.REFINE],
    TaskType.GENERAL: [AlgoType.MAP_REDUCE, AlgoType.REFINE],
    TaskType.TRANSLATION: [AlgoType.CONCAT],
    TaskType.NER_UNIQUE: [AlgoType.CONCAT],
    TaskType.SUM_CONFERENCE: [AlgoType.MAP_REDUCE, AlgoType.CONCAT],
}

TASKS_REQUIRING_QUERY = [TaskType.QA, TaskType.GENERAL]


class PreprocessorParameters(BaseModel):
    chunk_size: int
    chunk_overlap: int


TASK_SPECIFIC_PREPROC_PARAMS = {
    TaskType.SUM: PreprocessorParameters(chunk_size=4096, chunk_overlap=1024),
    TaskType.QA: PreprocessorParameters(chunk_size=4096, chunk_overlap=1024),
    TaskType.GENERAL: PreprocessorParameters(chunk_size=4096, chunk_overlap=1024),
    TaskType.NER_UNIQUE: PreprocessorParameters(chunk_size=200, chunk_overlap=0),
    TaskType.TRANSLATION: PreprocessorParameters(chunk_size=500, chunk_overlap=0),
    TaskType.SUM_CONFERENCE: PreprocessorParameters(chunk_size=4096, chunk_overlap=0)
}

AI_ROLE_NAME = "assistant"
USER_ROLE_NAME = "user"
