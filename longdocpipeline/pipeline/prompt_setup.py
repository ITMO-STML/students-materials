import json
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

from longdocpipeline.config.application_properties import application_properties
from longdocpipeline.pipeline.constants import (
    AI_ROLE_NAME,
    USER_ROLE_NAME,
    TASKS_REQUIRING_QUERY, AlgoType, TaskType,
)


class Prompts(BaseModel):
    system_prompt: str
    map_prompt: str
    iter_prompt: str


with open(application_properties.default_prompt_path, encoding="utf-8") as f:
    DEFAULT_PROMPTS = json.loads(f.read())


def validate_chat_history(history: List[Tuple[str]]):
    allowed_roles = [AI_ROLE_NAME, USER_ROLE_NAME]
    for role, message in history:
        assert (
                role in allowed_roles
        ), f"roles in history have to be one of '{allowed_roles}'"
        assert isinstance(
            message, str
        ), "wrong message type in history, should be 'str'"


def get_default_prompts(
        task: TaskType,
        algorithm: AlgoType,
        oneshot: bool,
) -> Prompts:
    system_prompt = DEFAULT_PROMPTS[f"{task}_system_prompt"]

    if oneshot and algorithm != "concat":
        # если чанк один, предложить oneshot-промпт
        # для 'concat' map-промпт и так вида oneshot
        map_prompt = DEFAULT_PROMPTS[f"{task}_oneshot_prompt"]
    else:
        map_prompt = DEFAULT_PROMPTS[f"{task}_map_prompt"]

    if algorithm == "map_reduce":
        iter_prompt = DEFAULT_PROMPTS[f"{task}_reduce_prompt"]
    elif algorithm == "refine":
        iter_prompt = DEFAULT_PROMPTS[f"{task}_refine_prompt"]
    elif algorithm == "concat":
        iter_prompt = ""
    else:
        raise ValueError("iter_prompt doesn't set")

    return Prompts(system_prompt=system_prompt,
                   map_prompt=map_prompt,
                   iter_prompt=iter_prompt)


def build_prompts(
        task: TaskType,
        algorithm: AlgoType,
        oneshot: bool,
        query: str = None,
        history: List[Tuple[str]] = None,
        map_parser: PydanticOutputParser = None,
        iter_parser: PydanticOutputParser = None,
        custom_system_prompt: str = None,
        custom_map_prompt: str = None,
        custom_iter_prompt: str = None,
) -> Tuple[ChatPromptTemplate, ChatPromptTemplate]:
    prompts = get_default_prompts(
        task=task,
        algorithm=algorithm,
        oneshot=oneshot,
    )

    system_prompt = custom_system_prompt or prompts.system_prompt
    map_prompt = custom_map_prompt or prompts.map_prompt
    iter_prompt = custom_iter_prompt or prompts.iter_prompt

    if history:
        validate_chat_history(history)

        # replace default system prompt with general
        # todo: решить, оптимально ли это
        system_prompt = get_default_prompts(
            task=TaskType.GENERAL,
            algorithm=algorithm,
            oneshot=oneshot,
        ).system_prompt

    if task in TASKS_REQUIRING_QUERY:
        if not query:
            raise ValueError(f"Task {task} requires a user query.")
        else:
            map_prompt = map_prompt.format(query=query, context="{context}")
            iter_prompt = iter_prompt.format(
                query=query,
                context="{context}",
                existing_result="{existing_result}",
            )

    map_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                *(history or []),
                ("user", map_prompt),
            ]
        )
    if map_parser:
        map_prompt = map_prompt.partial(format_instructions=map_parser.get_format_instructions())

    iter_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                *(history or []),
                ("user", iter_prompt),
            ]
        )
    if iter_parser:
        iter_prompt = iter_prompt.partial(format_instructions=iter_parser.get_format_instructions())
        
    return map_prompt, iter_prompt
