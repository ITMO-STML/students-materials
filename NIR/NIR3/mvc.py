from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import List
import json


class EstimationList(BaseModel):
    anec: List[str] = Field(
        default_factory=list,
        description="Список анекдотов"
    )


llm = ChatOpenAI(
    streaming=False,
    temperature=0,
    model='qwen3-14b',
    base_url="http://nid-sc-34:8765/v1",
    api_key="EMPTY",
)


prompt = ('Развлеки меня:'
          'Придерживайся строго формата выхода, не пиши лишнего.'
          'На выходе дай список анекдотов в формате'
          '{{"anec": ["1","2"]}}'
          ' /nothink')

response = llm.invoke(prompt)
answer = response.content

answer = answer.replace("```", "").replace("json\n", "")
answer = answer.split("</think>")[-1] if "</think>" in answer else answer.strip()

try:
    parsed = EstimationList.model_validate_json(answer)
    result = parsed.anec
except ValidationError:
    try:
        obj = json.loads(answer)
        if isinstance(obj, list):
            obj = {"anec": obj}
        parsed = EstimationList.model_validate(obj)
        result = parsed.anec
    except (json.JSONDecodeError, ValidationError):
        result = []

g = 1
