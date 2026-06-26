import json


class LLMJsonParser:
    @staticmethod
    def parse(content: str) -> dict:
        content = content.strip()

        if content.startswith("```json"):
            content = content.removeprefix("```json").strip()

        if content.startswith("```"):
            content = content.removeprefix("```").strip()

        if content.endswith("```"):
            content = content.removesuffix("```").strip()

        return json.loads(content)