import json

from src.json_parser import LLMJsonParser


class BaseExtractor:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    @property
    def name(self) -> str:
        raise NotImplementedError

    def extract(self, text: str) -> list[dict]:
        raise NotImplementedError

    def _safe_parse(self, content: str) -> list[dict]:
        try:
            data = LLMJsonParser.parse(content)
            return data.get("triples", [])
        except Exception as e:
            print(f"JSON ERROR in {self.name}:")
            print(e)
            print(content)
            return []


class NoOntologyExtractor(BaseExtractor):
    @property
    def name(self) -> str:
        return "A_no_ontology"

    def extract(self, text: str) -> list[dict]:
        prompt = f"""
Extract knowledge graph triples from the text.

Return ONLY raw JSON.
Do not use markdown.
Do not wrap the answer in ```json.

Format:
{{
  "triples": [
    {{
      "sub": "...",
      "rel": "...",
      "obj": "..."
    }}
  ]
}}

Rules:
- Extract facts explicitly stated in the text.
- Use natural relation names if needed.
- Do not add explanations.
- If there are no triples, return {{"triples": []}}.

Text:
{text}
"""

        content = self.llm_client.generate(prompt)
        return self._safe_parse(content)


class RelationNamesExtractor(BaseExtractor):
    def __init__(self, llm_client, ontology_relations: dict):
        super().__init__(llm_client)
        self.ontology_relations = ontology_relations

    @property
    def name(self) -> str:
        return "B_relation_names"

    def extract(self, text: str) -> list[dict]:
        relation_names = list(self.ontology_relations.keys())

        prompt = f"""
Extract knowledge graph triples from the text.

Use ONLY these relation names from the ontology:
{json.dumps(relation_names, indent=2, ensure_ascii=False)}

Return ONLY raw JSON.
Do not use markdown.
Do not wrap the answer in ```json.

Format:
{{
  "triples": [
    {{
      "sub": "...",
      "rel": "...",
      "obj": "..."
    }}
  ]
}}

Rules:
- Use only relation names from the ontology.
- Do not invent facts.
- Extract only facts explicitly stated in the text.
- Split compound locations when possible.
- For numbers, return only the number.
- For dates or years, return only the date/year.
- If there are no triples, return {{"triples": []}}.

Text:
{text}
"""

        content = self.llm_client.generate(prompt)
        return self._safe_parse(content)


class DomainRangeExtractor(BaseExtractor):
    def __init__(self, llm_client, ontology_relations: dict):
        super().__init__(llm_client)
        self.ontology_relations = ontology_relations

    @property
    def name(self) -> str:
        return "C_domain_range"

    def extract(self, text: str) -> list[dict]:
        prompt = f"""
Extract knowledge graph triples from the text.

Use ONLY these ontology relations with domain and range constraints:
{json.dumps(self.ontology_relations, indent=2, ensure_ascii=False)}

Return ONLY raw JSON.
Do not use markdown.
Do not wrap the answer in ```json.

Format:
{{
  "triples": [
    {{
      "sub": "...",
      "rel": "...",
      "obj": "..."
    }}
  ]
}}

Rules:
- Use only relation names from the ontology.
- Respect the meaning of domain and range constraints.
- Do not invent facts.
- Extract only facts explicitly stated in the text.
- Split compound locations when possible.
- For numbers, return only the number.
- For dates or years, return only the date/year.
- If there are no triples, return {{"triples": []}}.

Text:
{text}
"""

        content = self.llm_client.generate(prompt)
        return self._safe_parse(content)