import json


class OntologyLoader:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as f:
            ontology = json.load(f)

        return self._simplify(ontology)

    def _simplify(self, ontology: dict) -> dict:
        relations = {}

        for rel in ontology["relations"]:
            relations[rel["pid"]] = {
                "domain": rel.get("domain", ""),
                "range": rel.get("range", "")
            }

        return relations