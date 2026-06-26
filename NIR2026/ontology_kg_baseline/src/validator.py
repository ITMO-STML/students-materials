class OntologyValidator:
    def __init__(self, ontology_relations: dict):
        self.ontology_relations = ontology_relations
        self.allowed_relations = set(ontology_relations.keys())

    def validate(self, triples: list[dict]) -> tuple[list[dict], list[dict]]:
        valid = []
        rejected = []

        for triple in triples:
            rel = triple.get("rel")

            if rel in self.allowed_relations:
                valid.append(triple)
            else:
                rejected.append({
                    "triple": triple,
                    "reason": "relation_not_in_ontology"
                })

        return valid, rejected

    def ontology_validity(self, triples: list[dict]) -> float:
        if not triples:
            return 1.0

        valid_count = 0

        for triple in triples:
            if triple.get("rel") in self.allowed_relations:
                valid_count += 1

        return valid_count / len(triples)