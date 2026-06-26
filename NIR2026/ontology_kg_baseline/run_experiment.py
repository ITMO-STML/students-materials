from src.config import (
    MODEL_NAME,
    GROUND_TRUTH_PATH,
    ONTOLOGY_PATH,
    LIMIT,
    OUTPUT_DIR
)

from src.data_loader import DatasetLoader
from src.ontology_loader import OntologyLoader
from src.llm_client import LLMClient
from src.extractors import (
    NoOntologyExtractor,
    RelationNamesExtractor,
    DomainRangeExtractor
)
from src.validator import OntologyValidator
from src.result_writer import ResultWriter
from src.experiment import Experiment


def main():
    examples = DatasetLoader(
        path=GROUND_TRUTH_PATH,
        limit=LIMIT
    ).load()

    ontology_relations = OntologyLoader(
        path=ONTOLOGY_PATH
    ).load()

    llm_client = LLMClient(
        model_name=MODEL_NAME
    )

    extractors = [
        NoOntologyExtractor(llm_client),
        RelationNamesExtractor(llm_client, ontology_relations),
        DomainRangeExtractor(llm_client, ontology_relations)
    ]

    validator = OntologyValidator(
        ontology_relations=ontology_relations
    )

    writer = ResultWriter(
        output_dir=OUTPUT_DIR
    )

    experiment = Experiment(
        examples=examples,
        ontology_relations=ontology_relations,
        extractors=extractors,
        validator=validator,
        writer=writer,
        dataset_name="Text2KGBench / DBpedia-WebNLG / ont_1_university",
        model_name=MODEL_NAME
    )

    experiment.run()


if __name__ == "__main__":
    main()