from pathlib import Path


MODEL_NAME = "openai/gpt-4.1-mini"

GROUND_TRUTH_PATH = (
    "data/Text2KGBench/data/dbpedia_webnlg/ground_truth/"
    "ont_1_university_ground_truth.jsonl"
)

ONTOLOGY_PATH = (
    "data/Text2KGBench/data/dbpedia_webnlg/ontologies/"
    "1_university_ontology.json"
)

LIMIT = 70

OUTPUT_DIR = Path("outputs")