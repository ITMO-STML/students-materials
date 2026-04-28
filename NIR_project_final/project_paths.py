from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
EXAMPLES_DIR = DATA_DIR / "examples"
OUTPUTS_DIR = DATA_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
LOGS_DIR = OUTPUTS_DIR / "logs"
BASELINE_LOGS_DIR = LOGS_DIR / "baseline_runs"
CTC_LOGS_DIR = LOGS_DIR / "ctc_training"

MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODELS_DIR = MODELS_DIR / "baseline"
CTC_MODELS_DIR = MODELS_DIR / "ctc"
PHONETIC_MODELS_DIR = MODELS_DIR / "phonetic_features"

BASELINE_TRAINING_DIR = PROJECT_ROOT / "Baseline_Training"
CTC_NOTEBOOKS_DIR = PROJECT_ROOT / "CTC_models"
PHONETIC_TRAINING_DIR = PROJECT_ROOT / "PhoneticFeatures_Training"

DEFAULT_BASELINE_MODEL = BASELINE_MODELS_DIR / "VACXUXVEXO_model.pth"
DEFAULT_STRESS_MODEL = BASELINE_MODELS_DIR / "RGCAIQYZHB_model.pth"
DEFAULT_CTC_MODEL = CTC_MODELS_DIR / "TGDSLDZLTS" / "best_model.pth"
DEFAULT_CTC_ATTENTION_MODEL = CTC_MODELS_DIR / "WWAIMQTNYK" / "best_attention_model.pth"
DEFAULT_PHONETIC_MODEL_2CLASS = PHONETIC_MODELS_DIR / "ResistantBias_2class"
DEFAULT_PHONETIC_MODEL_3CLASS = PHONETIC_MODELS_DIR / "ResistantBias_3class"

EXAMPLE_WAV = EXAMPLES_DIR / "ata1101.wav"
EXAMPLE_SEG_B2 = EXAMPLES_DIR / "ata1101.seg_B2"
EXAMPLE_SEG_B4 = EXAMPLES_DIR / "ata1101.seg_B4"
EXAMPLE_PHONEMES = EXAMPLES_DIR / "ata1101_no_av_ph.txt"
EXAMPLE_EMBEDDINGS = EXAMPLES_DIR / "ata1101_no_av_embs.npy"
PHONEME_CHOICES_SUMMARY = TABLES_DIR / "phoneme_choices_summary.csv"


def ensure_project_dirs() -> None:
    for directory in (
        DATA_DIR,
        EXAMPLES_DIR,
        OUTPUTS_DIR,
        TABLES_DIR,
        LOGS_DIR,
        BASELINE_LOGS_DIR,
        CTC_LOGS_DIR,
        MODELS_DIR,
        BASELINE_MODELS_DIR,
        CTC_MODELS_DIR,
        PHONETIC_MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
