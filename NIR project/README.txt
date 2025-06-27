# HuBERT-based Acoustic Embedding Classifier

A toolkit for extracting and classifying acoustic embeddings from a speech corpus using HuBERT-based representations. Supports both averaged and non-averaged embeddings with detailed evaluation tools.

Project Structure

- `get_embbs.py` — Extracts averaged HuBERT embeddings over the entire corpus.
- `get_embbs_no_averaging.py` — Extracts non-averaged embeddings along with triplet labels (e.g., phoneme positions).
- `classification.py` — Classifies averaged embeddings.
- `classification_no_averaging.py` — Classifies non-averaged embeddings.
- `evaluation_utils.py` — Utilities for evaluating classification performance (accuracy, flexible match, mAP, etc.).
- `test_saved_model.py` — Script to test the trained model on a new audio file.
- `model_vow+sonants_var_borders.pth` — Trained PyTorch model for classifying vowelsonorant segments with variable boundaries.

Requirements

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- torchaudio
- pandas
- tqdm

Install dependencies
```bash
pip install -r requirements.txt
