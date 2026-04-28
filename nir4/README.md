# Structured Ontological Legal Analysis Reasoner (SOLAR)

This repository contains the evaluation code for the paper "On Verifiable Legal Reasoning: A Multi-Agent Framework with Formalized Knowledge Representations" accepted at CIKM 2025.

SOLAR is a two-stage framework that decomposes legal reasoning into knowledge acquisition (Stage I) and knowledge application (Stage II), demonstrating improved performance on statutory reasoning tasks.

## Setup

Create `.env` file with your API keys:

```
OPENAI_API_KEY=
FIREWORKS_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
```

## Usage

### Stage I: Knowledge Acquisition

Generate TBox ontology and interpreter from legal statute:

```bash
uv run stage_one.py \
    --statute ./eval/sara_statute.txt \
    --output-tbox ./tbox.json \
    --output-py ./interpreter.py \
    --model gpt-4.1-mini
```

Arguments:

- `--model`: Model to use (gpt-4.1-mini, o1 etc.)
- `--statute`: Path to input statute text file
- `--output-tbox`: Output path for generated TBox JSON
- `--output-py`: Output path for TBox interpreter Python code
- `--debug`: Enable debug output
- `--no-cache`: Disable LLM caching

## Stage II: Knowledge Application

Evaluate different reasoning approaches on the SARA numeric dataset:

```bash
uv run stage_two.py \
    --mode solar \
    --tbox-path ./tbox.json \
    --tbox-interpreter-path ./interpreter.py \
    --dataset test \
    --model gpt-4.1-mini
```

Arguments:
- `--model`: Model to evaluate
- `--mode`: Reasoning approach (`zero-shot`, `chain-of-code`, `solar`)
- `--dataset`: Dataset split (`train`, `test`)
- `--success-threshold`: Success tolerance percentage (default: 10.0%)
- `--limit`: Limit number of test cases (-1 for all)
- `--debug`: Enable debug output
- `--tbox-path`: Path to TBox JSON (for solar mode)
- `--tbox-interpreter-path`: Path to TBox interpreter (for solar mode)

## License

This project is licensed under the [MIT License](LICENSE).
