# Development

## Setup

```bash
git clone <repo-url>
cd reasoning_engine
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m spacy download en_core_web_trf
```

## Running Tests

```bash
pytest tests/ -v
```

All tests are deterministic and require no external services. The test suite covers:

| Test file | What it tests |
|-----------|---------------|
| `test_chunker.py` | Text chunking and proposition merging |
| `test_preprocessor.py` | spaCy preprocessing and coreference resolution |
| `test_pipeline.py` | Full pipeline integration |
| `test_type_mapper.py` | NodeType assignment rules |
| `test_demarcation_rules.py` | Demarcation dimension computation |
| `test_edge_assigner.py` | Edge lookup table and refinement rules |
| `test_reasoning_modes.py` | Mode filtering |
| `test_product_logic.py` | Category-theoretic checks |
| `test_sl_operators.py` | Subjective Logic operators |
| `test_validators.py` | Validation aggregation |
| `test_cta.py` | Conversation tree construction |

## Model Training

The DistilRoBERTa models need periodic retraining as the dataset grows.

### Requirements

```bash
pip install torch transformers datasets scikit-learn
```

### Training Scripts

**Proposition Tagger** (token classification):

```bash
python scripts/fine_tune_roberta.py
```

Flags:
- `--colab-notebook` — generates a Colab-ready `.ipynb` file instead of training locally
- `--tagger-epochs` — number of epochs for the tagger (default: 20)
- `--classifier-epochs` — number of epochs for the classifier (default: 8)

**Data Preparation:**

```bash
python scripts/prepare_ukp.py
```

Downloads and processes the UKP Sentential Argument Reasoning dataset.

## Project Conventions

- **Python 3.12+** — uses modern type hints (`list[X]`, `dict[K, V]`, `|` syntax)
- **No `__init__` logic** — constructors don't perform I/O or model loading
- **Lazy imports** — heavy dependencies (torch, transformers) are imported inside functions, not at module level
- **Determinism** — all functions are pure given their inputs; no random state or external calls
- **Logging, not printing** — use `logger.info/warning/error` from the `logging` module

## Adding a New Pipeline Stage

1. Create a new module in `src/cognitive_engine/`
2. Define the stage's function signature to accept `Graph` or relevant inputs
3. Add the call in `pipeline.py`'s `run()` function
4. Write tests in `tests/`
5. Run `pytest tests/ -v` to verify

## Adding a New NodeType

1. Add the value to `NodeType` enum in `models.py`
2. Add a detection rule in `type_mapper.py`'s `assign_type()` cascade
3. Add the type to the edge lookup table in `edge_assigner.py`
4. Add default opinion mapping in `config.py`'s `source_type_map`
5. Update tests in `test_type_mapper.py` and `test_edge_assigner.py`
