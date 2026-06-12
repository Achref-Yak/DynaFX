# Cognitive Reasoning Graph Engine

A **provably uncertainty-bounded** pipeline that extracts structured reasoning graphs from unstructured text and computes formal opinions via Subjective Logic. Fully deterministic — zero LLM calls.

```text
Text → [Chunker] → [spaCy Preprocessor] → [Sentence Tagger]
     → [Relation Classifier] → [Type Mapper] → [Demarcation Rules]
     → [Edge Assigner] → [SL Opinion Propagation] → [Validators]
     → [JSON Graph]
```

## Features

- **Deterministic pipeline** — no LLM dependencies. Sentence boundaries from spaCy, relation classification from DistilRoBERTa, all logic in pure Python.
- **Structured reasoning graphs** — 7 node types (AXIOM, CLAIM, EVIDENCE, CONDITION, COUNTERCLAIM, FALLACY, JUSTIFICATION) and 7 edge types (SUPPORTS, ATTACKS, CONTRADICTS, INFERS, QUALIFIES, REBUTS, JUSTIFIES).
- **Subjective Logic opinions** — every node and edge carries `(belief, disbelief, uncertainty, base_rate)` satisfying `b + d + u = 1`. Propagated via fusion and deduction.
- **Formal verification** — category-theoretic monotonicity checks, cycle detection via networkX, and SL invariant enforcement.
- **Demarcation dimensions** — each node annotated with cognitive/epistemic, institutional, affect, constraint, and temporal classifications.
- **4 reasoning modes** — Argument, Causal, Conditional, and Analogy mode filtering.

## Requirements

- Python 3.12+
- CUDA-capable GPU (optional; falls back to CPU)

## Install

```bash
git clone <repo-url>
cd reasoning_engine
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_trf
```

Model weights go in `models/` (see [Development](https://opencode.ai) for training instructions).

## Usage

```bash
# Analyze a document
cognitive-engine demo/complicated.txt

# Custom chunking
cognitive-engine demo/complicated.txt --chunk-size 256 --chunk-overlap 64

# Reasoning mode filter
cognitive-engine demo/complicated.txt --mode causal

# Custom priors
cognitive-engine demo/complicated.txt --config custom_priors.json

# Save output
cognitive-engine demo/complicated.txt --output result.json
```

## Output

A JSON graph with typed nodes, typed edges, and Subjective Logic opinion tuples:

```json
{
  "nodes": { "<uuid>": { "type": "AXIOM", "opinion": [0.7, 0.1, 0.2, 0.5], ... } },
  "edges": [ { "source_id": "...", "target_id": "...", "type": "ATTACKS", ... } ],
  "metadata": { "priors": { ... }, "modes": { ... } },
  "cta": { "root_id": "...", "node_ids": [...], "parent_map": {...} }
}
```

## Project Structure

```
src/cognitive_engine/
  __init__.py           — Public API exports
  orchestrator.py       — Top-level entry point
  pipeline.py           — Core extraction pipeline
  models.py             — Graph, Node, Edge, NodeType, EdgeType, Opinion
  chunker.py            — Text chunking via sliding window
  preprocessor.py       — spaCy preprocessing and coreference resolution
  tagger.py             — SentenceTagger and RelationClassifier
  type_mapper.py        — Rule-based NodeType assignment
  demarcation_rules.py  — Five cognitive-linguistic dimensions
  edge_assigner.py      — Edge type resolution via lookup table
  product_logic.py      — Category-theoretic validity checks
  reasoning_modes.py    — Mode-specific edge filtering
  sl_operators.py       — Subjective Logic calculus
  validators.py         — Category/cycle/opinion validation
  config.py             — Priors configuration
  cli.py                — Command-line entry point
  default_priors.json   — Built-in default priors
tests/                  — 161 tests covering all modules
docs/                   — MKDocs documentation
scripts/                — Model training scripts
demo/                   — Example input files
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Documentation

Full documentation is available via MKDocs:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

## Citation

If you use this in academic work, please cite:

```bibtex
@software{cognitive_engine,
  title = {Cognitive Reasoning Graph Engine},
  year = {2026},
  url = {https://github.com/Achref-Yak/reasoning_engine}
}
```

## License

MIT
