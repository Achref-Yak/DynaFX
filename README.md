# Cognitive Reasoning Graph Engine

A provably uncertainty-bounded pipeline that extracts structured reasoning graphs from unstructured text and computes formal opinions via Subjective Logic.

```text
Text → [Creator Agent (LLM)] → [Deterministic Validators] → [Reviewer Agent (LLM)]
       → [Subjective Logic propagation] → [JSON output]
       ↻ up to 5 rounds upon rejection
```

## Features

- **LLM extraction** — Groq-powered (`llama-3.3-70b-versatile`) extraction of typed nodes (CLAIM, EVIDENCE, CONDITION) and edges (SUPPORTS, CONTRADICTS, QUALIFIES, INFERS, JUSTIFIES)
- **Deterministic validation** — pure-Python validators (category hierarchy, cycle detection, opinion invariant) — zero LLM calls
- **Subjective Logic math** — opinion propagation via conditional deduction and cumulative fusion, with configurable priors from Jøsang (2016)
- **Fail-closed loop** — up to 5 correction rounds; `--sweep` flag for sensitivity analysis
- **No lock-in** — swap the LLM provider by changing one import

## Requirements

- Python 3.12+
- `GROQ_API_KEY` environment variable (get one at https://console.groq.com)

## Install

```bash
git clone <repo-url>
cd reasoning_engine
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # then edit .env with your GROQ_API_KEY
```

## Usage

```bash
# Analyze a document
cognitive-engine path/to/text.txt

# With custom priors
cognitive-engine path/to/text.txt --config custom_priors.json

# Sensitivity sweep (±0.1 on each prior)
cognitive-engine path/to/text.txt --sweep

# Save output to file
cognitive-engine path/to/text.txt --output result.json
```

## Output

A JSON graph with typed nodes, typed edges, and Subjective Logic opinion tuples `(belief, disbelief, uncertainty, base_rate)` that always satisfy `b + d + u = 1`.

```json
{
  "nodes": { "<uuid>": { "type": "CLAIM", "opinion": [0.41, 0.31, 0.28, 0.5], ... } },
  "edges": [ { "source_id": "...", "target_id": "...", "type": "SUPPORTS", ... } ],
  "metadata": { "priors": { ... } }
}
```

## Project Structure

```
src/cognitive_engine/
  __init__.py       — loads GROQ_API_KEY from .env
  models.py         — Graph, Node, Edge, Opinion, Violation, Span
  validators.py     — V1, V2, V3 validators (pure Python)
  sl_operators.py   — Subjective Logic math (conditional_deduction, cumulative_fusion, etc.)
  config.py         — Priors dataclass, load/sweep utilities
  extraction.py     — CreatorAgent (LLM → structured graph)
  reviewers.py      — ReviewerAgent (LLM review with validator context)
  orchestrator.py   — CAF-Gen orchestrator loop (5 rounds)
  cli.py            — Command-line entry point
  default_priors.json — Editable reference copy of all prior constants
tests/
  test_validators.py
  test_sl_operators.py
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Citation

If you use this in academic work, please cite:

```bibtex
@software{cognitive_engine,
  title = {Cognitive Reasoning Graph Engine},
  year = {2026},
  url = {https://github.com/anomalyco/reasoning_engine}
}
```

## License

MIT
