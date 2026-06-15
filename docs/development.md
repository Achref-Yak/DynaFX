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

All tests are deterministic and require no external services.

### Test coverage

| Test file | What it tests |
|-----------|---------------|
| `test_math.py` | All formulas in `core/math.py`: SL, Bayes, propagation, convergence, invariants |
| `test_assertion_gate.py` | AssertionGate processing, type check, invariant check, to_node/edge |
| `test_inference_cycle.py` | 9-step loop, convergence, stall detection, state snapshots |
| `test_policy_engine.py` | WhenCondition matching, rule/fallback selection, YAML loading |
| `test_hypothesis_generator.py` | Candidate scoring, dedup, relation inference, to_assertion |
| `test_tbox.py` | TBox types, axioms, valid edges, fallback, validation |
| `test_graph.py` | Graph construction, JSON serialization, entity/edge operations |
| `test_state.py` | State dataclass, trace recording |
| `test_operators.py` | Operator framework, core operators, schema/merge |
| `test_chunker.py` | Text chunking and proposition merging |
| `test_preprocessor.py` | spaCy preprocessing and coreference resolution |
| `test_type_mapper.py` | NodeType assignment rules |
| `test_demarcation_rules.py` | Demarcation dimension computation |
| `test_edge_assigner.py` | Edge lookup table and refinement rules |
| `test_reasoning_modes.py` | Mode filtering |
| `test_product_logic.py` | Category-theoretic checks |
| `test_sl_operators.py` | Subjective Logic operators |
| `test_validators.py` | Validation aggregation |
| `test_cta.py` | Conversation tree construction |
| `test_store.py` | CorpusStore (SQLite-backed graph storage) |
| `test_memory_retrieval.py` | Similarity-based retrieval |
| `test_memory_consolidate.py` | STM → LTM consolidation |
| `test_embeddings.py` | Embedding model initialization |
| `test_cognitive_operators.py` | Cognitive architecture operators |
| `test_fusion.py` | Opinion fusion strategies |

## Project Conventions

- **Python 3.12+** — modern type hints (`list[X]`, `dict[K, V]`, `|` syntax)
- **No `__init__` logic** — constructors don't perform I/O or model loading
- **Lazy imports** — heavy dependencies (torch, transformers) inside functions, not module level
- **Determinism** — pure functions given inputs; no random state or external calls
- **Logging, not printing** — use `logger.info/warning/error` from `logging`
- **Three-zone architecture** — new modules go in the appropriate zone:
  - `nlp/`, `extract/`, `perception/` → Zone 1 (perception)
  - `kernel/`, `operators/`, `memory/` → Zone 2 (kernel)
  - `policy/` → Zone 3 (policy)
  - `tbox/` → Domain layer
  - `core/` → Formula layer

## Adding a New Operator

1. Create a module in `operators/` with a callable class that takes `(state, **kwargs) → state`
2. Register it in the operator dict when constructing `InferenceCycle`
3. Add policy rules in `policy/builtin.py` for automatic selection
4. Write tests in `tests/test_operators.py`

## Adding a New Domain TBox

1. Create a module in `tbox/` defining your `TBox` instance
2. Register it in `tbox/loader.py`'s `BUILTIN_TBOXES`
3. Optionally create domain-specific config in `domains/`
4. Add YAML policy file for operator selection rules

## Adding a New Formula

1. Add the function to `core/math.py` with a pure signature
2. Write tests in `tests/test_math.py`
3. Add the function name to the lazy exports in `__init__.py`

## Code Health

Code Health 10.0 is the standard. Run the pre-commit safeguard before committing:

```bash
codescene pre-commit-safeguard /path/to/repo
```

If a regression is detected, use `code_health_review` and refactor until restored.
