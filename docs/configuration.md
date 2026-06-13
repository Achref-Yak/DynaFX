# Configuration

The engine uses a `Priors` dataclass that controls the default opinions, source-type-to-opinion mappings, edge warrants, and default warrant used during Subjective Logic propagation.

## Default Priors

Built-in defaults are defined in `src/cognitive_engine/config.py` and serialized to `src/cognitive_engine/default_priors.json`.

### Default Opinions per Source Type

| Type | Belief | Disbelief | Uncertainty | Base Rate |
|------|--------|-----------|-------------|-----------|
| Evidence (empirical pattern) | 0.8 | 0.1 | 0.1 | 0.5 |
| Condition (cognitive hypothesis) | 0.5 | 0.2 | 0.3 | 0.5 |
| Claim/Axiom (consensus principle) | 0.7 | 0.1 | 0.2 | 0.5 |
| Counterclaim/Fallacy (observational claim) | 0.4 | 0.3 | 0.3 | 0.5 |

### Edge Warrants

Each edge type has a pair of conditional opinions `(P(source → target), P(source → ¬target))`:

| Edge Type | P(→) | P(→¬) |
|-----------|------|-------|
| SUPPORTS | (0.85, 0.1, 0.05, 0.5) | (0.15, 0.8, 0.05, 0.5) |
| CONTRADICTS | (0.1, 0.85, 0.05, 0.5) | (0.8, 0.15, 0.05, 0.5) |
| QUALIFIES | (0.6, 0.2, 0.2, 0.5) | (0.4, 0.4, 0.2, 0.5) |
| INFERS | (0.9, 0.05, 0.05, 0.5) | (0.0, 1.0, 0.0, 0.5) |
| JUSTIFIES | (0.8, 0.1, 0.1, 0.5) | (0.2, 0.7, 0.1, 0.5) |
| ATTACKS | (0.05, 0.85, 0.1, 0.5) | (0.85, 0.1, 0.05, 0.5) |
| REBUTS | (0.6, 0.3, 0.1, 0.5) | (0.3, 0.6, 0.1, 0.5) |

### Default Warrant

Used when no specific warrant is defined: `((0.9, 0.05, 0.05, 0.5), (0.0, 1.0, 0.0, 0.5))`

## Custom Priors

Override any or all priors with a JSON file passed via `--config`:

```json
{
  "default_opinions": {
    "consensus_principle": [0.8, 0.1, 0.1, 0.5]
  },
  "edge_warrants": {
    "SUPPORTS": [[0.9, 0.05, 0.05, 0.5], [0.1, 0.85, 0.05, 0.5]]
  }
}
```

Missing keys fall back to the built-in defaults.

## Sensitivity Sweep

The `sweep_priors(base, delta)` function creates ±delta variants of each prior for sensitivity analysis:

```python
from cognitive_engine.core.config import Priors, sweep_priors

base = Priors()
variants = sweep_priors(base, delta=0.1)  # 40+ variants
```
