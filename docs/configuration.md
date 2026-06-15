# Configuration

## Domain Configuration (`domain.py`)

All tunable knobs are in the `DomainConfig` dataclass. A default instance is always active; domain-specific overrides are activated via the `Domain` context manager.

```python
from cognitive_engine.domain import Domain, DomainConfig, domain

cfg = domain.active()  # default config

with Domain("legal", DomainConfig(conflict_threshold=0.25)):
    cfg = domain.active()  # legal overrides
```

### Key thresholds

| Field | Default | Description |
|-------|---------|-------------|
| `opinion_positive_threshold` | 0.0 | Offset for `b > d` classification |
| `conflict_threshold` | 0.3 | Min belief+disbelief for conflicting opinions |
| `analogy_uncertainty_delta` | 0.2 | Belief → uncertainty shift in ANALOGY mode |
| `uncertainty_pseudocount` | 2.0 | Dirichlet pseudocount `W` |
| `clamp_epsilon` | 1e-9 | Tolerance for `b+d+u == 1` invariant |

### Category hierarchy

```python
category_levels = {
    1: "Necessity",
    2: "Fact",
    3: "Belief",
    4: "Concept",
}
```

### Default opinion templates

| Template | (b, d, u, a) |
|----------|--------------|
| `empirical_pattern` | (0.8, 0.1, 0.1, 0.5) |
| `cognitive_hypothesis` | (0.5, 0.2, 0.3, 0.5) |
| `consensus_principle` | (0.7, 0.1, 0.2, 0.5) |
| `observational_claim` | (0.4, 0.3, 0.3, 0.5) |
| `total_ignorance` | (0.0, 0.0, 1.0, 0.5) |

### Edge warrants

| Edge Type | P(source → target) | P(source → ¬target) |
|-----------|-------------------|---------------------|
| SUPPORTS | (0.85, 0.1, 0.05, 0.5) | (0.15, 0.8, 0.05, 0.5) |
| CONTRADICTS | (0.1, 0.85, 0.05, 0.5) | (0.8, 0.15, 0.05, 0.5) |
| INFERS | (0.9, 0.05, 0.05, 0.5) | (0.0, 1.0, 0.0, 0.5) |
| ATTACKS | (0.05, 0.85, 0.1, 0.5) | (0.85, 0.1, 0.05, 0.5) |

## Legal Domain Configuration (`domains/legal.py`)

`LegalCoefficients` extends the general config with legal-specific weights:

```python
from cognitive_engine.domains.legal import LegalCoefficients

coeffs = LegalCoefficients(alpha=0.25, beta=0.35)
```

| Coefficient | Default | Purpose |
|-------------|---------|---------|
| `alpha` | 0.25 | Probability evidence weight |
| `beta` | 0.35 | Graph propagation weight |
| `gamma` | 0.25 | Logic consistency weight |
| `delta` | 0.15 | Adversarial contradiction weight |

## YAML Policy Configuration

Operator selection policies are YAML files loaded by `PolicyEngine`:

```yaml
name: legal
rules:
  - when:
      cycle: "==1"
    then:
      operators: [extract, schema, graph]
      order: sequential
  - when:
      graph_has_contradictions: true
    then:
      operators: [constraint, debate, verify]
      order: sequential
fallback:
  operators: [propagate, verify]
  order: sequential
```

```python
from cognitive_engine.policy.engine import PolicyEngine
engine = PolicyEngine()
engine.load_yaml(open("policy.yaml").read())
```

## TBox Configuration

Domain type hierarchies are loaded from `tbox/loader.py`:

```python
from cognitive_engine.tbox.loader import load_tbox, TBox

tbox = load_tbox("general")  # built-in general TBox
tbox = load_tbox("legal")    # built-in legal TBox
tbox = load_tbox("path/to/tbox.json")  # custom JSON TBox
```

## Sensitivity Sweep

```python
from cognitive_engine.core.config import Priors, sweep_priors

base = Priors()
variants = sweep_priors(base, delta=0.1)  # 40+ ±delta variants
```
