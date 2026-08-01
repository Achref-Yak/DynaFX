# Tutorial 8 — Scenarios & Sensitivity

Once you have a model, the scientific questions begin: *what if?* and *what
matters most?* DynaFX answers both — `ScenarioComparison` for discrete
alternatives and `SensitivityAnalyzer` for parameter influence.

## Scenario comparison

A scenario is just a name plus a set of parameter overrides. Run many, then
compare:

```python
from dynafx.dynamics import parse_sysd
from dynafx.dynamics.scenario import ScenarioComparison, ScenarioDef

model = parse_sysd("""
Population
dt 0.1
from 0 to 10

stock "Population": 100
  + "births": pop_growth * Population
  - "deaths": Population * mortality

aux "pop_growth": 0.02
aux "mortality": 0.01
""")

comparison = ScenarioComparison(model, [
    ScenarioDef("Baseline", {}),
    ScenarioDef("High growth", {"pop_growth": 0.05}),
    ScenarioDef("High mortality", {"mortality": 0.03}),
])

for scenario in comparison.scenarios:
    final = scenario.result.values["Population"][-1]
    print(f"{scenario.name:<18} {final:.1f}")
```

`comparison.summary()` returns a `{scenario: {stock: final}}` table, and
`comparison.deviation_table()` gives absolute or relative deltas vs a baseline:

```python
summary = comparison.summary()
print(summary["High growth"]["Population"] > summary["Baseline"]["Population"])   # True

deviations = comparison.deviation_table(baseline=0, mode="relative")
print(deviations["High growth"]["Population"])   # > 0
```

## Grading and ranking

With a knowledge base attached, you can score each scenario by turning its
trajectory into an evidence triple, then rank:

```python
from dynafx.bridge import KBSimBridge
from dynafx.knowledge import TripleStore, NamedNode

fresh = TripleStore()
bridge = KBSimBridge(fresh)

def score(initial, final):
    return min(1.0, max(0.0, final[-1] / 200.0))

evidence_map = [("Population",
                 NamedNode("http://ex.org/Portfolio"),
                 NamedNode("http://ex.org/popScore"), score)]

grade_specs = [(
    "SELECT ?v WHERE { <http://ex.org/Portfolio> <http://ex.org/popScore> ?v }",
    "v", 0.8, 0.2,
)]

ranked = comparison.rank(grade_specs, fresh, evidence_map=evidence_map, bridge=bridge)
print([name for name, _ in ranked])   # scenarios sorted by grade
```

## Sensitivity analysis

`SensitivityAnalyzer` quantifies how much each parameter drives an output.
Each method takes `{param: (low, high)}` ranges and an `output` variable.

### Sobol (variance decomposition)

```python
from dynafx.dynamics.sensitivity import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(model)
result = analyzer.sobol(
    {"pop_growth": (0.0, 0.1), "mortality": (0.0, 0.05)},
    output="Population",
    n_base=256,
    seed=42,
)
print(result.first_order)      # {param: main-effect index}
print(result.total_order)      # {param: total-effect index}
```

### Morris (screening)

```python
result = analyzer.morris(
    {"pop_growth": (0.0, 0.1), "mortality": (0.0, 0.05)},
    output="Population",
    n_trajectories=15,
    seed=42,
)
print(result.mu_star)     # mean absolute elementary effect
```

### One-at-a-time

```python
result = analyzer.oat(
    {"pop_growth": (0.0, 0.1), "mortality": (0.0, 0.05)},
    output="Population",
)
print(result.oat_low["pop_growth"])
print(result.oat_high["pop_growth"])
```

### PRCC (partial rank correlation)

```python
result = analyzer.prcc(
    {"pop_growth": (0.0, 0.1), "mortality": (0.0, 0.05)},
    output="Population",
    n_samples=200,
    seed=42,
)
print(result.prcc)          # {param: prcc coefficient}
print(result.prcc_pvalue)   # {param: p-value}
```

All methods accept a `seed` for reproducibility — same seed, same numbers.

## What's next

- Turn your domain into inference rules in [Custom Ontology](09-custom-ontology.md).
- Run causal, optimization, and provenance for a publishable result in [Publishing Results](10-publishing-results.md).
