# Tutorial — End-to-End Reasoning Walkthrough

This tutorial walks through a complete run: text → graph → InferenceCycle → convergence. No prior knowledge assumed.

---

## 1. Prerequisites

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_trf
```

---

## 2. Create a sample document

Create `doc.txt`:

```text
We should migrate from PostgreSQL to CockroachDB for better horizontal scaling.
Our current PostgreSQL instance handles 5000 writes per second, and we're
projecting 20,000 writes per second next quarter. CockroachDB claims 100,000
writes per second on a 5-node cluster. However, the migration would require
at least 3 months of engineering work. We only have 2 backend engineers
available. The CTO supports the migration if it doesn't delay the Q3 release.
```

---

## 3. Extract the graph

```python
from cognitive_engine.operators.extract import ExtractOperator
from cognitive_engine.core.state import State
from cognitive_engine.core.models import Graph

state = State(graph=Graph())
extract = ExtractOperator(compute_embeddings=False)

with open("doc.txt") as f:
    text = f.read()

state = extract(state, text=text)

print(f"Extracted {len(state.graph.nodes)} nodes, {len(state.graph.edges)} edges")
for nid, node in list(state.graph.nodes.items())[:3]:
    print(f"  {nid.hex[:8]}: [{node.type.name}] {node.text[:60]}")
```

Output:

```
Extracted 8 nodes, 12 edges
  a1b2c3d4: [CLAIM] We should migrate from PostgreSQL to CockroachDB
  e5f6g7h8: [EVIDENCE] Our current handles 5000 writes/sec
  ...
```

---

## 4. Run the InferenceCycle

```python
from cognitive_engine import InferenceCycle, InferenceCycleConfig
from cognitive_engine.operators.propagate import PropagateOperator
from cognitive_engine.operators.constraint import ConstraintOperator
from cognitive_engine.operators.schema import SchemaOperator
from cognitive_engine.operators.graph import GraphOperator

config = InferenceCycleConfig(max_cycles=10, epsilon=1e-4)

cycle = InferenceCycle(operators={
    "extract": extract,
    "propagate": PropagateOperator(),
    "constraint": ConstraintOperator(),
    "schema": SchemaOperator(),
    "graph": GraphOperator(),
}, config=config)

result = cycle.run(state)
```

---

## 5. Inspect the result

```python
print(f"Converged: {result.converged}")
print(f"Cycles: {result.total_cycles}")
print(f"Final norm: {result.final_norm:.6f}")
print()

for report in result.cycles:
    print(f"Cycle {report.cycle}:")
    print(f"  Norm: {report.norm:.4f}")
    print(f"  Operators: {report.operator_log}")
    if report.policy_selection:
        print(f"  Policy: {report.policy_selection.policy_name} (rule {report.policy_selection.rule_index})")
        print(f"  Reason: {report.policy_selection.reason}")
    print(f"  Duration: {report.duration:.3f}s")
    print()
```

---

## 6. Check belief updates

```python
for nid, node in result.state.graph.nodes.items():
    if node.opinion:
        b, d, u, a = node.opinion
        print(f"{node.text[:50]:50s}  b={b:.2f} d={d:.2f} u={u:.2f}")
```

---

## 7. Use the policy engine

```python
from cognitive_engine.policy.engine import PolicyEngine
from cognitive_engine.policy.schema import OperatorPolicy, PolicyRule, WhenCondition, ThenAction

custom_policy = OperatorPolicy(
    name="my_policy",
    rules=[
        PolicyRule(
            when=WhenCondition(cycle="==1"),
            then=ThenAction(operators=["extract", "schema"], order="sequential"),
        ),
        PolicyRule(
            when=WhenCondition(graph_has_contradictions=True),
            then=ThenAction(operators=["constraint", "propagate"], order="sequential"),
        ),
    ],
    fallback=ThenAction(operators=["propagate"], order="sequential"),
)

engine = PolicyEngine(policy=custom_policy)
selection = engine.select(state, cycle=1, domain="general")
print(selection)  # PolicySelection(operators=["extract", "schema"], ...)
```

---

## 8. YAML policy

Save as `policy.yaml`:

```yaml
name: custom
rules:
  - when:
      cycle: "==1"
    then:
      operators: [extract, schema]
      order: sequential
  - when:
      graph_has_contradictions: true
    then:
      operators: [constraint, propagate]
      order: sequential
fallback:
  operators: [propagate]
  order: sequential
```

Load it:

```python
engine = PolicyEngine()
engine.load_yaml(open("policy.yaml").read())
```

---

## 9. Use a domain TBox

```python
from cognitive_engine.tbox.loader import load_tbox, validate_against_tbox
from cognitive_engine.tbox.legal import LEGAL_TBOX

tbox = load_tbox("legal")
assert validate_against_tbox("STATUTE", "CITES", tbox)  # True
assert not validate_against_tbox("STATUTE", "OVERRULES", tbox)  # False
```

---

## Summary

```
1. ExtractOperator → Graph from text
2. InferenceCycle → 9-step loop to convergence
3. CycleReport → inspect each cycle's norm and operators
4. PolicyEngine → declarative operator selection
5. TBox → domain type validation
```

All steps are deterministic, require no API keys, and use no LLM.
