# Cognitive Reasoning Engine

A **deterministic reasoning engine** that transforms unstructured text into a structured reasoning graph and iterates it to convergence via formal semantics — no LLM in the loop.

```
┌─────────────────────────────────────────────────────────────┐
│  Perception (Zone 1)     Kernel (Zone 2)     Policy (Z3)   │
│                           ┌──────────────────┐              │
│  chunker ──→ tagger ──→  │  InferenceCycle  │ ←── policy   │
│  classifier ──→ mapper   │  9-step loop     │    engine    │
│  extract_entities ──→    │  AssertionGate   │              │
│  HypothesisGenerator     │  TBox / ABox     │              │
│                           └──────────────────┘              │
│                                │                            │
│                          core/math.py (formulas)            │
└─────────────────────────────────────────────────────────────┘
```

## Quick Example

```python
from cognitive_engine import InferenceCycle, InferenceCycleConfig
from cognitive_engine.core.state import State
from cognitive_engine.core.models import Graph, Node, NodeType

g = Graph(nodes={
    nid: Node(type=NodeType.CLAIM, text="Example"),
})
state = State(graph=g)
config = InferenceCycleConfig(max_cycles=5)

cycle = InferenceCycle(operators={}, config=config)
result = cycle.run(state)

print(f"Converged: {result.converged}, cycles: {result.total_cycles}")
```

## Key Features

- **Zero LLM calls** — all reasoning is rule-based symbolic logic and formal semantics
- **3-zone architecture** — perception (extraction), kernel (inference), policy (operator selection)
- **Hard assertion gate** — every neural output is converted to an SL opinion before entering the kernel
- **Declarative YAML policies** — operator selection rules readable by domain experts
- **TBox/ABox** — OWL2-style domain type hierarchies with SWRL-like axioms
- **426+ tests** — fully deterministic test suite

## Where to Start

| Page | What it covers |
|------|---------------|
| [Getting Started](getting-started.md) | Install, first run, basic Python API |
| [Architecture](architecture.md) | 3-zone design, module map, data flow |
| [Pipeline](pipeline.md) | InferenceCycle 9-step loop + extraction sub-pipeline |
| [Models](models.md) | Data types: State, Assertion, TBox, Opinion, policies |
| [Theory](theory.md) | Formal semantics: SL, category theory, Dung, convergence |
| [Configuration](configuration.md) | DomainConfig, LegalCoefficients, YAML policies, TBox |
| [Development](development.md) | Running tests, adding modules, conventions |
