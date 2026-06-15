# Getting Started

## Requirements

- Python 3.12+
- A CUDA-capable GPU (optional, for RoBERTa models; falls back to CPU)
- spaCy `en_core_web_trf` model for text preprocessing

## Installation

```bash
git clone <repo-url>
cd reasoning_engine
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_trf
```

## First Run — Python API

The engine has no CLI. Everything runs through the Python API.

### Minimal example

```python
from cognitive_engine import InferenceCycle, InferenceCycleConfig
from cognitive_engine.core.state import State
from cognitive_engine.core.models import Graph, Node, NodeType
from uuid import uuid4

# Create a simple graph with one node
nid = uuid4()
graph = Graph(nodes={
    nid: Node(id=nid, type=NodeType.CLAIM, text="The sky appears blue during clear weather."),
})
state = State(graph=graph)

# Create the InferenceCycle
config = InferenceCycleConfig(max_cycles=10, epsilon=1e-4)
cycle = InferenceCycle(operators={}, config=config)

# Run to convergence
result = cycle.run(state)

print(f"Converged: {result.converged}")
print(f"Cycles: {result.total_cycles}")
print(f"Final norm: {result.final_norm:.6f}")
```

### Extract from text

```python
from cognitive_engine.operators.extract import ExtractOperator
from cognitive_engine.core.state import State
from cognitive_engine.core.models import Graph

state = State(graph=Graph())
extract = ExtractOperator(compute_embeddings=False)
state = extract(state, text="The defendant was seen near the scene. This contradicts her alibi.")

print(f"Nodes: {len(state.graph.nodes)}, Edges: {len(state.graph.edges)}")
```

### Full InferenceCycle with extraction

```python
from cognitive_engine import InferenceCycle, InferenceCycleConfig
from cognitive_engine.core.state import State
from cognitive_engine.core.models import Graph
from cognitive_engine.operators.extract import ExtractOperator
from cognitive_engine.operators.propagate import PropagateOperator
from cognitive_engine.operators.constraint import ConstraintOperator

state = State(graph=Graph())
config = InferenceCycleConfig(max_cycles=5)

cycle = InferenceCycle(operators={
    "extract": ExtractOperator(compute_embeddings=False),
    "propagate": PropagateOperator(),
    "constraint": ConstraintOperator(),
}, config=config)

result = cycle.run(state, text="We should migrate to CockroachDB. Our current system handles 5000 writes/sec.")
print(f"Converged: {result.converged}, cycles: {result.total_cycles}")
```

## Understanding the Output

The `InferenceResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `state` | `State` | Final reasoning state with graph and metadata |
| `cycles` | `list[CycleReport]` | Per-cycle reports with norm, operators, duration |
| `converged` | `bool` | Whether the loop converged (‖Δs‖ < ε) |
| `total_cycles` | `int` | Number of cycles executed |
| `final_norm` | `float` | Final state delta norm |

Each `CycleReport` records which operators ran, the convergence norm, policy selection provenance, and duration.

## Next Steps

- [Architecture](architecture.md) — 3-zone design explained
- [Pipeline](pipeline.md) — InferenceCycle 9-step loop in detail
- [Models](models.md) — all data types and their fields
- [Tutorial](tutorial.md) — end-to-end walkthrough with a real document
