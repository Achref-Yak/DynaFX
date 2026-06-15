# Data Models

All core models are in `core/models.py`. New architecture models are in their respective modules.

## NodeType

```python
class NodeType(Enum):
    AXIOM = auto()          # Necessity/obligation (has modal verb)
    EVIDENCE = auto()       # Empirical observation, supporting data
    CONDITION = auto()      # Conditional (if/unless/provided)
    CLAIM = auto()          # Root assertion, no specific marker
    COUNTERCLAIM = auto()   # Opposing claim (attacked or adversative)
    FALLACY = auto()        # Fallacious reasoning (keyword match)
    JUSTIFICATION = auto()  # Supporting reason (because/since/for)
    HYPOTHESIS = auto()     # Generated hypothesis (from hypothesis generator)
```

## EdgeType

```python
class EdgeType(Enum):
    INFERS = auto()         # Logical inference
    SUPPORTS = auto()       # Evidential support
    ATTACKS = auto()        # Direct attack
    REBUTS = auto()         # Rebuttal
    QUALIFIES = auto()      # Conditional qualification
    JUSTIFIES = auto()      # Justification
    CONTRADICTS = auto()    # Contradiction
```

## Opinion

```python
@dataclass
class Opinion:
    belief: float = 0.0       # Confidence proposition is true
    disbelief: float = 0.0     # Confidence proposition is false
    uncertainty: float = 1.0   # Vacuousness / lack of evidence
    prior: float = 0.5         # Base rate / prior probability
```

Invariant: `belief + disbelief + uncertainty = 1` (enforced by `check_opinion_invariant` + `normalize_sum`).
Projected probability: `P = belief + prior × uncertainty`.

Tuple-compatible: `to_tuple()`, `from_tuple()`, `__getitem__` for access by index.

## Node

```python
@dataclass
class Node:
    id: UUID
    type: NodeType
    text: str
    span: Optional[Span]
    opinion: Optional[Opinion]
    category: int = 2          # 1=Necessity…4=Concept
    metadata: dict = field(default_factory=dict)
```

## Edge

```python
@dataclass
class Edge:
    id: UUID
    source_id: UUID
    target_id: UUID
    type: EdgeType
    opinion: Optional[Opinion] = None
    warrant: Optional[Warrant] = None
    metadata: dict = field(default_factory=dict)
```

`Warrant = tuple[Opinion, Opinion]` — conditional opinions `(P(source→target), P(source→¬target))`.

## Graph

```python
@dataclass
class Graph:
    nodes: dict[UUID, Node]
    edges: dict[UUID, Edge]
    entities: dict[UUID, Entity]
    mode: ReasoningMode
    source_text: str
    metadata: dict
    interpretations: dict[str, Interpretation]
    cta: Optional[ConversationTree]
```

## State (`core/state.py`)

The `State` dataclass flows through the InferenceCycle:

```python
@dataclass
class State:
    graph: Graph
    metadata: dict = field(default_factory=dict)
    trace: TraceBuffer = field(default_factory=TraceBuffer)
    abox: list = field(default_factory=list)     # Assertion box
    tbox: Any = None                              # Terminological box
```

- `metadata`: cycle count, beliefs, norms, operator records
- `trace`: `StateDelta` history for each cycle
- `abox`: ABox assertions (accepted by the gate)
- `tbox`: Active domain TBox reference

## Assertion (`kernel/assertion_gate.py`)

```python
@dataclass
class Assertion:
    id: UUID
    source: str = ""          # Perception component name
    node_id: Optional[UUID] = None
    node_type: Optional[str] = None
    text: str = ""
    opinion: Optional[tuple[float,float,float,float]] = None
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
```

## GateResult (`kernel/assertion_gate.py`)

```python
@dataclass
class GateResult:
    passed: list[Assertion] = field(default_factory=list)
    quarantined: list[Assertion] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
```

## TBox (`tbox/loader.py`)

```python
@dataclass
class TBox:
    name: str = "general"
    node_types: dict[str, int] = field(default_factory=dict)    # type → category
    edge_types: dict[str, float] = field(default_factory=dict)  # type → weight
    axioms: list[dict] = field(default_factory=list)             # SWRL-like rules
    valid_edges: list[tuple[str, str, str]] = field(default_factory=list)  # (src, edge, tgt)
```

## Policy Models (`policy/schema.py`)

```python
@dataclass
class WhenCondition:
    cycle: Optional[str] = None
    domain: Optional[str] = None
    graph_node_count: Optional[str] = None
    graph_has_contradictions: Optional[bool] = None
    graph_mean_uncertainty: Optional[str] = None
    convergence_stalled: Optional[bool] = None
    last_operator: Optional[str] = None

@dataclass
class ThenAction:
    operators: list[str] = field(default_factory=list)
    order: str = "sequential"

@dataclass
class PolicyRule:
    when: WhenCondition = field(default_factory=WhenCondition)
    then: ThenAction = field(default_factory=ThenAction)

@dataclass
class OperatorPolicy:
    name: str = "default"
    description: str = ""
    rules: list[PolicyRule] = field(default_factory=list)
    fallback: ThenAction = field(default_factory=lambda: ThenAction(["propagate"]))
```

## PolicySelection (`policy/engine.py`)

```python
@dataclass
class PolicySelection:
    operators: list[str] = field(default_factory=list)
    order: str = "sequential"
    policy_name: str = "default"
    rule_index: int = -1
    reason: str = ""
    state_metrics: dict = field(default_factory=dict)
```

## Cycle Models (`kernel/inference_cycle.py`)

```python
@dataclass
class InferenceCycleConfig:
    epsilon: float = 1e-4
    max_cycles: int = 20
    stm_capacity: int = 128
    policy_name: str = "default"
    domain: str = "general"
    convergence_window: int = 3

@dataclass
class CycleReport:
    cycle: int
    norm: float
    converged: bool
    operator_log: list[str] = field(default_factory=list)
    policy_selection: Optional[PolicySelection] = None
    state_snapshot: Optional[dict] = None
    duration: float = 0.0

@dataclass
class InferenceResult:
    state: State
    cycles: list[CycleReport] = field(default_factory=list)
    converged: bool = False
    total_duration: float = 0.0
    final_norm: float = 0.0
    total_cycles: int = 0
```
