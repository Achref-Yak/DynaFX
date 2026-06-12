# Data Models

All data models are defined in `src/cognitive_engine/models.py`.

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

Which edge types are active depends on the reasoning mode:

| Mode | Active Edge Types |
|------|-------------------|
| `ARGUMENT` | SUPPORTS, CONTRADICTS, ATTACKS, REBUTS |
| `CAUSAL` | INFERS, SUPPORTS |
| `CONDITIONAL` | QUALIFIES, INFERS |
| `ANALOGY` | JUSTIFIES, SUPPORTS |

## Opinion

```python
Opinion = tuple[float, float, float, float]
#          (belief, disbelief, uncertainty, base_rate)
```

Represents a Subjective Logic opinion. Always satisfies:

- `belief + disbelief + uncertainty = 1.0`
- `base_rate` is the prior probability in `[0, 1]`
- Projected probability: `belief + base_rate * uncertainty`

## Node

```python
@dataclass
class Node:
    id: UUID
    type: NodeType
    text: str
    span: Optional[Span]       # Character offsets in source text
    abstraction_level: int      # 1 = concrete, 5 = abstract
    salience: float             # 0.0 to 1.0
    opinion: Opinion            # Subjective Logic opinion
    category: int               # 1=Necessity, 2=Fact, 3=Belief, 4=Concept
    metadata: dict              # Contains demarcation dimensions
```

## Edge

```python
@dataclass
class Edge:
    id: UUID
    source_id: UUID
    target_id: UUID
    type: EdgeType
    opinion: Opinion
    warrant: Optional[Warrant]  # Conditional opinion (if source, then target)
    metadata: dict
```

A `Warrant` is a pair of opinions `(Opinion, Opinion)` representing the conditional belief `(source → target, source → ¬target)`.

## Graph

```python
@dataclass
class Graph:
    nodes: Dict[UUID, Node]
    edges: List[Edge]
    mode: ReasoningMode
    source_text: str
    metadata: dict              # Priors config + mode views
    cta: Optional[ConversationTree]
```

The primary output of the pipeline. Serializes to JSON via `graph.to_json()`.

## ConversationTree

```python
@dataclass
class ConversationTree:
    root_id: UUID
    node_ids: set[UUID]
    parent_map: dict[UUID, UUID]
```

Isolates context windows to prevent logical context poisoning. Built from the graph by following INFERS/SUPPORTS/JUSTIFIES edges from leaf to root. Provides `get_context(node_id)` which returns the ancestor chain.

## ReasoningMode

```python
class ReasoningMode(Enum):
    CAUSAL = auto()
    CONDITIONAL = auto()
    ARGUMENT = auto()
    ANALOGY = auto()
```

Controls which edge types are retained during mode filtering.
