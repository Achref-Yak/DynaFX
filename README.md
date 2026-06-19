# Cognitive Reasoning Graph Engine

A **neuro-symbolic reasoning framework** that extracts structured world models from unstructured text and computes formal opinions via Subjective Logic. Builds things-and-their-relations world models with FrameNet/VerbNet/PropBank semantic resources, multi-agent orchestration, self-reflection, feedback fusion, and declarative type rules. Fully deterministic — zero LLM calls.

```text
Text → [Chunker] → [spaCy Preprocessor] → [Sentence Tagger]
     → [Heuristic Classifier] → [Type Mapper] → [FrameNet/VerbNet Rules]
     → [Edge Assigner] → [SL Opinion Propagation] → [Rule Engine]
     → [Self-Reflection] → [Multi-Agent Orchestration] → [JSON Graph]
```

## Features

- **World-model-first extraction** — 25+ node types (AGENT, PROCESS, STATE, GOAL, RESOURCE, CONSTRAINT, etc.) with BFO ontological grounding.
- **Semantic resource integration** — FrameNet (1221 frames), PropBank (112k rolesets), VerbNet (429 classes) for precise semantic typing.
- **Subjective Logic opinions** — every node and edge carries `(belief, disbelief, uncertainty, base_rate)` satisfying `b + d + u = 1`. Propagated via fusion and deduction.
- **Declarative type rules** — 22 TypeRule dataclass rules with frame-based matchers; argumentation keyword rules at higher priority than world-model frame rules.
- **Multi-agent orchestration** — ManagerAgent (lifecycle), BlackboardAgent (shared memory), SpecialistAgent (domain experts), HyperHeuristic (selector).
- **Self-reflection** — periodic and event-driven (Page-Hinkley) reflection with configurable frequency.
- **Feedback fusion** — SL Beta distribution via cumulative_fusion for belief updates.
- **Structured retrieval** — SQL-based FactStore with SCD Type 2 versioning; vector similarity as fallback only.
- **Rule engine** — pattern matching with negation-as-failure (Jena noValue semantics).
- **Systems analysis** — feedback loops, leverage points, system archetypes, causal chain analysis.
- **Stock-flow simulation** — inventory dynamics and what-if scenario modeling.
- **Risk attitude modeling** — risk-adjusted beliefs and stakeholder utility aggregation.
- **898 tests** — comprehensive test suite covering all modules.

## Requirements

- Python 3.12+
- CUDA-capable GPU (optional; falls back to CPU)

## Install

```bash
git clone <repo-url>
cd reasoning_engine
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_trf
```

## Usage

```python
from cognitive_engine.operators.extract import run_argumentation
from cognitive_engine.analysis import build_verifiable_summary

text = "Port X handles 60% of container throughput. Due to labor shortages, capacity dropped 40%."
graph = run_argumentation(text, spans=[], docs=[], source_text=text)
summary = build_verifiable_summary(graph)
```

```bash
# Run the logistics demo
python examples/logistics_demo.py | python -m json.tool
```

## Output

A JSON graph with typed nodes, typed edges, Subjective Logic opinions, and verifiable summaries:

```json
{
  "graph_analysis": {
    "node_count": 45,
    "edge_count": 114,
    "edge_types": {
      "CAUSES": 13,
      "SUPPORTS": 14,
      "DEPENDS": 1,
      "CONTRADICTS": 13
    }
  },
  "systems_analysis": {
    "feedback_loops": [...],
    "leverage_points": [...]
  },
  "pipeline_result": {
    "metadata": {
      "beliefs": {...},
      "bfo_violations": 0
    }
  }
}
```

## Project Structure

```
src/cognitive_engine/
  core/               — Data models, math, state, loom, workflow
    models.py         — Graph, Node, Edge, NodeType, EdgeType, Opinion, BfoCategory
    math.py           — SL ops, GNN primitives, neurosymbolic_fuse, complexity metrics
    state.py          — State (Graph, ABox, TBox, Metadata, Trace)
    loom.py           — Weave wraps State + log
    operator.py       — Operator protocol
    workflow.py       — PrimitiveRegistry, WorkflowEngine
  nlp/                — NLP pipeline
    chunker.py        — Text chunking, PropSpan
    preprocessor.py   — spaCy preprocessing
    tagger.py         — PropositionTagger, RelationClassifier
    heuristic_classifier.py — Causal/enablement/part-whole/dependency detection
    semantic_resources.py    — FrameNet/PropBank/VerbNet singleton
  extract/            — World model extraction
    types.py          — TypeRule dataclass rules (22 rules)
    frame_rules.py    — FRAMENodeType_MAP (35 frames), ARGUMENTATION_FRAMES
    verbnet_roles.py  — VN_CONCEPT_MAP, VN_EdgeType_MAP
    srl_relations.py  — SRL-based relation extraction
    edges.py          — Edge type resolution, BFO constraints
    entities.py       — Entity extraction + name-based dedup
    relations.py      — Relation classification with VerbNet
    concepts/         — Domain concept dispatch
  operators/          — 27 operators
    extract.py        — Text→Graph extraction pipeline
    relate.py         — Relation scoring (textual, semantic, numeric, entity, topic)
    propagate.py      — Belief propagation via Master Equation
    constraint.py     — BFO constraint enforcement
    rule_apply.py     — Rule engine integration
    systems.py        — FeedbackLoopDetector, LeveragePointScorer, SystemArchetypeClassifier
    stock_flow.py     — Stock-flow analysis
    simulate.py       — What-if simulation
  kernel/             — Inference cycle
    inference_cycle.py — 9-step loop with self-reflection
    assertion_gate.py  — Assertion validation
    self_reflect.py    — Periodic + event-driven reflection
  agents/             — Multi-agent orchestration
    manager.py        — ManagerAgent (lifecycle)
    blackboard.py     — BlackboardAgent (shared memory + optional SQLite)
    specialist.py     — SpecialistAgent + HyperHeuristic
  memory/             — Fact storage
    fact_store.py     — FactStore with SCD Type 2, facts_archive
    feedback.py       — FeedbackStore with SL Beta fusion
    consolidate.py    — Leiden communities, build_pattern()
    models.py         — Fact, FactArchive, LTMPattern
  rules/              — Declarative rules
    engine.py         — Pattern matching, negation-as-failure
    parser.py         — parse_rules() dict/JSON→Rule
  policy/             — Operator policy engine
    engine.py         — PolicyEngine
    builtin.py        — DEFAULT_POLICY, SCIENTIFIC_POLICY
  tbox/               — TBox axioms
    loader.py         — GENERAL_TBOX (10 world-model axioms)
tests/                — 898 tests
examples/             — Logistics demo, output examples
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Documentation

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

## Citation

```bibtex
@software{cognitive_engine,
  title = {Cognitive Reasoning Graph Engine},
  year = {2026},
  url = {https://github.com/Achref-Yak/reasoning_engine}
}
```

## License

MIT
