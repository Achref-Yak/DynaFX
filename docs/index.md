# Cognitive Reasoning Graph Engine

A **provably uncertainty-bounded** pipeline that extracts structured reasoning graphs from unstructured text and computes formal opinions via [Subjective Logic](https://doi.org/10.1007/978-3-319-42337-1).

```mermaid
flowchart LR
    A[Input Text] --> B[Chunker]
    B --> C[spaCy Preprocessor]
    C --> D[Sentence Tagger]
    D --> E[Relation Classifier]
    E --> F[Type Mapper]
    F --> G[Demarcation Rules]
    G --> H[Edge Assigner]
    H --> I[SL Opinion Propagation]
    I --> J[Validators]
    J --> K{{JSON Graph}}
```

## Key Features

- **Fully deterministic pipeline** — zero LLM calls. Sentence boundaries from spaCy, relations from a DistilRoBERTa classifier, all logic in pure Python.
- **Structured reasoning graphs** — typed nodes (CLAIM, AXIOM, EVIDENCE, CONDITION, FALLACY, COUNTERCLAIM, JUSTIFICATION) connected by typed edges (SUPPORTS, ATTACKS, CONTRADICTS, INFERS, QUALIFIES, REBUTS, JUSTIFIES).
- **Subjective Logic opinions** — every node and edge carries a 4-tuple `(belief, disbelief, uncertainty, base_rate)` satisfying `b + d + u = 1`, propagated via fusion and deduction operators.
- **Formal verification** — category-theoretic monotonicity checks, topological cycle detection, and opinion invariant enforcement.
- **Five demarcation dimensions** — each node is annotated with cognitive/epistemic, institutional, affect, constraint, and temporal classifications.

## Quick Example

```bash
cognitive-engine demo/complicated.txt
```

Outputs a JSON graph with typed nodes, edges, opinions, and metadata. See [Getting Started](getting-started.md) for a full walkthrough.

## Project Status

Alpha. The pipeline produces correct structural output; the relation classifier and some type assignments depend on model quality and are actively being improved through retraining.
