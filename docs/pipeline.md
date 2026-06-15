# InferenceCycle and Extraction Pipeline

## The 9-Step InferenceCycle

The `InferenceCycle` (`kernel/inference_cycle.py`) is the main reasoning loop. Each cycle runs three passes in sequence:

### Pass 1 — Structural (steps 1-3)

| Step | Operator | Purpose |
|------|----------|---------|
| 1. Extract | `Ξ` (extract) | Text → Graph (idempotent, runs only once) |
| 2. Schema | `Σ` (schema) | Apply domain schema / TBox validation |
| 3. Structural | `graph`, `constraint`, `temporal` | Build graph structure, check constraints |

### Pass 2 — Evidential (step 4)

| Operator | Purpose |
|----------|---------|
| `propagate` | Belief propagation via Master Equation (mandatory) |
| `abduce`, `induce`, `analogy` | Abductive / inductive inference |
| `reason`, `align`, `attention` | Evidence alignment and attention |
| `merge`, `simulate` | Graph merge and what-if simulation |

Selected by the PolicyEngine based on current state metrics.

### Pass 3 — Conflict (step 5)

| Operator | Purpose |
|----------|---------|
| `debate` | Adversarial argument evaluation |
| `verify` | Verification of inference chains |
| `constraint` | Logical constraint violation detection |
| `compress` | Graph compression / simplification |

### Convergence Check (steps 6-9)

6. **State delta** — compute ‖Δs‖ = G_dist + A_dist + H_dist + op_change
7. **Memory consolidation** — STM → LTM if capacity exceeded
8. **Convergence** — ‖Δs‖ < ε or stall_count ≥ window
9. **Tick** — increment cycle, emit `CycleReport`

```python
for cycle in range(1, max_cycles + 1):
    # Step 1-2: Extract + Schema
    # Step 3: Structural pass
    # Step 4: Evidential pass (policy-selected operators)
    # Step 5: Conflict pass (policy-selected operators)
    # Step 6: Δs = snapshot diff
    # Step 7: Memory store
    # Step 8: converged = ‖Δs‖ < ε
    # Step 9: CycleReport + state.record()
```

## The Extraction Sub-Pipeline

Text → Graph conversion happens in `operators/extract.py` (inlined from the old `pipeline/` module). It runs once and is idempotent.

### 1. Chunking (`nlp/chunker.py`)

Long documents split into overlapping 512-token windows using a RoBERTa tokenizer.

```python
from cognitive_engine.nlp.chunker import chunk_text
chunks = chunk_text(text, tokenizer, max_tokens=512, overlap=128)
```

### 2. Preprocessing (`nlp/preprocessor.py`)

Each chunk processed with spaCy `en_core_web_trf` for tokenization, POS tagging, dependency parsing, NER. A lightweight coreference resolver replaces pronouns.

### 3. Sentence Detection (`nlp/tagger.py` — SentenceTagger)

Propositions extracted as sentence boundaries from spaCy `Doc.sents`. Duplicate spans across chunk boundaries deduplicated by character range.

### 4. Relation Classification (`nlp/tagger.py` — RelationClassifier)

Every pair `(A, B)` classified as Support / Attack / None using a fine-tuned DistilRoBERTa model.

### 5. Type Mapping (`extract/types.py`)

Rule cascade assigns `NodeType`:

| Type | Signal |
|------|--------|
| CONDITION | `if`/`unless`/`provided` |
| FALLACY | Keyword match |
| JUSTIFICATION | `because`/`since`/`for` |
| AXIOM | Modal verb |
| COUNTERCLAIM | Adversative (but/however) |
| CLAIM | Root dependency, no marker |
| EVIDENCE | Non-root, no marker |

### 6. Demarcation Rules (`extract/demarcation.py`)

Five cognitive-linguistic dimensions per node: cognitive vs epistemic, epistemic vs institutional, affect vs cognition, constraint vs enablement, synchronic vs diachronic.

### 7. Edge Assignment (`extract/edges.py`)

Relations resolved to directed edges via a 3D lookup table `(source_type, target_type, label) → EdgeType` with demarcation-based refinement (SUPPORTS → QUALIFIES / ATTACKS → REBUTS when modalities differ).
