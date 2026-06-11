# Tutorial — Running the Reasoning Graph Engine

This tutorial walks you through your first run, step by step. No prior
knowledge assumed.

---

## 1. Prerequisites

- **Python 3.12** installed (`python3.12 --version` should show 3.12.x)
- **pip** (comes with Python)
- **A Groq API key** — free account at https://console.groq.com/keys
  (the free tier gives you enough credits to run this many times over)

---

## 2. Set up the environment

From the project root:

```bash
# Create a virtual environment (optional but recommended)
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

If you want to run tests too:

```bash
pip install -e ".[dev]"
```

---

## 3. Configure your API key

Copy the example env file and edit it:

```bash
cp .env.example .env
```

Open `.env` and replace `gsk_your_api_key_here` with your real key from
https://console.groq.com/keys.

```
GROQ_API_KEY="gsk_abc123..."  # <-- your real key
```

The engine loads `.env` automatically on startup. No need to `export`.

---

## 4. Create a sample document

Create a file called `demo.txt`:

```text
We should migrate from PostgreSQL to CockroachDB for better horizontal scaling.
Our current PostgreSQL instance handles 5000 writes per second, and we're
projecting 20,000 writes per second next quarter. CockroachDB claims 100,000
writes per second on a 5-node cluster. However, the migration would require
at least 3 months of engineering work. We only have 2 backend engineers
available. The CTO supports the migration if it doesn't delay the Q3 release.
```

This is a realistic mini-PRD — it has claims, evidence, conditions, and
even a contradiction.

---

## 5. Run the pipeline

```bash
PYTHONPATH=src python3.12 -m cognitive_engine.cli demo.txt
```

What happens:

```
INFO | Processing file: demo.txt (421 chars)
INFO | Round 1/3: extracting graph...
INFO | Round 1: 0 violations (0 errors)
INFO | Round 1: reviewer says 'accept'
INFO | Graph accepted after 1 round(s)
```

Then it prints a JSON graph to stdout. All in one shot (if the graph is
clean on the first try).

If you want to save the output to a file:

```bash
PYTHONPATH=src python3.12 -m cognitive_engine.cli demo.txt --output result.json
```

---

## 6. Understand the output

The JSON has three main sections:

### Nodes

Each idea in your text becomes a node:

```json
"nodes": {
    "abc123...": {
        "id": "abc123...",
        "type": "EVIDENCE",
        "text": "Current PostgreSQL handles 5000 writes/sec",
        "category": 2,
        "opinion": [0.0, 0.0, 1.0, 0.5]
    },
    ...
}
```

Three types of nodes: `CLAIM`, `EVIDENCE`, `CONDITION`.

### Edges

Relationships between ideas:

```json
"edges": [
    {
        "id": "def456...",
        "source_id": "abc123...",
        "target_id": "ghi789...",
        "type": "SUPPORTS"
    },
    ...
]
```

Five edge types: `SUPPORTS`, `CONTRADICTS`, `QUALIFIES`, `INFERS`,
`JUSTIFIES`.

### Opinions

Every node and edge has an opinion `[b, d, u, a]`:
- `b` = belief (0–1)
- `d` = disbelief (0–1)
- `u` = uncertainty (0–1)
- `a` = base rate / prior (0–1)

`b + d + u` always equals 1. Initially all opinions are `[0, 0, 1, 0.5]`
(total ignorance). In a real system these would be updated with evidence.

---

## 7. What if the graph gets rejected?

Try making a deliberately hard document. Create `bad.txt`:

```text
The sky is green. Therefore the sky is green because the sky is green.
The sky being green proves that grass must also be green.
The concept of greenness implies the color blue.
```

This has circular reasoning and category violations. Run it:

```bash
PYTHONPATH=src python3.12 -m cognitive_engine.cli bad.txt
```

You'll see the loop retry:

```
INFO | Round 1/3: extracting graph...
INFO | Round 1: 1 violations (1 errors)
INFO | Round 1: reviewer says 'reject'
INFO | Round 2/3: extracting graph...
...
```

After 3 failed rounds, it exits with an error. The engine prefers to say
"no" rather than return garbage. That's **fail-closed**.

---

## 8. Using a different model

By default it uses `llama-3.3-70b-versatile` via Groq. You can change it:

```bash
PYTHONPATH=src python3.12 -m cognitive_engine.cli demo.txt --model llama-3.1-8b-instant
```

Smaller models are faster but may produce lower quality graphs. The
70B model is the recommended default.

---

## 9. Run the tests

To make sure everything is wired correctly:

```bash
PYTHONPATH=src python3.12 -m pytest tests/ -v
```

All 22 tests should pass.

---

## Summary

```
1. cp .env.example .env        # configure your API key
2. pip install -e .            # install dependencies
3. create demo.txt             # write some text
4. python3.12 -m ... demo.txt  # run the pipeline
5. inspect the JSON output     # see the reasoning graph
```

That's it. The engine reads your text, extracts a reasoning graph, checks it
for logical errors, and gives you back a structured map of every argument it
found.
