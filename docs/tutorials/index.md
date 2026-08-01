# Tutorials

The fastest way to learn DynaFX is to run code. Every tutorial below is a
**verified, runnable walkthrough**: the code blocks were executed against the
installed package before publishing, so you can copy them into a REPL or a
script and they work.

!!! tip "One environment, no servers"
    DynaFX is a Python library. Everything here runs in a single process — no
    database, no HTTP server, no external connectors. Data is generated or
    loaded from local CSV/Turtle files.

## Track

| # | Tutorial | Builds on | What you'll be able to do |
|---|----------|-----------|---------------------------|
| 1 | [Hello World](01-hello-world.md) | — | Parse a `.sysd` model and read trajectories |
| 2 | [System Dynamics](02-system-dynamics.md) | 1 | Model stocks, flows, auxes, and parameters |
| 3 | [Agent Behavior](03-agent-based-modeling.md) | 1 | Program agents with strategies and message passing |
| 4 | [Discrete Events](04-discrete-event-simulation.md) | 1 | Model queues, resources, and events |
| 5 | [Knowledge Graphs](05-knowledge-graph.md) | — | Build an RDF knowledge base from Turtle |
| 6 | [Semantic Queries](06-semantic-queries.md) | 5 | Query the KB with SPARQL, from inside expressions |
| 7 | [Closed-Loop Twin](07-closed-loop-twin.md) | 5, 6 | Connect KB → simulation → KB into a loop |
| 8 | [Scenarios & Sensitivity](08-scenarios-and-sensitivity.md) | 2 | Compare scenarios and rank parameter influence |
| 9 | [Custom Ontology](09-custom-ontology.md) | 5 | Define inference rules and type hierarchies |
| 10 | [Publishing Results](10-publishing-results.md) | 2, 8 | Run causal analysis, optimize, and record provenance |

## How to run the examples

Install the package (editable) and open a Python session:

```bash
uv sync
uv run python
```

Or run any tutorial file directly:

```bash
uv run python -c "from dynafx.dynamics import parse_sysd; print(parse_sysd('T\ndt 1\nfrom 0 to 5\nstock X: 100\n  - O: X*0.1\n').simulate().values['X'][-1])"
```

The examples import from the top-level namespaces (`dynafx.dynamics`,
`dynafx.knowledge`, `dynafx.bridge`, `dynafx.patterns`), matching the public
API. See [Development](../development.md) for the test and lint commands, and
[Scientific Foundations](../foundations.md) for the design rationale behind
what you're using.
