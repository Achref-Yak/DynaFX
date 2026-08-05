# Examples

## The Digital Twin

The flagship example is `examples/global_solar_epc_twin.py` — a living digital twin of a solar EPC supply chain reasoning through a typhoon port closure. It spans L1 (sense) through L5 (decide). See [Digital Twin](digital-twin.md) for the full walkthrough.

```bash
python examples/global_solar_epc_twin.py
```

## Model Library

The `.sysd` models under `data/models/`:

| Model | Role |
|-------|------|
| `global_solar_epc.sysd` | The twin's SD + ABM + DES supply-chain model (11 stocks, 57 auxes, 5 DES queues, 3 resources, 2 agent types, `KB_QUERY` auxes) |
| `vmi.sysd` | Vendor-managed inventory — test fixture |
| `reverse_logistics.sysd` | Reverse logistics — test fixture |
| `cold_chain.sysd` | Cold-chain distribution — test fixture |
| `supply_chain_demo.sysd` | Supply-chain demonstration — test fixture |

The library fixtures are exercised by `tests/test_reference_models.py` and `tests/test_delays.py`.

## Data

All demo resources live under `data/`:

- `data/epc_*.csv` — 7 enterprise CSVs (suppliers, projects, ports, ships, containers, warehouses, workers)
- `data/epc-ontology.ttl` — the EPC ontology (9 classes, ~40 properties)
- `data/mappings/*.yaml` — 7 YAML ingestion mappings
- `data/models/*.sysd` — the `.sysd` model library

Data is regenerable via `scripts/generate_epc_csvs.py` (seed=42).

## Patterns

`dynafx.patterns` provides reusable cross-paradigm model factories:

- **`SignalChain`** — builds a `SysdModel` for a leading-indicator → outcome pattern. Parameters: `trace_expr`, `detection_delay` (list for multi-hop), `decision_lag`, `outcome_threshold`, `outcome_sensitivity`, `threshold_direction`, `has_feedback`, `has_tracking`.
- **`DisruptionCascade`** — supply-chain disruption propagation model.

Both return a wired `SysdModel` ready to `simulate()`.
