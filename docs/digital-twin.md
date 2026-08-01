# Digital Twin: Global Solar EPC

`examples/global_solar_epc_twin.py` is a living digital twin of a solar EPC (engineering, procurement, construction) enterprise reasoning through a typhoon-induced port closure disruption. It closes the full KB→Sim→Evidence loop in one runnable script.

```bash
python examples/global_solar_epc_twin.py      # exit 0, ~8 s, 12 sections
```

## The 12 Sections

| # | Section | Proves |
|---|---------|--------|
| 1 | SENSE | 7 EPC CSVs → named graphs (`ingest_csv`), ontology, RDFS inference, Portfolio aggregates |
| 2 | ASSEMBLE | `KBSimBridge.params_from_kb` — reliability, risk, MW, in-transit → sim params |
| 3 | MODEL | 11 SD stocks + 17 flows + 57 auxes + 5 DES queues + 3 resources + 2 agent types |
| 4 | **LIVE** | Baseline run; typhoon = flip `GlobalDisruption active` flag in the KB; ABM agents write live triples mid-run |
| 5 | LEARN | `evidence_from_result` writes revenue/cost/penalty/supply evidence triples |
| 6 | PREDICT | 6 disruption scenarios graded, ranked, constraint-filtered on a fresh store |
| 7 | ACT | Production rules on an isolated evidence snapshot → `requiresMitigation` |
| 8 | OPTIMIZE | `kb_lp_minimize` allocates a mitigation budget from SPARQL coeffs/bounds |
| 9 | DIAGNOSE | `causal_trace` revenue decomposition + `detect_feedback_loops` (5 loops) |
| 10 | PROVENANCE | `record_provenance` — PROV run entity with audit trail |
| 11 | MAP | which section proves L1 (sense) … L5 (decide) |
| 12 | TAKEAWAY | the loop closes; $2,795K cost of a 30-day closure |

## The Live Knowledge Base

Disruption is modeled as a knowledge-graph fact, not a parameter. Section 4 removes the `active=false` triple and adds `epc:GlobalDisruption epc:active true` to the `meta` graph. Each timestep the DSL `KB_QUERY` builtins re-read the store:

| Aux | Query (as param) | Returns |
|-----|------------------|---------|
| `kb_disruption_active` | `ASK { epc:GlobalDisruption epc:active true }` | 1.0 / 0.0 |
| `kb_supplier_reliability` | `SELECT ?v WHERE { epc:Portfolio epc:aggregateSupplierReliability ?v }` | 0.82 |
| `kb_projects_at_risk` | `SELECT ?v WHERE { epc:Portfolio epc:projectsAtRisk ?v }` | 22.0 |

The KB values multiply into the flow equations (`supply_to_port_rate`, `asia_outflow_rate` gated by `port_capacity × disruption_mult × kb_supplier_reliability`), so the KB steers the dynamics numerically.

## Verified Outputs

- Baseline profit **$931,425K**, completion 84.0%
- 30-day typhoon port closure: **−$2,795K** (supplier reliability 0.82, 22 projects at risk)
- Scenario grades: Baseline 0.956 … Extended Recovery 0.854
- LP: `port_capacity x=1.67`, objective $1.67K under a $2.0K budget
- 5 feedback loops (2 reinforcing, 3 balancing); 20 PROV provenance triples
