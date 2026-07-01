"""Validate EPC KB generator — run after any CSV or generator changes."""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from dynafx.knowledge import sparql_evaluate, parse_sparql
from dynafx.knowledge.model import TriplePattern, NamedNode
from epc_kb_generator import load_all, GRAPHS, EPC_NS

store = load_all()

def _sparql(q: str):
    """Run SPARQL query, return first binding or None."""
    algebra = parse_sparql(q)
    result = sparql_evaluate(algebra, store)
    if result.bindings:
        b = result.bindings[0]
        return {k: v.value for k, v in b.items()}
    return None

errors = []

# 1. Graph counts
for gname, guri in GRAPHS.items():
    cnt = len(list(store.triples(TriplePattern(), graph=guri)))
    print(f"  {gname}: {cnt} triples")
    if cnt == 0:
        errors.append(f"Empty graph: {gname}")

def count_type(typename: str) -> int:
    n = 0
    for g in store.graphs():
        for t in store.triples(TriplePattern(object_=NamedNode(f"{EPC_NS}{typename}")), graph=g):
            n += 1
    return n

print(f"\n  Projects: {count_type('Project')}")
print(f"  Suppliers: {count_type('Supplier')}")
print(f"  Ports: {count_type('Port')}")
print(f"  Ships: {count_type('Ship')}")
print(f"  Warehouses: {count_type('Warehouse')}")
print(f"  Workers: {count_type('Worker')}")
print(f"  Containers: {count_type('Container')}")

# 3. Aggregate values
agg_rel = _sparql(f"PREFIX epc: <{EPC_NS}> SELECT ?v WHERE {{ epc:Portfolio epc:aggregateSupplierReliability ?v }}")
at_risk = _sparql(f"PREFIX epc: <{EPC_NS}> SELECT ?v WHERE {{ epc:Portfolio epc:projectsAtRisk ?v }}")
active = _sparql(f"PREFIX epc: <{EPC_NS}> SELECT ?v WHERE {{ epc:Portfolio epc:activeProjects ?v }}")
total_mw = _sparql(f"PREFIX epc: <{EPC_NS}> SELECT ?v WHERE {{ epc:Portfolio epc:totalCapacityMW ?v }}")
in_transit = _sparql(f"PREFIX epc: <{EPC_NS}> SELECT ?v WHERE {{ epc:Portfolio epc:containersInTransit ?v }}")
disruption = _sparql(f"PREFIX epc: <{EPC_NS}> ASK {{ epc:GlobalDisruption epc:active true }}")

print(f"\n  Supplier reliability: {agg_rel}")
print(f"  Projects at risk: {at_risk}")
print(f"  Active projects: {active}")
print(f"  Total MW: {total_mw}")
print(f"  Containers in transit: {in_transit}")
print(f"  Disruption active: {disruption}")

# 4. Assertions
if agg_rel:
    v = float(agg_rel["v"])
    if not (0.70 <= v <= 0.95):
        errors.append(f"Supplier reliability {v} out of range [0.70, 0.95]")
else:
    errors.append("Missing aggregateSupplierReliability")

if at_risk:
    v = int(at_risk["v"])
    if v < 5 or v > 40:
        errors.append(f"Projects at risk {v} seems off (expected 5-40)")
else:
    errors.append("Missing projectsAtRisk")

if total_mw:
    v = float(total_mw["v"])
    if abs(v - 420) > 20:
        errors.append(f"Total MW {v} too far from 420")
else:
    errors.append("Missing totalCapacityMW")

if disruption is None:
    pass  # No disruption is correct
elif isinstance(disruption, dict) and disruption.get("v") == "true":
    errors.append("Disruption should not be active initially")

# 5. Chokepoint check
choke_ports = []
for g in store.graphs():
    for t in store.triples(TriplePattern(predicate=NamedNode(f"{EPC_NS}isChokepoint")), graph=g):
        if t.object_.value == "true":
            choke_ports.append((str(t.subject.iri), g))
if choke_ports:
    print(f"\n  Chokepoint ports: {choke_ports}")
else:
    errors.append("No chokepoint port found")

if errors:
    print(f"\n  FAILED ({len(errors)} errors):")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print(f"\n  ALL CHECKS PASSED")
