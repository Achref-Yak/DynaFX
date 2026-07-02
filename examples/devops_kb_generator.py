"""Read DevOps telemetry CSVs and populate a TripleStore.

Uses ingest_csv() with YAML mappings + TTL ontology.
"""

from pathlib import Path
from statistics import mean

from dynafx.knowledge.ingest_csv import ingest_csv, load_all_mappings
from dynafx.knowledge.model import NamedNode, Literal, Triple, XSD_DOUBLE, XSD_INTEGER, XSD_STRING, XSD_BOOLEAN
from dynafx.knowledge.store import TripleStore
from dynafx.knowledge.turtle import parse_turtle

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAPPINGS_DIR = DATA_DIR / "mappings"
ONTOLOGY_FILE = DATA_DIR / "devops-ontology.ttl"

DEVOPS_NS = "http://devops.org/"

GRAPHS = {
    "metrics": "http://devops.org/graphs/metrics",
    "events": "http://devops.org/graphs/events",
    "infra": "http://devops.org/graphs/infra",
    "meta": "http://devops.org/graphs/meta",
}


def _devops(name: str) -> NamedNode:
    return NamedNode(f"{DEVOPS_NS}{name}")


def _lit_num(val, dtype=XSD_DOUBLE) -> Literal:
    return Literal(str(val), datatype=dtype)


def _lit_str(val: str) -> Literal:
    return Literal(val, datatype=XSD_STRING)


def _load_csv(filename: str) -> list[dict]:
    import csv
    path = DATA_DIR / filename
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_ontology(store: TripleStore):
    if not ONTOLOGY_FILE.exists():
        raise FileNotFoundError(f"Ontology file not found: {ONTOLOGY_FILE}")
    ontology_store = parse_turtle(ONTOLOGY_FILE.read_text())
    meta_graph = GRAPHS["meta"]
    for triple in ontology_store.all_triples():
        store.add(triple, meta_graph)


def _compute_aggregates(store: TripleStore):
    """Pre-compute aggregates that SPARQL can't compute (no AVG/COUNT)."""
    g = GRAPHS["meta"]
    infra = _devops("InfrastructureSummary")

    store.add(Triple(infra, _devops("type"), _devops("Infrastructure")), g)

    metrics = _load_csv("devops_metrics.csv")

    cpus = [float(r["cpu"]) for r in metrics]
    mems = [float(r["memory"]) for r in metrics]
    lats = [float(r["latency"]) for r in metrics]
    reqs = [int(r["requests"]) for r in metrics]
    insts = [int(r["instances"]) for r in metrics]
    queues = [int(r["queue_length"]) for r in metrics]
    errs = [float(r["error_rate"]) for r in metrics]

    avg_cpu = round(mean(cpus), 1)
    avg_mem = round(mean(mems), 1)
    avg_lat = round(mean(lats), 1)
    max_lat = round(max(lats), 1)
    peak_reqs = max(reqs)
    total_inst_hrs = round(sum(insts) / 60.0, 3)

    store.add(Triple(infra, _devops("averageCPU"), _lit_num(avg_cpu)), g)
    store.add(Triple(infra, _devops("memoryUsage"), _lit_num(avg_mem)), g)
    store.add(Triple(infra, _devops("averageLatency"), _lit_num(avg_lat)), g)

    latency_slo = 500
    compliant = sum(1 for v in lats if v <= latency_slo)
    slo_pct = round(compliant / max(len(lats), 1) * 100, 1)
    store.add(Triple(infra, _devops("sloCompliance"), _lit_num(slo_pct)), g)

    infra_config = _load_csv("devops_infra.csv")
    app_cfg = next((r for r in infra_config if r["service"] == "app"), None)
    cost_per_hour = float(app_cfg["cost_per_hour"]) if app_cfg else 0.50
    total_cost = round(total_inst_hrs * cost_per_hour, 2)
    store.add(Triple(infra, _devops("totalCost"), _lit_num(total_cost)), g)

    events = _load_csv("devops_events.csv")
    autoscale_events = sum(1 for r in events if "scale" in r["event_type"])
    store.add(Triple(infra, _devops("autoscaleEventCount"), _lit_num(autoscale_events, XSD_INTEGER)), g)

    peak_instances = max(insts)
    final_instances = insts[-1] if insts else 0
    min_needed = 2
    idle_ratio = round(max(0, final_instances - min_needed) / max(final_instances, 1), 2)
    store.add(Triple(infra, _devops("idleInstanceRatio"), _lit_num(idle_ratio)), g)

    store.add(Triple(infra, _devops("totalInstances"), _lit_num(peak_instances, XSD_INTEGER)), g)

    return {
        "avg_cpu": avg_cpu,
        "avg_latency": avg_lat,
        "peak_latency": max_lat,
        "peak_requests": peak_reqs,
        "slo_compliance": slo_pct,
        "total_cost": total_cost,
        "autoscale_events": autoscale_events,
        "idle_ratio": idle_ratio,
    }


def load_all(store: TripleStore | None = None) -> TripleStore:
    if store is None:
        store = TripleStore()

    _load_ontology(store)

    for mapping_def in load_all_mappings(MAPPINGS_DIR):
        if not mapping_def.csv.startswith("devops_"):
            continue
        csv_path = DATA_DIR / mapping_def.csv
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        ingest_csv(mapping_def, str(csv_path), store, strict=False)

    _compute_aggregates(store)
    return store


def print_stats(store: TripleStore):
    from dynafx.knowledge.model import TriplePattern
    total = sum(len(list(store.triples(TriplePattern(), graph=g))) for g in store.graphs())
    print(f"Total triples: {total}")
    print(f"Named graphs: {len(store.graphs())}")
    for g in sorted(store.graphs()):
        cnt = len(list(store.triples(TriplePattern(), graph=g)))
        print(f"  {g}: {cnt} triples")


if __name__ == "__main__":
    store = load_all()
    print_stats(store)
