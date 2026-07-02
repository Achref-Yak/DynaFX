"""Read EPC enterprise CSVs and populate a TripleStore.

Uses ingest_csv() with YAML mappings + TTL ontology instead of
procedural load_* functions.
"""

from pathlib import Path
from statistics import mean

from dynafx.knowledge.ingest_csv import ingest_csv, load_all_mappings
from dynafx.knowledge.model import NamedNode, Literal, Triple, XSD_DOUBLE, XSD_INTEGER, XSD_STRING, XSD_BOOLEAN
from dynafx.knowledge.store import TripleStore
from dynafx.knowledge.turtle import parse_turtle

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAPPINGS_DIR = DATA_DIR / "mappings"
ONTOLOGY_FILE = DATA_DIR / "epc-ontology.ttl"

EPC_NS = "http://epc.org/"

GRAPHS = {
    "projects": "http://epc.org/graphs/projects",
    "suppliers": "http://epc.org/graphs/suppliers",
    "logistics": "http://epc.org/graphs/logistics",
    "workforce": "http://epc.org/graphs/workforce",
    "meta": "http://epc.org/graphs/meta",
}


def _epc(name: str) -> NamedNode:
    return NamedNode(f"{EPC_NS}{name}")


def _lit_num(val, dtype=XSD_DOUBLE) -> Literal:
    return Literal(str(val), datatype=dtype)


def _lit_str(val: str) -> Literal:
    return Literal(val, datatype=XSD_STRING)


def _lit_bool(val: bool) -> Literal:
    return Literal("true" if val else "false", datatype=XSD_BOOLEAN)


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
    supp_rows = _load_csv("epc_suppliers.csv")
    proj_rows = _load_csv("epc_projects.csv")
    ctn_rows = _load_csv("epc_containers.csv")
    g = GRAPHS["meta"]
    portfolio = _epc("Portfolio")
    disruption = _epc("GlobalDisruption")

    store.add(Triple(portfolio, _epc("type"), _epc("Portfolio")), g)
    store.add(Triple(disruption, _epc("type"), _epc("Disruption")), g)

    rels = [float(r["reliability"]) for r in supp_rows]
    avg_rel = round(mean(rels), 3)
    store.add(Triple(portfolio, _epc("aggregateSupplierReliability"), _lit_num(avg_rel)), g)

    at_risk = sum(1 for r in proj_rows if r["status"] in ("at_risk", "delayed"))
    store.add(Triple(portfolio, _epc("projectsAtRisk"), _lit_num(at_risk, XSD_INTEGER)), g)

    active = sum(1 for r in proj_rows if r["status"] == "active")
    store.add(Triple(portfolio, _epc("activeProjects"), _lit_num(active, XSD_INTEGER)), g)

    total_mw = round(sum(float(r["capacity_mw"]) for r in proj_rows), 1)
    store.add(Triple(portfolio, _epc("totalCapacityMW"), _lit_num(total_mw)), g)

    store.add(Triple(disruption, _epc("active"), _lit_bool(False)), g)

    in_transit = sum(1 for r in ctn_rows if r["status"] == "in_transit")
    store.add(Triple(portfolio, _epc("containersInTransit"), _lit_num(in_transit, XSD_INTEGER)), g)

    return avg_rel, at_risk, active, total_mw, in_transit


def load_all(store: TripleStore | None = None) -> TripleStore:
    if store is None:
        store = TripleStore()

    _load_ontology(store)

    for mapping_def in load_all_mappings(MAPPINGS_DIR):
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
