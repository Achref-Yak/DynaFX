"""Read EPC enterprise CSVs and populate a TripleStore."""

import csv
import os
from pathlib import Path
from statistics import mean

from dynafx.knowledge.model import NamedNode, Literal, Triple, XSD_DOUBLE, XSD_INTEGER, XSD_BOOLEAN, XSD_STRING
from dynafx.knowledge.store import TripleStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

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
    path = DATA_DIR / filename
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_suppliers(store: TripleStore):
    rows = _load_csv("epc_suppliers.csv")
    g = GRAPHS["suppliers"]
    for r in rows:
        s = _epc(r["id"])
        store.add(Triple(s, _epc("type"), _epc("Supplier")), g)
        store.add(Triple(s, _epc("name"), _lit_str(r["name"])), g)
        store.add(Triple(s, _epc("region"), _lit_str(r["region"])), g)
        store.add(Triple(s, _epc("material"), _lit_str(r["material"])), g)
        store.add(Triple(s, _epc("reliability"), _lit_num(r["reliability"])), g)
        store.add(Triple(s, _epc("volumePerDay"), _lit_num(r["volume_per_day"])), g)
        store.add(Triple(s, _epc("costPerPanel"), _lit_num(r["cost_per_panel"])), g)
        store.add(Triple(s, _epc("leadTimeDays"), _lit_num(r["lead_time_days"], XSD_INTEGER)), g)
    return rows


def load_projects(store: TripleStore, supp_rows: list[dict]):
    rows = _load_csv("epc_projects.csv")
    g = GRAPHS["projects"]
    supp_map = {r["id"]: r for r in supp_rows}
    for r in rows:
        s = _epc(r["id"])
        store.add(Triple(s, _epc("type"), _epc("Project")), g)
        store.add(Triple(s, _epc("name"), _lit_str(r["name"])), g)
        store.add(Triple(s, _epc("region"), _lit_str(r["region"])), g)
        store.add(Triple(s, _epc("capacityMW"), _lit_num(r["capacity_mw"])), g)
        store.add(Triple(s, _epc("budgetK"), _lit_num(r["budget_k"], XSD_INTEGER)), g)
        store.add(Triple(s, _epc("deadlineDay"), _lit_num(r["deadline_day"], XSD_INTEGER)), g)
        store.add(Triple(s, _epc("status"), _lit_str(r["status"])), g)
        partner = supp_map.get(r["partner_id"])
        if partner:
            store.add(Triple(s, _epc("partner"), _epc(partner["id"])), g)
    return rows


def load_ports(store: TripleStore):
    rows = _load_csv("epc_ports.csv")
    g = GRAPHS["logistics"]
    for r in rows:
        s = _epc(r["id"])
        store.add(Triple(s, _epc("type"), _epc("Port")), g)
        store.add(Triple(s, _epc("name"), _lit_str(r["name"])), g)
        store.add(Triple(s, _epc("region"), _lit_str(r["region"])), g)
        store.add(Triple(s, _epc("capacityPerDay"), _lit_num(r["capacity_per_day"], XSD_INTEGER)), g)
        store.add(Triple(s, _epc("isChokepoint"), _lit_bool(bool(int(r["is_chokepoint"])))), g)
    return rows


def load_ships(store: TripleStore):
    rows = _load_csv("epc_ships.csv")
    g = GRAPHS["logistics"]
    for r in rows:
        s = _epc(r["id"])
        store.add(Triple(s, _epc("type"), _epc("Ship")), g)
        store.add(Triple(s, _epc("name"), _lit_str(r["name"])), g)
        store.add(Triple(s, _epc("route"), _lit_str(r["route"])), g)
        store.add(Triple(s, _epc("capacityContainers"), _lit_num(r["capacity_containers"], XSD_INTEGER)), g)
        store.add(Triple(s, _epc("speedKnots"), _lit_num(r["speed_knots"])), g)
        store.add(Triple(s, _epc("currentPort"), _epc(r["current_port_id"])), g)
        store.add(Triple(s, _epc("nextPort"), _epc(r["next_port_id"])), g)
        store.add(Triple(s, _epc("transitDays"), _lit_num(r["transit_days"], XSD_INTEGER)), g)
    return rows


def load_containers(store: TripleStore):
    rows = _load_csv("epc_containers.csv")
    g = GRAPHS["logistics"]
    for r in rows:
        s = _epc(r["id"])
        store.add(Triple(s, _epc("type"), _epc("Container")), g)
        store.add(Triple(s, _epc("contents"), _lit_str(r["contents"])), g)
        store.add(Triple(s, _epc("quantity"), _lit_num(r["quantity"], XSD_INTEGER)), g)
        store.add(Triple(s, _epc("etaDay"), _lit_num(r["eta_day"], XSD_INTEGER)), g)
        store.add(Triple(s, _epc("status"), _lit_str(r["status"])), g)
        store.add(Triple(s, _epc("onShip"), _epc(r["ship_id"])), g)
        store.add(Triple(s, _epc("originPort"), _epc(r["origin_port_id"])), g)
        store.add(Triple(s, _epc("destPort"), _epc(r["dest_port_id"])), g)
    return rows


def load_warehouses(store: TripleStore):
    rows = _load_csv("epc_warehouses.csv")
    g = GRAPHS["logistics"]
    for r in rows:
        s = _epc(r["id"])
        store.add(Triple(s, _epc("type"), _epc("Warehouse")), g)
        store.add(Triple(s, _epc("name"), _lit_str(r["name"])), g)
        store.add(Triple(s, _epc("region"), _lit_str(r["region"])), g)
        store.add(Triple(s, _epc("capacityPanels"), _lit_num(r["capacity_panels"], XSD_INTEGER)), g)
        store.add(Triple(s, _epc("inventoryPanels"), _lit_num(r["inventory_panels"], XSD_INTEGER)), g)
    return rows


def load_workers(store: TripleStore):
    rows = _load_csv("epc_workers.csv")
    g = GRAPHS["workforce"]
    for r in rows:
        s = _epc(r["id"])
        store.add(Triple(s, _epc("type"), _epc("Worker")), g)
        store.add(Triple(s, _epc("name"), _lit_str(r["name"])), g)
        store.add(Triple(s, _epc("role"), _lit_str(r["role"])), g)
        store.add(Triple(s, _epc("region"), _lit_str(r["region"])), g)
        store.add(Triple(s, _epc("skillLevel"), _lit_num(r["skill_level"], XSD_INTEGER)), g)
        if r.get("project_id"):
            store.add(Triple(s, _epc("assignedTo"), _epc(r["project_id"])), g)
    return rows


def compute_aggregates(store: TripleStore, proj_rows: list[dict], supp_rows: list[dict]):
    """Pre-compute aggregates that SPARQL can't compute (no AVG/COUNT)."""
    g = GRAPHS["meta"]
    portfolio = _epc("Portfolio")
    disruption = _epc("GlobalDisruption")

    store.add(Triple(portfolio, _epc("type"), _epc("Portfolio")), g)
    store.add(Triple(disruption, _epc("type"), _epc("Disruption")), g)

    # Aggregate supplier reliability
    rels = [float(r["reliability"]) for r in supp_rows]
    avg_rel = round(mean(rels), 3)
    store.add(Triple(portfolio, _epc("aggregateSupplierReliability"), _lit_num(avg_rel)), g)

    # Projects at risk (status = at_risk or delayed)
    at_risk = sum(1 for r in proj_rows if r["status"] in ("at_risk", "delayed"))
    store.add(Triple(portfolio, _epc("projectsAtRisk"), _lit_num(at_risk, XSD_INTEGER)), g)

    # Active projects
    active = sum(1 for r in proj_rows if r["status"] == "active")
    store.add(Triple(portfolio, _epc("activeProjects"), _lit_num(active, XSD_INTEGER)), g)

    # Total capacity
    total_mw = round(sum(float(r["capacity_mw"]) for r in proj_rows), 1)
    store.add(Triple(portfolio, _epc("totalCapacityMW"), _lit_num(total_mw)), g)

    # No disruption initially
    store.add(Triple(disruption, _epc("active"), _lit_bool(False)), g)

    # Count containers in_transit
    ctn_rows = _load_csv("epc_containers.csv")
    in_transit = sum(1 for r in ctn_rows if r["status"] == "in_transit")
    store.add(Triple(portfolio, _epc("containersInTransit"), _lit_num(in_transit, XSD_INTEGER)), g)

    return avg_rel, at_risk, active, total_mw, in_transit


def load_all(store: TripleStore | None = None) -> TripleStore:
    """Read all CSVs and populate the TripleStore. Returns the store."""
    if store is None:
        store = TripleStore()

    supp = load_suppliers(store)
    proj = load_projects(store, supp)
    load_ports(store)
    load_ships(store)
    load_containers(store)
    load_warehouses(store)
    load_workers(store)
    compute_aggregates(store, proj, supp)

    return store


def print_stats(store: TripleStore):
    """Print summary statistics about the KB."""
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
