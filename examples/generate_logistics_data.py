"""Generate deterministic logistics network historical CSVs + ontology TTL."""

import csv, math, os
from pathlib import Path

DATA_DIR = Path("data/logistics")
MAPPINGS_DIR = Path("data/mappings")
ONTOLOGY_PATH = Path("data/logistics-ontology.ttl")
MONTHS = 12
SEED = 42
NS = "logi:"
NS_URI = "http://logistics-network.org/"
BASE_IRI = "http://logistics-network.org/"
os.makedirs(DATA_DIR, exist_ok=True)


def _lin(start, end, t, total):
    return start + (end - start) * t / max(1, total - 1)


def _perturb(base, month, amplitude=0.02):
    return base * (1 + amplitude * math.sin(2 * math.pi * month / 12 + 0.5 * SEED))


def write_csv(filename, rows):
    path = DATA_DIR / filename
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  {path} ({len(rows) - 1} data rows)")


def write_yaml(filename, entity_class, columns, target_graph):
    lines = [
        "prefixes:",
        f'  logi: "{BASE_IRI}"',
        '  xsd: "http://www.w3.org/2001/XMLSchema#"',
        '  rdf: "http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
        f'csv: "logistics/{filename.replace(".yaml", ".csv")}"',
        f'target_graph: "{target_graph}"',
        "entity:",
        f"  class: {entity_class}",
        '  id_column: "id"',
        f'  id_prefix: "{NS}"',
        "columns:",
    ]
    for col_name, col_type, predicate in columns:
        lines.append(f"  {col_name}:")
        lines.append(f"    predicate: {predicate}")
        lines.append(f"    type: {col_type}")
    path = MAPPINGS_DIR / filename
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  {path}")


def generate_warehouse():
    rows = [("id", "month", "region", "inventory_load", "queue_hours", "throughput")]
    for m in range(MONTHS):
        for r, start_inv, end_inv, start_q, end_q, start_t, end_t in [
            ("A", 0.58, 0.78, 1.2, 3.8, 420, 510),
            ("B", 0.48, 0.55, 0.8, 1.2, 380, 400),
            ("C", 0.35, 0.45, 0.5, 0.9, 280, 320),
        ]:
            inv = _lin(start_inv, end_inv, m, MONTHS)
            qh = _lin(start_q, end_q, m, MONTHS)
            tp = _lin(start_t, end_t, m, MONTHS)
            inv = _perturb(inv, m, 0.03)
            qh = _perturb(qh, m, 0.04)
            tp = _perturb(tp, m, 0.02)
            rows.append((f"wh_{m}_{r}", m, r, round(inv, 4), round(qh, 2), round(tp, 1)))
    return rows


def generate_fleet():
    rows = [("id", "month", "region", "vehicles_active", "utilization_pct", "avg_trip_hours")]
    for m in range(MONTHS):
        for r, start_v, end_v, start_u, end_u in [
            ("A", 42, 48, 0.76, 0.88),
            ("B", 28, 30, 0.58, 0.65),
            ("C", 18, 22, 0.42, 0.52),
        ]:
            v = round(_lin(start_v, end_v, m, MONTHS))
            u = _lin(start_u, end_u, m, MONTHS)
            u = _perturb(u, m, 0.025)
            trip = 3.2 + 1.8 * u
            rows.append((f"fl_{m}_{r}", m, r, v, round(u, 4), round(trip, 2)))
    return rows


def generate_deliveries():
    rows = [("id", "month", "route_from", "route_to", "volume", "on_time_pct", "avg_delay_hours")]
    for m in range(MONTHS):
        for f, t, vol_s, vol_e, ot_s, ot_e in [
            ("A", "A", 2800, 4200, 0.94, 0.82),
            ("A", "B", 1200, 1800, 0.95, 0.86),
            ("A", "C", 800, 1400, 0.96, 0.88),
            ("B", "B", 1800, 2100, 0.96, 0.93),
            ("C", "C", 1400, 1800, 0.97, 0.94),
        ]:
            vol = round(_lin(vol_s, vol_e, m, MONTHS))
            ot = _lin(ot_s, ot_e, m, MONTHS)
            ot = _perturb(ot, m, 0.015)
            delay = 0.5 + 3.5 * (1 - ot)
            rows.append((f"del_{m}_{f}_{t}", m, f, t, vol, round(ot, 4), round(delay, 2)))
    return rows


def generate_demand():
    rows = [("id", "month", "demand_index_a", "demand_index_b", "demand_index_c", "competitor_activity")]
    for m in range(MONTHS):
        da = _lin(0.65, 1.0, m, MONTHS)
        db = _lin(0.50, 0.55, m, MONTHS)
        dc = _lin(0.40, 0.70, m, MONTHS)
        comp = 0.2 + 0.6 / (1 + math.exp(-(m - 8) * 0.8))
        rows.append((f"dem_{m}", m, round(da, 4), round(db, 4), round(dc, 4), round(comp, 4)))
    return rows


def generate_events():
    rows = [("id", "month", "region", "event_type", "impact_value", "delay_days")]
    for i, (m, r, etype, impact, delay) in enumerate([
        (4, "B", "fleet_addition", 4.0, 0),
        (7, "A", "warehouse_expansion", 15.0, 90),
        (9, "C", "fleet_addition", 3.0, 30),
        (10, "A", "automation_upgrade", 8.0, 60),
    ]):
        rows.append((f"evt_{i}", m, r, etype, impact, delay))
    return rows


def generate_all():
    os.makedirs(MAPPINGS_DIR, exist_ok=True)
    print(f"Ontology: {ONTOLOGY_PATH}")

    rows = generate_warehouse()
    write_csv("warehouse_inventory.csv", rows)
    write_yaml("warehouse_inventory.yaml", "logi:WarehouseSnapshot", [
        ("month", "integer", "logi:month"),
        ("region", "string", "logi:region"),
        ("inventory_load", "float", "logi:inventoryLoad"),
        ("queue_hours", "float", "logi:queueHours"),
        ("throughput", "float", "logi:throughput"),
    ], "http://logistics-network.org/graphs/warehouse")

    rows = generate_fleet()
    write_csv("fleet_status.csv", rows)
    write_yaml("fleet_status.yaml", "logi:FleetSnapshot", [
        ("month", "integer", "logi:month"),
        ("region", "string", "logi:region"),
        ("vehicles_active", "integer", "logi:vehiclesActive"),
        ("utilization_pct", "float", "logi:fleetUtilization"),
        ("avg_trip_hours", "float", "logi:avgTripHours"),
    ], "http://logistics-network.org/graphs/fleet")

    rows = generate_deliveries()
    write_csv("delivery_performance.csv", rows)
    write_yaml("delivery_performance.yaml", "logi:DeliveryRecord", [
        ("month", "integer", "logi:month"),
        ("route_from", "string", "logi:region"),
        ("route_to", "string", "logi:region"),
        ("volume", "integer", "logi:deliveryVolume"),
        ("on_time_pct", "float", "logi:onTimeRate"),
        ("avg_delay_hours", "float", "logi:avgDelayHours"),
    ], "http://logistics-network.org/graphs/delivery")

    rows = generate_demand()
    write_csv("demand_forecast.csv", rows)
    write_yaml("demand_forecast.yaml", "logi:DemandSignal", [
        ("month", "integer", "logi:month"),
        ("demand_index_a", "float", "logi:demandIndex"),
        ("demand_index_b", "float", "logi:demandIndex"),
        ("demand_index_c", "float", "logi:demandIndex"),
        ("competitor_activity", "float", "logi:competitorActivity"),
    ], "http://logistics-network.org/graphs/demand")

    rows = generate_events()
    write_csv("infrastructure_events.csv", rows)
    write_yaml("infrastructure_events.yaml", "logi:InfrastructureEvent", [
        ("month", "integer", "logi:month"),
        ("region", "string", "logi:region"),
        ("event_type", "string", "logi:eventType"),
        ("impact_value", "float", "logi:impactValue"),
        ("delay_days", "integer", "logi:delayDays"),
    ], "http://logistics-network.org/graphs/events")


if __name__ == "__main__":
    generate_all()
    print("\nLogistics data generation complete.")
