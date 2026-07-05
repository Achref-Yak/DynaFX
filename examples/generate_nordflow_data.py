"""Generate deterministic NordFlow logistics historical CSVs + ontology TTL."""

import csv, math, os
from pathlib import Path

DATA_DIR = Path("data/nordflow")
MAPPINGS_DIR = Path("data/mappings")
ONTO_PATH = Path("data/nordflow-ontology.ttl")
MONTHS = 12
SEED = 42
NS_URI = "http://nordflow-logistics.org/"
BASE_IRI = "http://nordflow-logistics.org/"
os.makedirs(DATA_DIR, exist_ok=True)

REGIONS = ["North", "South", "East", "West"]

def _perturb(base, month, amplitude=0.015):
    return base * (1 + amplitude * math.sin(2 * math.pi * month / 12 + 0.5 * SEED))

def write_csv(filename, rows):
    path = DATA_DIR / filename
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  {path} ({len(rows) - 1} data rows)")

def generate_warehouses():
    rows = [("id", "month", "region", "inventory_load", "queue_hours", "throughput")]
    for m in range(MONTHS):
        for ri, r in enumerate(REGIONS):
            inv_starts = [0.55, 0.42, 0.38, 0.45]
            inv_ends = [0.78, 0.52, 0.46, 0.56]
            q_starts = [1.2, 0.6, 0.4, 0.7]
            q_ends = [3.8, 1.1, 0.8, 1.8]
            t_starts = [420, 220, 170, 280]
            t_ends = [510, 250, 200, 310]
            t = m / max(1, MONTHS - 1)
            inv = (inv_starts[ri] + (inv_ends[ri] - inv_starts[ri]) * t) * _perturb(1, m)
            q = (q_starts[ri] + (q_ends[ri] - q_starts[ri]) * t) * _perturb(1, m, 0.03)
            tp = (t_starts[ri] + (t_ends[ri] - t_starts[ri]) * t) * _perturb(1, m, 0.01)
            rows.append((f"wh_{r.lower()}_{m}", m, r, round(inv, 4), round(q, 2), round(tp, 1)))
    return rows

def generate_fleet():
    rows = [("id", "month", "region", "vehicles_active", "fleet_utilization", "avg_trip_hours")]
    fleet_by_reg = {"North": 60, "South": 40, "East": 35, "West": 45}
    for m in range(MONTHS):
        for r in REGIONS:
            fc = fleet_by_reg[r]
            util_starts = {"North": 0.82, "South": 0.65, "East": 0.60, "West": 0.70}
            util_ends = {"North": 0.93, "South": 0.72, "East": 0.68, "West": 0.78}
            trip_starts = {"North": 3.2, "South": 2.8, "East": 2.5, "West": 3.0}
            trip_ends = {"North": 4.2, "South": 3.1, "East": 2.8, "West": 3.4}
            t = m / max(1, MONTHS - 1)
            util = (util_starts[r] + (util_ends[r] - util_starts[r]) * t) * _perturb(1, m, 0.01)
            trip = (trip_starts[r] + (trip_ends[r] - trip_starts[r]) * t) * _perturb(1, m, 0.02)
            rows.append((f"fl_{r.lower()}_{m}", m, r, fc, round(util, 4), round(trip, 2)))
    return rows

def generate_delivery():
    rows = [("id", "month", "region", "delivery_volume", "on_time_rate", "avg_delay_hours")]
    vol_by_reg = {"North": 4200, "South": 1800, "East": 1350, "West": 1650}
    ot_starts = {"North": 0.94, "South": 0.97, "East": 0.98, "West": 0.96}
    ot_ends = {"North": 0.87, "South": 0.95, "East": 0.96, "West": 0.94}
    delay_starts = {"North": 0.4, "South": 0.2, "East": 0.15, "West": 0.25}
    delay_ends = {"North": 0.8, "South": 0.3, "East": 0.25, "West": 0.4}
    for m in range(MONTHS):
        for r in REGIONS:
            t = m / max(1, MONTHS - 1)
            ot = (ot_starts[r] + (ot_ends[r] - ot_starts[r]) * t) * _perturb(1, m, 0.005)
            dv = vol_by_reg[r] * _perturb(1, m, 0.02)
            dl = (delay_starts[r] + (delay_ends[r] - delay_starts[r]) * t) * _perturb(1, m, 0.03)
            rows.append((f"dv_{r.lower()}_{m}", m, r, int(dv), round(ot, 4), round(dl, 2)))
    return rows

def generate_demand():
    rows = [("id", "month", "region", "demand_index", "competitor_activity")]
    di_starts = {"North": 1.00, "South": 0.90, "East": 0.85, "West": 0.92}
    di_ends = {"North": 1.18, "South": 0.93, "East": 0.92, "West": 0.97}
    comp_starts = {"North": 0.30, "South": 0.20, "East": 0.15, "West": 0.25}
    comp_ends = {"North": 0.45, "South": 0.25, "East": 0.20, "West": 0.30}
    for m in range(MONTHS):
        for r in REGIONS:
            t = m / max(1, MONTHS - 1)
            di = (di_starts[r] + (di_ends[r] - di_starts[r]) * t) * _perturb(1, m, 0.008)
            ca = (comp_starts[r] + (comp_ends[r] - comp_starts[r]) * t) * _perturb(1, m, 0.02)
            rows.append((f"dm_{r.lower()}_{m}", m, r, round(di, 4), round(ca, 4)))
    return rows

def generate_events():
    rows = [("id", "month", "region", "event_type", "impact_value", "delay_days")]
    events = [
        ("ev_north_exp_1", 2, "North", "warehouse_expansion", 0.15, 45),
        ("ev_south_fleet_1", 4, "South", "fleet_addition", 8, 0),
        ("ev_east_road_1", 6, "East", "road_closure", 0.0, 14),
        ("ev_west_auto_1", 8, "West", "automation_upgrade", 0.10, 30),
        ("ev_north_staff_1", 10, "North", "staff_shortage", 0.0, 7),
    ]
    rows.extend(events)
    return rows

def generate_all():
    print("Generating NordFlow data...")
    wh = generate_warehouses()
    write_csv("warehouse_inventory.csv", wh)
    fl = generate_fleet()
    write_csv("fleet_status.csv", fl)
    dv = generate_delivery()
    write_csv("delivery_performance.csv", dv)
    dm = generate_demand()
    write_csv("demand_forecast.csv", dm)
    ev = generate_events()
    write_csv("infrastructure_events.csv", ev)
    total = (len(wh) - 1) + (len(fl) - 1) + (len(dv) - 1) + (len(dm) - 1) + (len(ev) - 1)
    print(f"  Total: {total} data rows across 5 CSVs")
    print("Done.")

if __name__ == "__main__":
    generate_all()
