"""Generate deterministic EPC enterprise CSV data."""

import csv
import os
import numpy as np

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)

REGIONS = ["europe", "na", "me", "asia"]
EPC_REGIONS = ["europe", "na", "me"]  # regions with projects/suppliers/workers
REGION_PORTS = {
    "europe": ["Rotterdam", "Antwerp", "Hamburg", "Valencia", "Piraeus", "Felixstowe"],
    "na": ["Tangier", "Casablanca", "Algiers", "Tunis", "Tripoli", "Alexandria"],
    "me": ["Dubai", "Dammam", "Muscat", "Aqaba", "Jeddah", "Doha"],
    "asia": ["Shanghai"],
}
REGION_CAPACITY_MW = {"europe": 180, "na": 140, "me": 100}  # ~420 MW total
REGION_BUDGET_PANEL_RATIO = {"europe": 155, "na": 140, "me": 125}  # $K/panel

MATERIALS = ["panels", "inverters", "racking", "cabling"]
ROLES = [
    ("installer", 0.60),
    ("electrician", 0.15),
    ("supervisor", 0.10),
    ("engineer", 0.10),
    ("safety", 0.05),
]
STATUSES = ["active", "at_risk", "delayed", "on_hold"]
SHIP_ROUTES = [
    ("asia_to_europe", "asia", "europe", 25),
    ("asia_to_na", "asia", "na", 20),
    ("asia_to_me", "asia", "me", 18),
    ("intra_europe", "europe", "europe", 5),
]


def pick(seq):
    return seq[rng.integers(len(seq))]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def trunc_norm(mean, std, lo, hi):
    return clamp(rng.normal(mean, std), lo, hi)


def generate_suppliers():
    suppliers = []
    for region in EPC_REGIONS:
        for i in range(42):
            sid = f"S_{region[:2].upper()}_{i+1:03d}"
            name_parts = [
                "Solar",
                "Green",
                "Eco",
                "Power",
                "Sun",
                "Energy",
                "Volt",
                "Watt",
                "Grid",
                "Panel",
                "Inno",
                "Nova",
                "Prime",
                "Core",
                "Peak",
                "Star",
            ]
            name = f"{pick(name_parts)} {pick(name_parts)} {pick(['Ltd', 'GmbH', 'Inc', 'BV', 'SA', 'Corp', 'LLC', 'SpA'])}"
            reliability = round(trunc_norm(0.82, 0.10, 0.60, 1.0), 3)
            volume = round(trunc_norm(300, 200, 30, 1200), 1)
            cost = round(trunc_norm(0.18, 0.06, 0.07, 0.35), 3)
            lead_time = int(round(trunc_norm(20, 10, 3, 50)))
            material = pick(MATERIALS)
            suppliers.append(
                {
                    "id": sid,
                    "name": name,
                    "region": region,
                    "material": material,
                    "reliability": reliability,
                    "volume_per_day": volume,
                    "cost_per_panel": cost,
                    "lead_time_days": lead_time,
                }
            )
    return suppliers


def generate_projects(suppliers):
    projects = []
    pid = 0
    supps_by_region = {r: [s for s in suppliers if s["region"] == r] for r in EPC_REGIONS}
    for region in EPC_REGIONS:
        target_mw = REGION_CAPACITY_MW[region]
        n = 12 if region == "me" else (19 if region == "europe" else 16)
        capacities = np.random.dirichlet(np.ones(n)) * target_mw
        for j in range(n):
            pid += 1
            name = pick(
                [
                    "Solar Farm",
                    "PV Plant",
                    "Solar Park",
                    "Solar Array",
                    "Solar Field",
                    "Solar Complex",
                    "Solar Hub",
                    "Solar Station",
                ]
            ) + f" {pick(['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Omicron', 'Pi', 'Rho', 'Sigma', 'Tau'])}"
            cap_mw = round(float(capacities[j]), 1)
            budget_k = int(cap_mw * REGION_BUDGET_PANEL_RATIO[region] * 2000)
            deadline = int(round(trunc_norm(365, 90, 200, 600)))
            status = pick(STATUSES)
            partner = pick(supps_by_region[region])
            projects.append(
                {
                    "id": f"P_{pid:03d}",
                    "name": name,
                    "region": region,
                    "capacity_mw": cap_mw,
                    "budget_k": budget_k,
                    "deadline_day": deadline,
                    "status": status,
                    "partner_id": partner["id"],
                }
            )
    return projects


def generate_ports():
    ports = []
    for region in EPC_REGIONS + ["asia"]:
        for i, name in enumerate(REGION_PORTS[region]):
            capacity = int(round(trunc_norm(2500, 1200, 300, 6000)))
            is_chokepoint = 1 if name == "Shanghai" else 0
            ports.append(
                {
                    "id": f"PORT_{name.upper()[:4]}",
                    "name": name + (" (Shanghai)" if name == "Shanghai" else ""),
                    "region": region,
                    "country": name,
                    "capacity_per_day": capacity,
                    "is_chokepoint": is_chokepoint,
                }
            )
    return ports


def collect_port_ids(ports):
    """Return list of port IDs per region."""
    by_region = {r: [] for r in set(REGIONS + EPC_REGIONS)}
    for p in ports:
        by_region[p["region"]].append(p["id"])
    return by_region


def generate_ships(ports):
    ships = []
    by_region = collect_port_ids(ports)
    all_port_ids = [p["id"] for p in ports]
    route_pool = []
    for route_name, src_reg, dst_reg, transit_days in SHIP_ROUTES:
        for _ in range(10 if route_name != "intra_europe" else 10):
            route_pool.append((route_name, src_reg, dst_reg, transit_days))
    rng.shuffle(route_pool)
    for i in range(min(40, len(route_pool))):
        route_name, src_reg, dst_reg, transit_days = route_pool[i]
        name = pick(
            [
                "MSC",
                "Maersk",
                "CMA CGM",
                "COSCO",
                "Evergreen",
                "Hapag",
                "ONE",
                "Yang Ming",
                "ZIM",
                "HMM",
            ]
        ) + f" {pick(['Apollo', 'Artemis', 'Atlas', 'Aurora', 'Borealis', 'Celeste', 'Comet', 'Cosmos', 'Crest', 'Crown'])}"
        capacity = int(round(trunc_norm(250, 150, 40, 600)))
        speed = round(trunc_norm(20, 3, 14, 27), 1)
        current_port = pick(all_port_ids)
        next_port = pick(all_port_ids)
        ships.append(
            {
                "id": f"SHIP_{i+1:03d}",
                "name": name,
                "route": route_name,
                "capacity_containers": capacity,
                "speed_knots": speed,
                "current_port_id": current_port,
                "next_port_id": next_port,
                "transit_days": transit_days,
            }
        )
    return ships


def generate_containers(ships, ports):
    containers = []
    port_map = {p["id"]: p for p in ports}
    contents_opts = ["panels", "inverters", "racking", "mixed"]
    status_opts = ["in_transit", "delivered", "delayed"]
    cid = 0
    for ship in ships:
        n = int(round(trunc_norm(17, 8, 2, 40)))
        for _ in range(n):
            cid += 1
            content = pick(contents_opts)
            qty = int(round(trunc_norm(300, 150, 20, 600)))
            eta = int(round(trunc_norm(30, 15, 2, 90)))
            status = pick(status_opts)
            origin = pick(
                [p["id"] for p in ports if p["region"] != "me"]
            )  # origin from non-ME
            dest = pick(
                [p["id"] for p in ports if p["region"] == "europe"]
            ) if "europe" in ship["route"] else pick(
                [p["id"] for p in ports if p["region"] == ship["route"].split("_")[-1]]
            )
            containers.append(
                {
                    "id": f"CTN_{cid:04d}",
                    "ship_id": ship["id"],
                    "origin_port_id": origin,
                    "dest_port_id": dest,
                    "contents": content,
                    "quantity": qty,
                    "eta_day": eta,
                    "status": status,
                }
            )
    return containers


def generate_warehouses():
    warehouses = []
    for region in EPC_REGIONS:
        for i in range(14):
            cap = int(round(trunc_norm(25000, 12000, 3000, 60000)))
            inv = int(round(trunc_norm(4000, 3000, 0, cap)))
            warehouses.append(
                {
                    "id": f"WH_{region[:2].upper()}_{i+1:02d}",
                    "name": f"{region.capitalize()} Warehouse {i+1}",
                    "region": region,
                    "capacity_panels": cap,
                    "inventory_panels": inv,
                }
            )
    return warehouses


def generate_workers(projects):
    workers = []
    wid = 0
    for region in EPC_REGIONS:
        region_projects = [p for p in projects if p["region"] == region]
        n_workers = 1850 // 3
        for _ in range(n_workers):
            wid += 1
            role_name = pick([r for r, _ in ROLES])
            skill = int(round(clamp(rng.normal(3.0, 1.0), 1, 5)))
            proj = pick(region_projects) if region_projects else None
            workers.append(
                {
                    "id": f"W_{wid:04d}",
                    "name": f"Worker_{wid}",
                    "role": role_name,
                    "region": region,
                    "project_id": proj["id"] if proj else "",
                    "skill_level": skill,
                }
            )
    return workers


def write_csv(filename, fieldnames, rows):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  {filename}: {len(rows)} rows")


def main():
    print("Generating EPC enterprise data (seed=42)...")
    suppliers = generate_suppliers()
    projects = generate_projects(suppliers)
    ports = generate_ports()
    ships = generate_ships(ports)
    containers = generate_containers(ships, ports)
    warehouses = generate_warehouses()
    workers = generate_workers(projects)

    write_csv(
        "epc_suppliers.csv",
        [
            "id",
            "name",
            "region",
            "material",
            "reliability",
            "volume_per_day",
            "cost_per_panel",
            "lead_time_days",
        ],
        suppliers,
    )
    write_csv(
        "epc_projects.csv",
        [
            "id",
            "name",
            "region",
            "capacity_mw",
            "budget_k",
            "deadline_day",
            "status",
            "partner_id",
        ],
        projects,
    )
    write_csv(
        "epc_ports.csv",
        [
            "id",
            "name",
            "region",
            "country",
            "capacity_per_day",
            "is_chokepoint",
        ],
        ports,
    )
    write_csv(
        "epc_ships.csv",
        [
            "id",
            "name",
            "route",
            "capacity_containers",
            "speed_knots",
            "current_port_id",
            "next_port_id",
            "transit_days",
        ],
        ships,
    )
    write_csv(
        "epc_containers.csv",
        [
            "id",
            "ship_id",
            "origin_port_id",
            "dest_port_id",
            "contents",
            "quantity",
            "eta_day",
            "status",
        ],
        containers,
    )
    write_csv(
        "epc_warehouses.csv",
        [
            "id",
            "name",
            "region",
            "capacity_panels",
            "inventory_panels",
        ],
        warehouses,
    )
    write_csv(
        "epc_workers.csv",
        [
            "id",
            "name",
            "role",
            "region",
            "project_id",
            "skill_level",
        ],
        workers,
    )

    total_mw = sum(p["capacity_mw"] for p in projects)
    total_suppliers = len(suppliers)
    by_material = {}
    for s in suppliers:
        by_material.setdefault(s["material"], 0)
        by_material[s["material"]] += 1
    chokepoints = [p for p in ports if p["is_chokepoint"]]
    print(f"\nSummary:")
    print(f"  Projects: {len(projects)} ({total_mw:.1f} MW total)")
    print(f"  Suppliers: {len(suppliers)} (by material: {by_material})")
    print(f"  Ports: {len(ports)} (chokepoints: {len(chokepoints)}: {[p['name'] for p in chokepoints]})")
    print(f"  Ships: {len(ships)}")
    print(f"  Containers: {len(containers)}")
    print(f"  Warehouses: {len(warehouses)}")
    print(f"  Workers: {len(workers)}")


if __name__ == "__main__":
    main()
