"""Generate deterministic ISP broadband historical CSVs + ontology TTL."""

import csv
import os
from pathlib import Path

DATA_DIR = Path("data/isp_broadband")
MAPPINGS_DIR = Path("data/mappings")
ONTOLOGY_PATH = Path("data/isp-ontology.ttl")
MONTHS = 24
SEED = 42

os.makedirs(DATA_DIR, exist_ok=True)


def _lin(start, end, t, total):
    return start + (end - start) * t / max(1, total - 1)


def _perturb(base, month, amplitude=0.02):
    import math
    return base * (1 + amplitude * math.sin(2 * math.pi * month / 12 + 0.5 * SEED))


def generate_subscribers():
    rows = [("id", "month", "region", "subscriber_count")]
    for m in range(MONTHS):
        for r, start, end in [("A", 28000, 38000), ("B", 30000, 42000), ("C", 28000, 35000)]:
            val = round(_lin(start, end, m, MONTHS) * (1 + 0.01 * (m / MONTHS - 0.5)))
            rows.append((f"sub_{m}_{r}", m, r, val))
    return rows


def generate_churn():
    rows = [("id", "month", "region", "churn_rate")]
    for m in range(MONTHS):
        for r, start, end in [
            ("A", 0.00018, 0.00025),
            ("B", 0.00012, 0.00030),
            ("C", 0.00015, 0.00020),
        ]:
            if r == "B" and m < 18:
                val = _lin(start, 0.00018, m, 18)
            elif r == "B":
                val = _lin(0.00018, end, m - 18, MONTHS - 18)
            else:
                val = _lin(start, end, m, MONTHS)
            val = _perturb(val, m, 0.03)
            rows.append((f"churn_{m}_{r}", m, r, round(val, 8)))
    return rows


def generate_qos():
    rows = [("id", "month", "region", "qos_score", "avg_utilization")]
    for m in range(MONTHS):
        for r, start_qos, end_qos, start_util, end_util in [
            ("A", 92, 73, 0.55, 0.78),
            ("B", 95, 55, 0.50, 0.88),
            ("C", 93, 78, 0.52, 0.72),
        ]:
            qos = _lin(start_qos, end_qos, m, MONTHS)
            util = _lin(start_util, end_util, m, MONTHS)
            qos = _perturb(qos, m, 0.02)
            util = _perturb(util, m, 0.03)
            rows.append((f"qos_{m}_{r}", m, r, round(qos, 1), round(util, 6)))
    return rows


def generate_capacity():
    rows = [("id", "month", "region", "units_added", "delay_days")]
    events = [
        (10, "B", 80, 120),
        (16, "A", 50, 90),
        (20, "C", 40, 60),
    ]
    for i, (m, r, units, delay) in enumerate(events):
        rows.append((f"cap_{i}", m, r, units, delay))
    return rows


def generate_indicators():
    rows = [("id", "month", "building_permits", "competitor_active", "marketing_spend")]
    for m in range(MONTHS):
        bp = 0.3 if m < 6 else min(1.0, 0.3 + (m - 6) * 0.1)
        bp = max(0, min(1, bp - (m - 14) * 0.05 if m > 14 else bp))
        comp = 1 if m >= 16 else 0
        mkt = 0.2 + 0.8 / (1 + 2.7 ** (-(m - 20) * 0.5)) if m >= 18 else 0.1
        rows.append((f"ind_{m}", m, round(bp, 4), comp, round(mkt, 4)))
    return rows


def write_csv(filename, rows):
    path = DATA_DIR / filename
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"  {path} ({len(rows) - 1} data rows)")


def write_yaml_mapping(filename, entity_class, columns, target_graph, id_prefix="isp:"):
    path = MAPPINGS_DIR / filename
    lines = [
        "prefixes:",
        '  isp: "http://isp-broadband.org/"',
        '  xsd: "http://www.w3.org/2001/XMLSchema#"',
        '  rdf: "http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
        f'csv: "isp_broadband/{filename.replace(".yaml", ".csv")}"',
        f'target_graph: "{target_graph}"',
        "entity:",
        f"  class: {entity_class}",
        '  id_column: "id"',
        f'  id_prefix: "{id_prefix}"',
        "columns:",
    ]
    for col_name, col_type, predicate in columns:
        lines.append(f"  {col_name}:")
        lines.append(f"    predicate: {predicate}")
        lines.append(f"    type: {col_type}")
    content = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(content)
    print(f"  {path}")


def generate_all():
    os.makedirs(MAPPINGS_DIR, exist_ok=True)

    # ontology already written
    print(f"Ontology: {ONTOLOGY_PATH}")

    # --- Subscribers ---
    rows = generate_subscribers()
    write_csv("isp_subscribers.csv", rows)
    write_yaml_mapping(
        "isp_subscribers.yaml",
        "isp:SubscriberRecord",
        [
            ("month", "integer", "isp:month"),
            ("region", "string", "isp:region"),
            ("subscriber_count", "integer", "isp:subscriberCount"),
        ],
        "http://isp-broadband.org/graphs/subscribers"
    )

    # --- Churn ---
    rows = generate_churn()
    write_csv("isp_churn_rate.csv", rows)
    write_yaml_mapping(
        "isp_churn_rate.yaml",
        "isp:ChurnRecord",
        [
            ("month", "integer", "isp:month"),
            ("region", "string", "isp:region"),
            ("churn_rate", "float", "isp:churnRate"),
        ],
        "http://isp-broadband.org/graphs/churn"
    )

    # --- QoS ---
    rows = generate_qos()
    write_csv("isp_qos_score.csv", rows)
    write_yaml_mapping(
        "isp_qos_score.yaml",
        "isp:QoSRecord",
        [
            ("month", "integer", "isp:month"),
            ("region", "string", "isp:region"),
            ("qos_score", "float", "isp:qosScore"),
            ("avg_utilization", "float", "isp:avgUtilization"),
        ],
        "http://isp-broadband.org/graphs/qos"
    )

    # --- Capacity events ---
    rows = generate_capacity()
    write_csv("isp_capacity_events.csv", rows)
    write_yaml_mapping(
        "isp_capacity_events.yaml",
        "isp:CapacityEvent",
        [
            ("month", "integer", "isp:month"),
            ("region", "string", "isp:region"),
            ("units_added", "integer", "isp:unitsAdded"),
            ("delay_days", "integer", "isp:delayDays"),
        ],
        "http://isp-broadband.org/graphs/capacity"
    )

    # --- Leading indicators ---
    rows = generate_indicators()
    write_csv("isp_leading_indicators.csv", rows)
    write_yaml_mapping(
        "isp_leading_indicators.yaml",
        "isp:LeadingIndicator",
        [
            ("month", "integer", "isp:month"),
            ("building_permits", "float", "isp:buildingPermits"),
            ("competitor_active", "integer", "isp:competitorActive"),
            ("marketing_spend", "float", "isp:marketingSpend"),
        ],
        "http://isp-broadband.org/graphs/indicators"
    )


if __name__ == "__main__":
    generate_all()
    print("\nISP data generation complete.")
