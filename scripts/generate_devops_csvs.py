"""Generate deterministic DevOps telemetry CSV data for the autoscaling cascade scenario.

Scenario: 120-minute window
  t=0-30:   Normal traffic (~100 req/s, 2 instances, CPU ~45%)
  t=30-50:  Gradual ramp to 150 req/s
  t=60:     Spike hits 300 req/s (3x), autoscaler triggers
  t=62:     Autoscaler detects → scale-out event (120s startup)
  t=80:     DB connection pool saturates, latency spikes
  t=82:     New instances come online (4→6 instances)
  t=85:     User retries amplify traffic (~350 req/s effective)
  t=100:    Traffic normalizes, instances slowly scale down
  t=105:    Scale-in begins (6→4 instances)
  t=120:    End (2 idle instances, cost waste detected)
"""

import csv
import os
import numpy as np

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)
MINUTES = 120
SECONDS_PER_MINUTE = 60


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def add_noise(base, pct: float):
    return round(base + base * rng.normal(0, pct / 3), 1)


def generate_metrics():
    rows = []
    instances = 2
    cpu = 45.0
    memory = 55.0
    latency = 120.0
    requests = 100
    queue = 0
    error_rate = 0.1
    throughput = 100

    for t in range(MINUTES):
        # Traffic phase logic
        if t < 30:
            target_requests = 100
        elif t < 50:
            target_requests = 100 + (t - 30) * 2.5  # ramp 100→150
        elif t < 60:
            target_requests = 150
        elif t < 80:
            target_requests = 300  # spike
        elif t < 85:
            target_requests = 300  # sustained
        elif t < 100:
            target_requests = 350  # retry amplification
        elif t < 110:
            target_requests = 200  # normalizing
        else:
            target_requests = 120  # near normal

        # Add noise
        requests = int(round(add_noise(target_requests, 8)))
        requests = max(requests, 10)

        # Instance management
        if t == 62 and instances < 8:
            instances += 2  # scale-out starts
        elif t == 82:
            instances += 2  # scale-out completes (+2 more already in progress)
        elif t == 105 and instances > 4:
            instances -= 1  # begin scale-in
        elif t == 112 and instances > 4:
            instances -= 1  # continue scale-in

        # CPU: proportional to requests/instance
        req_per_inst = requests / max(instances, 1)
        cpu_target = clamp(req_per_inst * 0.8, 5, 98)
        if req_per_inst > 80:
            cpu_target = min(98, cpu_target)
        cpu = round(clamp(add_noise(cpu_target, 5), 1, 99), 1)

        # Memory: climbs with queue pressure
        mem_target = 55.0 + (queue / 10)
        mem_target = min(mem_target, 92)
        memory = round(clamp(add_noise(mem_target, 3), 10, 99), 1)

        # Latency: low when queue is small, spikes with queue
        if queue < 10:
            lat_target = 100 + queue * 5
        elif queue < 50:
            lat_target = 150 + queue * 8
        else:
            lat_target = 400 + queue * 6
        latency = round(clamp(add_noise(lat_target, 10), 20, 5000), 1)

        # Queue: accumulates when demand > throughput capacity
        processing_capacity = instances * 80
        if requests > processing_capacity:
            queue += requests - processing_capacity
        else:
            queue = max(0, queue - (processing_capacity - requests) // 2)
        queue = int(clamp(queue, 0, 500))

        # Throughput: min(requests, processing_capacity)
        throughput_val = min(requests, processing_capacity) + (
            requests - processing_capacity if requests > processing_capacity else 0
        )
        throughput_val = max(throughput_val, 0)
        throughput = int(round(add_noise(throughput_val, 3)))

        # Error rate: climbs with latency
        if latency > 2000:
            err_target = 5.0 + (latency - 2000) * 0.005
        elif latency > 1000:
            err_target = 2.0
        else:
            err_target = 0.1
        error_rate = round(clamp(add_noise(err_target, 15), 0.0, 25.0), 1)

        rows.append({
            "timestamp": str(t),
            "service": "app",
            "cpu": str(cpu),
            "memory": str(memory),
            "latency": str(latency),
            "requests": str(requests),
            "throughput": str(throughput),
            "queue_length": str(queue),
            "instances": str(instances),
            "error_rate": str(error_rate),
        })

    return rows


def generate_events():
    return [
        {"timestamp": "60", "event_type": "request_spike", "service": "app",
         "value": "3x", "severity": "warning"},
        {"timestamp": "62", "event_type": "scale_out", "service": "app",
         "value": "+2", "severity": "info"},
        {"timestamp": "80", "event_type": "db_slowdown", "service": "db",
         "value": "2x_latency", "severity": "critical"},
        {"timestamp": "82", "event_type": "scale_out", "service": "app",
         "value": "+2", "severity": "info"},
        {"timestamp": "85", "event_type": "retry_storm", "service": "app",
         "value": "1.2x", "severity": "warning"},
        {"timestamp": "100", "event_type": "traffic_normalized", "service": "app",
         "value": "1x", "severity": "info"},
        {"timestamp": "105", "event_type": "scale_in", "service": "app",
         "value": "-1", "severity": "info"},
        {"timestamp": "110", "event_type": "idle_detected", "service": "app",
         "value": "2_idle", "severity": "warning"},
        {"timestamp": "112", "event_type": "scale_in", "service": "app",
         "value": "-1", "severity": "info"},
    ]


def generate_infra():
    return [
        {"service": "app", "instance_type": "standard",
         "cpu_cores": "4", "memory_gb": "16",
         "cost_per_hour": "0.50", "max_capacity": "1000",
         "startup_time_sec": "120"},
        {"service": "db", "instance_type": "large",
         "cpu_cores": "8", "memory_gb": "32",
         "cost_per_hour": "1.20", "max_capacity": "2000",
         "startup_time_sec": "180"},
        {"service": "cache", "instance_type": "small",
         "cpu_cores": "2", "memory_gb": "8",
         "cost_per_hour": "0.20", "max_capacity": "500",
         "startup_time_sec": "60"},
    ]


def write_csv(filename, fieldnames, rows):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  {filename}: {len(rows)} rows")


def main():
    print("Generating DevOps telemetry data (seed=42)...")
    metrics = generate_metrics()
    events = generate_events()
    infra = generate_infra()

    write_csv("devops_metrics.csv", [
        "timestamp", "service", "cpu", "memory", "latency",
        "requests", "throughput", "queue_length", "instances", "error_rate",
    ], metrics)
    write_csv("devops_events.csv", [
        "timestamp", "event_type", "service", "value", "severity",
    ], events)
    write_csv("devops_infra.csv", [
        "service", "instance_type", "cpu_cores", "memory_gb",
        "cost_per_hour", "max_capacity", "startup_time_sec",
    ], infra)

    print(f"\nSummary:")
    print(f"  Metrics: {len(metrics)} rows (t=0..{MINUTES-1})")
    print(f"  Events: {len(events)} operational events")
    print(f"  Infra: {len(infra)} services")


if __name__ == "__main__":
    main()
