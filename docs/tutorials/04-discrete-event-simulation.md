# Tutorial 4 — Discrete Event Simulation

Discrete Event Simulation (DES) models **queues**, **resources**, and
**events** — the operational, transactional layer of a system: orders arriving,
being processed, waiting on capacity. Like ABM, DES coexists with SD in the
same model.

## Queues

A queue holds waiting items; arrivals are driven by an `arrival_rate`
expression (constant or a function of the SD state).

```sysd
ServiceDesk
dt 1
from 0 to 20

queue "Support": capacity 50, service_time 2.0, servers 3
  arrival_rate 5
```

```python
from dynafx.dynamics import SysdModel

model = SysdModel("ServiceDesk")
model.dt = 1.0
model.t_span = (0, 20)
model.queue("Support", capacity=50, service_time="2.0",
            servers=3, arrival_rate="5")

result = model.simulate()
stats = result.des_engine.get_all_stats()
print(stats["Support"]["avg_length"])     # average queue length
print(stats["Support"]["utilization"])    # fraction of time non-empty
print(stats["Support"]["total_arrivals"])
```

## Resources

Resources have capacity and optional cost. Queues and resources share names in
DES stats, so inspect the dict keys.

```python
model = SysdModel("Desk")
model.dt = 1.0
model.t_span = (0, 20)
model.queue("Support", capacity=50, service_time="2.0", servers=3, arrival_rate="5")
model.resource("Agents", capacity=5, cost_per_unit=1.0)

result = model.simulate()
stats = result.des_engine.get_all_stats()
print(sorted(stats.keys()))      # ['Agents', 'Support']
```

## DES metrics in aux expressions

Queue metrics are exposed to SD auxes with the naming
`{queue}_{metric}` — e.g. `Support_length`, `Support_departed`. This lets
financial or operational auxes react to congestion:

```python
model = SysdModel("monitor")
model.dt = 1.0
model.t_span = (0, 10)
model.queue("Orders", capacity=-1, service_time="1.0", servers=2, arrival_rate="10")
model.aux("watch", "Orders_length")

result = model.simulate()
print(max(result.aux_values["watch"]))    # peak queue length > 0
```

## Combined SD + DES + ABM

All three paradigms share one state dict. This model has a stock, a queue, and
agents, all evolving together:

```python
model = parse_sysd("""
Clinic
dt 1
from 0 to 5

stock "population": 1000
  + "birth": 10

queue "Clinic": capacity 5
resource "Doctor": capacity 2

agent "Patient": 3
  property "healthy": 1
  rule "check": always
    healthy += 1
""")

result = model.simulate()
print(result.values["population"][-1])          # SD: > 1000
print(result.des_engine is not None)            # DES present
print(len(result.abm_engine.instances))         # ABM: 3 patients
```

## What's next

- Store what the system *knows* in [Knowledge Graphs](05-knowledge-graph.md).
- Drive DES arrival rates from KB queries in [Closed-Loop Simulation](07-closed-loop-simulation.md).
