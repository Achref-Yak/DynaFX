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

model = SysdModel("ServiceDesk", dt=1.0, t_span=(0, 20))
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
model = SysdModel("Desk", dt=1.0, t_span=(0, 20))
model.queue("Support", capacity=50, service_time="2.0", servers=3, arrival_rate="5")
model.resource("Agents", capacity=5, cost_per_unit=1.0)

result = model.simulate()
stats = result.des_engine.get_all_stats()
print(sorted(stats.keys()))      # ['Agents', 'Support']
```

## Routing between queues

Use `model.route(from_queue, condition, to_queue)` to route entities that
complete service in one queue into another. The condition has `entity` (or `e`),
`t`, `state`, `len`, `abs`, `min`, `max`, and all registered builtins in scope:

```python
model = SysdModel("Desk", dt=0.5, t_span=(0, 20))
model.queue("triage", capacity=-1, service_time="1.0", arrival_rate="4")
model.queue("intensive", capacity=-1, service_time="4.0")
model.queue("routine", capacity=-1, service_time="2.0")

model.route("triage", "entity.get('priority', 0) >= 3", "intensive")
model.route("triage", "True", "routine")            # fallback: everything else

result = model.simulate()
print(result.des_engine.queue_stats("intensive").total_arrivals)
```

Rules are evaluated in registration order; the first match wins. A routed entity
is re-enqueued into the target queue (an arrival there) rather than departing
the system. Route conditions are also available in the `.sysd` text DSL:

```sysd
queue "triage": capacity -1, service_time 1.0
  arrival_rate 4
  route entity.priority >= 3 -> "intensive"
```

## Ordering disciplines

Set the `discipline` when declaring a queue: `FIFO` (default), `SPT`
(shortest processing time), `EDD` (earliest due date), or `PRIORITY`
(entity `priority` field, lowest first):

## DES metrics in aux expressions

Queue metrics are exposed to SD auxes with the naming
`{queue}_{metric}` — e.g. `Support_length`, `Support_departed`. This lets
financial or operational auxes react to congestion:

```python
model = SysdModel("monitor", dt=1.0, t_span=(0, 10))
model.queue("Orders", capacity=-1, service_time="1.0", servers=2, arrival_rate="10")
model.aux("watch", "Orders_length")

result = model.simulate()
print(max(result.aux_values["watch"]))    # peak queue length > 0
```

## Reading aligned time series: `result.series(name)`

`series()` is the canonical way to pull a time series from a result, regardless of
whether the quantity is a stock, an aux, a DES metric (`{queue}_{metric}`), or an
ABM metric (`{AgentType}_{prop}_{agg}`). It always returns an aligned
`(times, values)` pair:

```python
model = SysdModel("monitor", dt=1.0, t_span=(0, 10))
model.queue("Support", capacity=50, service_time="2.0", servers=3, arrival_rate="5")
model.aux("congestion", "Support_length")

result = model.simulate()

t, length = result.series("Support_length")  # DES metric
t, watch  = result.series("congestion")       # aux
print(max(length))     # peak queue depth
print(len(t) == len(length))  # True — aligned with result.times
```

Behind the scenes `series()` skips the DES seed step, fills sparse
`_departed`/`_arrivals` keys with `0`, and raises `KeyError` listing the
available names for typos.

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

## Typed DES statistics

Installed queues and resources accumulate statistics. Pull them per-instance
with dedicated typed accessors:

```python
result = model.simulate()

q = result.queue_stats("Clinic")       # QueueStats — queue fields only
r = result.resource_stats("Doctor")    # ResourceStats — resource fields only

print(q.avg_length, q.max_length, q.avg_wait)
print(r.utilization, r.total_granted)
```

`result.stats(name)` dispatches on the name and never mixes kinds: it returns
queue stats for a queue name and resource stats for a resource name, raising
`KeyError` (listing the valid names) otherwise. `Engine.get_all_stats()`
returns every entry's `summary()` dict, each tagged with a `kind` key:

```python
for name, s in result.des_engine.get_all_stats().items():
    print(name, s["kind"])             # "queue" or "resource"
```

## What's next

- Store what the system *knows* in [Knowledge Graphs](05-knowledge-graph.md).
- Drive DES arrival rates from KB queries in [Closed-Loop Simulation](07-closed-loop-simulation.md).
