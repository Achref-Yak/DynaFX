# QueueDef / DESEngine

**What they are:** `QueueDef` defines a DES queue with capacity and service time. `DESEngine` runs the event-driven simulation.

**When to use them:** When you need discrete event simulation — queues, resources, and event-driven processing.

**Quick start:**
```python
model.queue("checkout", capacity=20, service_time="3 + random()", servers=3)
model.resource("cashier", capacity=3, cost_per_unit=50)
model.event("arrival", rate="5", target_queue="checkout")
```

---

## Import

```python
from dynafx.dynamics import QueueDef, DESEngine, Queue, Resource, QueueStats, ResourceStats
```

---

## QueueDef

Defines a DES queue.

### Usage via SysdModel

```python
model.queue(
    name: str,                     # Queue name
    capacity: int = -1,            # Max length (-1 = unlimited)
    service_time: str = "",        # Service time expression
    arrival_rate: str = "",        # Arrival rate expression
    servers: int = 1,              # Number of parallel servers
    event_driven: bool = False,    # Event-driven vs time-sliced
)
```

### Examples

```python
# Basic queue
model.queue("checkout", capacity=20, service_time="3 + random()", servers=3)

# Unlimited queue with arrival rate
model.queue("assembly", capacity=-1, arrival_rate="10")

# Event-driven queue
model.queue("processing", service_time="5", event_driven=True)
```

---

## ResourceDef

Defines a DES resource (e.g., machine or worker pool).

### Usage via SysdModel

```python
model.resource(
    name: str,                     # Resource name
    capacity: int = 1,             # Number of units
    cost_per_unit: float = 0.0,    # Cost per unit time
)
```

### Examples

```python
model.resource("machine", capacity=5, cost_per_unit=100)
model.resource("worker", capacity=20, cost_per_unit=50)
```

---

## EventDef

Defines a DES event.

### Usage via SysdModel

```python
model.event(
    name: str,                     # Event name
    rate: str = "",                # Event rate expression
    target_queue: str = "",        # Queue to send entities to
    effects: list[str] = None,     # Effects to apply
)
```

### Examples

```python
model.event("arrival", rate="5", target_queue="checkout")
model.event("failure", rate="0.1", effects=["downtime += 1"])
```

---

## DESEngine

Runtime engine for event-driven simulation.

### Usage

The DESEngine is used internally by `SysdModel.simulate()` when queues/resources are defined.

```python
model = SysdModel(dt=0.5, t_span=(0, 100))
model.queue("checkout", capacity=20, service_time="3 + random()")
model.resource("cashier", capacity=3)

# DESEngine is created internally during simulate()
result = model.simulate()

# Access DES metrics
print(result.des_metrics_history)
```

---

## Common Patterns

### Pattern 1: Basic Queue

```python
model.queue("checkout", capacity=20, service_time="3 + random()", servers=3)
model.event("arrival", rate="5", target_queue="checkout")
```

### Pattern 2: With Resources

```python
model.queue("processing", capacity=10, service_time="5")
model.resource("machine", capacity=3, cost_per_unit=100)
```

### Pattern 3: Multi-Queue

```python
model.queue("receiving", capacity=50, arrival_rate="10")
model.queue("inspection", capacity=20, service_time="2")
model.queue("storage", capacity=100, service_time="1")
```

---

## See Also

- [`SysdModel`](SysdModel.md) — How to add queues to a model
- [`AgentDef`](AgentDef.md) — Agent-based modeling
- [Tutorial 4: Discrete Event Simulation](../../tutorials/04-discrete-event-simulation.md) — Full walkthrough
