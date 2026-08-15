# Video 6: Queues, Resources, and Event-Driven Simulation

**What it does:** Model operational systems with queues, resources, and events.

**Duration:** 12-15 minutes

**Hook:** "When you need to model a factory floor, a hospital, or a supply chain - you need queues and resources."

---

## Script Outline

### 0:00 - Hook (30 sec)
- "Real systems have waiting lines, limited capacity, and random arrivals. Let's model that."

### 0:30 - Queues (3 min)
```python
model.queue("checkout", capacity=20, service_time="3 + random()", servers=3)
```
- Capacity limits
- Service time expressions
- Multiple servers

### 3:30 - Resources (2 min)
```python
model.resource("barista", capacity=3, cost_per_unit=15)
```
- Capacity constraints
- Cost tracking

### 5:30 - Events (2 min)
```python
model.event("arrival", rate="5", target_queue="checkout")
```
- Scheduled arrivals
- Random events

### 7:30 - DES Metrics (3 min)
```python
result = model.simulate()
stats = result.des_engine.get_all_stats()
print(stats["checkout"]["avg_length"])
print(stats["checkout"]["utilization"])
```
- Queue length over time
- Utilization tracking

### 10:30 - Integration with SD (2 min)
```python
model.aux("revenue", "checkout_length * price_per_item")
```
- DES metrics feed into SD auxiliaries
- "The queue length affects revenue."

### 12:30 - Challenge (1 min)
- "Model a coffee shop with 3 baristas. What's the average wait time?"

### 13:30 - Outro (30 sec)
- "Next time: knowledge graphs."

---

## Code Samples

### Basic Queue
```python
from dynafx.dynamics import SysdModel

model = SysdModel(dt=0.5, t_span=(0, 100))
model.queue("checkout", capacity=20, service_time="3 + random()", servers=3)
model.event("arrival", rate="5", target_queue="checkout")

result = model.simulate()
stats = result.des_engine.get_all_stats()
print(stats["checkout"])
```

### With Resources
```python
model = SysdModel(dt=0.5, t_span=(0, 100))
model.queue("ordering", capacity=20, service_time="1 + random()", servers=1)
model.queue("pickup", capacity=10, service_time="2 + random()", servers=3)
model.resource("barista", capacity=3, cost_per_unit=15)
model.event("arrival", rate="5", target_queue="ordering")

result = model.simulate()
```

### DES Metrics in SD
```python
model = SysdModel(dt=0.5, t_span=(0, 100))
model.queue("processing", capacity=50, service_time="2", servers=5)
model.aux("backlog_cost", "processing_length * 10")
model.aux("utilization_cost", "processing_utilization * 100")

with model.stock("TotalCost", 0) as s:
    s.inflow("cost_rate", "backlog_cost + utilization_cost")
```

---

## Thumbnail
- Left: Queue diagram
- Right: Utilization chart
- Bottom: "Queues and resources"

---

## Description
```
Model queues, resources, and events in DynaFX - discrete event simulation.

Docs: https://achref-yak.github.io/DynaFX/api/dynamics/DESEngine/
GitHub: https://github.com/Achref-Yak/DynaFX
```

---

## Pin Comment
```
Challenge #6: Model a hospital ER with:
- Triage queue (capacity 30)
- Treatment rooms (5)
- Doctors (3)
- Arrival rate: 8 per hour
What's the average wait time?
```
