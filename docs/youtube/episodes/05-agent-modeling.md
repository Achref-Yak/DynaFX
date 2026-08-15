# Video 5: Agents with Brains (Behavioral Modeling)

**What it does:** Create agents with properties, rules, and strategy switching.

**Duration:** 15-18 minutes

**Hook:** "What if your customers don't all behave the same? What if they have strategies, budgets, and moods?"

---

## Script Outline

### 0:00 - Hook (30 sec)
- "System Dynamics models averages. Agent-Based Modeling models individuals. Today: agents with brains."

### 0:30 - When ABM Beats SD (2 min)
- Heterogeneity matters (customers differ)
- Decision-making matters (agents choose)
- Adaptive behavior (strategies change)

### 2:30 - Agent Properties (3 min)
```python
with model.agent("Customer", 100) as a:
    a.prop("budget", 100, min_val=0, max_val=1000)
    a.prop("satisfaction", 1.0, min_val=0, max_val=1)
```
- State variables with constraints
- Each agent instance has its own values

### 5:30 - Behavioral Rules (4 min)
```python
    a.rule("buy", "budget > 10 AND inventory > 0",
           ["budget -= 10", "satisfaction += 0.05"])
    a.rule("churn", "satisfaction < 0.3",
           ["SWITCH_STRATEGY('inactive')"])
```
- Condition: boolean expression
- Effects: list of assignments
- Strategy switching: adaptive behavior

### 9:30 - Strategies (4 min)
```python
    a.strategy("active").rule("buy", "budget > 10", ["budget -= 10"])
    a.strategy("inactive").rule("leave", "always", ["budget = 0"])
    a.meta_rule("recover", "satisfaction > 0.5", ["SWITCH_STRATEGY('active')"])
```
- Named rule sets
- Meta-rules: always evaluated regardless of active strategy

### 13:30 - Agent Metrics (2 min)
```python
result = model.simulate()
print(result.abm_metrics_history)
```
- Aggregated metrics per step
- "How many agents bought? How many churned?"

### 15:30 - Challenge (1 min)
- "Build a market with 50 buyers. How many survive 100 steps?"

### 16:30 - Outro (30 sec)
- "Next time: queues and resources."

---

## Code Samples

### Basic Agents
```python
from dynafx.dynamics import SysdModel

model = SysdModel(dt=0.5, t_span=(0, 100))

with model.stock("Inventory", 500) as s:
    s.inflow("production", "100")
    s.outflow("sales", "total_spend")

model.aux("total_spend", "SUM(agent_budget)")

with model.agent("Customer", 100) as a:
    a.prop("budget", 100, min_val=0)
    a.prop("satisfaction", 1.0, min_val=0, max_val=1)
    a.rule("buy", "budget > 10 AND Inventory > 0",
           ["budget -= 10", "satisfaction += 0.05"])
    a.rule("churn", "satisfaction < 0.3",
           ["SWITCH_STRATEGY('inactive')"])
    a.strategy("inactive").rule("leave", "always", ["budget = 0"])

result = model.simulate()
```

### Strategy Switching
```python
with model.agent("Worker", 50) as a:
    a.prop("stress", 0.0, min_val=0, max_val=1)
    a.prop("output", 0.0)
    a.strategy("normal").rule("work", "stress < 0.7", ["output += 10"])
    a.strategy("crisis").rule("rest", "stress > 0.7", ["stress -= 0.2", "output = 0"])
    a.meta_rule("switch_to_crisis", "stress > 0.8", ["SWITCH_STRATEGY('crisis')"])
    a.meta_rule("switch_to_normal", "stress < 0.3", ["SWITCH_STRATEGY('normal')"])
```

### Message Passing
```python
with model.agent("Sender", 10) as a:
    a.rule("send", "always", ["SEND('alert', 1.0)"])

with model.agent("Receiver", 10) as a:
    a.rule("receive", "inbox > 0", ["satisfaction += 0.1"])
```

---

## Thumbnail
- Left: Agent icons with different colors
- Right: Strategy switching diagram
- Bottom: "Agents with strategies"

---

## Description
```
Build agents with properties, rules, and strategy switching in DynaFX.

Docs: https://achref-yak.github.io/DynaFX/api/dynamics/AgentDef/
GitHub: https://github.com/Achref-Yak/DynaFX
```

---

## Pin Comment
```
Challenge #5: Build a market with 50 buyers. Each has:
- budget (50-200)
- satisfaction (0-1)
Rules: buy when budget > 10, churn when satisfaction < 0.3
How many survive 100 steps?
```
