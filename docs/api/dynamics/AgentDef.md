# AgentDef / ABMEngine

**What they are:** `AgentDef` defines an agent type with properties and rules. `ABMEngine` runs the agent simulation.

**When to use them:** When you need agent-based modeling — heterogeneous actors with behaviors, strategies, and interactions.

**Quick start:**
```python
with model.agent("Customer", 100) as a:
    a.prop("satisfaction", 1.0, min_val=0, max_val=1)
    a.rule("buy", "budget > 10", ["budget -= 10"])
    a.strategy("crisis").rule("rest", "stress > 0.7", ["stress -= 0.2"])
```

---

## Import

```python
from dynafx.dynamics import AgentDef, ABMEngine, AgentInstance, Message
```

---

## AgentDef

Defines an agent type with properties and behavioral rules.

### Constructor

```python
AgentDef(
    name: str,                              # Agent type name
    count: int = 1,                         # Number of instances
    properties: list[AgentPropDef] = [],    # Agent properties
    rules: list[AgentRuleDef] = [],         # Behavioral rules
    strategies: list[AgentStrategy] = [],   # Named rule sets
    meta_rules: list[AgentRuleDef] = [],    # Always-evaluated rules
    network_type: str = "none",             # Network topology
)
```

### Usage via Context Manager

```python
with model.agent("Customer", 100) as a:
    # Properties
    a.prop("satisfaction", 1.0, min_val=0, max_val=1)
    a.prop("budget", 100, min_val=0)

    # Rules
    a.rule("buy", "budget > 10", ["budget -= 10", "satisfaction += 0.1"])
    a.rule("churn", "satisfaction < 0.3", ["churn_risk += 0.1"])

    # Strategy switching
    a.strategy("normal").rule("work", "stress < 0.7", ["output += 10"])
    a.strategy("crisis").rule("rest", "stress > 0.7", ["stress -= 0.2"])

    # Meta rules (always evaluated)
    a.meta_rule("detect", "KB_QUERY(stress_level) > 0.8", ["SWITCH_STRATEGY('crisis')"])

    # Network topology
    a.network("small-world")
```

---

## AgentPropDef

Defines an agent property.

```python
AgentPropDef(
    name: str,              # Property name
    initial: float = 0.0,   # Initial value
    min: float = 0.0,       # Minimum value
    max: float = 1e18,      # Maximum value
)
```

---

## AgentRuleDef

Defines a behavioral rule.

```python
AgentRuleDef(
    name: str,                          # Rule name
    condition: str,                     # Condition expression
    effects: list[str] = [],            # Effect expressions
    priority: int = 0,                  # Evaluation priority
)
```

### Condition Examples

```python
# Simple comparison
"budget > 10"
"satisfaction < 0.3"

# Compound conditions
"budget > 10 AND satisfaction < 0.5"

# KB query
"KB_QUERY('SELECT ?v WHERE { ex:level ?v }') > 0.8"
```

### Effect Examples

```python
# Simple assignment
"budget -= 10"
"satisfaction += 0.1"
"output = 100"

# Multiple effects
["budget -= 10", "satisfaction += 0.1"]

# Strategy switching
["SWITCH_STRATEGY('crisis')"]
```

---

## AgentStrategy

Named rule set for strategy switching.

```python
AgentStrategy(
    name: str,                          # Strategy name
    rules: list[AgentRuleDef] = [],     # Rules in this strategy
)
```

### Usage

```python
with model.agent("Worker", 50) as a:
    a.strategy("normal").rule("work", "stress < 0.7", ["output += 10"])
    a.strategy("crisis").rule("rest", "stress > 0.7", ["stress -= 0.2"])
    a.meta_rule("switch", "KB_QUERY(stress) > 0.8", ["SWITCH_STRATEGY('crisis')"])
```

---

## ABMEngine

Runtime engine that runs agent step cycles.

### Usage

The ABMEngine is used internally by `SysdModel.simulate()` when agents are defined. You don't typically create it directly.

```python
model = SysdModel(dt=0.5, t_span=(0, 100))
with model.agent("Customer", 100) as a:
    a.prop("satisfaction", 1.0)
    a.rule("buy", "budget > 10", ["budget -= 10"])

# ABMEngine is created internally during simulate()
result = model.simulate()

# Access ABM metrics
print(result.abm_metrics_history)
```

---

## Common Patterns

### Pattern 1: Basic Agents

```python
with model.agent("Customer", 100) as a:
    a.prop("budget", 100, min_val=0)
    a.prop("satisfaction", 1.0, min_val=0, max_val=1)
    a.rule("buy", "budget > 10", ["budget -= 10", "satisfaction += 0.1"])
```

### Pattern 2: Strategy Switching

```python
with model.agent("Worker", 50) as a:
    a.prop("stress", 0.0, min_val=0, max_val=1)
    a.strategy("normal").rule("work", "stress < 0.7", ["output += 10"])
    a.strategy("crisis").rule("rest", "stress > 0.7", ["stress -= 0.2"])
    a.meta_rule("switch", "stress > 0.8", ["SWITCH_STRATEGY('crisis')"])
```

### Pattern 3: With KB Integration

```python
with model.agent("Customer", 100) as a:
    a.prop("satisfaction", 1.0)
    a.rule("react", "KB_QUERY('SELECT ?v WHERE { ex:threat ?v }') > 0.5",
           ["satisfaction -= 0.2"])
```

---

## See Also

- [`SysdModel`](SysdModel.md) — How to add agents to a model
- [`DESEngine`](DESEngine.md) — Discrete event simulation
- [Tutorial 3: Agent-Based Modeling](../../tutorials/03-agent-based-modeling.md) — Full walkthrough
