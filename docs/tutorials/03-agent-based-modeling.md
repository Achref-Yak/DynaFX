# Tutorial 3 — Agent-Based Modeling

Where System Dynamics models *averages*, Agent-Based Modeling (ABM) models
**individual actors** — each with its own state, rules, and messages. In DynaFX
both paradigms run in the *same simulation*, sharing one state dict.

## Agents, properties, and rules

An agent type has a count, named properties, and rules. A rule fires when its
condition is true, then applies its effects.

```sysd
Market
dt 1
from 0 to 5

agent "Trader": 10
  property "cash": 1000
  property "risk": 0.0
  rule "gamble": always
    risk += 0.1
```

```python
from dynafx.dynamics import parse_sysd

model = parse_sysd("""
Market
dt 1
from 0 to 5

agent "Trader": 10
  property "cash": 1000
  property "risk": 0.0
  rule "gamble": always
    risk += 0.1
""")

result = model.simulate()
engine = result.abm_engine
print(len(engine.instances))   # 10 traders
print(engine.instances[0].state["risk"])   # 0.5 after 5 steps
```

## Conditions

Conditions are boolean expressions over the agent's own properties, shared
model state, time, and inbox:

- `always` — always true.
- `cash < 500` — compare a property to a constant.
- `t > 10` — time-based trigger.
- `inbox > 0`, `inbox_warning > 0` — message-related triggers (see below).

```python
model = parse_sysd("""
Market
dt 1
from 0 to 10

agent "Trader": 5
  property "cash": 1000
  property "mode": 0.0
  rule "panic": cash < 200
    mode = 1.0
  rule "recover": cash >= 200
    mode = 0.0
""")
```

## Strategies and meta-rules

An agent can switch between **strategies** (bundles of rules). A **meta-rule**
decides when to switch.

```sysd
A
dt 0.1
from 0 to 3

agent "X": 1
  property "v": 0
  strategy "normal"
    rule "r": when v < 100
      v += 1
  strategy "crisis"
    rule "r2": when v < 5
      v += 10
  meta_rule "switch": when v > 50
    SWITCH_STRATEGY("crisis")
```

```python
from dynafx.dynamics import parse_sysd

model = parse_sysd("""
A
dt 0.1
from 0 to 3

agent "X": 1
  property "v": 0
  strategy "normal"
    rule "r": when v < 100
      v += 1
  strategy "crisis"
    rule "r2": when v < 5
      v += 10
  meta_rule "switch": when v > 50
    SWITCH_STRATEGY("crisis")
""")

result = model.simulate()
agent = result.abm_engine.instances[0]
print(agent.state["v"])
```

## Message passing

Agents send messages with `SEND(target_type, topic, key=value, ...)`. Messages
arrive in the target's inbox next step; inbox size is exposed to conditions as
`inbox`, `inbox_{topic}`, and `inbox_total`.

```python
model = parse_sysd("""
A
dt 1
from 0 to 5

agent "Sender": 1
  property "count": 0
  rule "r": always
    count += 1
    SEND(Receiver, "order", qty=count)

agent "Receiver": 1
  property "got": 0
  strategy "normal"
    rule "listen": when inbox_order > 0
      got += inbox_order
""")

result = model.simulate()
receiver = [i for i in result.abm_engine.instances if i.agent_def.name == "Receiver"][0]
print(receiver.state["got"])   # messages accumulated over the run
```

## The Python API

The same agent can be defined programmatically, without the text DSL:

```python
from dynafx.dynamics import SysdModel

model = SysdModel("p")
model.dt = 1.0
model.t_span = (0, 5)

with model.agent("Buyer", 3) as a:
    a.prop("budget", 500.0, min_val=0.0)
    a.rule("order", "budget > 100", ["budget -= 50"])
    with a.strategy("crisis") as s:
        s.rule("hoard", "budget < 200", ["budget -= 10"])
    a.meta_rule("detect", "inbox_warning > 0", ["SWITCH_STRATEGY('crisis')"])

result = model.simulate()
```

## Reading agent metrics

`engine.get_metrics()` aggregates properties across instances:

```python
metrics = result.abm_engine.get_metrics()
print(metrics["Buyer_budget_avg"])    # mean budget across buyers
print(metrics["Buyer_budget_min"])    # minimum budget
```

## What's next

- Layer discrete events on top in [Discrete Events](04-discrete-event-simulation.md).
- Have agents write live KB triples mid-run in [Closed-Loop Simulation](07-closed-loop-simulation.md).
