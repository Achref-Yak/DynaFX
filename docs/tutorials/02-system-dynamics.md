# Tutorial 2 — System Dynamics

System Dynamics (SD) models the *aggregate* behavior of a system using
**stocks** (accumulations), **flows** (rates into/out of stocks), and **auxes**
(derived quantities). This is the workhorse paradigm of DynaFX.

## Stocks, flows, and auxes

A stock accumulates; a flow changes it each step; an aux is a pure function of
other variables, recomputed every step.

```sysd
Population
dt 0.1
from 0 to 20

stock "Population" = 1000
  + "births": pop_growth * Population
  - "deaths": Population * mortality

aux "pop_growth": 0.02
aux "mortality": 0.01
```

!!! note "Aux syntax"
    Use the colon form `aux "name": expr`. (The `.sysd` files in
    `data/models/` use the same convention.)

```python
from dynafx.dynamics import parse_sysd

model = parse_sysd("""
Population
dt 0.1
from 0 to 20

stock "Population" = 1000
  + "births": pop_growth * Population
  - "deaths": Population * mortality

aux "pop_growth": 0.02
aux "mortality": 0.01
""")

result = model.simulate()
pop = result.values["Population"]
print(round(pop[0], 1))    # 1000.0
print(round(pop[-1], 1))   # ~1221.4 (net growth of 1% per year)
```

Note the `+` and `-` markers on the flows: `+` adds to the stock, `-` removes.

## Parameters override flow expressions

Auxes that are *numeric literals* behave as parameters: you can override them
at `simulate()` time without touching the model.

```python
fast = model.simulate(params={"pop_growth": 0.05})
print(round(fast.values["Population"][-1], 1))   # ~2225.5
```

This is the mechanism `KBSimBridge.params_from_kb` uses to inject knowledge
facts as model inputs (see [Tutorial 7](07-closed-loop-simulation.md)).

## Auxes can reference stocks, flows, and time

Auxes see every other variable, plus the special variable `t` (current time).

```python
model = parse_sysd("""
T
dt 1
from 0 to 5

stock "Level": 0
  + "fill": 10

aux "ramp": 2 * t
aux "headroom": MAX(0, 100 - Level)
""")

result = model.simulate()
print(result.aux_values["ramp"][-1])        # 10.0
print(result.aux_values["headroom"][-1])    # 50.0
```

## Built-in functions

Expressions support common math (`MIN`, `MAX`, `IF`, `ABS`, `EXP`, `LOG`,
`SQRT`, `SIN`, `COS`), lookup tables, and higher-order dynamic functions:

- `SMOOTH(x, t)` — first-order exponential smoothing.
- `DELAY3(x, t)` — third-order material delay (a pipe with lag).
- `STEP(h, t0)` — step function: 0 before `t0`, `h` after.
- `PULSE(h, t0, w)` — a pulse of height `h` starting at `t0`, lasting `w`.

```python
model = parse_sysd("""
T
dt 0.5
from 0 to 30

stock "Inventory": 100
  - "orders": SMOOTH(10, 3)

aux "pulse": PULSE(50, 10, 2)
""")

result = model.simulate()
print(round(result.values["Inventory"][-1], 1))
```

## RK4 vs Euler

Simulation defaults to RK4 (accurate for smooth systems). Euler is faster and
sometimes desirable for event-heavy or discontinuous models:

```python
rk4 = model.simulate(method="rk4")
eul = model.simulate(method="euler", dt=0.25)
```

When `dt` is large relative to flow timescales, use a smaller step or switch to
RK4 to avoid numerical drift.

## Unit checking

You can annotate units and verify dimensional consistency. Units use Vensim
`~unit~` syntax:

```python
from dynafx.dynamics.units import UnitChecker

model = parse_sysd("""
T
dt 1
from 0 to 5

stock "Inv": 100 ~panel~
  - "drain": Inv * 0.1 ~panel/day~
aux "prod": 5 ~panel/day~
""")

check = UnitChecker().check(model)
print(check.violations)   # []  (Inv = drain × time, dimensionally consistent)
```

A violation is reported when, e.g., a flow's unit does not equal the stock's
unit per time step.

## What's next

- Add individual agents on top of stocks in [Agent Behavior](03-agent-based-modeling.md).
- Compare alternative futures in [Scenarios & Sensitivity](08-scenarios-and-sensitivity.md).
