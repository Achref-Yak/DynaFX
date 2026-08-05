# Tutorial 1 — Hello World

This is the shortest path from zero to a running model. You'll parse a model
described in the `.sysd` language, simulate it, and read the trajectory.

## What a model looks like

A DynaFX model is a small text program. This one is a **stock** (`X`) that
drains at 10% per time unit:

```sysd
T
dt 1
from 0 to 10

stock X: 100
  - Out: X * 0.1
```

- `T` is the model name (optional).
- `dt 1` is the integration time step.
- `from 0 to 10` is the time horizon.
- `stock X: 100` declares a stock starting at 100.
- `- Out: X * 0.1` declares an *outflow* that removes 10% of `X` per step.

## Parse and simulate

```python
from dynafx.dynamics import parse_sysd

model = parse_sysd("""
T
dt 1
from 0 to 10

stock X: 100
  - Out: X * 0.1
""")

result = model.simulate()
print(result.stocks)        # ['X']
print(len(result.times))    # number of steps
```

You can also build the same model in pure Python, without any text parsing:

```python
from dynafx.dynamics import SysdModel

model = SysdModel("T")
model.dt = 1.0
model.t_span = (0, 10)

with model.stock("X", 100) as s:
    s.outflow("Out", "X * 0.1")

result = model.simulate()
```

## Read the trajectory

The result object stores one list of values per stock:

```python
x = result.values["X"]
print(x[0])          # 100.0  (initial value)
print(x[-1])         # ~36.8  (10% per-unit-time decay over 10 units)
print(result.times[0], result.times[-1])   # 0.0  10.0
```

The outflow `X * 0.1` describes the *rate* `dX/dt = -0.1·X`, so the model
solves the continuous exponential decay `100·e^(−1) ≈ 36.79` (RK4 integration
by default). A discrete `dt 1` step would give `100·0.9^10 ≈ 34.87` instead —
the difference matters when you state your integration method.

## Also read auxes

Any derived quantity you define with `aux` is stored in `result.aux_values`:

```python
model = parse_sysd("""
T
dt 1
from 0 to 10

stock X: 100
  - Out: X * 0.1
aux half_life: X / 2
""")

result = model.simulate()
print(result.aux_values["half_life"][-1])   # ~18.4
```

## Inspecting the model

```python
print(model.name)              # 'T'
print(len(model.stocks))       # 1
print(model.stocks[0].name)    # 'X'
```

## What's next

- Add parameters and richer stocks/flows in [System Dynamics](02-system-dynamics.md).
- Skip ahead to a complete KB + simulation loop in [Closed-Loop Twin](07-closed-loop-twin.md).
