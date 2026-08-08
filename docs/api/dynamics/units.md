# Unit Checking

**What it is:** Verify dimensional consistency — check that `stock = flow × time` holds.

**When to use it:** When verifying model equations are dimensionally correct.

**Quick start:**
```python
from dynafx.dynamics import UnitChecker

checker = UnitChecker(model)
result = checker.check()
if result.violations:
    for v in result.violations:
        print(f"Violation: {v.message}")
```

---

## Import

```python
from dynafx.dynamics import UnitChecker, Unit, UnitRegistry, UnitCheckResult, UnitViolation
```

## Classes

### UnitChecker

```python
UnitChecker(model: SysdModel)
```

#### Methods

```python
checker.check()  # Returns UnitCheckResult
```

### UnitCheckResult

```python
UnitCheckResult(
    valid: bool,
    violations: list[UnitViolation],
)
```

### UnitViolation

```python
UnitViolation(
    variable: str,
    expected: str,
    actual: str,
    message: str,
)
```

### Unit / UnitRegistry

For defining and managing units.

---

## See Also

- [`SysdModel`](SysdModel.md) — How to add units to stocks/flows
- [Tutorial 2: System Dynamics](../../tutorials/02-system-dynamics.md) — Units deep dive
