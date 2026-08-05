"""Dimensional analysis and unit checking for system dynamics models.

Provides Vensim-style unit annotation (~Unit~ syntax) with automatic
unit propagation and consistency checking.

Unit syntax:
    stock "Population": 1000 ~people~
    aux "birth_rate": 0.02 ~people/person/year~
    + births: Pop * birth_rate ~people/year~

Rules enforced:
    - Stock units = flow units × time
    - Inflow units must match outflow units (per stock)
    - Expression units must match declared units
    - Multiplication/division: units combine algebraically
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ─── Unit representation ────────────────────────────────────────

@dataclass(frozen=True)
class Unit:
    """Dimensional unit as a product of base-dimension exponents.

    Example: people/year^0.5 → Unit(factors={"people": 1, "year": -0.5})
    """
    factors: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        # Sort by name for canonical ordering
        if self.factors and not isinstance(self.factors, tuple):
            object.__setattr__(self, "factors", tuple(sorted(self.factors)))

    @classmethod
    def from_str(cls, s: str) -> Unit:
        """Parse unit string like 'people/person/year' or 'kg*m/s^2'."""
        s = s.strip()
        if not s or s.lower() == "dimensionless":
            return cls()

        # Normalize: replace ^ with **
        s = s.replace("**", "^")

        # Parse: split on / for denominator, then * for multiplication within parts
        # Example: kg*m/s^2 → numerator: [kg, m], denominator: [s^2]
        parts = s.split("/")
        numerator: dict[str, float] = {}
        denominator: dict[str, float] = {}

        for i, part in enumerate(parts):
            # Parse multiplication within part (e.g., "kg*m")
            for term in re.split(r"[*]", part):
                term = term.strip()
                if not term:
                    continue
                # Handle exponents: people^2 → people with exp 2
                m = re.match(r"^([A-Za-z_]\w*)(?:\^(-?\d+(?:\.\d+)?))?$", term)
                if m:
                    name = m.group(1)
                    # Skip pure numeric names (e.g., "1" from "1/year")
                    if re.match(r"^\d+(\.\d+)?$", name):
                        continue
                    exp = float(m.group(2)) if m.group(2) else 1.0
                else:
                    name = term
                    exp = 1.0

                if i == 0:
                    numerator[name] = numerator.get(name, 0) + exp
                else:
                    denominator[name] = denominator.get(name, 0) + exp

        # Combine: numerator exponents stay, denominator exponents are subtracted
        factors: dict[str, float] = {}
        for name, exp in numerator.items():
            factors[name] = factors.get(name, 0) + exp
        for name, exp in denominator.items():
            factors[name] = factors.get(name, 0) - exp

        # Remove zero exponents
        factors = {k: v for k, v in factors.items() if abs(v) > 1e-10}

        return cls(factors=tuple(sorted(factors.items())))

    def __mul__(self, other: Unit) -> Unit:
        """Multiply two units."""
        factors: dict[str, float] = {}
        for name, exp in self.factors:
            factors[name] = factors.get(name, 0) + exp
        for name, exp in other.factors:
            factors[name] = factors.get(name, 0) + exp
        factors = {k: v for k, v in factors.items() if abs(v) > 1e-10}
        return Unit(factors=tuple(sorted(factors.items())))

    def __truediv__(self, other: Unit) -> Unit:
        """Divide two units."""
        factors: dict[str, float] = {}
        for name, exp in self.factors:
            factors[name] = factors.get(name, 0) + exp
        for name, exp in other.factors:
            factors[name] = factors.get(name, 0) - exp
        factors = {k: v for k, v in factors.items() if abs(v) > 1e-10}
        return Unit(factors=tuple(sorted(factors.items())))

    def __pow__(self, exp: float) -> Unit:
        """Raise unit to a power."""
        if abs(exp) < 1e-10:
            return Unit()
        factors = tuple((name, e * exp) for name, e in self.factors
                       if abs(e * exp) > 1e-10)
        return Unit(factors=factors)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented
        return self.factors == other.factors

    def __hash__(self) -> int:
        return hash(self.factors)

    def is_dimensionless(self) -> bool:
        return len(self.factors) == 0

    def __str__(self) -> str:
        if not self.factors:
            return "dimensionless"
        parts = []
        for name, exp in self.factors:
            if abs(exp - 1.0) < 1e-10:
                parts.append(name)
            elif abs(exp + 1.0) < 1e-10:
                parts.append(f"1/{name}")
            else:
                # Check if exp is negative
                if exp < 0:
                    parts.append(f"1/{name}^{abs(exp):.0f}")
                else:
                    parts.append(f"{name}^{exp:.0f}")
        return "*".join(parts) if parts else "dimensionless"

    def __repr__(self) -> str:
        return f"Unit({self!s})"


# ─── Built-in units ─────────────────────────────────────────────

BUILTIN_UNITS: dict[str, Unit] = {
    # Time
    "second": Unit.from_str("second"),
    "minute": Unit.from_str("minute"),
    "hour": Unit.from_str("hour"),
    "day": Unit.from_str("day"),
    "week": Unit.from_str("week"),
    "month": Unit.from_str("month"),
    "year": Unit.from_str("year"),
    # Amount
    "person": Unit.from_str("person"),
    "people": Unit.from_str("person"),
    # Currency
    "dollar": Unit.from_str("dollar"),
    "$": Unit.from_str("dollar"),
    # Mass
    "kg": Unit.from_str("kg"),
    "g": Unit.from_str("g"),
    "ton": Unit.from_str("ton"),
    # Volume
    "liter": Unit.from_str("liter"),
    "L": Unit.from_str("liter"),
    "gallon": Unit.from_str("gallon"),
    # Count
    "unit": Unit.from_str("unit"),
}

# Time base units for flow dimensionality checking
_TIME_UNITS = {"second", "minute", "hour", "day", "week", "month", "year"}


# ─── Unit Registry ──────────────────────────────────────────────

class UnitRegistry:
    """Registry of known units with conversion factors."""

    def __init__(self):
        self.units: dict[str, Unit] = dict(BUILTIN_UNITS)
        self.aliases: dict[str, str] = {
            "people": "person",
            "$": "dollar",
            "L": "liter",
            "sec": "second",
            "min": "minute",
            "hr": "hour",
            "yr": "year",
        }

    def register(self, name: str, unit: Unit) -> None:
        """Register a custom unit."""
        self.units[name] = unit

    def resolve(self, name: str) -> Unit | None:
        """Look up a unit by name."""
        if name in self.units:
            return self.units[name]
        if name in self.aliases:
            return self.units.get(self.aliases[name])
        return None

    def is_time_unit(self, name: str) -> bool:
        return name in _TIME_UNITS or self.aliases.get(name) in _TIME_UNITS


# ─── Unit Checker ───────────────────────────────────────────────

@dataclass
class UnitViolation:
    """A unit inconsistency found during checking."""
    name: str
    expected: Unit
    actual: Unit
    message: str
    severity: str = "error"  # "error" or "warning"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "expected": str(self.expected),
            "actual": str(self.actual),
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class UnitCheckResult:
    """Result of unit checking."""
    violations: list[UnitViolation] = field(default_factory=list)
    checked_names: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(v.severity != "error" for v in self.violations)

    @property
    def errors(self) -> list[UnitViolation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> list[UnitViolation]:
        return [v for v in self.violations if v.severity == "warning"]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
            "violations": [v.to_dict() for v in self.violations],
            "checked_names": self.checked_names,
        }


class UnitChecker:
    """Checks dimensional consistency of a SysdModel."""

    def __init__(self, registry: UnitRegistry | None = None):
        self.registry = registry or UnitRegistry()
        self._units: dict[str, Unit] = {}  # variable name → unit
        self._declared: dict[str, Unit] = {}  # explicitly annotated units
        self._result = UnitCheckResult()

    def check(self, model) -> UnitCheckResult:
        """Run full unit check on a SysdModel.

        Steps:
        1. Collect declared units from stocks, auxes, flows
        2. Propagate units through expressions
        3. Verify stock units = flow units × time
        4. Verify inflow/outflow unit consistency
        5. Verify expression result units match declarations
        """
        self._result = UnitCheckResult()
        self._units.clear()
        self._declared.clear()

        # Step 1: Collect declared units
        self._collect_declarations(model)

        # Step 2: Propagate units through expressions
        self._propagate_units(model)

        # Step 3: Check stock-flow consistency
        self._check_stock_flow_consistency(model)

        # Step 4: Check expression units
        self._check_expression_units(model)

        return self._result

    def _collect_declarations(self, model) -> None:
        """Collect explicitly declared units from the model."""
        for stock in model.stocks:
            if stock.units:
                unit = Unit.from_str(stock.units)
                self._units[stock.name] = unit
                self._declared[stock.name] = unit
                self._result.checked_names.append(stock.name)

        for aux in model.aux_vars:
            if aux.units:
                unit = Unit.from_str(aux.units)
                self._units[aux.name] = unit
                self._declared[aux.name] = unit
                self._result.checked_names.append(aux.name)

        for stock in model.stocks:
            for flow in stock.flows:
                if flow.units:
                    unit = Unit.from_str(flow.units)
                    self._units[flow.name] = unit
                    self._declared[flow.name] = unit
                    self._result.checked_names.append(flow.name)

        # Flows without explicit units inherit from their expression references
        for stock in model.stocks:
            for flow in stock.flows:
                if flow.name in self._units:
                    continue
                # Try to find units from the flow expression
                refs = re.findall(r"[A-Za-z_]\w*", flow.expr)
                for ref in refs:
                    if ref in self._units:
                        self._units[flow.name] = self._units[ref]
                        break

    def _propagate_units(self, model) -> None:
        """Propagate units through expressions using inference rules."""
        # Build name → unit map from what we know
        changed = True
        max_iter = 10
        while changed and max_iter > 0:
            changed = False
            max_iter -= 1

            for aux in model.aux_vars:
                if aux.name in self._units:
                    continue
                unit = self._infer_expression_unit(aux.expr, model)
                if unit is not None:
                    self._units[aux.name] = unit
                    self._result.checked_names.append(aux.name)
                    changed = True

    def _infer_expression_unit(self, expr: str, model) -> Unit | None:
        """Infer the unit of an expression from its components."""
        expr = expr.strip()

        # Simple variable reference
        if expr in self._units:
            return self._units[expr]

        # Number literal → dimensionless
        try:
            float(expr)
            return Unit()
        except ValueError:
            pass

        # Binary operations: a + b, a - b, a * b, a / b
        # Try to split on outermost operator
        for op in ["+", "-", "*", "/"]:
            depth = 0
            for i in range(len(expr) - 1, -1, -1):
                if expr[i] in ")]":
                    depth += 1
                elif expr[i] in "([":
                    depth -= 1
                elif depth == 0 and expr[i] == op and i > 0:
                    left = expr[:i].strip()
                    right = expr[i + 1:].strip()
                    left_unit = self._infer_expression_unit(left, model)
                    right_unit = self._infer_expression_unit(right, model)
                    if left_unit is not None and right_unit is not None:
                        if op in ("+", "-"):
                            # Addition/subtraction: units must match
                            if left_unit == right_unit:
                                return left_unit
                            return None
                        elif op == "*":
                            return left_unit * right_unit
                        elif op == "/":
                            return left_unit / right_unit

        # Function calls: MIN, MAX, IF, SMOOTH, etc.
        m = re.match(r"^([A-Z_]+)\((.+)\)$", expr)
        if m:
            func_name = m.group(1)
            args_str = m.group(2)
            args = self._split_args(args_str)
            if func_name in ("MIN", "MAX", "IF"):
                # Return unit of first argument (or matching units)
                for arg in args:
                    unit = self._infer_expression_unit(arg, model)
                    if unit is not None:
                        return unit
            elif func_name in ("SMOOTH", "SMOOTHI", "DELAY3", "DELAYN",
                              "DELAY_FIXED"):
                # First argument determines output unit
                if args:
                    return self._infer_expression_unit(args[0], model)
            elif func_name in ("ABS", "SQRT", "EXP", "LN", "SIN", "COS"):
                # Mathematical functions: return dimensionless or same unit
                if args:
                    return self._infer_expression_unit(args[0], model)
            elif func_name == "PULSE" or func_name == "NOISE" or func_name in ("STEP", "RAMP"):
                return Unit()  # dimensionless

        # Parenthesized expression
        if expr.startswith("(") and expr.endswith(")"):
            return self._infer_expression_unit(expr[1:-1], model)

        # Reference to known variable
        # Extract all words and check if any are known
        words = re.findall(r"[A-Za-z_]\w*", expr)
        for word in words:
            if word in self._units and expr == word:
                return self._units[word]

        return None

    def _split_args(self, s: str) -> list[str]:
        """Split function arguments respecting nested parentheses."""
        args = []
        depth = 0
        current = []
        for ch in s:
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            args.append("".join(current).strip())
        return args

    def _check_stock_flow_consistency(self, model) -> None:
        """Verify stock units = flow units × time for each stock.

        In SD: stock = ∫flow dt, so stock_unit = flow_unit × time_unit.
        Equivalently: flow_unit = stock_unit / time_unit.
        """
        time_unit = self._find_time_unit(model)

        for stock in model.stocks:
            stock_unit = self._units.get(stock.name)
            if stock_unit is None:
                continue

            stock_has_time = any(name in _TIME_UNITS for name, _ in stock_unit.factors)
            time_is_dim = time_unit.is_dimensionless()

            for flow in stock.flows:
                flow_unit = self._units.get(flow.name)
                if flow_unit is None:
                    continue

                if stock_has_time and not time_is_dim:
                    # Stock has time dimension, time unit exists: verify flow = stock / time
                    expected_flow_unit = stock_unit / time_unit
                    if flow_unit != expected_flow_unit:
                        self._result.violations.append(UnitViolation(
                            name=flow.name,
                            expected=expected_flow_unit,
                            actual=flow_unit,
                            message=(
                                f"Flow '{flow.name}' has units {flow_unit}, "
                                f"but stock '{stock.name}' has units {stock_unit}. "
                                f"Expected flow units: {expected_flow_unit} "
                                f"(stock_units / time)"
                            ),
                        ))
                elif not stock_has_time and not time_is_dim:
                    # Stock has no time, time unit exists: flow should = stock / time
                    expected_flow_unit = stock_unit / time_unit
                    if flow_unit != expected_flow_unit:
                        self._result.violations.append(UnitViolation(
                            name=flow.name,
                            expected=expected_flow_unit,
                            actual=flow_unit,
                            message=(
                                f"Flow '{flow.name}' has units {flow_unit}, "
                                f"but stock '{stock.name}' has units {stock_unit}. "
                                f"Expected flow units: {expected_flow_unit} "
                                f"(stock_units / time)"
                            ),
                        ))
                # If time is dimensionless, we can't verify stock/flow consistency

    def _check_expression_units(self, model) -> None:
        """Verify declared units match inferred units for all variables."""
        for aux in model.aux_vars:
            if aux.name not in self._declared:
                continue
            declared = self._declared[aux.name]
            inferred = self._units.get(aux.name)
            if inferred is not None and declared != inferred:
                self._result.violations.append(UnitViolation(
                    name=aux.name,
                    expected=declared,
                    actual=inferred,
                    message=(
                        f"Aux '{aux.name}' declared as {declared}, "
                        f"but expression infers {inferred}"
                    ),
                    severity="warning",
                ))

        for stock in model.stocks:
            for flow in stock.flows:
                if flow.name not in self._declared:
                    continue
                declared = self._declared[flow.name]
                inferred = self._units.get(flow.name)
                if inferred is not None and declared != inferred:
                    self._result.violations.append(UnitViolation(
                        name=flow.name,
                        expected=declared,
                        actual=inferred,
                        message=(
                            f"Flow '{flow.name}' declared as {declared}, "
                            f"but expression infers {inferred}"
                        ),
                        severity="warning",
                    ))

    def _find_time_unit(self, model) -> Unit | None:
        """Find the time unit used in the model.

        Prefers variables whose name suggests they are time parameters,
        then any explicitly declared time unit.
        """
        # First pass: look for variables with "time" in the name
        for name, unit in self._declared.items():
            if "time" in name.lower():
                for factor_name, _ in unit.factors:
                    if factor_name in _TIME_UNITS:
                        return Unit(factors=((factor_name, 1.0),))

        # Second pass: look for explicitly declared time units
        for _name, unit in self._declared.items():
            for factor_name, _ in unit.factors:
                if factor_name in _TIME_UNITS:
                    return Unit(factors=((factor_name, 1.0),))

        # Third pass: look for any variable with a time unit (inferred)
        for name, unit in self._units.items():
            if name in self._declared:
                continue
            for factor_name, _ in unit.factors:
                if factor_name in _TIME_UNITS:
                    return Unit(factors=((factor_name, 1.0),))

        # Default: assume 'time' is dimensionless
        return Unit()
