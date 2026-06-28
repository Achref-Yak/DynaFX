"""Emergent properties and stock-flow consistency for system dynamics.

EmergentProperty captures threshold-triggered behavioral shifts that are
currently buried in IF-gates inside aux expression strings.  Making them
first-class dataclasses means they are queryable, traceable, and
distinguishable from ordinary parameter computation.

StockFlowConsistency validates structural invariants that, if violated,
produce the class of bugs we hit twice in the pandemic model:
  - outflow partitions not summing to 1.0
  - flows with only one side (conservation violation)
  - stocks with initial=0 used as divisors
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from dynafx.system.dsl import (
    AuxDef,
    FlowDef,
    StockDef,
    SysdModel,
    ValidationResult,
    ValidationIssue,
)


# ── Emergent Property ────────────────────────────────────────────

class ComparisonOp(Enum):
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    EQ = "=="
    NE = "!="


class EffectType(Enum):
    MULTIPLY = auto()
    ADD = auto()
    SET = auto()
    REPLACE_EXPR = auto()


@dataclass(frozen=True)
class Condition:
    """A threshold condition on an aux or stock variable.

    Example: ``healthcare_stress_avg > 1.2``
    """
    variable: str
    op: ComparisonOp
    threshold: float

    def evaluate(self, state: dict[str, float]) -> bool:
        val = state.get(self.variable, 0.0)
        ops = {
            ComparisonOp.GT: val > self.threshold,
            ComparisonOp.GE: val >= self.threshold,
            ComparisonOp.LT: val < self.threshold,
            ComparisonOp.LE: val <= self.threshold,
            ComparisonOp.EQ: val == self.threshold,
            ComparisonOp.NE: val != self.threshold,
        }
        return ops[self.op]

    def __str__(self) -> str:
        return f"{self.variable} {self.op.value} {self.threshold}"


@dataclass(frozen=True)
class Effect:
    """A parameter modification that activates when a condition triggers.

    Example: effective_mortality is multiplied by overload_multiplier.
    """
    target: str
    effect_type: EffectType
    value: float = 0.0
    expr: str = ""

    def apply(self, base: float, state: dict[str, float]) -> float:
        if self.effect_type == EffectType.MULTIPLY:
            return base * self.value
        if self.effect_type == EffectType.ADD:
            return base + self.value
        if self.effect_type == EffectType.SET:
            return self.value
        return base


@dataclass
class EmergentProperty:
    """A threshold-triggered behavioral shift in the system.

    Captures phenomena like "healthcare overload doubles mortality" that
    are currently buried in IF-gates.  Fields:
      - condition: the threshold test (e.g. healthcare_stress_avg > 1.2)
      - effects: parameter modifications when triggered
      - provenance: which stocks/flows/auxes are involved
      - severity: how critical the threshold crossing is (0-1)
      - active: whether the condition is currently met (updated during sim)
      - activation_times: time points where state changed (off→on / on→off)

    This is a pure SD structure — no SL types.
    """
    name: str
    description: str
    condition: Condition
    effects: list[Effect] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)
    severity: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    # Runtime state (updated during simulation)
    active: bool = False
    activation_times: list[float] = field(default_factory=list)

    def check(self, state: dict[str, float], t: float) -> bool:
        """Evaluate condition against current state, track transitions.

        Returns True if state changed (transition occurred).
        """
        now_active = self.condition.evaluate(state)
        transition = now_active != self.active
        if transition:
            self.activation_times.append(t)
        self.active = now_active
        return transition

    def apply_effects(self, state: dict[str, float]) -> dict[str, float]:
        """Apply effects to state dict when active. Returns modified state."""
        if not self.active:
            return state
        modified = dict(state)
        for eff in self.effects:
            base = modified.get(eff.target, 0.0)
            modified[eff.target] = eff.apply(base, state)
        return modified

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "condition": str(self.condition),
            "active": self.active,
            "severity": self.severity,
            "effects": [
                {"target": e.target, "type": e.effect_type.name, "value": e.value}
                for e in self.effects
            ],
            "provenance": self.provenance,
            "transitions": len(self.activation_times),
        }


# ── Stock-Flow Consistency Checker ──────────────────────────────

@dataclass
class ConsistencyViolation:
    """A structural violation found by the consistency checker."""
    level: str      # "error" or "warning"
    rule: str       # which check failed
    message: str
    location: str   # e.g. "stock 'Infected': flow 'Recovery'"


@dataclass
class ConsistencyResult:
    """Result of running stock-flow consistency checks."""
    violations: list[ConsistencyViolation] = field(default_factory=list)
    checks_run: int = 0
    checks_passed: int = 0

    @property
    def is_valid(self) -> bool:
        return not any(v.level == "error" for v in self.violations)

    def to_validation(self) -> ValidationResult:
        """Convert to existing ValidationResult for CLI reporting."""
        r = ValidationResult()
        for v in self.violations:
            issue = ValidationIssue(level=v.level, message=v.message, location=v.location)
            if v.level == "error":
                r.errors.append(issue)
            elif v.level == "warning":
                r.warnings.append(issue)
            else:
                r.infos.append(issue)
        return r

    def print_report(self) -> None:
        for v in self.violations:
            prefix = "ERROR" if v.level == "error" else "WARNING"
            print(f"  {prefix} [{v.rule}] {v.message}")
        if self.is_valid:
            print(f"  ✅ All {self.checks_run} checks passed")
        else:
            errs = sum(1 for v in self.violations if v.level == "error")
            print(f"  ❌ {errs} error(s) in {self.checks_run} checks")


def _extract_fraction_parts(expr_str: str) -> list[tuple[str, float]]:
    """Try to extract fraction parts from an expression like 'Infected * 0.3 / 7'.

    Handles two formats:
      1. '<word> * <number> / <number>' (full expression)
      2. '<number> / <number>' or '<number>' (after stock name stripping)

    Returns list of (multiplier_name, fraction_value) for simple cases.
    Returns empty list for complex expressions we can't statically analyse.
    """
    parts: list[tuple[str, float]] = []
    for segment in re.split(r"\s*\+\s*", expr_str):
        segment = segment.strip()
        # Try: <word> * <number> / <number>
        m = re.match(
            r"(\w+)\s*\*\s*([\d.]+)(?:\s*/\s*([\d.]+))?",
            segment,
        )
        if m:
            stock_name = m.group(1)
            numerator = float(m.group(2))
            denominator = float(m.group(3)) if m.group(3) else 1.0
            parts.append((stock_name, numerator / denominator))
            continue
        # Try: <number> / <number> (after stock name stripping)
        m = re.match(r"([\d.]+)\s*/\s*([\d.]+)", segment)
        if m:
            numerator = float(m.group(1))
            denominator = float(m.group(2))
            parts.append(("", numerator / denominator))
            continue
        # Try: bare <number>
        m = re.match(r"([\d.]+)", segment)
        if m:
            parts.append(("", float(m.group(1))))
    return parts


def check_outflow_partitions(model: SysdModel) -> list[ConsistencyViolation]:
    """Check that outflow fractions from each stock sum to 1.0.

    For each stock, collect all outflows that have the form:
        Stock * <number> / <number>
    If the numeric fractions (summed) don't equal 1.0, flag it.

    This catches the bug where recovery_fraction + severe_fraction +
    effective_mortality doesn't sum to 1.0.
    """
    violations: list[ConsistencyViolation] = []

    for stock in model.stocks:
        outflows = [f for f in stock.flows if f.direction == "-"]
        if len(outflows) < 2:
            continue

        # Check if any outflow uses a symbolic variable as the fraction
        # (not just as the stock multiplier). E.g. "A * recovery_fraction / period"
        # is dynamic; "A * 0.5 / 10" is static.
        has_dynamic = False
        for f in outflows:
            stripped = re.sub(rf"^{re.escape(stock.name)}\s*\*\s*", "", f.expr)
            # If the stripped expression contains letters (variables) in the
            # fraction position, it's dynamic
            if re.search(r"[a-zA-Z_]", stripped):
                has_dynamic = True
                break
        if has_dynamic:
            # Dynamic fractions — check if aux variables enforce sum-to-1
            # Look for aux like: recovery_fraction = 1.0 - severe_fraction - effective_mortality
            for aux in model.aux_vars:
                if re.search(
                    rf"1\.0\s*-\s*.*{re.escape(stock.name)}",
                    aux.expr,
                    re.IGNORECASE,
                ):
                    # Found a compensating aux — good, but warn if it's
                    # the only safeguard
                    violations.append(ConsistencyViolation(
                        level="info",
                        rule="dynamic_partition_compensator",
                        message=(
                            f"Stock '{stock.name}' has dynamic outflow fractions "
                            f"compensated by aux '{aux.name}' — verify it always "
                            f"sums to 1.0 at runtime"
                        ),
                        location=f"stock '{stock.name}'",
                    ))
            continue

        # Static fractions — extract and sum
        total = 0.0
        for f in outflows:
            # Strip the stock name prefix: "Infected * 0.3 / 7" → "0.3 / 7"
            stripped = re.sub(rf"^{re.escape(stock.name)}\s*\*\s*", "", f.expr)
            parts = _extract_fraction_parts(stripped)
            if parts:
                for _, frac in parts:
                    total += frac
            else:
                # Can't parse — skip with info
                violations.append(ConsistencyViolation(
                    level="info",
                    rule="unparseable_outflow",
                    message=f"Cannot statically parse outflow expression: {f.expr}",
                    location=f"stock '{stock.name}': flow '{f.name}'",
                ))

        if total > 0 and abs(total - 1.0) > 1e-6:
            violations.append(ConsistencyViolation(
                level="error",
                rule="partition_sum",
                message=(
                    f"Outflow fractions from stock '{stock.name}' sum to "
                    f"{total:.6f}, expected 1.0 — "
                    f"{'excess' if total > 1.0 else 'leakage'} of "
                    f"{abs(total - 1.0):.6f}"
                ),
                location=f"stock '{stock.name}'",
            ))

    return violations


def check_flow_sides(model: SysdModel) -> list[ConsistencyViolation]:
    """Check each flow appears exactly twice (once as +, once as -)."""
    violations: list[ConsistencyViolation] = []
    flow_sides: dict[str, list[tuple[str, str]]] = {}

    for stock in model.stocks:
        for f in stock.flows:
            flow_sides.setdefault(f.name, []).append((f.direction, stock.name))

    for fname, sides in flow_sides.items():
        if len(sides) == 1:
            violations.append(ConsistencyViolation(
                level="warning",
                rule="one_sided_flow",
                message=(
                    f"Flow '{fname}' has only one side "
                    f"({sides[0][0]} in stock '{sides[0][1]}') — "
                    f"check conservation"
                ),
                location=f"flow '{fname}'",
            ))
        elif len(sides) > 2:
            violations.append(ConsistencyViolation(
                level="error",
                rule="too_many_sides",
                message=(
                    f"Flow '{fname}' appears {len(sides)} times, "
                    f"expected exactly 2"
                ),
                location=f"flow '{fname}'",
            ))

    return violations


def check_zero_divisors(model: SysdModel) -> list[ConsistencyViolation]:
    """Check for stocks with initial=0 used as divisors."""
    violations: list[ConsistencyViolation] = []
    zero_stocks = {s.name for s in model.stocks if s.initial == 0}

    for stock in model.stocks:
        for f in stock.flows:
            if "/" in f.expr:
                for zs in zero_stocks:
                    if re.search(rf"\b{re.escape(zs)}\b", f.expr.split("/", 1)[1]):
                        violations.append(ConsistencyViolation(
                            level="warning",
                            rule="zero_divisor",
                            message=(
                                f"Stock '{zs}' has initial value 0 and "
                                f"appears as a divisor in flow '{f.name}'"
                            ),
                            location=f"stock '{stock.name}': flow '{f.name}'",
                        ))

    return violations


def check_cross_type_flows(model: SysdModel) -> list[ConsistencyViolation]:
    """Check that flows connect stocks of compatible subtypes.

    Uses name-based inference (same logic as ontology.py) but applied
    structurally to the flow graph.
    """
    violations: list[ConsistencyViolation] = []
    STOCK_PATTERNS: dict[str, list[str]] = {
        "MATERIAL": ["inventory", "population", "goods", "stock", "supply", "product",
                      "susceptible", "exposed", "infected", "recovered", "hospitalized",
                      "fatalities", "capacity"],
        "FINANCIAL": ["capital", "funds", "debt", "cash", "budget"],
        "INFORMATION": ["belief", "knowledge", "awareness", "perception"],
    }

    def infer(name: str) -> str:
        low = name.lower()
        for sub, patterns in STOCK_PATTERNS.items():
            for p in patterns:
                if p in low:
                    return sub
        return "GENERIC"

    flow_sides: dict[str, list[tuple[str, str]]] = {}
    for stock in model.stocks:
        for f in stock.flows:
            flow_sides.setdefault(f.name, []).append((f.direction, stock.name))

    for fname, sides in flow_sides.items():
        if len(sides) == 2:
            src = sides[0][1] if sides[0][0] == "-" else sides[1][1]
            tgt = sides[1][1] if sides[1][0] == "+" else sides[0][1]
            src_type = infer(src)
            tgt_type = infer(tgt)
            if src_type != "GENERIC" and tgt_type != "GENERIC" and src_type != tgt_type:
                violations.append(ConsistencyViolation(
                    level="warning",
                    rule="cross_type_flow",
                    message=(
                        f"Flow '{fname}' connects {src_type} stock '{src}' "
                        f"to {tgt_type} stock '{tgt}' — may be incompatible"
                    ),
                    location=f"flow '{fname}'",
                ))

    return violations


def run_consistency_checks(model: SysdModel) -> ConsistencyResult:
    """Run all stock-flow consistency checks on a SysdModel.

    Returns ConsistencyResult with violations and summary counts.
    """
    checks = [
        check_outflow_partitions,
        check_flow_sides,
        check_zero_divisors,
        check_cross_type_flows,
    ]

    result = ConsistencyResult()
    for check_fn in checks:
        violations = check_fn(model)
        result.violations.extend(violations)
        result.checks_run += 1
        if not violations:
            result.checks_passed += 1

    return result
