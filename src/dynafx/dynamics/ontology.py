"""Lightweight SD ontology — stock/flow subtypes and constraint rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dynafx.dynamics.dsl import (
    SysdModel,
    ValidationResult,
    ValidationIssue,
)

STOCK_TYPES: dict[str, list[str]] = {
    "MATERIAL": ["Inventory", "Population", "Goods", "Stock", "Supply", "Product"],
    "INFORMATION": ["Belief", "Knowledge", "Awareness", "Perception"],
    "FINANCIAL": ["Capital", "Funds", "Debt", "Cash", "Budget"],
}

FLOW_TYPES: dict[str, list[str]] = {
    "MATERIAL": ["Production", "Shipment", "Consumption", "Delivery", "Flow"],
    "INFORMATION": ["Communication", "Advertising", "Signal", "Info"],
    "FINANCIAL": ["Revenue", "Cost", "Investment", "Spending", "Income"],
}

RULES: list[tuple[str, str]] = [
    ("material_flow_conservation",
     "A material flow should connect stocks of compatible subtypes"),
    ("financial_balance",
     "Financial flows into a stock should balance outflows"),
]


def _infer_subtype(name: str, type_map: dict[str, list[str]]) -> str:
    """Infer stock/flow subtype by matching name against known patterns."""
    name_lower = name.lower()
    for subtype, patterns in type_map.items():
        for p in patterns:
            if p.lower() in name_lower:
                return subtype
    return "GENERIC"


def validate_ontology(model: SysdModel) -> ValidationResult:
    """Run SD ontology validation rules on a SysdModel."""
    result = ValidationResult()

    stock_subtypes: dict[str, str] = {}
    for s in model.stocks:
        subtype = _infer_subtype(s.name, STOCK_TYPES)
        stock_subtypes[s.name] = subtype
        if subtype != "GENERIC":
            result.infos.append(ValidationIssue(
                "info", f"Stock '{s.name}' inferred as {subtype}", f"stock '{s.name}'"
            ))

    flow_subtypes: dict[str, str] = {}
    flow_sides: dict[str, list[tuple[str, str]]] = {}
    for s in model.stocks:
        for f in s.flows:
            subtype = _infer_subtype(f.name, FLOW_TYPES)
            flow_subtypes[f.name] = subtype
            flow_sides.setdefault(f.name, [])
            flow_sides[f.name].append((f.direction, s.name))
            if subtype != "GENERIC":
                result.infos.append(ValidationIssue(
                    "info", f"Flow '{f.name}' inferred as {subtype}", f"stock '{s.name}': flow '{f.name}'"
                ))

    for fname, sides in flow_sides.items():
        if len(sides) == 2:
            src_name = sides[0][1]
            tgt_name = sides[1][1]
            src_sub = stock_subtypes.get(src_name, "GENERIC")
            tgt_sub = stock_subtypes.get(tgt_name, "GENERIC")
            f_sub = flow_subtypes.get(fname, "GENERIC")
            if f_sub == "MATERIAL" and src_sub != "GENERIC" and tgt_sub != "GENERIC":
                if src_sub != tgt_sub:
                    result.warnings.append(ValidationIssue(
                        "warning",
                        f"Material flow '{fname}' connects {src_sub} stock '{src_name}' "
                        f"to {tgt_sub} stock '{tgt_name}' — may be incompatible",
                        f"flow '{fname}'",
                    ))

            if f_sub == "FINANCIAL":
                result.infos.append(ValidationIssue(
                    "info",
                    f"Financial flow '{fname}' — verify balance check on '{src_name}' ↔ '{tgt_name}'",
                    f"flow '{fname}'",
                ))

    return result
