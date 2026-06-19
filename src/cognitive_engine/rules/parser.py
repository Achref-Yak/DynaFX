"""Rule parser — converts dict/JSON to Rule objects.

Parses rules from a dictionary format consistent with TBox axiom format.
Supports negation via the "negated" field in pattern definitions.
"""

from __future__ import annotations

from cognitive_engine.rules.engine import Action, Pattern, Rule


def parse_rules(data: dict) -> list[Rule]:
    """Parse rules from dict format.

    Input format::

        {
            "rules": [
                {
                    "name": "transitivity",
                    "when": [
                        {"source": "?a", "edge": "SUPPORTS", "target": "?b"},
                        {"source": "?b", "edge": "SUPPORTS", "target": "?c"}
                    ],
                    "then": [
                        {"source": "?a", "edge": "SUPPORTS", "target": "?c", "weight": 0.7}
                    ],
                    "confidence": 0.8
                }
            ]
        }

    Negation example::

        {
            "name": "missing_causal",
            "when": [
                {"source": "?a", "edge": "CAUSES", "target": "?b", "negated": true},
                {"source": "?a", "edge": "DEPENDS", "target": "?b"}
            ],
            "then": [
                {"source": "?a", "edge": "CAUSES", "target": "?b", "weight": 0.5}
            ],
            "confidence": 0.6
        }

    Args:
        data: Dictionary with a "rules" key containing rule definitions.

    Returns:
        List of parsed Rule objects.
    """
    rules: list[Rule] = []

    for rule_data in data.get("rules", []):
        when: list[Pattern] = []
        for p in rule_data.get("when", []):
            when.append(Pattern(
                source_type=p.get("source_type"),
                edge_type=p.get("edge"),
                target_type=p.get("target_type"),
                source_var=p.get("source"),
                target_var=p.get("target"),
                negated=p.get("negated", False),
                min_belief=p.get("min_belief"),
                max_belief=p.get("max_belief"),
            ))

        then: list[Action] = []
        for a in rule_data.get("then", []):
            then.append(Action(
                source_var=a["source"],
                target_var=a["target"],
                edge_type=a["edge"],
                weight=a.get("weight", 0.5),
                confidence=a.get("confidence", 0.5),
            ))

        rules.append(Rule(
            name=rule_data["name"],
            when=when,
            then=then,
            confidence=rule_data.get("confidence", 0.5),
            enabled=rule_data.get("enabled", True),
        ))

    return rules
