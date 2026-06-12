from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Dict, Tuple

from cognitive_engine.models import Opinion

OpinionPair = Tuple[Opinion, Opinion]


@dataclass
class Priors:
    default_opinions: Dict[str, Opinion] = field(default_factory=lambda: {
        "empirical_pattern": (0.8, 0.1, 0.1, 0.5),
        "cognitive_hypothesis": (0.5, 0.2, 0.3, 0.5),
        "consensus_principle": (0.7, 0.1, 0.2, 0.5),
        "observational_claim": (0.4, 0.3, 0.3, 0.5),
        "total_ignorance": (0.0, 0.0, 1.0, 0.5),
    })

    source_type_map: Dict[str, str] = field(default_factory=lambda: {
        "EVIDENCE": "empirical_pattern",
        "CLAIM": "consensus_principle",
        "COUNTERCLAIM": "observational_claim",
        "CONDITION": "cognitive_hypothesis",
        "AXIOM": "consensus_principle",
        "FALLACY": "observational_claim",
        "JUSTIFICATION": "empirical_pattern",
    })

    edge_warrants: Dict[str, OpinionPair] = field(default_factory=lambda: {
        "SUPPORTS": ((0.85, 0.1, 0.05, 0.5), (0.15, 0.8, 0.05, 0.5)),
        "CONTRADICTS": ((0.1, 0.85, 0.05, 0.5), (0.8, 0.15, 0.05, 0.5)),
        "QUALIFIES": ((0.6, 0.2, 0.2, 0.5), (0.4, 0.4, 0.2, 0.5)),
        "INFERS": ((0.9, 0.05, 0.05, 0.5), (0.0, 1.0, 0.0, 0.5)),
        "JUSTIFIES": ((0.8, 0.1, 0.1, 0.5), (0.2, 0.7, 0.1, 0.5)),
        "ATTACKS": ((0.05, 0.85, 0.1, 0.5), (0.85, 0.1, 0.05, 0.5)),
        "REBUTS": ((0.6, 0.3, 0.1, 0.5), (0.3, 0.6, 0.1, 0.5)),
    })

    default_warrant: OpinionPair = ((0.9, 0.05, 0.05, 0.5), (0.0, 1.0, 0.0, 0.5))

    def to_dict(self) -> dict:
        return {
            "default_opinions": {k: list(v) for k, v in self.default_opinions.items()},
            "source_type_map": dict(self.source_type_map),
            "edge_warrants": {
                k: [list(a), list(b)]
                for k, (a, b) in self.edge_warrants.items()
            },
            "default_warrant": [list(self.default_warrant[0]), list(self.default_warrant[1])],
        }


def bundled_priors_path() -> Path:
    pkg = resource_files("cognitive_engine")
    return Path(str(pkg.joinpath("default_priors.json")))


def load_priors(path: str | Path | None = None) -> Priors:
    if path is None:
        return Priors()

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Priors config not found: {p}")

    data = json.loads(p.read_text())
    default_opinions = {
        k: tuple(v) for k, v in data.get("default_opinions", {}).items()
    }
    source_type_map = dict(data.get("source_type_map", {}))
    edge_warrants = {
        k: (tuple(v[0]), tuple(v[1]))
        for k, v in data.get("edge_warrants", {}).items()
    }
    dw = data.get("default_warrant", [])
    default_warrant = (tuple(dw[0]), tuple(dw[1])) if len(dw) == 2 else Priors().default_warrant

    return Priors(
        default_opinions=default_opinions,
        source_type_map=source_type_map,
        edge_warrants=edge_warrants,
        default_warrant=default_warrant,
    )


def sweep_priors(base: Priors, delta: float = 0.1) -> list[Priors]:
    variants: list[Priors] = [base]

    for key in list(base.default_opinions):
        b, d, u, a = base.default_opinions[key]
        for sign in [-1, 1]:
            variant = deepcopy(base)
            nb = max(0.0, min(1.0, b + sign * delta))
            nd = max(0.0, min(1.0, d))
            nu = max(0.0, min(1.0, 1.0 - nb - nd))
            variant.default_opinions[key] = (nb, nd, nu, a)
            variants.append(variant)

    for key in list(base.edge_warrants):
        (b1, d1, u1, a1), (b2, d2, u2, a2) = base.edge_warrants[key]
        for sign in [-1, 1]:
            variant = deepcopy(base)
            nb1 = max(0.0, min(1.0, b1 + sign * delta))
            nd1 = max(0.0, min(1.0, d1))
            nu1 = max(0.0, min(1.0, 1.0 - nb1 - nd1))
            variant.edge_warrants[key] = ((nb1, nd1, nu1, a1), (b2, d2, u2, a2))
            variants.append(variant)

    return variants
