from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Optional

from dynafx.domain import domain as _domain


@dataclass
class Priors:
    source_type_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cfg = _domain.active()
        if not self.source_type_map:
            self.source_type_map = dict(cfg.source_type_map)

    def to_dict(self) -> dict:
        return {
            "source_type_map": dict(self.source_type_map),
        }


def bundled_priors_path() -> Path:
    pkg = resource_files("dynafx")
    return Path(str(pkg.joinpath("default_priors.json")))


def load_priors(path: str | Path | None = None) -> Priors:
    if path is None:
        return Priors()

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Priors config not found: {p}")

    data = json.loads(p.read_text())

    source_type_map = dict(data.get("source_type_map", {}))

    return Priors(
        source_type_map=source_type_map,
    )
