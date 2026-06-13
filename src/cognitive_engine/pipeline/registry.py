from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Set

from cognitive_engine.core.models import Graph

logger = logging.getLogger(__name__)

ModuleFn = Callable[..., None]


class ModuleDef:
    def __init__(
        self,
        name: str,
        dependencies: List[str],
        fn: ModuleFn,
    ) -> None:
        self.name = name
        self.dependencies = dependencies
        self.fn = fn


class ModuleRegistry:
    def __init__(self) -> None:
        self._modules: Dict[str, ModuleDef] = {}

    def register(self, mod: ModuleDef) -> None:
        self._modules[mod.name] = mod

    def run(
        self,
        names: List[str],
        graph: Graph,
        **context: Any,
    ) -> Graph:
        ordered = self._resolve(names)
        for name in ordered:
            mod = self._modules[name]
            logger.debug("Running module: %s", name)
            mod.fn(graph, **context)
        return graph

    def _resolve(self, names: List[str]) -> List[str]:
        all_mods: Dict[str, ModuleDef] = {}
        for n in names:
            if n in self._modules:
                all_mods[n] = self._modules[n]
            else:
                raise ValueError(f"Unknown module: {n}")

        resolved: List[str] = []
        visited: Set[str] = set()

        def dfs(name: str) -> None:
            if name in resolved:
                return
            if name in visited:
                raise ValueError(f"Circular dependency detected: {name}")
            visited.add(name)
            mod = all_mods.get(name)
            if mod:
                for dep in mod.dependencies:
                    if dep not in all_mods:
                        raise ValueError(
                            f"Module '{name}' depends on '{dep}' which is not in the run list"
                        )
                    dfs(dep)
                if name not in resolved:
                    resolved.append(name)

        for n in names:
            dfs(n)

        return resolved
