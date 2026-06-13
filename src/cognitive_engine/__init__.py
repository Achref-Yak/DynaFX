import importlib as _importlib

_lazy_modules: dict[str, list[str]] = {
    "core.models": [
        "Graph", "Node", "NodeType", "Edge", "EdgeType",
        "Opinion", "Warrant", "Violation", "Severity",
        "ReasoningMode", "FusionSituation", "ConversationTree",
        "EvidenceCounts",
    ],
    "reason.evidence": ["CorpusResult"],
    "pipeline.extraction": ["extract_graph"],
    "pipeline.orchestrator": ["orchestrator_run"],
}

_lazy_map: dict[str, str] = {}
for _mod_path, _names in _lazy_modules.items():
    for _name in _names:
        _lazy_map[_name] = _mod_path

__all__ = list(_lazy_map.keys())


def __getattr__(name: str):
    if name in _lazy_map:
        mod = _importlib.import_module(f"cognitive_engine.{_lazy_map[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
