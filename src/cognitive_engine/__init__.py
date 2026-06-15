import importlib as _importlib

_lazy_modules: dict[str, list[str]] = {
    "core.models": [
        "Graph", "Node", "NodeType", "Edge", "EdgeType",
        "Opinion", "Warrant", "Violation", "Severity",
        "ReasoningMode", "FusionSituation", "ConversationTree",
        "EvidenceCounts",
    ],
    "core.math": [
        "master_equation_all", "master_equation", "global_objective",
        "propagate_step", "build_adjacency", "initialize_beliefs",
        "compute_support_sum", "compute_attack_sum", "count_violations",
        "compute_logic_penalty", "convergence_norm",
        "modus_ponens_strength", "inference_closure",
        "conjunction", "disjunction", "conditional_deduction",
        "cumulative_fusion", "consensus_compromise",
        "bayes_rule", "joint_probability", "expectation",
        "argument_acceptability", "dung_semantics",
        "opinion_conflict", "reverse_warrant", "subjective_abduction",
        "analogy_warrant_transform", "similarity_diffusion",
        "neurosymbolic_fuse", "graph_distance", "hidden_state_distance",
        "memory_similarity", "graph_diff_score",
        "check_opinion_invariant", "check_cycle_free",
        "check_category_monotonicity", "betweenness_centrality",
        "leverage_score", "classify_feedback_loop",
    ],
    "core.state": ["State"],
    "core.trace": ["StateDelta", "TraceBuffer"],
    "reason.evidence": ["CorpusResult"],
    "kernel.assertion_gate": ["AssertionGate", "Assertion", "GateResult"],
    "kernel.inference_cycle": ["InferenceCycle", "CycleReport", "InferenceResult"],
    "policy.schema": ["WhenCondition", "ThenAction", "PolicyRule", "OperatorPolicy"],
    "policy.engine": ["PolicyEngine"],
    "policy.builtin": ["DEFAULT_POLICY", "LEGAL_POLICY", "SCIENTIFIC_POLICY", "BUILTIN_POLICIES"],
    "tbox.loader": ["TBox", "load_tbox", "GENERAL_TBOX"],
    "tbox.legal": ["LEGAL_TBOX"],
    "perception.hypothesis_generator": ["HypothesisGenerator"],
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
