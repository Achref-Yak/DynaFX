import importlib as _importlib

def _load_dotenv():
    import os
    from pathlib import Path
    cwd = Path.cwd()
    for p in [cwd, *cwd.parents]:
        env_path = p / ".env"
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip()
                            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                                v = v[1:-1]
                            if k and k not in os.environ:
                                os.environ[k] = v
                break
            except Exception:
                pass
    if "HG_TOKEN" in os.environ and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HG_TOKEN"]

_load_dotenv()

_lazy_modules: dict[str, list[str]] = {
    "core.models": [
        "Graph", "Node", "NodeType", "Edge", "EdgeType",
        "Opinion", "Warrant", "Violation", "Severity",
        "ReasoningMode", "FusionSituation", "ConversationTree",
        "EvidenceCounts",
    ],
    "core.state": ["State"],
    "core.trace": ["StateDelta", "TraceBuffer"],
    "core.loom": ["Weave", "weave"],
    "reason.evidence": ["CorpusResult"],
    "kernel.assertion_gate": ["AssertionGate", "Assertion", "GateResult"],
    "kernel.inference_cycle": ["InferenceCycle", "CycleReport", "InferenceResult"],
    "policy.schema": ["WhenCondition", "ThenAction", "PolicyRule", "OperatorPolicy"],
    "policy.engine": ["PolicyEngine"],
    "policy.builtin": ["DEFAULT_POLICY", "SCIENTIFIC_POLICY", "BUILTIN_POLICIES"],
    "tbox.loader": ["TBox", "load_tbox", "GENERAL_TBOX"],
    "tbox.hierarchy": ["TypeNode", "TypeHierarchy", "MDM_TYPE_HIERARCHY"],
    "core.higraph": ["Blob", "Interaction", "StructuralRelationship", "Hierarchy", "Partitions"],
    "core.events": ["Event", "EventBus", "get_event_bus"],
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
        "extract_max_dag", "topological_sort", "tna_propagate",
        "check_category_monotonicity", "betweenness_centrality",
        "leverage_score", "classify_feedback_loop",
        "shannon_entropy", "normalized_shannon_entropy",
        "pointwise_mutual_information", "normalized_pmi",
        "mutual_information_from_counts", "entropy_of_distribution",
        "interaction_complexity", "domain_entanglement", "blob_type_entropy",
    ],
    "mp.scenario": ["Event", "ScenarioGenerator", "Assertion"],
    "perception.hypothesis_generator": ["HypothesisGenerator"],
}

_lazy_map: dict[str, str] = {}
for _mod_path, _names in _lazy_modules.items():
    for _name in _names:
        _lazy_map[_name] = _mod_path

__all__ = [*list(_lazy_map.keys()), "loom"]


def __getattr__(name: str):
    if name in _lazy_map:
        mod = _importlib.import_module(f"cognitive_engine.{_lazy_map[name]}")
        return getattr(mod, name)
    if name == "loom":
        return _importlib.import_module("cognitive_engine.core.loom")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
