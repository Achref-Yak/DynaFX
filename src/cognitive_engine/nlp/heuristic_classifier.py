"""Rule-based relation classifier to replace the broken DistilRoBERTa model.

The original RelationClassifier has eval_attack_recall=0.012 — it outputs
"Support" for ~99% of pairs. This module uses linguistic heuristics to
produce meaningful Support/Attack/None labels.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Negation markers ──────────────────────────────────────────────
_NEGATION_WORDS = frozenset({
    "not", "never", "no", "neither", "nor", "cannot", "can't",
    "won't", "doesn't", "didn't", "isn't", "aren't", "wasn't",
    "weren't", "haven't", "hasn't", "hadn't", "wouldn't", "shouldn't",
    "couldn't", "mustn't", "nobody", "nothing", "nowhere", "hardly",
    "barely", "scarcely",
})

# ── Contradictory pairs ───────────────────────────────────────────
_CONTRADICTION_LEXICAL = [
    ({"always", "always"}, {"never", "never"}),
    ({"all", "every", "everyone", "everything"}, {"none", "no one", "nobody", "nothing"}),
    ({"increase", "increased", "increases", "rise", "rose", "grow", "grew"},
     {"decrease", "decreased", "decreases", "fall", "fell", "decline", "declined"}),
    ({"safe", "safety", "secure", "security"},
     {"dangerous", "danger", "hazard", "hazardous", "risky", "risk"}),
    ({"guilty", "liable", "responsible"},
     {"innocent", "not liable", "not responsible"}),
    ({"before", "prior to"},
     {"after", "subsequent to", "following"}),
    ({"more", "greater", "higher", "larger"},
     {"less", "fewer", "smaller", "lower"}),
    ({"must", "shall", "required", "mandatory"},
     {"may", "might", "optional", "permitted"}),
]

# ── Modal strength ────────────────────────────────────────────────
_STRONG_MODAL = {"must", "shall", "will", "always", "certainly", "definitely", "undoubtedly"}
_WEAK_MODAL = {"may", "might", "could", "possibly", "perhaps", "sometimes", "possibly"}

# ── Adversative conjunctions ──────────────────────────────────────
_ADVERSATIVES = {"but", "however", "although", "nevertheless", "conversely",
                 "yet", "whereas", "nonetheless", "on the other hand",
                 "in contrast", "despite", "actually", "in fact"}

# ── Q/A detection ─────────────────────────────────────────────────
_Q_STARTERS = re.compile(
    r"^(Q:|Who|What|When|Where|Why|How|Did|Does|Do|Is|Are|Was|Were|"
    r"Have|Has|Had|Can|Could|Would|Should|Will|Shall|Mr\.|Mrs\.|Ms\.)",
    re.IGNORECASE,
)

# ── Hearsay markers ───────────────────────────────────────────────
_HEARSAY_MARKERS = {"said", "told", "stated", "mentioned", "claimed", "reported",
                    "according to", "he said", "she said", "they said"}

# ── Shared content nouns (for support detection) ──────────────────
_MIN_SHARED_NOUNS = 2

# ── World-model lexical signals ──────────────────────────────────
_CAUSAL_LEXICAL = frozenset({
    "cause", "caused", "leads", "results", "produces", "triggers",
    "because", "due to", "therefore", "consequently", "thus",
})
_ENABLEMENT_LEXICAL = frozenset({
    "enable", "enables", "allows", "facilitates", "supports",
    "empowers", "permits", "makes possible",
})
_PART_WHOLE_LEXICAL = frozenset({
    "part of", "consists of", "contains", "includes",
    "member of", "component of", "subset of",
})
_DEPENDENCY_LEXICAL = frozenset({
    "depends on", "requires", "needs", "relies on", "contingent on",
})


def _normalize(text: str) -> str:
    return text.lower().strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", _normalize(text))


def _has_negation(text: str) -> bool:
    tokens = set(_tokenize(text))
    return bool(tokens & _NEGATION_WORDS)


def _get_nouns(text: str) -> set[str]:
    """Extract simple noun-like tokens (words > 3 chars, not stopwords)."""
    _stopwords = {"the", "and", "that", "this", "with", "from", "have", "been",
                  "were", "they", "their", "there", "then", "than", "what",
                  "when", "where", "which", "while", "about", "would", "could",
                  "should", "will", "just", "also", "into", "over", "only"}
    tokens = _tokenize(text)
    return {t for t in tokens if len(t) > 3 and t not in _stopwords}


def _contradiction_score(text_a: str, text_b: str) -> float:
    """Check for contradictory lexical items between two texts."""
    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))

    for set_a, set_b in _CONTRADICTION_LEXICAL:
        if (tokens_a & set_a and tokens_b & set_b) or (tokens_a & set_b and tokens_b & set_a):
            return 1.0
    return 0.0


def _negation_asymmetry(text_a: str, text_b: str) -> float:
    """If one text negates and the other affirms the same topic."""
    neg_a = _has_negation(text_a)
    neg_b = _has_negation(text_b)

    if neg_a == neg_b:
        return 0.0

    nouns_a = _get_nouns(text_a)
    nouns_b = _get_nouns(text_b)
    shared = nouns_a & nouns_b

    if len(shared) >= 2:
        return 0.8
    return 0.0


def _modal_conflict(text_a: str, text_b: str) -> float:
    """Strong vs weak modal assertion of same topic."""
    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))

    strong_a = bool(tokens_a & _STRONG_MODAL)
    weak_a = bool(tokens_a & _WEAK_MODAL)
    strong_b = bool(tokens_b & _STRONG_MODAL)
    weak_b = bool(tokens_b & _WEAK_MODAL)

    if (strong_a and weak_b) or (strong_b and weak_a):
        nouns_a = _get_nouns(text_a)
        nouns_b = _get_nouns(text_b)
        if len(nouns_a & nouns_b) >= 2:
            return 0.6
    return 0.0


def _adversative_score(text_a: str, text_b: str) -> float:
    """If one text starts with an adversative conjunction."""
    for text in (text_a, text_b):
        first_words = _normalize(text).split()[:3]
        if any(adj in " ".join(first_words) for adj in _ADVERSATIVES):
            return 0.7
    return 0.0


def _qa_structure_score(text_a: str, text_b: str) -> Optional[float]:
    """Detect Q/A structure and classify based on answer content."""
    is_q_a = bool(_Q_STARTERS.match(text_a.strip()))
    is_q_b = bool(_Q_STARTERS.match(text_b.strip()))

    if is_q_a and not is_q_b:
        answer = _normalize(text_b)
        if _has_negation(answer) or any(w in answer for w in
            {"no", "not", "never", "none", "neither", "nobody", "nothing"}):
            return -0.5
        return 0.5
    return None


def _topic_support_score(text_a: str, text_b: str) -> float:
    """Shared nouns without negation conflict suggest support."""
    nouns_a = _get_nouns(text_a)
    nouns_b = _get_nouns(text_b)
    shared = nouns_a & nouns_b

    if len(shared) >= _MIN_SHARED_NOUNS:
        if not _has_negation(text_a) and not _has_negation(text_b):
            return 0.4
    return 0.0


def _hearsay_score(text_a: str, text_b: str) -> float:
    """Detect hearsay patterns."""
    for text in (text_a, text_b):
        lower = _normalize(text)
        if any(marker in lower for marker in _HEARSAY_MARKERS):
            return 0.3
    return 0.0


def _causal_score(text_a: str, text_b: str) -> float:
    """Detect causal relations between two texts."""
    norm_a = _normalize(text_a)
    norm_b = _normalize(text_b)
    for text in (norm_a, norm_b):
        if any(phrase in text for phrase in _CAUSAL_LEXICAL):
            return 0.7
    return 0.0


def _enablement_score(text_a: str, text_b: str) -> float:
    """Detect enablement relations between two texts."""
    text = (text_a + " " + text_b).lower()
    if any(phrase in text for phrase in _ENABLEMENT_LEXICAL):
        return 0.7
    return 0.0


def _part_whole_score(text_a: str, text_b: str) -> float:
    """Detect part-whole relations between two texts."""
    text = (text_a + " " + text_b).lower()
    if any(p in text for p in _PART_WHOLE_LEXICAL):
        return 0.7
    return 0.0


def _dependency_score(text_a: str, text_b: str) -> float:
    """Detect dependency relations between two texts."""
    text = (text_a + " " + text_b).lower()
    if any(p in text for p in _DEPENDENCY_LEXICAL):
        return 0.7
    return 0.0


class HeuristicClassifier:
    """Rule-based relation classifier.

    Interface-compatible with RelationClassifier: exposes classify(text_a, text_b) -> str.
    """

    def __init__(self, same_section: bool = False):
        """Args:
            same_section: If True, indicates both texts are from the same section.
                         Increases support likelihood.
        """
        self.same_section = same_section

    def classify(self, text_a: str, text_b: str) -> str:
        """Classify the relation between two text spans.

        Returns "Support", "Attack", "Causes", "Enables", "Depends",
        "PartOf", or "None".
        """
        # 1. Contradictory lexical items (strongest signal)
        c = _contradiction_score(text_a, text_b)
        if c > 0:
            return "Attack"

        # 2. Negation asymmetry (strong signal — one negates, other affirms)
        neg_a = _has_negation(text_a)
        neg_b = _has_negation(text_b)
        if neg_a != neg_b:
            nouns_a = _get_nouns(text_a)
            nouns_b = _get_nouns(text_b)
            shared = nouns_a & nouns_b
            if len(shared) >= 1:
                return "Attack"

        # 3. Adversative conjunctions (override support detection)
        adv = _adversative_score(text_a, text_b)
        if adv > 0:
            return "Attack"

        # 4. Q/A structure (supportive by default)
        qa = _qa_structure_score(text_a, text_b)
        if qa is not None:
            if qa >= 0.3:
                return "Support"
            if qa <= -0.3:
                return "Attack"

        # 5. Modal conflict
        modal = _modal_conflict(text_a, text_b)
        if modal > 0:
            return "Attack"

        # 6. Causal detection (before topic support — more specific)
        causal = _causal_score(text_a, text_b)
        if causal > 0:
            return "Causes"

        # 7. Enablement detection
        enable = _enablement_score(text_a, text_b)
        if enable > 0:
            return "Enables"

        # 8. Part-whole detection
        part = _part_whole_score(text_a, text_b)
        if part > 0:
            return "PartOf"

        # 9. Dependency detection
        dep = _dependency_score(text_a, text_b)
        if dep > 0:
            return "Depends"

        # 10. Topic support (weaker signal — after semantic-specific checks)
        topic = _topic_support_score(text_a, text_b)
        if topic >= 0.3:
            return "Support"

        # 11. Same section support — answers in same section are related
        if self.same_section:
            nouns_a = _get_nouns(text_a)
            nouns_b = _get_nouns(text_b)
            if nouns_a & nouns_b:
                return "Support"

        return "None"
