"""Lazy-loading singleton for NLTK semantic resources.

Provides O(1) lookup for FrameNet frames, PropBank rolesets,
and VerbNet classes after initial load. Thread-safe via double-checked
locking pattern.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level singleton
_instance: Optional["SemanticResources"] = None
_lock = threading.Lock()


class SemanticResources:
    """Lazy-loading wrapper for NLTK FrameNet, PropBank, VerbNet.

    Usage:
        res = SemanticResources.instance()
        frames = res.frames_for_lemma("fry")
        vn_classes = res.vn_classes_for_lemma("give")
        roles = res.vn_themroles("give-13.1")
    """

    def __init__(self) -> None:
        self._fn_loaded = False
        self._pb_loaded = False
        self._vn_loaded = False

        # FrameNet: lemma → list of frame names
        self._frames_by_lemma: dict[str, list[str]] = {}
        # FrameNet: frame name → frame object (lazy)
        self._frame_cache: dict[str, object] = {}
        # FrameNet: frame name → list of frame element names
        self._fe_by_frame: dict[str, list[str]] = {}

        # VerbNet: lemma → list of class IDs
        self._vn_by_lemma: dict[str, list[str]] = {}
        # VerbNet: class_id → list of thematic role dicts
        self._themroles_by_class: dict[str, list[dict]] = {}
        # VerbNet: class_id → list of syntactic frame dicts
        self._frames_by_class: dict[str, list[dict]] = {}

        # PropBank: roleset_id → roleset object (lazy)
        self._pb_cache: dict[str, object] = {}

    @classmethod
    def instance(cls) -> "SemanticResources":
        """Get or create the singleton instance."""
        global _instance
        if _instance is None:
            with _lock:
                if _instance is None:
                    _instance = cls()
        return _instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        global _instance
        with _lock:
            _instance = None

    # ── FrameNet ──────────────────────────────────────────────────

    def _ensure_fn(self) -> None:
        if self._fn_loaded:
            return
        try:
            from nltk.corpus import framenet as fn
            for frame in fn.frames():
                name = frame.name
                self._frame_cache[name] = frame
                # Index by all lexical unit lemmas
                for lu in frame.lexUnit.values():
                    lemma = lu.name.split(".")[0]
                    self._frames_by_lemma.setdefault(lemma.lower(), []).append(name)
                # Index frame elements
                self._fe_by_frame[name] = [fe.name for fe in frame.FE.values()]
            self._fn_loaded = True
            logger.info("FrameNet loaded: %d frames, %d lemma entries",
                        len(self._frame_cache), len(self._frames_by_lemma))
        except Exception as e:
            logger.warning("FrameNet unavailable: %s", e)
            self._fn_loaded = True  # Don't retry

    def frames_for_lemma(self, lemma: str) -> list[str]:
        """Get frame names evoked by a verb lemma."""
        self._ensure_fn()
        return self._frames_by_lemma.get(lemma.lower(), [])

    def frame_elements(self, frame_name: str) -> list[str]:
        """Get frame element names for a frame."""
        self._ensure_fn()
        return self._fe_by_frame.get(frame_name, [])

    def frame_object(self, frame_name: str) -> Optional[object]:
        """Get the full FrameNet frame object."""
        self._ensure_fn()
        return self._frame_cache.get(frame_name)

    def has_frame(self, frame_name: str) -> bool:
        """Check if a frame exists."""
        self._ensure_fn()
        return frame_name in self._frame_cache

    # ── VerbNet ───────────────────────────────────────────────────

    def _ensure_vn(self) -> None:
        if self._vn_loaded:
            return
        try:
            from nltk.corpus import verbnet as vn
            for class_id in vn.classids():
                # Index by all lemmas in the class
                lemmas = vn.lemmas(class_id)
                for lemma in lemmas:
                    self._vn_by_lemma.setdefault(lemma.lower(), []).append(class_id)
                # Index thematic roles
                themroles = vn.themroles(class_id)
                self._themroles_by_class[class_id] = themroles
                # Index syntactic frames
                frames = vn.frames(class_id)
                self._frames_by_class[class_id] = [
                    {"description": getattr(f, "description", ""),
                     "syntax": getattr(f, "syntax", [])}
                    for f in frames
                ]
            self._vn_loaded = True
            logger.info("VerbNet loaded: %d classes, %d lemma entries",
                        len(self._themroles_by_class), len(self._vn_by_lemma))
        except Exception as e:
            logger.warning("VerbNet unavailable: %s", e)
            self._vn_loaded = True

    def vn_classes_for_lemma(self, lemma: str) -> list[str]:
        """Get VerbNet class IDs for a verb lemma."""
        self._ensure_vn()
        return self._vn_by_lemma.get(lemma.lower(), [])

    def vn_themroles(self, class_id: str) -> list[dict]:
        """Get thematic roles for a VerbNet class.

        Returns list of dicts with keys: 'type', 'modifiers'.
        """
        self._ensure_vn()
        return self._themroles_by_class.get(class_id, [])

    def vn_frames(self, class_id: str) -> list[dict]:
        """Get syntactic frames for a VerbNet class."""
        self._ensure_vn()
        return self._frames_by_class.get(class_id, [])

    def vn_all_roles(self, lemma: str) -> list[str]:
        """Get all thematic roles across all VerbNet classes for a lemma."""
        classes = self.vn_classes_for_lemma(lemma)
        roles = set()
        for cid in classes:
            for mr in self.vn_themroles(cid):
                roles.add(mr.get("type", ""))
        return sorted(roles - {""})

    # ── PropBank ──────────────────────────────────────────────────

    def _ensure_pb(self) -> None:
        if self._pb_loaded:
            return
        try:
            from nltk.corpus import propbank as pb
            # Pre-index rolesets by verb lemma (skip corrupt ones)
            for inst in pb.instances():
                rs_id = inst.roleset
                if rs_id not in self._pb_cache:
                    try:
                        self._pb_cache[rs_id] = pb.roleset(rs_id)
                    except Exception:
                        logger.debug("Skipping corrupt PropBank roleset: %s", rs_id)
            self._pb_loaded = True
            logger.info("PropBank loaded: %d rolesets", len(self._pb_cache))
        except Exception as e:
            logger.warning("PropBank unavailable: %s", e)
            self._pb_loaded = True

    def pb_roleset(self, roleset_id: str) -> Optional[object]:
        """Get a PropBank roleset by ID (e.g., 'give.01')."""
        self._ensure_pb()
        return self._pb_cache.get(roleset_id)

    def pb_rolesets_for_lemma(self, lemma: str) -> list[object]:
        """Get all PropBank rolesets for a verb lemma."""
        self._ensure_pb()
        prefix = lemma.lower() + "."
        return [rs for rid, rs in self._pb_cache.items() if rid.startswith(prefix)]
