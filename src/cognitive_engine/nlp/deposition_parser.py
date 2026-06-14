"""Deposition-specific pre-parser.

Parses deposition transcripts to extract structure: header, Q/A pairs,
examiner identity, and examination type. Filters non-propositional text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ExaminationType(Enum):
    DIRECT = auto()
    CROSS = auto()
    REDIRECT = auto()
    REBUTTAL = auto()
    UNKNOWN = auto()


@dataclass
class QA:
    question: str
    answer: str
    speaker_q: str = ""
    speaker_a: str = ""
    char_start: int = 0
    char_end: int = 0


@dataclass
class DepositionSection:
    examiner: str
    exam_type: ExaminationType
    qa_pairs: List[QA] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0


@dataclass
class ParsedDeposition:
    header: Dict[str, str] = field(default_factory=dict)
    sections: List[DepositionSection] = field(default_factory=list)
    raw_text: str = ""
    is_deposition: bool = False


_HEADER_PATTERNS = [
    (re.compile(r"^CASE\s+NO\.?\s*:?\s*(.+)$", re.IGNORECASE), "case_number"),
    (re.compile(r"^DEPOSITION\s+OF\s+(.+)$", re.IGNORECASE), "deponent"),
    (re.compile(r"^DATE\s*:?\s*(.+)$", re.IGNORECASE), "date"),
    (re.compile(r"^VOLUME?\s*:?\s*(.+)$", re.IGNORECASE), "volume"),
    (re.compile(r"^Pages?\s*:?\s*(.+)$", re.IGNORECASE), "pages"),
]

_EXAMINATION_PATTERNS = [
    (re.compile(r"^Direct\s+Examination\s+by\s+(.+):", re.IGNORECASE), ExaminationType.DIRECT),
    (re.compile(r"^Cross\s*-?\s*Examination\s+by\s+(.+):", re.IGNORECASE), ExaminationType.CROSS),
    (re.compile(r"^Redirect\s+Examination\s+by\s+(.+):", re.IGNORECASE), ExaminationType.REDIRECT),
    (re.compile(r"^Rebuttal\s+Examination\s+by\s+(.+):", re.IGNORECASE), ExaminationType.REBUTTAL),
    (re.compile(r"^Recross\s*-?\s*Examination\s+by\s+(.+):", re.IGNORECASE), ExaminationType.CROSS),
]

_QA_PATTERN = re.compile(
    r"^(Q|A)\s*:\s*(.+)$", re.IGNORECASE,
)


def _is_deposition_header(text: str) -> bool:
    """Check if the text looks like a deposition header line."""
    t = text.strip()
    return bool(
        re.match(r"^CASE\s+NO", t, re.IGNORECASE) or
        re.match(r"^DEPOSITION\s+OF", t, re.IGNORECASE) or
        re.match(r"^DATE\s*:", t, re.IGNORECASE) or
        re.match(r"^VOLUME?\s*:", t, re.IGNORECASE) or
        re.match(r"^Pages?\s*:", t, re.IGNORECASE) or
        re.match(r"^\d+\s*$", t)
    )


def _is_examination_line(text: str) -> bool:
    return any(p.match(text.strip()) for p, _ in _EXAMINATION_PATTERNS)


def parse_deposition(text: str) -> ParsedDeposition:
    """Parse a deposition transcript into structured sections.

    Returns ParsedDeposition with header, sections (each containing
    Q/A pairs), and examination metadata.
    """
    lines = text.split("\n")
    result = ParsedDeposition(raw_text=text)

    header_end = 0
    current_section: Optional[DepositionSection] = None
    current_q: Optional[str] = None
    current_q_start = 0
    char_offset = 0

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        line_start = char_offset
        line_end = char_offset + len(line)
        char_offset = line_end + 1  # +1 for \n

        if not line_stripped:
            continue

        # ── Header parsing ──────────────────────────────────────
        for pattern, key in _HEADER_PATTERNS:
            m = pattern.match(line_stripped)
            if m:
                result.header[key] = m.group(1).strip()
                result.is_deposition = True
                header_end = line_end
                break
        else:
            # ── Examination line ────────────────────────────────
            exam_matched = False
            for pattern, exam_type in _EXAMINATION_PATTERNS:
                m = pattern.match(line_stripped)
                if m:
                    examiner = m.group(1).strip().rstrip(":")
                    current_section = DepositionSection(
                        examiner=examiner,
                        exam_type=exam_type,
                        char_start=line_start,
                    )
                    result.sections.append(current_section)
                    result.is_deposition = True
                    exam_matched = True
                    break

            if exam_matched:
                continue

            # ── Q/A parsing ─────────────────────────────────────
            m = _QA_PATTERN.match(line_stripped)
            if m:
                qa_type = m.group(1).upper()
                content = m.group(2).strip()

                if qa_type == "Q":
                    current_q = content
                    current_q_start = line_start
                elif qa_type == "A" and current_q is not None and current_section is not None:
                    qa = QA(
                        question=current_q,
                        answer=content,
                        speaker_q=current_section.examiner,
                        char_start=current_q_start,
                        char_end=line_end,
                    )
                    current_section.qa_pairs.append(qa)
                    current_q = None
                continue

            # ── Continuation of previous answer ─────────────────
            if current_q is None and current_section and current_section.qa_pairs:
                last_qa = current_section.qa_pairs[-1]
                if last_qa.answer and not _is_deposition_header(line_stripped):
                    last_qa.answer += " " + line_stripped
                    last_qa.char_end = line_end

    if not result.sections:
        result.is_deposition = False

    logger.info(
        "Parsed deposition: is_deposition=%s, sections=%d, total_qa=%d",
        result.is_deposition,
        len(result.sections),
        sum(len(s.qa_pairs) for s in result.sections),
    )
    return result


def get_propositional_text(parsed: ParsedDeposition) -> str:
    """Extract only the propositional content (answers) from a parsed deposition.

    Filters out header, questions, and examination markers.
    Returns clean text suitable for NLP extraction.
    """
    parts: List[str] = []
    for section in parsed.sections:
        for qa in section.qa_pairs:
            parts.append(qa.answer)
    return "\n".join(parts)


def get_full_propositional_text(parsed: ParsedDeposition) -> str:
    """Extract questions AND answers as propositional text.

    Questions can contain implicit claims (e.g., "you said you had a beer"
    implies the deponent did have a beer).
    """
    parts: List[str] = []
    for section in parsed.sections:
        for qa in section.qa_pairs:
            parts.append(qa.question)
            parts.append(qa.answer)
    return "\n".join(parts)
