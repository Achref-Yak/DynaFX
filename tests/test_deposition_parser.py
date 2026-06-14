"""Tests for the deposition parser."""

from __future__ import annotations

import pytest
from cognitive_engine.nlp.deposition_parser import (
    ExaminationType,
    ParsedDeposition,
    parse_deposition,
    get_propositional_text,
    get_full_propositional_text,
)


SAMPLE_DEPOSITION = """DEPOSITION OF MARK HARRISON
CASE NO. 2024-CV-0342
DATE: MARCH 12, 2026

Direct Examination by Ms. Reynolds:

Q: Mr. Harrison, where were you on the evening of October 15th?
A: I was at the intersection of Main Street and Oak Avenue.

Q: What did you observe at approximately 8:15 PM?
A: I saw a blue sedan run the red light and strike a pedestrian in the crosswalk.

Cross Examination by Mr. Thompson:

Q: Mr. Harrison, you said you had a beer at six o'clock. Is that correct?
A: Yes.

Q: And the accident was at 8:15 PM. So over two hours had passed?
A: That's right.
"""


def test_is_deposition():
    parsed = parse_deposition(SAMPLE_DEPOSITION)
    assert parsed.is_deposition is True


def test_header_extraction():
    parsed = parse_deposition(SAMPLE_DEPOSITION)
    assert "case_number" in parsed.header
    assert "2024-CV-0342" in parsed.header["case_number"]
    assert "deponent" in parsed.header
    assert "MARK HARRISON" in parsed.header["deponent"].upper()


def test_section_extraction():
    parsed = parse_deposition(SAMPLE_DEPOSITION)
    assert len(parsed.sections) == 2
    assert parsed.sections[0].exam_type == ExaminationType.DIRECT
    assert parsed.sections[0].examiner == "Ms. Reynolds"
    assert parsed.sections[1].exam_type == ExaminationType.CROSS
    assert parsed.sections[1].examiner == "Mr. Thompson"


def test_qa_pairs():
    parsed = parse_deposition(SAMPLE_DEPOSITION)
    assert len(parsed.sections[0].qa_pairs) == 2
    assert len(parsed.sections[1].qa_pairs) == 2

    first_qa = parsed.sections[0].qa_pairs[0]
    assert "where were you" in first_qa.question.lower()
    assert "Main Street" in first_qa.answer
    assert first_qa.speaker_q == "Ms. Reynolds"


def test_propositional_text():
    parsed = parse_deposition(SAMPLE_DEPOSITION)
    text = get_propositional_text(parsed)
    assert "Main Street" in text
    assert "blue sedan" in text
    assert "Q:" not in text
    assert "Direct Examination" not in text


def test_full_propositional_text():
    parsed = parse_deposition(SAMPLE_DEPOSITION)
    text = get_full_propositional_text(parsed)
    assert "where were you" in text
    assert "Main Street" in text


def test_non_deposition():
    parsed = parse_deposition("This is just a plain text document with no structure.")
    assert parsed.is_deposition is False
    assert len(parsed.sections) == 0


def test_redirect_examination():
    text = """DEPOSITION OF JOHN DOE
CASE NO. 2024-CV-0001

Direct Examination by Ms. Smith:

Q: What did you see?
A: I saw the accident.

Redirect Examination by Ms. Smith:

Q: You mentioned earlier it was raining?
A: Yes, it was raining.
"""
    parsed = parse_deposition(text)
    assert parsed.is_deposition is True
    assert len(parsed.sections) == 2
    assert parsed.sections[1].exam_type == ExaminationType.REDIRECT


def test_empty_deposition():
    parsed = parse_deposition("")
    assert parsed.is_deposition is False
    assert len(parsed.sections) == 0


def test_qa_char_positions():
    parsed = parse_deposition(SAMPLE_DEPOSITION)
    qa = parsed.sections[0].qa_pairs[0]
    assert qa.char_start < qa.char_end
    assert qa.char_start >= 0
