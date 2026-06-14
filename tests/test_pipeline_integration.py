"""Integration tests for the NLP extraction pipeline.

Tests the full pipeline from text → graph with the new heuristic classifier
and deposition parser.
"""

from __future__ import annotations

import pytest
from cognitive_engine.nlp.heuristic_classifier import HeuristicClassifier
from cognitive_engine.nlp.deposition_parser import parse_deposition


class TestHeuristicClassifierIntegration:
    """Test the classifier with realistic deposition content."""

    @pytest.fixture
    def clf(self):
        return HeuristicClassifier()

    def test_eyewitness_support(self, clf):
        q = "What did you observe at approximately 8:15 PM?"
        a = "I saw a blue sedan run the red light and strike a pedestrian in the crosswalk."
        assert clf.classify(q, a) == "Support"

    def test_cross_examination_attack(self, clf):
        q = "You were fifty feet away. Could you be mistaken about the color of the light?"
        a = "I'm certain it was red. I've been through that intersection every day for five years."
        # The question implies doubt, answer affirms — but no contradiction in content
        result = clf.classify(q, a)
        assert result in ("Support", "None")

    def test_denial_attack(self, clf):
        q = "Did you see the driver's face?"
        a = "No, it was too dark to see clearly."
        assert clf.classify(q, a) == "Attack"

    def test_conflicting_statements(self, clf):
        a1 = "The light was always red at that intersection."
        a2 = "The light was never red at that intersection."
        assert clf.classify(a1, a2) == "Attack"

    def test_corroboration(self, clf):
        a1 = "I saw the blue sedan run the red light."
        a2 = "The blue sedan was definitely running the red light."
        result = clf.classify(a1, a2)
        assert result in ("Support", "None")

    def test_qualification(self, clf):
        a1 = "He was speeding."
        a2 = "However, the road conditions were poor."
        assert clf.classify(a1, a2) == "Attack"

    def test_irrelevant_statements(self, clf):
        a1 = "I had coffee that morning."
        a2 = "The weather was clear and sunny."
        assert clf.classify(a1, a2) == "None"


class TestDepositionParserIntegration:
    """Test the parser with the full sample deposition."""

    SAMPLE = """DEPOSITION OF MARK HARRISON
CASE NO. 2024-CV-0342
DATE: MARCH 12, 2026

Direct Examination by Ms. Reynolds:

Q: Mr. Harrison, where were you on the evening of October 15th?
A: I was at the intersection of Main Street and Oak Avenue.

Q: What did you observe at approximately 8:15 PM?
A: I saw a blue sedan run the red light and strike a pedestrian in the crosswalk.

Q: How far away were you when this happened?
A: I was about fifty feet away, waiting at the bus stop.

Q: Had you consumed any alcohol that evening?
A: I had one beer with dinner around six o'clock.

Cross Examination by Mr. Thompson:

Q: Mr. Harrison, you said you had a beer at six o'clock. Is that correct?
A: Yes.

Q: And the accident was at 8:15 PM. So over two hours had passed?
A: That's right.

Q: You were fifty feet away. Could you be mistaken about the color of the light?
A: I'm certain it was red. I've been through that intersection every day for five years.
"""

    def test_full_parse(self):
        parsed = parse_deposition(self.SAMPLE)
        assert parsed.is_deposition is True
        assert len(parsed.sections) == 2
        total_qa = sum(len(s.qa_pairs) for s in parsed.sections)
        assert total_qa == 7

    def test_no_header_in_propositions(self):
        parsed = parse_deposition(self.SAMPLE)
        text = "\n".join(qa.answer for s in parsed.sections for qa in s.qa_pairs)
        assert "DEPOSITION OF" not in text
        assert "CASE NO." not in text
        assert "DATE:" not in text

    def test_no_questions_in_propositions(self):
        parsed = parse_deposition(self.SAMPLE)
        text = "\n".join(qa.answer for s in parsed.sections for qa in s.qa_pairs)
        assert "Q:" not in text

    def test_answers_preserved(self):
        parsed = parse_deposition(self.SAMPLE)
        text = "\n".join(qa.answer for s in parsed.sections for qa in s.qa_pairs)
        assert "Main Street" in text
        assert "blue sedan" in text
        assert "fifty feet away" in text


class TestClassifierWithDomainData:
    """Test classifier with domain-specific patterns."""

    @pytest.fixture
    def clf(self):
        return HeuristicClassifier()

    def test_hearsay_pattern(self, clf):
        a1 = "He said the light was yellow."
        a2 = "The traffic camera shows the light was red."
        result = clf.classify(a1, a2)
        assert result in ("Attack", "None")

    def test_expert_opinion(self, clf):
        q = "What is your expert opinion on the cause of the accident?"
        a = "Based on the skid marks and vehicle damage, the sedan was traveling at approximately 45 mph."
        assert clf.classify(q, a) == "Support"

    def test_impeachment(self, clf):
        a1 = "I was completely sober that night."
        a2 = "But you just admitted you had three beers."
        assert clf.classify(a1, a2) == "Attack"

    def test_prior_inconsistent(self, clf):
        a1 = "I saw the accident from my apartment window."
        a2 = "Actually, I wasn't really watching at the time."
        assert clf.classify(a1, a2) == "Attack"
