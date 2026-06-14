"""Tests for the heuristic relation classifier."""

from __future__ import annotations

import pytest
from cognitive_engine.nlp.heuristic_classifier import HeuristicClassifier


@pytest.fixture
def clf():
    return HeuristicClassifier()


class TestContradictionDetection:
    def test_always_vs_never(self, clf):
        assert clf.classify("The light was always red", "The light was never red") == "Attack"

    def test_all_vs_none(self, clf):
        assert clf.classify("All witnesses saw the accident", "None of the witnesses saw the accident") == "Attack"

    def test_increased_vs_decreased(self, clf):
        assert clf.classify("Speed increased before impact", "Speed decreased before impact") == "Attack"

    def test_safe_vs_dangerous(self, clf):
        assert clf.classify("The intersection is safe", "The intersection is dangerous") == "Attack"

    def test_guilty_vs_innocent(self, clf):
        assert clf.classify("The driver was guilty", "The driver was innocent") == "Attack"

    def test_before_vs_after(self, clf):
        assert clf.classify("The light was green before", "The light was green after") == "Attack"


class TestNegationAsymmetry:
    def test_one_negates(self, clf):
        result = clf.classify("The driver ran the red light", "The driver did not run the red light")
        assert result == "Attack"

    def test_both_affirm(self, clf):
        result = clf.classify("I saw the accident", "I witnessed the accident")
        assert result in ("Support", "None")

    def test_negation_with_shared_nouns(self, clf):
        result = clf.classify("The car was speeding", "The car was not speeding")
        assert result == "Attack"


class TestModalConflict:
    def test_strong_vs_weak(self, clf):
        result = clf.classify("The light must have been red", "The light may have been yellow")
        assert result == "Attack"

    def test_same_strength(self, clf):
        result = clf.classify("He will testify", "She will testify")
        assert result in ("Support", "None")


class TestAdversativeConjunctions:
    def test_however(self, clf):
        result = clf.classify("The witness was credible", "However, his memory was questionable")
        assert result == "Attack"

    def test_but(self, clf):
        result = clf.classify("It was dark outside", "But the streetlights were on")
        assert result == "Attack"

    def test_nevertheless(self, clf):
        result = clf.classify("He had been drinking", "Nevertheless, he was sober")
        assert result == "Attack"


class TestQAStructure:
    def test_question_and_answer(self, clf):
        result = clf.classify("Q: Were you at the scene?", "A: Yes, I was at the scene")
        assert result == "Support"

    def test_question_and_negative_answer(self, clf):
        result = clf.classify("Q: Did you see the accident?", "A: No, I did not see anything")
        assert result == "Attack"

    def test_question_with_who(self, clf):
        result = clf.classify("Who was driving the car?", "The defendant was driving")
        assert result == "Support"


class TestTopicSupport:
    def test_shared_nouns(self, clf):
        result = clf.classify(
            "The blue sedan ran the red light at the intersection",
            "The blue sedan was speeding through the intersection"
        )
        assert result in ("Support", "None")

    def test_no_shared_nouns(self, clf):
        result = clf.classify("The weather was clear", "The defendant pleaded guilty")
        assert result == "None"


class TestHearsay:
    def test_he_said(self, clf):
        result = clf.classify("He said the light was yellow", "The light was actually red")
        assert result in ("Attack", "None")

    def test_according_to(self, clf):
        result = clf.classify("According to the officer, he was speeding", "He was not speeding")
        assert result == "Attack"


class TestDefault:
    def test_unrelated_texts(self, clf):
        result = clf.classify("The sky is blue", "I like ice cream")
        assert result == "None"

    def test_empty_texts(self, clf):
        result = clf.classify("", "")
        assert result == "None"
