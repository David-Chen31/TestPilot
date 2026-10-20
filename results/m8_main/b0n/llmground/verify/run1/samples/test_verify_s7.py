"""Tests for llmground.verify."""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest

from llmground.matchers import Match, MatchTier
from llmground.verify import FieldResult, Report, Verdict, verify


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

class TestVerdict:
    def test_members(self):
        assert Verdict.GROUNDED.value == "grounded"
        assert Verdict.WEAK.value == "weak"
        assert Verdict.UNGROUNDED.value == "ungrounded"
        assert Verdict.SKIPPED.value == "skipped"

    def test_is_str_enum(self):
        assert isinstance(Verdict.GROUNDED, str)
        assert Verdict.GROUNDED == "grounded"


# ---------------------------------------------------------------------------
# FieldResult
# ---------------------------------------------------------------------------

class TestFieldResult:
    def _make(self, verdict=Verdict.GROUNDED, value="x", path="field"):
        return FieldResult(path, value, verdict, Match(MatchTier.EXACT, 1.0), "text")

    def test_is_hallucination_true(self):
        r = self._make(verdict=Verdict.UNGROUNDED)
        assert r.is_hallucination is True

    def test_is_hallucination_false(self):
        for v in (Verdict.GROUNDED, Verdict.WEAK, Verdict.SKIPPED):
            assert self._make(verdict=v).is_hallucination is False

    def test_default_kind_is_text(self):
        r = FieldResult("p", "v", Verdict.GROUNDED, Match(MatchTier.EXACT))
        assert r.kind == "text"

    def test_to_dict_plain(self):
        r = self._make(value="hello", path="a.b")
        d = r.to_dict()
        assert d["path"] == "a.b"
        assert d["value"] == "hello"
        assert d["verdict"] == "grounded"
        assert d["kind"] == "text"
        assert d["match"]["tier"] == "exact"

    def test_to_dict_decimal_value_becomes_string(self):
        r = self._make(value=Decimal("1.5"))
        assert r.to_dict()["value"] == "1.5"

    def test_to_dict_other_value_str(self):
        class Foo:
            def __str__(self):
                return "foo-str"

        r = self._make(value=Foo())
        assert r.to_dict()["value"] == "foo-str"

    def test_to_dict_none_value(self):
        r = self._make(value=None)
        assert r.to_dict()["value"] is None

    def test_to_dict_bool_value(self):
        r = self._make(value=True)
        assert r.to_dict()["value"] is True

    def test_to_dict_match_detail(self):
        r = FieldResult("p", "v", Verdict.WEAK, Match(MatchTier.FUZZY, 0.9), "text")
        md = r.to_dict()["match"]
        assert md["tier"] == "fuzzy"
        assert md["score"] == 0.9


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _result(path, verdict, value="v", kind="text"):
    return FieldResult(path, value, verdict, Match(MatchTier.EXACT), kind)


class TestReport:
    def test_empty_report(self):
        r = Report()
        assert len(r) == 0
        assert list(r) == []
        assert r.checked == []
        assert r.ungrounded == []
        assert r.weak == []
        assert r.grounded == []
        assert r.has_hallucinations is False
        assert r.grounding_rate == 1.0

    def test_iter_and_len(self):
        r = Report([_result("a", Verdict.GROUNDED), _result("b", Verdict.UNGROUNDED)])
        assert len(r) == 2
        assert [x.path for x in r] == ["a", "b"]

    def test_filter_properties(self):
        r = Report([
            _result("a", Verdict.GROUNDED),
            _result("b", Verdict.WEAK),
            _result("c", Verdict.UNGROUNDED),
            _result("d", Verdict.SKIPPED),
        ])
        assert [x.path for x in r.grounded] == ["a"]
        assert [x.path for x in r.weak] == ["b"]
        assert [x.path for x in r.ungrounded] == ["c"]
        assert [x.path for x in r.checked] == ["a", "b", "c"]

    def test_has_hallucinations(self):
        assert Report([_result("a", Verdict.GROUNDED)]).has_hallucinations is False
        assert Report([_result("a", Verdict.UNGROUNDED)]).has_hallucinations is True

    def test_grounding_rate_over_checked(self):
        r = Report([
            _result("a", Verdict.GROUNDED),
            _result("b", Verdict.GROUNDED),
            _result("c", Verdict.UNGROUNDED),
            _result("d", Verdict.SKIPPED),
        ])
        # 2 grounded / 3 checked
        assert r.grounding_rate == pytest.approx(2 / 3)

    def test_grounding_rate_all_skipped_is_one(self):
        r = Report([_result("a", Verdict.SKIPPED)])
        assert r.grounding_rate == 1.0

    def test_get(self):
        r = Report([_result("a", Verdict.GROUNDED), _result("b", Verdict.WEAK)])
        assert r.get("a").path == "a"
        assert r.get("b").verdict is Verdict.WEAK
        assert r.get("missing") is None

    def test_to_dict(self):
        r = Report([
            _result("a", Verdict.GROUNDED),
            _result("b", Verdict.WEAK),
            _result("c", Verdict.UNGROUNDED),
            _result("d", Verdict.SKIPPED),
        ])
        d = r.to_dict()
        assert d["grounding_rate"] == round(1 / 3, 4)
        assert d["fields_checked"] == 3
        assert d["grounded"] == 1
        assert d["weak"] == 1
        assert d["ungrounded"] == 1
        assert len(d["results"]) == 4

    def test_summary(self):
        r = Report([
            _result("a", Verdict.GROUNDED),
            _result("b", Verdict.UNGROUNDED),
        ])
        s = r.summary()
        assert "1/2 fields grounded" in s
        assert "50%" in s
        assert "0 weak" in s
        assert "1 ungrounded" in s


# ---------------------------------------------------------------------------
# verify — type / argument handling
# ---------------------------------------------------------------------------

class TestVerifyArguments:
    def test_source_must_be_string(self):
        with pytest.raises(TypeError):
            verify({"a": "b"}, 123)

    def test_source_none_raises(self):
        with pytest.raises(TypeError):
            verify({"a": "b"}, None)

    def test_empty_extracted(self):
        report = verify({}, "some source")
        assert len(report) == 0
        assert report.grounding_rate == 1.0


# ---------------------------------------------------------------------------
# verify — traversal
# ---------------------------------------------------------------------------

class TestTraversal:
    def test_flat_dict_paths(self):
        report = verify({"name": "Alice", "city": "Paris"}, "Alice went to Paris")
        assert sorted(r.path for r in report) == ["city", "name"]

    def test_nested_dict_paths(self):
        report = verify({"a": {"b": "Alice"}}, "Alice is here")
        assert report.get("a.b").verdict is Verdict.GROUNDED

    def test_list_paths(self):
        report = verify({"items": ["Alice", "Bob"]}, "Alice and Bob")
        assert report.get("items[0]").verdict is Verdict.GROUNDED
        assert report.get("items[1]").verdict is Verdict.GROUNDED

    def test_tuple_paths(self):
        report = verify({"items": ("Alice",)}, "Alice")
        assert report.get("items[0]").verdict is Verdict.GROUNDED

    def test_set_values_are_walked(self):
        report = verify({"items": {"Alice"}}, "Alice")
        # single element set -> index 0
        assert report.get("items[0]").verdict is Verdict.GROUNDED

    def test_dataclass_traversal(self):
        @dataclasses.dataclass
        class Person:
            name: str
            age: int

        report = verify(Person(name="Alice", age=30), "Alice is 30")
        assert report.get("name").verdict is Verdict.GROUNDED
        assert report.get("age").verdict is Verdict.GROUNDED

    def test_pydantic_like_model_dump(self):
        class FakeModel:
            def model_dump(self):
                return {"name": "Alice"}

        report = verify(FakeModel(), "Alice")
        assert report.get("name").verdict is Verdict.GROUNDED

    def test_pydantic_dump_raises_falls_through(self):
        class BadModel:
            def model_dump(self):
                raise ValueError("boom")

        report = verify(BadModel(), "anything")
        # Not a mapping: treated as a single leaf with empty path; str() of object
        assert len(report) == 1
        assert report.results[0].path == ""


# ---------------------------------------------------------------------------
# verify — verdicts
# ---------------------------------------------------------------------------

class TestVerdicts:
    def test_grounded_exact(self):
        report = verify({"name": "Alice"}, "Alice is here")
        r = report.get("name")
        assert r.verdict is Verdict.GROUNDED
        assert r.kind == "text"
        assert r.match.tier is MatchTier.EXACT

    def test_ungrounded(self):
        report = verify({"name": "Zelda"}, "Alice is here")
        r = report.get("name")
        assert r.verdict is Verdict.UNGROUNDED
        assert r.is_hallucination is True
        assert report.has_hallucinations is True

    def test_weak_fuzzy(self):
        report = verify({"name": "Alicia"}, "Alice is here")
        r = report.get("name")
        if r.match.tier is MatchTier.FUZZY:
            assert r.verdict is Verdict.WEAK
        else:
            # If not fuzzy enough, it is simply ungrounded.
            assert r.verdict is Verdict.UNGROUNDED

    def test_accept_fuzzy_turns_weak_grounded(self):
        report = verify(
            {"name": "Alicey"},
            "Alice is here",
            fuzzy_threshold=0.5,
            accept_fuzzy=True,
        )
        r = report.get("name")
        assert r.match.tier is MatchTier.FUZZY
        assert r.verdict is Verdict.GROUNDED

    def test_fuzzy_without_accept_is_weak(self):
        report = verify(
            {"name": "Alicey"},
            "Alice is here",
            fuzzy_threshold=0.5,
        )
        r = report.get("name")
        assert r.match.tier is MatchTier.FUZZY
        assert r.verdict is Verdict.WEAK


# ---------------------------------------------------------------------------
# verify — skipping
# ---------------------------------------------------------------------------

class TestSkipping:
    def test_boolean_skipped(self):
        report = verify({"flag": True}, "anything")
        r = report.get("flag")
        assert r.verdict is Verdict.SKIPPED
        assert r.kind == "skip"

    def test_none_skipped(self):
        report = verify({"x": None}, "anything")
        r = report.get("x")
        assert r.verdict is Verdict.SKIPPED

    def test_empty_string_skipped(self):
        report = verify({"x": ""}, "anything")
        assert report.get("x").verdict is Verdict.SKIPPED

    def test_whitespace_string_skipped(self):
        report = verify({"x": "   "}, "anything")
        assert report.get("x").verdict is Verdict.SKIPPED

    def test_ignore_paths(self):
        report = verify({"a": "Alice", "b": "Nobody"}, "Alice", ignore={"b"})
        assert report.get("b") is None
        assert report.get("a").verdict is Verdict.GROUNDED

    def test_ignore_nested_path(self):
        report = verify(
            {"a": {"b": "Zelda", "c": "Alice"}},
            "Alice",
            ignore={"a.b"},
        )
        assert report.get("a.b") is None
        assert report.get("a.c").verdict is Verdict.GROUNDED

    def test_skipped_not_counted_in_grounding_rate(self):
        report = verify({"a": "Alice", "b": True}, "Alice")
        assert len(report.checked) == 1
        assert report.grounding_rate == 1.0


# ---------------------------------------------------------------------------
# verify — numbers
# ---------------------------------------------------------------------------

class TestNumbers:
    def test_numeric_value_grounded(self):
        report = verify({"amount": 4500}, "The total is $4,500.00")
        r = report.get("amount")
        assert r.kind == "number"
        assert r.verdict is Verdict.GROUNDED

    def test_float_grounded(self):
        report = verify({"amount": 4500.0}, "value: 4500")
        assert report.get("amount").verdict is Verdict.GROUNDED

    def test_decimal_grounded(self):
        report = verify({"amount": Decimal("4500")}, "value: 4500")
        assert report.get("amount").verdict is Verdict.GROUNDED

    def test_number_ungrounded(self):
        report = verify({"amount": 9999}, "The total is 4500")
        assert report.get("amount").verdict is Verdict.UNGROUNDED

    def test_number_tolerance(self):
        report = verify(
            {"amount": 100},
            "value: 105",
            number_tolerance=5,
        )
        assert report.get("amount").verdict is Verdict.GROUNDED

    def test_number_tolerance_decimal(self):
        report = verify(
            {"amount": Decimal("100")},
            "value: 105.5",
            number_tolerance=Decimal("5.5"),
        )
        assert report.get("amount").verdict is Verdict.GROUNDED

    def test_number_tolerance_exceeded(self):
        report = verify(
            {"amount": 100},
            "value: 200",
            number_tolerance=5,
        )
        assert report.get("amount").verdict is Verdict.UNGROUNDED

    def test_string_that_is_number_classified_number(self):
        report = verify({"amount": "4500"}, "$4,500.00")
        r = report.get("amount")
        assert r.kind == "number"
        assert r.verdict is Verdict.GROUNDED

    def test_id_like_string_not_number(self):
        # field name suggests id -> treated as text
        report = verify({"user_id": "00123"}, "user 00123")
        r = report.get("user_id")
        assert r.kind == "text"


# ---------------------------------------------------------------------------
# verify — dates
# ---------------------------------------------------------------------------

class TestDates:
    def test_date_semantic_match(self):
        report = verify({"invoice_date": "2026-01-05"}, "Issued 5 January 2026")
        r = report.get("invoice_date")
        assert r.kind == "date"
        assert r.verdict is Verdict.GROUNDED

    def test_date_exact_match(self):
        report = verify({"due": "2026-01-05"}, "Due 2026-01-05")
        r = report.get("due")
        assert r.kind == "date"
        assert r.verdict is Verdict.GROUNDED
        assert r.match.tier is MatchTier.EXACT

    def test_date_ungrounded(self):
        report = verify({"invoice_date": "2026-01-05"}, "Issued 9 February 2027")
        r = report.get("invoice_date")
        assert r.verdict is Verdict.UNGROUNDED

    def test_day_first(self):
        report = verify(
            {"date": "03/04/2026"},
            "on 3 April 2026",
            day_first=True,
        )
        assert report.get("date").verdict is Verdict.GROUNDED

    def test_day_first_false(self):
        report = verify(
            {"date": "03/04/2026"},
            "on March 4, 2026",
            day_first=False,
        )
        assert report.get("date").verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# verify — classifier override
# ---------------------------------------------------------------------------

class TestClassifierOverride:
    def test_custom_classifier_skip(self):
        report = verify(
            {"a": "Alice"},
            "Alice",
            classifier=lambda value, path: "skip",
        )
        assert report.get("a").verdict is Verdict.SKIPPED

    def test_custom_classifier_used_for_kind(self):
        calls = []

        def classifier(value, path):
            calls.append((value, path))
            return "text"

        report = verify({"a": "Alice"}, "Alice", classifier=classifier)
        assert calls == [("Alice", "a")]
        assert report.get("a").kind == "text"
        assert report.get("a").verdict is Verdict.GROUNDED

    def test_classifier_returning_number(self):
        report = verify(
            {"a": "Alice"},
            "the value 99",
            classifier=lambda value, path: "number",
        )
        r = report.get("a")
        # "Alice" isn't numeric -> match_number returns NONE with "value is not numeric"
        assert r.verdict is Verdict.SKIPPED


# ---------------------------------------------------------------------------
# verify — match note handling
# ---------------------------------------------------------------------------

class TestMatchNoteHandling:
    def test_trivial_value_skipped(self):
        # "the" is a trivial stopword for match_text
        report = verify({"x": "the"}, "the cat")
        assert report.get("x").verdict is Verdict.SKIPPED

    def test_non_numeric_number_skipped(self):
        # classifier forces number; match_number says "value is not numeric".
        report = verify(
            {"x": "abc"},
            "abc",
            classifier=lambda v, p: "number",
        )
        assert report.get("x").verdict is Verdict.SKIPPED
        assert report.get("x").kind == "number"


# ---------------------------------------------------------------------------
# verify — end-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_mixed_document(self):
        extracted = {
            "vendor": "Acme Corp",
            "total": 1250.00,
            "invoice_date": "2026-03-15",
            "paid": False,
            "notes": "Net 30 terms apply",
            "fake_field": "Totally fabricated text that isn't here",
        }
        source = (
            "Invoice from Acme Corp\n"
            "Total: $1,250.00\n"
            "Date: 15 March 2026\n"
            "Payment not received.\n"
            "Net 30 terms apply.\n"
        )
        report = verify(extracted, source)

        assert report.get("vendor").verdict is Verdict.GROUNDED
        assert report.get("total").verdict is Verdict.GROUNDED
        assert report.get("invoice_date").verdict is Verdict.GROUNDED
        assert report.get("paid").verdict is Verdict.SKIPPED
        assert report.get("notes").verdict is Verdict.GROUNDED
        assert report.get("fake_field").verdict is Verdict.UNGROUNDED

        assert report.has_hallucinations is True
        assert len(report.ungrounded) == 1
        assert len(report.checked) == 5
        assert report.to_dict()["fields_checked"] == 5

    def test_report_to_dict_roundtrip_shape(self):
        report = verify({"name": "Alice"}, "Alice")
        d = report.to_dict()
        assert set(d.keys()) == {
            "grounding_rate",
            "fields_checked",
            "grounded",
            "weak",
            "ungrounded",
            "results",
        }
        assert d["results"][0]["path"] == "name"
