"""Tests for llmground.verify."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

import pytest

from llmground.matchers import Match, MatchTier
from llmground.verify import (
    FieldResult,
    Report,
    Verdict,
    verify,
)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def test_verdict_values_are_strings():
    assert Verdict.GROUNDED == "grounded"
    assert Verdict.WEAK == "weak"
    assert Verdict.UNGROUNDED == "ungrounded"
    assert Verdict.SKIPPED == "skipped"


def test_verdict_is_str_enum():
    assert isinstance(Verdict.GROUNDED, str)
    assert Verdict.GROUNDED.value == "grounded"


# ---------------------------------------------------------------------------
# FieldResult
# ---------------------------------------------------------------------------

def _make_field_result(verdict=Verdict.GROUNDED, value="x", kind="text", match=None):
    m = match if match is not None else Match(MatchTier.EXACT, 1.0)
    return FieldResult(path="a.b", value=value, verdict=verdict, match=m, kind=kind)


def test_field_result_is_hallucination_true_for_ungrounded():
    fr = _make_field_result(verdict=Verdict.UNGROUNDED)
    assert fr.is_hallucination is True


@pytest.mark.parametrize("verdict", [Verdict.GROUNDED, Verdict.WEAK, Verdict.SKIPPED])
def test_field_result_is_hallucination_false_for_others(verdict):
    fr = _make_field_result(verdict=verdict)
    assert fr.is_hallucination is False


def test_field_result_default_kind_is_text():
    fr = FieldResult(path="p", value="v", verdict=Verdict.GROUNDED, match=Match(MatchTier.EXACT))
    assert fr.kind == "text"


def test_field_result_to_dict_structure():
    m = Match(MatchTier.SEMANTIC, 0.9, note="n")
    fr = FieldResult(path="p.q", value="hello", verdict=Verdict.GROUNDED, match=m, kind="text")
    d = fr.to_dict()
    assert d["path"] == "p.q"
    assert d["value"] == "hello"
    assert d["verdict"] == "grounded"
    assert d["kind"] == "text"
    assert d["match"] == m.to_dict()


def test_field_result_to_dict_decimal_value_is_str():
    fr = _make_field_result(value=Decimal("1.50"))
    d = fr.to_dict()
    assert d["value"] == "1.50"


def test_field_result_to_dict_none_value():
    fr = _make_field_result(value=None)
    assert fr.to_dict()["value"] is None


def test_field_result_to_dict_non_jsonable_value_becomes_str():
    class Foo:
        def __str__(self):
            return "FOO"

    fr = _make_field_result(value=Foo())
    assert fr.to_dict()["value"] == "FOO"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fr(path, verdict, value="v"):
    return FieldResult(path=path, value=value, verdict=verdict, match=Match(MatchTier.EXACT))


def test_report_defaults_empty():
    r = Report()
    assert r.results == []
    assert len(r) == 0
    assert list(r) == []


def test_report_iter_and_len():
    r = Report(results=[_fr("a", Verdict.GROUNDED), _fr("b", Verdict.UNGROUNDED)])
    assert len(r) == 2
    assert [x.path for x in r] == ["a", "b"]


def test_report_checked_excludes_skipped():
    r = Report(results=[
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.SKIPPED),
        _fr("c", Verdict.UNGROUNDED),
    ])
    assert [x.path for x in r.checked] == ["a", "c"]


def test_report_properties_filter_by_verdict():
    r = Report(results=[
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.WEAK),
        _fr("c", Verdict.UNGROUNDED),
        _fr("d", Verdict.SKIPPED),
    ])
    assert [x.path for x in r.grounded] == ["a"]
    assert [x.path for x in r.weak] == ["b"]
    assert [x.path for x in r.ungrounded] == ["c"]


def test_report_has_hallucinations():
    assert Report(results=[_fr("a", Verdict.GROUNDED)]).has_hallucinations is False
    assert Report(results=[_fr("a", Verdict.UNGROUNDED)]).has_hallucinations is True


def test_report_grounding_rate_all_grounded():
    r = Report(results=[_fr("a", Verdict.GROUNDED), _fr("b", Verdict.GROUNDED)])
    assert r.grounding_rate == 1.0


def test_report_grounding_rate_mixed():
    r = Report(results=[
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.GROUNDED),
        _fr("c", Verdict.WEAK),
        _fr("d", Verdict.UNGROUNDED),
    ])
    # 2 grounded out of 4 checked
    assert r.grounding_rate == 0.5


def test_report_grounding_rate_skipped_not_counted():
    r = Report(results=[
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.SKIPPED),
        _fr("c", Verdict.SKIPPED),
    ])
    assert r.grounding_rate == 1.0


def test_report_grounding_rate_empty_is_one():
    assert Report().grounding_rate == 1.0


def test_report_grounding_rate_only_skipped_is_one():
    r = Report(results=[_fr("a", Verdict.SKIPPED)])
    assert r.grounding_rate == 1.0


def test_report_get_found():
    r = Report(results=[_fr("a", Verdict.GROUNDED), _fr("b", Verdict.UNGROUNDED)])
    assert r.get("b").path == "b"
    assert r.get("missing") is None


def test_report_to_dict_counts():
    r = Report(results=[
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.GROUNDED),
        _fr("c", Verdict.WEAK),
        _fr("d", Verdict.UNGROUNDED),
        _fr("e", Verdict.SKIPPED),
    ])
    d = r.to_dict()
    assert d["grounding_rate"] == 0.5
    assert d["fields_checked"] == 4
    assert d["grounded"] == 2
    assert d["weak"] == 1
    assert d["ungrounded"] == 1
    assert len(d["results"]) == 5


def test_report_to_dict_rounds_rate():
    # one of three grounded -> 0.3333
    r = Report(results=[
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.UNGROUNDED),
        _fr("c", Verdict.UNGROUNDED),
    ])
    assert r.to_dict()["grounding_rate"] == 0.3333


def test_report_summary_text():
    r = Report(results=[
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.WEAK),
        _fr("c", Verdict.UNGROUNDED),
        _fr("d", Verdict.SKIPPED),
    ])
    assert r.summary() == "1/3 fields grounded (33%), 1 weak, 1 ungrounded"


# ---------------------------------------------------------------------------
# verify: error handling
# ---------------------------------------------------------------------------

def test_verify_raises_on_non_string_source():
    with pytest.raises(TypeError, match="source must be a string"):
        verify({"a": "b"}, 123)  # type: ignore[arg-type]


def test_verify_raises_on_none_source():
    with pytest.raises(TypeError):
        verify({"a": "b"}, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# verify: basic grounding
# ---------------------------------------------------------------------------

def test_verify_exact_text_grounded():
    report = verify({"name": "Alice"}, "The applicant is Alice.")
    assert len(report) == 1
    fr = report.results[0]
    assert fr.path == "name"
    assert fr.value == "Alice"
    assert fr.verdict is Verdict.GROUNDED
    assert fr.kind == "text"
    assert fr.match.tier is MatchTier.EXACT
    assert fr.match.spans
    assert fr.match.spans[0].text == "Alice"


def test_verify_ungrounded_text():
    report = verify({"name": "Bob"}, "The applicant is Alice.")
    fr = report.results[0]
    assert fr.verdict is Verdict.UNGROUNDED
    assert fr.is_hallucination is True
    assert report.has_hallucinations is True


def test_verify_multiple_fields():
    report = verify(
        {"first": "Alice", "last": "Smith"},
        "Alice Smith lives here.",
    )
    assert len(report) == 2
    assert report.get("first").verdict is Verdict.GROUNDED
    assert report.get("last").verdict is Verdict.GROUNDED
    assert report.grounding_rate == 1.0


# ---------------------------------------------------------------------------
# verify: numeric fields
# ---------------------------------------------------------------------------

def test_verify_int_number_grounded_semantically():
    report = verify({"amount": 4500}, "Total: $4,500.00")
    fr = report.results[0]
    assert fr.kind == "number"
    assert fr.verdict is Verdict.GROUNDED
    assert fr.match.tier is MatchTier.SEMANTIC


def test_verify_float_number_grounded():
    report = verify({"price": 4500.0}, "The price is $4,500.00.")
    assert report.results[0].verdict is Verdict.GROUNDED


def test_verify_decimal_number_grounded():
    report = verify({"price": Decimal("4500")}, "The price is $4,500.00.")
    assert report.results[0].verdict is Verdict.GROUNDED


def test_verify_number_ungrounded():
    report = verify({"amount": 9999}, "Total: $4,500.00")
    fr = report.results[0]
    assert fr.kind == "number"
    assert fr.verdict is Verdict.UNGROUNDED


def test_verify_number_within_tolerance():
    report = verify({"amount": 100}, "Total: 105", number_tolerance=10)
    assert report.results[0].verdict is Verdict.GROUNDED


def test_verify_number_outside_tolerance():
    report = verify({"amount": 100}, "Total: 200", number_tolerance=10)
    assert report.results[0].verdict is Verdict.UNGROUNDED


def test_verify_number_tolerance_float_converted():
    report = verify({"amount": 100}, "Total: 105", number_tolerance=10.0)
    assert report.results[0].verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# verify: date fields
# ---------------------------------------------------------------------------

def test_verify_date_semantic_match():
    report = verify({"invoice_date": "2026-01-05"}, "Due on 5 January 2026.")
    fr = report.results[0]
    assert fr.kind == "date"
    assert fr.verdict is Verdict.GROUNDED
    assert fr.match.tier is MatchTier.SEMANTIC


def test_verify_date_exact_match():
    report = verify({"invoice_date": "2026-01-05"}, "Due on 2026-01-05.")
    fr = report.results[0]
    assert fr.kind == "date"
    assert fr.verdict is Verdict.GROUNDED
    assert fr.match.tier is MatchTier.EXACT


def test_verify_date_not_found():
    report = verify({"invoice_date": "2026-01-05"}, "Due on 1 March 2027.")
    fr = report.results[0]
    assert fr.kind == "date"
    assert fr.verdict is Verdict.UNGROUNDED


def test_verify_ambiguous_date_classified_as_text():
    # An ambiguous value like "03/04/2026" cannot be parsed during
    # classification (day_first defaults to None there), so it falls back to
    # text. Its textual form does not appear in the source verbatim, so the
    # end result is UNGROUNDED -- day_first does not rescue it.
    report = verify({"due_date": "03/04/2026"}, "Pay on 04/03/2026.", day_first=True)
    fr = report.results[0]
    assert fr.kind == "text"
    assert fr.verdict is Verdict.UNGROUNDED


# ---------------------------------------------------------------------------
# verify: skipped fields
# ---------------------------------------------------------------------------

def test_verify_boolean_skipped():
    report = verify({"active": True}, "anything")
    fr = report.results[0]
    assert fr.kind == "skip"
    assert fr.verdict is Verdict.SKIPPED
    assert fr.match.tier is MatchTier.NONE


def test_verify_none_skipped():
    report = verify({"note": None}, "anything")
    fr = report.results[0]
    assert fr.verdict is Verdict.SKIPPED


def test_verify_empty_string_skipped():
    report = verify({"name": ""}, "anything")
    fr = report.results[0]
    assert fr.verdict is Verdict.SKIPPED


def test_verify_whitespace_string_skipped():
    report = verify({"name": "   "}, "anything")
    fr = report.results[0]
    assert fr.verdict is Verdict.SKIPPED


def test_verify_skipped_not_in_checked():
    report = verify({"a": True, "b": "Alice"}, "Alice")
    assert len(report.checked) == 1
    assert report.grounding_rate == 1.0


# ---------------------------------------------------------------------------
# verify: single-char value present verbatim is grounded
# ---------------------------------------------------------------------------

def test_verify_single_char_present_verbatim_grounded():
    # A single character that literally appears in the source is matched
    # verbatim -- the triviality short-circuit in match_text is checked but
    # only fires for values that are actually in the trivial set. A plain
    # "x" is found exactly here.
    report = verify({"v": "x"}, "some document with x inside")
    fr = report.results[0]
    assert fr.verdict is Verdict.GROUNDED
    assert fr.match.tier is MatchTier.EXACT


# ---------------------------------------------------------------------------
# verify: ignore
# ---------------------------------------------------------------------------

def test_verify_ignore_path():
    report = verify({"a": "Alice", "b": "ghost"}, "Alice", ignore=["b"])
    assert len(report) == 1
    assert report.get("b") is None
    assert report.get("a").verdict is Verdict.GROUNDED


def test_verify_ignore_multiple_paths():
    report = verify(
        {"a": "Alice", "b": "ghost", "c": "phantom"},
        "Alice",
        ignore=("b", "c"),
    )
    assert len(report) == 1


# ---------------------------------------------------------------------------
# verify: fuzzy matches
# ---------------------------------------------------------------------------

def test_verify_fuzzy_is_weak_by_default():
    report = verify({"name": "Alicee"}, "The applicant is Alicee.")
    # exact match because it's verbatim
    assert report.results[0].verdict is Verdict.GROUNDED


def test_verify_fuzzy_near_match_weak():
    # "Alicia" vs "Alice" fuzzy at some similarity
    report = verify({"name": "Alice"}, "Someone called Alica wrote this.")
    fr = report.results[0]
    if fr.match.tier is MatchTier.FUZZY:
        assert fr.verdict is Verdict.WEAK
    # Either FUZZY -> WEAK, or NONE -> UNGROUNDED
    assert fr.verdict in (Verdict.WEAK, Verdict.UNGROUNDED)


def test_verify_fuzzy_accepted_when_accept_fuzzy():
    # craft source that yields a fuzzy match for a chosen threshold
    source = "Acme Corporation Ltd"
    report = verify(
        {"company": "Acme Corporaton Ltd"},
        source,
        fuzzy_threshold=0.8,
        accept_fuzzy=True,
    )
    fr = report.results[0]
    if fr.match.tier is MatchTier.FUZZY:
        assert fr.verdict is Verdict.GROUNDED


def test_verify_fuzzy_rejected_when_not_accept_fuzzy():
    report = verify(
        {"company": "Acme Corporaton Ltd"},
        "Acme Corporation Ltd",
        fuzzy_threshold=0.8,
        accept_fuzzy=False,
    )
    fr = report.results[0]
    if fr.match.tier is MatchTier.FUZZY:
        assert fr.verdict is Verdict.WEAK


# ---------------------------------------------------------------------------
# verify: traversal
# ---------------------------------------------------------------------------

def test_verify_nested_dict_paths():
    report = verify(
        {"customer": {"name": "Alice", "city": "Paris"}},
        "Alice lives in Paris.",
    )
    paths = {r.path for r in report.results}
    assert paths == {"customer.name", "customer.city"}
    assert report.get("customer.name").verdict is Verdict.GROUNDED
    assert report.get("customer.city").verdict is Verdict.GROUNDED


def test_verify_list_paths():
    report = verify(
        {"items": ["Alice", "Bob"]},
        "Alice and Bob are here.",
    )
    paths = {r.path for r in report.results}
    assert paths == {"items[0]", "items[1]"}
    assert report.get("items[0]").verdict is Verdict.GROUNDED
    assert report.get("items[1]").verdict is Verdict.GROUNDED


def test_verify_tuple_traversal():
    report = verify({"pair": ("Alice", "Bob")}, "Alice, Bob")
    assert report.get("pair[0]").verdict is Verdict.GROUNDED
    assert report.get("pair[1]").verdict is Verdict.GROUNDED


def test_verify_top_level_list():
    report = verify(["Alice", "Bob"], "Alice, Bob")
    assert len(report) == 2
    assert report.get("[0]").value == "Alice"


def test_verify_top_level_scalar():
    report = verify("Alice", "Alice is here.")
    assert len(report) == 1
    assert report.results[0].path == ""
    assert report.results[0].verdict is Verdict.GROUNDED


def test_verify_nested_list_of_dicts():
    report = verify(
        {"rows": [{"name": "Alice"}, {"name": "Bob"}]},
        "Alice, Bob",
    )
    assert report.get("rows[0].name").verdict is Verdict.GROUNDED
    assert report.get("rows[1].name").verdict is Verdict.GROUNDED


def test_verify_dataclass_input():
    @dataclass
    class Person:
        name: str
        age: int

    report = verify(Person(name="Alice", age=30), "Alice is 30 years old.")
    assert report.get("name").verdict is Verdict.GROUNDED
    assert report.get("age").kind == "number"
    assert report.get("age").verdict is Verdict.GROUNDED


def test_verify_nested_dataclass():
    @dataclass
    class Inner:
        value: str

    @dataclass
    class Outer:
        inner: Inner

    report = verify(Outer(inner=Inner(value="Alice")), "Alice")
    assert report.get("inner.value").verdict is Verdict.GROUNDED


def test_verify_pydantic_like_object():
    class FakeModel:
        def __init__(self, data):
            self._data = data

        def model_dump(self):
            return self._data

    report = verify(FakeModel({"name": "Alice"}), "Alice is here.")
    assert report.get("name").verdict is Verdict.GROUNDED


def test_verify_pydantic_dump_failure_falls_through():
    class Broken:
        def model_dump(self):
            raise RuntimeError("boom")

        def __str__(self):
            return "Alice"

    # As a leaf value, it will be stringified for matching.
    report = verify({"obj": Broken()}, "Alice is here.")
    fr = report.get("obj")
    assert fr.kind == "text"
    assert fr.verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# verify: classifier override
# ---------------------------------------------------------------------------

def test_verify_custom_classifier_forces_text():
    def classifier(value, path):
        return "text"

    report = verify({"amount": 4500}, "Total: $4,500.00", classifier=classifier)
    fr = report.results[0]
    assert fr.kind == "text"
    # "4500" as text isn't verbatim; likely ungrounded
    assert fr.verdict in (Verdict.GROUNDED, Verdict.WEAK, Verdict.UNGROUNDED)


def test_verify_custom_classifier_forces_skip():
    def classifier(value, path):
        return "skip"

    report = verify({"amount": 4500}, "Total: $4,500.00", classifier=classifier)
    assert report.results[0].verdict is Verdict.SKIPPED


def test_verify_string_number_classified_as_number():
    report = verify({"total": "4500"}, "Total was $4,500.00.")
    assert report.results[0].kind == "number"
    assert report.results[0].verdict is Verdict.GROUNDED


def test_verify_id_field_not_classified_as_number():
    report = verify({"id": "00123"}, "Reference 00123 here.")
    fr = report.results[0]
    assert fr.kind == "text"
    assert fr.verdict is Verdict.GROUNDED


def test_verify_leading_zero_number_not_treated_as_number():
    report = verify({"value": "00123"}, "some 00123 here")
    fr = report.results[0]
    assert fr.kind == "text"


# ---------------------------------------------------------------------------
# verify: empty structures
# ---------------------------------------------------------------------------

def test_verify_empty_dict():
    report = verify({}, "anything")
    assert len(report) == 0
    assert report.grounding_rate == 1.0
    assert report.has_hallucinations is False


def test_verify_empty_list():
    report = verify([], "anything")
    assert len(report) == 0


# ---------------------------------------------------------------------------
# Integration-ish
# ---------------------------------------------------------------------------

def test_verify_report_to_dict_roundtrip():
    report = verify(
        {"name": "Alice", "amount": 4500, "ok": True, "miss": "Bob"},
        "Alice was charged $4,500.00.",
    )
    d = report.to_dict()
    assert d["grounded"] == 2  # name, amount
    assert d["ungrounded"] == 1  # miss
    assert d["fields_checked"] == 3  # boolean skipped
    assert any(r["path"] == "miss" for r in d["results"])
    assert all("match" in r for r in d["results"])


def test_verify_enum_leaf_stringified():
    class Color(str, Enum):
        RED = "red"

    # An Enum subclass of str is a str instance, but str(Color.RED) yields
    # "Color.RED", not "red". So the text matcher looks for "Color.RED" in the
    # source, does not find it, and the field is ungrounded.
    report = verify({"color": Color.RED}, "The color is red.")
    fr = report.results[0]
    assert fr.kind == "text"
    assert fr.verdict is Verdict.UNGROUNDED
