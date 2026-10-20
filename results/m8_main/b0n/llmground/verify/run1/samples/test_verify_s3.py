"""Unit tests for llmground.verify."""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest

from llmground.verify import (
    FieldResult,
    Report,
    Verdict,
    verify,
)
from llmground.matchers import Match, MatchTier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Person:
    name: str
    age: int


class FakeModel:
    """Duck-typed stand-in for a Pydantic model."""

    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


class BrokenModel:
    def model_dump(self):
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def test_verdict_values():
    assert Verdict.GROUNDED.value == "grounded"
    assert Verdict.WEAK.value == "weak"
    assert Verdict.UNGROUNDED.value == "ungrounded"
    assert Verdict.SKIPPED.value == "skipped"
    # str-enum behaviour
    assert Verdict.GROUNDED == "grounded"


# ---------------------------------------------------------------------------
# FieldResult
# ---------------------------------------------------------------------------

def _make_result(verdict, value="x", path="a", kind="text"):
    return FieldResult(path=path, value=value, verdict=verdict,
                       match=Match(MatchTier.NONE), kind=kind)


def test_field_result_is_hallucination():
    assert _make_result(Verdict.UNGROUNDED).is_hallucination is True
    assert _make_result(Verdict.GROUNDED).is_hallucination is False
    assert _make_result(Verdict.WEAK).is_hallucination is False
    assert _make_result(Verdict.SKIPPED).is_hallucination is False


def test_field_result_to_dict_basic():
    result = FieldResult(
        path="a.b",
        value="hello",
        verdict=Verdict.GROUNDED,
        match=Match(MatchTier.EXACT, 1.0, note="ok"),
        kind="text",
    )
    d = result.to_dict()
    assert d["path"] == "a.b"
    assert d["value"] == "hello"
    assert d["verdict"] == "grounded"
    assert d["kind"] == "text"
    assert d["match"] == {
        "tier": "exact",
        "score": 1.0,
        "spans": [],
        "note": "ok",
    }


def test_field_result_to_dict_decimal_value():
    result = FieldResult(
        path="amount",
        value=Decimal("12.50"),
        verdict=Verdict.GROUNDED,
        match=Match(MatchTier.EXACT),
        kind="number",
    )
    d = result.to_dict()
    assert d["value"] == "12.50"


def test_field_result_to_dict_non_jsonable_value():
    class Obj:
        def __str__(self):
            return "obj-str"

    result = FieldResult(
        path="x",
        value=Obj(),
        verdict=Verdict.GROUNDED,
        match=Match(MatchTier.EXACT),
    )
    assert result.to_dict()["value"] == "obj-str"


def test_field_result_to_dict_none_and_bool():
    r_none = FieldResult("x", None, Verdict.SKIPPED, Match(MatchTier.NONE), "skip")
    assert r_none.to_dict()["value"] is None
    r_bool = FieldResult("y", True, Verdict.SKIPPED, Match(MatchTier.NONE), "skip")
    assert r_bool.to_dict()["value"] is True


def test_field_result_default_kind():
    r = FieldResult("x", "v", Verdict.GROUNDED, Match(MatchTier.EXACT))
    assert r.kind == "text"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def test_report_default_empty():
    report = Report()
    assert report.results == []
    assert len(report) == 0
    assert list(report) == []


def test_report_iter_and_len():
    results = [
        _make_result(Verdict.GROUNDED),
        _make_result(Verdict.UNGROUNDED),
    ]
    report = Report(results=results)
    assert len(report) == 2
    assert list(report) == results


def test_report_categories():
    results = [
        _make_result(Verdict.GROUNDED, path="a"),
        _make_result(Verdict.GROUNDED, path="b"),
        _make_result(Verdict.WEAK, path="c"),
        _make_result(Verdict.UNGROUNDED, path="d"),
        _make_result(Verdict.SKIPPED, path="e"),
    ]
    report = Report(results=results)
    assert [r.path for r in report.grounded] == ["a", "b"]
    assert [r.path for r in report.weak] == ["c"]
    assert [r.path for r in report.ungrounded] == ["d"]
    assert [r.path for r in report.checked] == ["a", "b", "c", "d"]
    assert report.has_hallucinations is True


def test_report_no_hallucinations():
    report = Report(results=[_make_result(Verdict.GROUNDED)])
    assert report.has_hallucinations is False


def test_report_grounding_rate():
    results = [
        _make_result(Verdict.GROUNDED),
        _make_result(Verdict.GROUNDED),
        _make_result(Verdict.UNGROUNDED),
        _make_result(Verdict.SKIPPED),
    ]
    report = Report(results=results)
    # 2 grounded out of 3 checked
    assert report.grounding_rate == pytest.approx(2 / 3)


def test_report_grounding_rate_empty():
    assert Report().grounding_rate == 1.0


def test_report_grounding_rate_all_skipped():
    report = Report(results=[_make_result(Verdict.SKIPPED)])
    assert report.grounding_rate == 1.0


def test_report_get():
    results = [
        _make_result(Verdict.GROUNDED, path="a.b"),
        _make_result(Verdict.UNGROUNDED, path="c.d"),
    ]
    report = Report(results=results)
    assert report.get("a.b") is results[0]
    assert report.get("c.d") is results[1]
    assert report.get("missing") is None


def test_report_to_dict():
    results = [
        _make_result(Verdict.GROUNDED, path="a"),
        _make_result(Verdict.UNGROUNDED, path="b"),
    ]
    report = Report(results=results)
    d = report.to_dict()
    assert d["grounding_rate"] == 0.5
    assert d["fields_checked"] == 2
    assert d["grounded"] == 1
    assert d["weak"] == 0
    assert d["ungrounded"] == 1
    assert len(d["results"]) == 2
    assert d["results"][0]["path"] == "a"


def test_report_summary():
    results = [
        _make_result(Verdict.GROUNDED),
        _make_result(Verdict.GROUNDED),
        _make_result(Verdict.WEAK),
        _make_result(Verdict.UNGROUNDED),
    ]
    report = Report(results=results)
    s = report.summary()
    assert s == "2/4 fields grounded (50%), 1 weak, 1 ungrounded"


# ---------------------------------------------------------------------------
# verify - source validation
# ---------------------------------------------------------------------------

def test_verify_source_not_string():
    with pytest.raises(TypeError):
        verify({"a": "b"}, None)
    with pytest.raises(TypeError):
        verify({"a": "b"}, 123)


# ---------------------------------------------------------------------------
# verify - basic behavior
# ---------------------------------------------------------------------------

def test_verify_exact_text():
    source = "The invoice was issued to Acme Corp."
    report = verify({"customer": "Acme Corp"}, source)
    assert len(report) == 1
    result = report.results[0]
    assert result.path == "customer"
    assert result.verdict is Verdict.GROUNDED
    assert result.match.tier is MatchTier.EXACT


def test_verify_ungrounded_text():
    source = "The invoice was issued to Acme Corp."
    report = verify({"customer": "Globex"}, source)
    result = report.results[0]
    assert result.verdict is Verdict.UNGROUNDED
    assert result.match.tier is MatchTier.NONE
    assert report.has_hallucinations is True


def test_verify_fuzzy_default_weak():
    source = "The quick brown fox jumps."
    # slightly different string -> fuzzy
    report = verify({"animal": "quick brown fox jump"}, source)
    result = report.results[0]
    # Either weak or ungrounded depending on threshold; with high similarity
    # default threshold 0.85 should match fuzzily.
    assert result.verdict is Verdict.WEAK
    assert result.match.tier is MatchTier.FUZZY


def test_verify_fuzzy_accepted_as_grounded():
    source = "The quick brown fox jumps."
    report = verify(
        {"animal": "quick brown fox jump"}, source, accept_fuzzy=True
    )
    result = report.results[0]
    assert result.verdict is Verdict.GROUNDED
    assert result.match.tier is MatchTier.FUZZY


def test_verify_fuzzy_threshold_too_high():
    source = "The quick brown fox jumps."
    report = verify(
        {"animal": "quick brown fox jump"}, source, fuzzy_threshold=0.99
    )
    result = report.results[0]
    assert result.verdict is Verdict.UNGROUNDED


# ---------------------------------------------------------------------------
# verify - skip cases
# ---------------------------------------------------------------------------

def test_verify_skips_bool_and_none():
    source = "anything"
    report = verify({"flag": True, "nothing": None}, source)
    assert len(report) == 2
    for r in report.results:
        assert r.verdict is Verdict.SKIPPED
        assert r.match.tier is MatchTier.NONE


def test_verify_skips_empty_string():
    source = "some source"
    report = verify({"name": "   "}, source)
    result = report.results[0]
    assert result.verdict is Verdict.SKIPPED


def test_verify_skips_trivial_text():
    source = "a b c"
    report = verify({"val": "a"}, source)
    result = report.results[0]
    # "a" is trivial per match_text
    assert result.verdict is Verdict.SKIPPED


# ---------------------------------------------------------------------------
# verify - numbers
# ---------------------------------------------------------------------------

def test_verify_number_semantic():
    source = "Total: $4,500.00"
    report = verify({"total": 4500.0}, source)
    result = report.results[0]
    assert result.kind == "number"
    assert result.verdict is Verdict.GROUNDED
    assert result.match.tier is MatchTier.SEMANTIC


def test_verify_number_exact():
    source = "The count was 42 here"
    report = verify({"count": 42}, source)
    result = report.results[0]
    assert result.kind == "number"
    assert result.verdict is Verdict.GROUNDED
    assert result.match.tier is MatchTier.EXACT


def test_verify_number_ungrounded():
    source = "Total: $100.00"
    report = verify({"total": 999}, source)
    result = report.results[0]
    assert result.verdict is Verdict.UNGROUNDED
    assert "nearest number" in result.match.note


def test_verify_number_tolerance():
    source = "Total: $100.00"
    report = verify({"total": 105}, source, number_tolerance=10)
    result = report.results[0]
    assert result.verdict is Verdict.GROUNDED
    assert result.match.tier is MatchTier.SEMANTIC


def test_verify_number_tolerance_decimal():
    source = "Total: $100.00"
    report = verify({"total": Decimal("102.50")}, source,
                    number_tolerance=Decimal("3"))
    result = report.results[0]
    assert result.verdict is Verdict.GROUNDED


def test_verify_number_trivial_skipped():
    source = "some text"
    report = verify({"count": 0}, source)
    result = report.results[0]
    assert result.verdict is Verdict.SKIPPED


def test_verify_number_string_classified_as_number():
    source = "The amount is $4,500.00"
    report = verify({"amount": "4500"}, source)
    result = report.results[0]
    assert result.kind == "number"
    assert result.verdict is Verdict.GROUNDED


def test_verify_id_hint_not_number():
    source = "The amount is 4500"
    report = verify({"order_id": "4500"}, source)
    result = report.results[0]
    # id_hint -> matched as text
    assert result.kind == "text"


def test_verify_leading_zero_not_number():
    source = "code is 00123"
    report = verify({"value": "00123"}, source)
    result = report.results[0]
    assert result.kind == "text"


# ---------------------------------------------------------------------------
# verify - dates
# ---------------------------------------------------------------------------

def test_verify_date_semantic():
    source = "Issued on 5 January 2026."
    report = verify({"invoice_date": "2026-01-05"}, source)
    result = report.results[0]
    assert result.kind == "date"
    assert result.verdict is Verdict.GROUNDED
    assert result.match.tier is MatchTier.SEMANTIC


def test_verify_date_exact():
    source = "On 2026-01-05 we shipped"
    report = verify({"invoice_date": "2026-01-05"}, source)
    result = report.results[0]
    assert result.kind == "date"
    assert result.verdict is Verdict.GROUNDED
    assert result.match.tier is MatchTier.EXACT


def test_verify_date_ungrounded():
    source = "Issued on 5 January 2026."
    report = verify({"invoice_date": "2026-02-09"}, source)
    result = report.results[0]
    assert result.verdict is Verdict.UNGROUNDED


def test_verify_date_not_parsed_is_text():
    source = "value not a date here"
    report = verify({"invoice_date": "notadate"}, source)
    result = report.results[0]
    assert result.kind == "text"


# ---------------------------------------------------------------------------
# verify - nested structures
# ---------------------------------------------------------------------------

def test_verify_nested_paths():
    source = "Acme Corp 12345 Main Street"
    report = verify(
        {
            "company": {"name": "Acme Corp", "address": "Main Street"},
            "items": ["Acme Corp"],
        },
        source,
    )
    paths = {r.path for r in report.results}
    assert paths == {"company.name", "company.address", "items[0]"}


def test_verify_dataclass_input():
    source = "Alice is 30 years old"
    report = verify(Person(name="Alice", age=30), source)
    paths = {r.path for r in report.results}
    assert paths == {"name", "age"}
    assert report.get("name").verdict is Verdict.GROUNDED
    assert report.get("age").verdict is Verdict.GROUNDED


def test_verify_pydantic_ducktype():
    source = "Alice is 30 years old"
    model = FakeModel({"name": "Alice", "age": 30})
    report = verify(model, source)
    paths = {r.path for r in report.results}
    assert paths == {"name", "age"}


def test_verify_broken_model_falls_through_to_leaf():
    # model_dump raises -> _to_mapping returns None, value treated as leaf.
    source = "whatever"
    report = verify({"m": BrokenModel()}, source)
    assert len(report) == 1
    result = report.results[0]
    assert result.path == "m"
    # non-str, non-number, non-bool -> kind "text"
    assert result.kind == "text"


def test_verify_list_of_numbers():
    source = "1 2 3"
    report = verify({"nums": [1, 2, 3]}, source)
    paths = [r.path for r in report.results]
    assert paths == ["nums[0]", "nums[1]", "nums[2]"]
    for r in report.results:
        assert r.verdict is Verdict.GROUNDED


def test_verify_tuple_and_set():
    source = "alpha beta"
    report = verify({"t": ("alpha",), "s": {"beta"}}, source)
    paths = {r.path for r in report.results}
    assert "t[0]" in paths
    assert any(p.startswith("s[") for p in paths)


# ---------------------------------------------------------------------------
# verify - ignore
# ---------------------------------------------------------------------------

def test_verify_ignore_paths():
    source = "Acme Corp"
    report = verify(
        {"name": "Acme Corp", "computed": "anything"},
        source,
        ignore=["computed"],
    )
    paths = [r.path for r in report.results]
    assert paths == ["name"]


# ---------------------------------------------------------------------------
# verify - custom classifier
# ---------------------------------------------------------------------------

def test_verify_custom_classifier_force_skip():
    source = "Acme Corp"
    report = verify(
        {"name": "Acme Corp"},
        source,
        classifier=lambda value, path: "skip",
    )
    result = report.results[0]
    assert result.verdict is Verdict.SKIPPED
    assert result.kind == "skip"


def test_verify_custom_classifier_force_number():
    source = "value 42"
    report = verify(
        {"name": "42"},
        source,
        classifier=lambda value, path: "number",
    )
    result = report.results[0]
    assert result.kind == "number"
    assert result.verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# verify - non-string leaf type coercion
# ---------------------------------------------------------------------------

def test_verify_non_string_leaf_is_text():
    class Custom:
        def __str__(self):
            return "Acme Corp"

    source = "Acme Corp is here"
    report = verify({"obj": Custom()}, source)
    result = report.results[0]
    assert result.kind == "text"
    assert result.verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# verify - day_first
# ---------------------------------------------------------------------------

def test_verify_day_first():
    source = "Issued 05/01/2026"
    report = verify({"invoice_date": "2026-05-01"}, source, day_first=True)
    result = report.results[0]
    assert result.kind == "date"
    assert result.verdict is Verdict.GROUNDED


def test_verify_day_first_mismatch():
    source = "Issued 05/01/2026"
    report = verify({"invoice_date": "2026-01-05"}, source, day_first=True)
    result = report.results[0]
    assert result.verdict is Verdict.UNGROUNDED
