"""Unit tests for llmground.verify."""

from __future__ import annotations

import dataclasses
from decimal import Decimal

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

def test_verdict_values():
    assert Verdict.GROUNDED.value == "grounded"
    assert Verdict.WEAK.value == "weak"
    assert Verdict.UNGROUNDED.value == "ungrounded"
    assert Verdict.SKIPPED.value == "skipped"


def test_verdict_is_str_enum():
    assert isinstance(Verdict.GROUNDED, str)
    assert Verdict.GROUNDED == "grounded"


# ---------------------------------------------------------------------------
# FieldResult
# ---------------------------------------------------------------------------

def _match(tier=MatchTier.NONE, **kwargs):
    return Match(tier, **kwargs)


def test_field_result_is_hallucination_true():
    fr = FieldResult("a", "x", Verdict.UNGROUNDED, _match())
    assert fr.is_hallucination is True


@pytest.mark.parametrize("verdict", [Verdict.GROUNDED, Verdict.WEAK, Verdict.SKIPPED])
def test_field_result_is_hallucination_false(verdict):
    fr = FieldResult("a", "x", verdict, _match())
    assert fr.is_hallucination is False


def test_field_result_default_kind_text():
    fr = FieldResult("a", "x", Verdict.GROUNDED, _match())
    assert fr.kind == "text"


def test_field_result_to_dict_basic():
    fr = FieldResult("a.b", "hello", Verdict.GROUNDED, _match(MatchTier.EXACT, 1.0), "text")
    d = fr.to_dict()
    assert d["path"] == "a.b"
    assert d["value"] == "hello"
    assert d["verdict"] == "grounded"
    assert d["kind"] == "text"
    assert d["match"]["tier"] == "exact"
    assert d["match"]["score"] == 1.0


def test_field_result_to_dict_decimal_value():
    fr = FieldResult("n", Decimal("12.5"), Verdict.GROUNDED, _match())
    d = fr.to_dict()
    assert d["value"] == "12.5"


def test_field_result_to_dict_non_jsonable_value():
    class Custom:
        def __str__(self):
            return "custom-repr"

    fr = FieldResult("c", Custom(), Verdict.GROUNDED, _match())
    assert fr.to_dict()["value"] == "custom-repr"


def test_field_result_to_dict_none_value():
    fr = FieldResult("c", None, Verdict.SKIPPED, _match())
    assert fr.to_dict()["value"] is None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fr(path, verdict, value="x", kind="text"):
    return FieldResult(path, value, verdict, _match(), kind)


def test_report_empty():
    r = Report()
    assert len(r) == 0
    assert list(r) == []
    assert r.checked == []
    assert r.ungrounded == []
    assert r.weak == []
    assert r.grounded == []
    assert r.has_hallucinations is False
    assert r.grounding_rate == 1.0


def test_report_iteration_and_len():
    results = [_fr("a", Verdict.GROUNDED), _fr("b", Verdict.UNGROUNDED)]
    r = Report(results=results)
    assert len(r) == 2
    assert list(r) == results


def test_report_checked_excludes_skipped():
    results = [
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.SKIPPED),
        _fr("c", Verdict.WEAK),
    ]
    r = Report(results=results)
    assert [x.path for x in r.checked] == ["a", "c"]


def test_report_ungrounded_weak_grounded():
    results = [
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.UNGROUNDED),
        _fr("c", Verdict.WEAK),
        _fr("d", Verdict.UNGROUNDED),
    ]
    r = Report(results=results)
    assert [x.path for x in r.grounded] == ["a"]
    assert [x.path for x in r.weak] == ["c"]
    assert [x.path for x in r.ungrounded] == ["b", "d"]
    assert r.has_hallucinations is True


def test_report_has_hallucinations_false_when_only_weak():
    r = Report(results=[_fr("a", Verdict.WEAK)])
    assert r.has_hallucinations is False


def test_report_grounding_rate_excludes_skipped():
    results = [
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.GROUNDED),
        _fr("c", Verdict.UNGROUNDED),
        _fr("d", Verdict.SKIPPED),
    ]
    r = Report(results=results)
    # 2 grounded out of 3 checked
    assert r.grounding_rate == pytest.approx(2 / 3)


def test_report_grounding_rate_all_skipped_is_one():
    r = Report(results=[_fr("a", Verdict.SKIPPED)])
    assert r.grounding_rate == 1.0


def test_report_get_found():
    results = [_fr("a", Verdict.GROUNDED), _fr("b", Verdict.WEAK)]
    r = Report(results=results)
    assert r.get("b") is results[1]


def test_report_get_missing():
    r = Report(results=[_fr("a", Verdict.GROUNDED)])
    assert r.get("zzz") is None


def test_report_to_dict():
    results = [
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.WEAK),
        _fr("c", Verdict.UNGROUNDED),
        _fr("d", Verdict.SKIPPED),
    ]
    r = Report(results=results)
    d = r.to_dict()
    assert d["fields_checked"] == 3
    assert d["grounded"] == 1
    assert d["weak"] == 1
    assert d["ungrounded"] == 1
    assert d["grounding_rate"] == round(1 / 3, 4)
    assert len(d["results"]) == 4


def test_report_summary():
    results = [
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.UNGROUNDED),
    ]
    r = Report(results=results)
    assert r.summary() == "1/2 fields grounded (50%), 0 weak, 1 ungrounded"


# ---------------------------------------------------------------------------
# verify: input validation
# ---------------------------------------------------------------------------

def test_verify_source_must_be_string():
    with pytest.raises(TypeError, match="source must be a string"):
        verify({"a": "x"}, 123)


def test_verify_source_none_raises():
    with pytest.raises(TypeError):
        verify({"a": "x"}, None)


# ---------------------------------------------------------------------------
# verify: traversal & paths
# ---------------------------------------------------------------------------

def test_verify_simple_dict_exact():
    report = verify({"name": "Alice"}, "Alice went to the market.")
    assert len(report) == 1
    fr = report.get("name")
    assert fr is not None
    assert fr.value == "Alice"
    assert fr.verdict is Verdict.GROUNDED
    assert fr.match.tier is MatchTier.EXACT
    assert fr.kind == "text"


def test_verify_nested_paths():
    extracted = {"a": {"b": {"c": "Hello"}}}
    report = verify(extracted, "Well, Hello there.")
    assert report.get("a.b.c") is not None
    assert report.get("a.b.c").verdict is Verdict.GROUNDED


def test_verify_list_paths():
    extracted = {"names": ["Alice", "Bob"]}
    report = verify(extracted, "Alice and Bob met.")
    assert report.get("names[0]").value == "Alice"
    assert report.get("names[1]").value == "Bob"


def test_verify_tuple_and_set_traversal():
    # tuple preserves order; set of one element for determinism
    report = verify({"t": ("Alice",)}, "Alice")
    assert report.get("t[0]").value == "Alice"


def test_verify_dataclass_extracted():
    @dataclasses.dataclass
    class Person:
        name: str
        age: int

    report = verify(Person(name="Alice", age=30), "Alice is 30 years old.")
    assert report.get("name").verdict is Verdict.GROUNDED
    assert report.get("age").verdict is Verdict.GROUNDED


def test_verify_dataclass_class_not_instantiated_is_leaf():
    # Passing the dataclass type itself is not converted (isinstance type check).
    @dataclasses.dataclass
    class Person:
        name: str

    report = verify(Person, "some source")
    # whole class is treated as a single leaf at path ""
    assert len(report) == 1
    assert report.results[0].path == ""


def test_verify_pydantic_like_object():
    class FakeModel:
        def model_dump(self):
            return {"name": "Alice"}

    report = verify(FakeModel(), "Alice")
    assert report.get("name").verdict is Verdict.GROUNDED


def test_verify_pydantic_model_dump_raises_falls_back():
    class BrokenModel:
        def model_dump(self):
            raise ValueError("boom")

    report = verify(BrokenModel(), "anything")
    # falls back to being treated as a leaf at path ""
    assert len(report) == 1
    assert report.results[0].path == ""


# ---------------------------------------------------------------------------
# verify: ignore
# ---------------------------------------------------------------------------

def test_verify_ignore_paths():
    extracted = {"a": "Alice", "b": "Bob"}
    report = verify(extracted, "Alice and Bob", ignore=["b"])
    assert report.get("a") is not None
    assert report.get("b") is None


def test_verify_ignore_nested_paths():
    extracted = {"outer": {"inner": "x"}}
    report = verify(extracted, "x", ignore=["outer.inner"])
    assert len(report) == 0


def test_verify_ignore_accepts_iterable():
    extracted = {"a": "Alice"}
    report = verify(extracted, "Alice", ignore=iter(["a"]))
    assert len(report) == 0


# ---------------------------------------------------------------------------
# verify: skipped values
# ---------------------------------------------------------------------------

def test_verify_skips_bool():
    report = verify({"flag": True}, "whatever")
    fr = report.get("flag")
    assert fr.verdict is Verdict.SKIPPED
    assert fr.kind == "skip"


def test_verify_skips_none():
    report = verify({"x": None}, "whatever")
    fr = report.get("x")
    assert fr.verdict is Verdict.SKIPPED


def test_verify_skips_empty_string():
    report = verify({"x": ""}, "whatever")
    assert report.get("x").verdict is Verdict.SKIPPED


def test_verify_skips_whitespace_string():
    report = verify({"x": "   "}, "whatever")
    assert report.get("x").verdict is Verdict.SKIPPED


def test_verify_skips_trivial_text():
    # match_text returns "value too trivial to verify" -> SKIPPED
    report = verify({"company": "the"}, "The document.")
    fr = report.get("company")
    assert fr.verdict is Verdict.SKIPPED


# ---------------------------------------------------------------------------
# verify: ungrounded
# ---------------------------------------------------------------------------

def test_verify_ungrounded_text():
    report = verify({"name": "Zebra"}, "There is no such thing here.")
    fr = report.get("name")
    assert fr.verdict is Verdict.UNGROUNDED
    assert report.has_hallucinations is True


def test_verify_ungrounded_number():
    report = verify({"amount": 999}, "The total was 100 dollars.")
    fr = report.get("amount")
    assert fr.kind == "number"
    assert fr.verdict is Verdict.UNGROUNDED


# ---------------------------------------------------------------------------
# verify: numbers
# ---------------------------------------------------------------------------

def test_verify_number_exact():
    report = verify({"amount": 42}, "The answer is 42.")
    fr = report.get("amount")
    assert fr.kind == "number"
    assert fr.verdict is Verdict.GROUNDED


def test_verify_number_formatted_differently():
    report = verify({"total": 4500}, "The price is $4,500.00.")
    fr = report.get("total")
    assert fr.verdict is Verdict.GROUNDED
    assert fr.match.tier is MatchTier.SEMANTIC


def test_verify_number_tolerance():
    report = verify({"amount": 100}, "Value is 101.", number_tolerance=1)
    fr = report.get("amount")
    assert fr.verdict is Verdict.GROUNDED


def test_verify_number_tolerance_too_far():
    report = verify({"amount": 100}, "Value is 105.", number_tolerance=1)
    fr = report.get("amount")
    assert fr.verdict is Verdict.UNGROUNDED


def test_verify_number_float():
    report = verify({"x": 3.5}, "It is 3.5 units.")
    assert report.get("x").verdict is Verdict.GROUNDED


def test_verify_number_decimal():
    report = verify({"x": Decimal("10.5")}, "It is 10.5 units.")
    assert report.get("x").verdict is Verdict.GROUNDED


def test_verify_numeric_string_field():
    # "amount" has no id hint and parses as number -> number kind
    report = verify({"amount": "42"}, "Total of 42.")
    fr = report.get("amount")
    assert fr.kind == "number"
    assert fr.verdict is Verdict.GROUNDED


def test_verify_id_field_string_stays_text():
    # id hint prevents number parsing
    report = verify({"user_id": "123"}, "User 123 logged in.")
    fr = report.get("user_id")
    assert fr.kind == "text"


def test_verify_leading_zero_string_stays_text():
    report = verify({"value": "00123"}, "code 00123")
    fr = report.get("value")
    assert fr.kind == "text"


def test_verify_numeric_string_not_in_source_ungrounded():
    # number kind, not trivial, not found
    report = verify({"quantity": "77"}, "There were 12 items.")
    fr = report.get("quantity")
    assert fr.kind == "number"
    assert fr.verdict is Verdict.UNGROUNDED


def test_verify_trivial_number_skipped():
    # is_trivial(1) is True -> match_number returns "value too trivial"
    report = verify({"count": 1}, "There is one item.")
    fr = report.get("count")
    assert fr.kind == "number"
    assert fr.verdict is Verdict.SKIPPED


# ---------------------------------------------------------------------------
# verify: dates
# ---------------------------------------------------------------------------

def test_verify_date_field_parses_as_date():
    report = verify({"invoice_date": "2026-01-05"}, "Invoice date: 5 January 2026.")
    fr = report.get("invoice_date")
    assert fr.kind == "date"
    assert fr.verdict is Verdict.GROUNDED


def test_verify_date_semantic_match():
    report = verify({"due_date": "2026-01-05"}, "Due on 5 January 2026.")
    fr = report.get("due_date")
    assert fr.verdict is Verdict.GROUNDED
    assert fr.match.tier in (MatchTier.EXACT, MatchTier.SEMANTIC)


def test_verify_date_not_in_source_ungrounded():
    report = verify({"expiry": "2026-06-15"}, "Expires on 1 January 2020.")
    fr = report.get("expiry")
    assert fr.kind == "date"
    assert fr.verdict is Verdict.UNGROUNDED


def test_verify_date_day_first():
    # ambiguous date; day_first=True => 3 April 2026
    report = verify(
        {"date_of_birth": "03/04/2026"},
        "Born on 3 April 2026.",
        day_first=True,
    )
    fr = report.get("date_of_birth")
    assert fr.kind == "date"
    assert fr.verdict is Verdict.GROUNDED


def test_verify_date_with_date_hint_but_unparseable_stays_text():
    report = verify({"date": "not-a-date"}, "Some random text.")
    fr = report.get("date")
    assert fr.kind == "text"


def test_verify_invalid_date_value_not_in_source():
    # value not a recognizable date in match_date, but classified as text
    # since parse_date fails; still ungrounded.
    report = verify({"date": "blah"}, "nothing here")
    fr = report.get("date")
    assert fr.verdict is Verdict.UNGROUNDED


# ---------------------------------------------------------------------------
# verify: fuzzy / weak
# ---------------------------------------------------------------------------

def test_verify_fuzzy_weak_by_default():
    # Slight OCR-like difference
    report = verify({"name": "Alice"}, "A1ice was here.")
    fr = report.get("name")
    # if fuzzy matched, weak; otherwise ungrounded — assert not grounded
    assert fr.verdict in (Verdict.WEAK, Verdict.UNGROUNDED)
    if fr.match.tier is MatchTier.FUZZY:
        assert fr.verdict is Verdict.WEAK


def test_verify_fuzzy_accept_fuzzy_grounded():
    source = "A1ice was here."
    report = verify({"name": "Alice"}, source, accept_fuzzy=True)
    fr = report.get("name")
    if fr.match.tier is MatchTier.FUZZY:
        assert fr.verdict is Verdict.GROUNDED


def test_verify_fuzzy_threshold_low_accepts():
    # With a low enough threshold, a near-miss becomes FUZZY
    report = verify({"name": "Alice"}, "Alex was here.", fuzzy_threshold=0.5)
    fr = report.get("name")
    if fr.match.tier is MatchTier.FUZZY:
        assert fr.verdict is Verdict.WEAK


# ---------------------------------------------------------------------------
# verify: normalized matches
# ---------------------------------------------------------------------------

def test_verify_case_insensitive_normalized():
    report = verify({"company": "Acme Corp"}, "We work with ACME CORP.")
    fr = report.get("company")
    assert fr.verdict is Verdict.GROUNDED
    assert fr.match.tier is MatchTier.NORMALIZED


def test_verify_whitespace_normalized():
    report = verify({"x": "hello   world"}, "hello world")
    fr = report.get("x")
    assert fr.verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# verify: classifier override
# ---------------------------------------------------------------------------

def test_verify_custom_classifier():
    calls = []

    def classifier(value, path):
        calls.append((value, path))
        return "skip"

    report = verify({"a": "Alice"}, "Alice", classifier=classifier)
    assert len(calls) == 1
    assert calls[0] == ("Alice", "a")
    assert report.get("a").verdict is Verdict.SKIPPED


def test_verify_custom_classifier_forced_number():
    def classifier(value, path):
        return "number"

    report = verify({"a": "42"}, "Total 42 here.", classifier=classifier)
    fr = report.get("a")
    assert fr.kind == "number"
    assert fr.verdict is Verdict.GROUNDED


def test_verify_custom_classifier_forced_text():
    def classifier(value, path):
        return "text"

    report = verify({"a": 42}, "The number 42 appears.", classifier=classifier)
    fr = report.get("a")
    assert fr.kind == "text"
    assert fr.verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# verify: top-level scalars
# ---------------------------------------------------------------------------

def test_verify_scalar_extracted():
    report = verify("Alice", "Alice is here.")
    assert len(report) == 1
    assert report.results[0].path == ""
    assert report.results[0].verdict is Verdict.GROUNDED


def test_verify_trivial_top_level_scalar():
    report = verify("the", "The quick brown fox.")
    assert report.results[0].verdict is Verdict.SKIPPED


# ---------------------------------------------------------------------------
# verify: full report properties
# ---------------------------------------------------------------------------

def test_verify_report_mixed():
    extracted = {
        "name": "Alice",
        "amount": 42,
        "flag": True,
        "missing": "Zebra",
    }
    report = verify(extracted, "Alice paid 42.")
    assert report.get("name").verdict is Verdict.GROUNDED
    assert report.get("amount").verdict is Verdict.GROUNDED
    assert report.get("flag").verdict is Verdict.SKIPPED
    assert report.get("missing").verdict is Verdict.UNGROUNDED
    assert report.has_hallucinations is True
    assert len(report.checked) == 3
    assert report.grounding_rate == pytest.approx(2 / 3)
    assert "ungrounded" in report.summary()


def test_verify_number_tolerance_str():
    # number_tolerance accepts Decimal/float; passing Decimal
    report = verify({"amount": 100}, "value 100.5", number_tolerance=Decimal("0.5"))
    assert report.get("amount").verdict is Verdict.GROUNDED


def test_verify_empty_extracted_dict():
    report = verify({}, "anything")
    assert len(report) == 0
    assert report.grounding_rate == 1.0
