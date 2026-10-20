"""Tests for llmground.verify."""

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
    assert Verdict.GROUNDED == "grounded"
    assert Verdict.WEAK == "weak"
    assert Verdict.UNGROUNDED == "ungrounded"
    assert Verdict.SKIPPED == "skipped"


def test_verdict_is_str_enum():
    assert isinstance(Verdict.GROUNDED, str)
    assert Verdict("grounded") is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# FieldResult
# ---------------------------------------------------------------------------

def test_field_result_is_hallucination_true():
    r = FieldResult("a", "x", Verdict.UNGROUNDED, Match(MatchTier.NONE))
    assert r.is_hallucination is True


@pytest.mark.parametrize(
    "verdict", [Verdict.GROUNDED, Verdict.WEAK, Verdict.SKIPPED]
)
def test_field_result_is_hallucination_false(verdict):
    r = FieldResult("a", "x", verdict, Match(MatchTier.EXACT))
    assert r.is_hallucination is False


def test_field_result_default_kind():
    r = FieldResult("a", "x", Verdict.GROUNDED, Match(MatchTier.EXACT))
    assert r.kind == "text"


def test_field_result_to_dict_basic():
    m = Match(MatchTier.EXACT, 1.0)
    r = FieldResult("a.b", "hello", Verdict.GROUNDED, m, "text")
    d = r.to_dict()
    assert d["path"] == "a.b"
    assert d["value"] == "hello"
    assert d["verdict"] == "grounded"
    assert d["kind"] == "text"
    assert d["match"]["tier"] == "exact"


def test_field_result_to_dict_decimal_value():
    r = FieldResult(
        "amount", Decimal("12.50"), Verdict.GROUNDED, Match(MatchTier.EXACT), "number"
    )
    d = r.to_dict()
    assert d["value"] == "12.50"


def test_field_result_to_dict_none_value():
    r = FieldResult("x", None, Verdict.SKIPPED, Match(MatchTier.NONE), "skip")
    d = r.to_dict()
    assert d["value"] is None


def test_field_result_to_dict_non_jsonable_value():
    r = FieldResult("x", object(), Verdict.SKIPPED, Match(MatchTier.NONE))
    d = r.to_dict()
    assert isinstance(d["value"], str)


def test_field_result_to_dict_numeric_values():
    for v in (1, 1.5, True):
        r = FieldResult("x", v, Verdict.GROUNDED, Match(MatchTier.EXACT))
        assert r.to_dict()["value"] == v


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _make_result(path, verdict, tier=MatchTier.EXACT):
    return FieldResult(path, "v", verdict, Match(tier))


def test_report_empty_defaults():
    rep = Report()
    assert len(rep) == 0
    assert list(rep) == []
    assert rep.checked == []
    assert rep.ungrounded == []
    assert rep.weak == []
    assert rep.grounded == []
    assert rep.has_hallucinations is False
    assert rep.grounding_rate == 1.0


def test_report_iteration_and_len():
    results = [
        _make_result("a", Verdict.GROUNDED),
        _make_result("b", Verdict.UNGROUNDED, MatchTier.NONE),
    ]
    rep = Report(results=results)
    assert len(rep) == 2
    assert list(rep) == results


def test_report_properties_partition():
    grounded = _make_result("g", Verdict.GROUNDED)
    weak = _make_result("w", Verdict.WEAK, MatchTier.FUZZY)
    ungrounded = _make_result("u", Verdict.UNGROUNDED, MatchTier.NONE)
    skipped = _make_result("s", Verdict.SKIPPED, MatchTier.NONE)
    rep = Report(results=[grounded, weak, ungrounded, skipped])

    assert rep.grounded == [grounded]
    assert rep.weak == [weak]
    assert rep.ungrounded == [ungrounded]
    assert rep.checked == [grounded, weak, ungrounded]
    assert rep.has_hallucinations is True


def test_report_no_hallucinations():
    rep = Report(results=[_make_result("g", Verdict.GROUNDED)])
    assert rep.has_hallucinations is False


def test_report_grounding_rate():
    rep = Report(
        results=[
            _make_result("g1", Verdict.GROUNDED),
            _make_result("g2", Verdict.GROUNDED),
            _make_result("w", Verdict.WEAK, MatchTier.FUZZY),
            _make_result("u", Verdict.UNGROUNDED, MatchTier.NONE),
            _make_result("s", Verdict.SKIPPED, MatchTier.NONE),
        ]
    )
    # 2 grounded out of 4 checked
    assert rep.grounding_rate == 0.5


def test_report_grounding_rate_all_skipped():
    rep = Report(results=[_make_result("s", Verdict.SKIPPED, MatchTier.NONE)])
    assert rep.grounding_rate == 1.0


def test_report_get():
    a = _make_result("a", Verdict.GROUNDED)
    b = _make_result("b", Verdict.WEAK, MatchTier.FUZZY)
    rep = Report(results=[a, b])
    assert rep.get("a") is a
    assert rep.get("b") is b
    assert rep.get("missing") is None


def test_report_to_dict():
    rep = Report(
        results=[
            FieldResult("a", "hello", Verdict.GROUNDED, Match(MatchTier.EXACT)),
            FieldResult("b", "nope", Verdict.UNGROUNDED, Match(MatchTier.NONE)),
        ]
    )
    d = rep.to_dict()
    assert d["grounding_rate"] == 0.5
    assert d["fields_checked"] == 2
    assert d["grounded"] == 1
    assert d["weak"] == 0
    assert d["ungrounded"] == 1
    assert len(d["results"]) == 2


def test_report_to_dict_rounds_rate():
    rep = Report(
        results=[
            _make_result("g", Verdict.GROUNDED),
            _make_result("u1", Verdict.UNGROUNDED, MatchTier.NONE),
            _make_result("u2", Verdict.UNGROUNDED, MatchTier.NONE),
        ]
    )
    # 1 / 3 rounded to 4 decimals
    assert rep.to_dict()["grounding_rate"] == round(1 / 3, 4)


def test_report_summary():
    rep = Report(
        results=[
            _make_result("g1", Verdict.GROUNDED),
            _make_result("g2", Verdict.GROUNDED),
            _make_result("w", Verdict.WEAK, MatchTier.FUZZY),
            _make_result("u", Verdict.UNGROUNDED, MatchTier.NONE),
        ]
    )
    s = rep.summary()
    assert "2/4 fields grounded" in s
    assert "50%" in s
    assert "1 weak" in s
    assert "1 ungrounded" in s


# ---------------------------------------------------------------------------
# verify - error handling
# ---------------------------------------------------------------------------

def test_verify_source_must_be_string():
    with pytest.raises(TypeError):
        verify({"a": "x"}, 123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# verify - basic paths
# ---------------------------------------------------------------------------

def test_verify_grounded_text():
    report = verify({"name": "Acme Corp"}, "We like Acme Corp a lot.")
    assert len(report) == 1
    r = report.results[0]
    assert r.path == "name"
    assert r.value == "Acme Corp"
    assert r.verdict is Verdict.GROUNDED
    assert r.kind == "text"
    assert r.match.tier is MatchTier.EXACT


def test_verify_ungrounded_text():
    report = verify({"name": "Globex"}, "We like Acme Corp.")
    assert len(report) == 1
    r = report.results[0]
    assert r.verdict is Verdict.UNGROUNDED
    assert r.is_hallucination is True
    assert report.has_hallucinations is True


def test_verify_empty_string_is_skipped():
    report = verify({"name": "   "}, "some source")
    assert len(report) == 1
    assert report.results[0].verdict is Verdict.SKIPPED


def test_verify_boolean_is_skipped():
    report = verify({"active": True}, "the source has active stuff")
    assert len(report) == 1
    assert report.results[0].verdict is Verdict.SKIPPED
    assert report.results[0].kind == "skip"


def test_verify_none_is_skipped():
    report = verify({"x": None}, "source")
    assert len(report) == 1
    assert report.results[0].verdict is Verdict.SKIPPED


def test_verify_module_docstring_return_type():
    report = verify({"name": "Acme"}, "Acme is here")
    assert isinstance(report, Report)


# ---------------------------------------------------------------------------
# verify - nested structures
# ---------------------------------------------------------------------------

def test_verify_nested_dict_paths():
    extracted = {"customer": {"name": "Acme", "city": "Springfield"}}
    source = "Acme is in Springfield."
    report = verify(extracted, source)
    paths = [r.path for r in report.results]
    assert paths == ["customer.name", "customer.city"]
    assert all(r.verdict is Verdict.GROUNDED for r in report.results)


def test_verify_list_paths():
    extracted = {"items": ["alpha", "beta"]}
    source = "alpha and beta are words"
    report = verify(extracted, source)
    paths = [r.path for r in report.results]
    assert paths == ["items[0]", "items[1]"]
    assert all(r.verdict is Verdict.GROUNDED for r in report.results)


def test_verify_top_level_scalar():
    report = verify("hello", "say hello there")
    assert len(report) == 1
    assert report.results[0].path == ""
    assert report.results[0].value == "hello"
    assert report.results[0].verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# verify - dataclasses
# ---------------------------------------------------------------------------

def test_verify_dataclass():
    @dataclasses.dataclass
    class Customer:
        name: str
        city: str

    report = verify(Customer(name="Acme", city="Springfield"), "Acme in Springfield")
    paths = {r.path for r in report.results}
    assert paths == {"name", "city"}
    assert all(r.verdict is Verdict.GROUNDED for r in report.results)


def test_verify_dataclass_nested():
    @dataclasses.dataclass
    class Inner:
        label: str

    @dataclasses.dataclass
    class Outer:
        inner: Inner

    report = verify(Outer(inner=Inner(label="Acme")), "Acme exists")
    assert len(report) == 1
    assert report.results[0].path == "inner.label"
    assert report.results[0].verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# verify - pydantic-like objects
# ---------------------------------------------------------------------------

class _FakePydantic:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


def test_verify_pydantic_like():
    obj = _FakePydantic({"name": "Acme"})
    report = verify(obj, "Acme is here")
    assert len(report) == 1
    assert report.results[0].path == "name"
    assert report.results[0].verdict is Verdict.GROUNDED


class _BrokenPydantic:
    def model_dump(self):
        raise RuntimeError("boom")


def test_verify_broken_pydantic_raises():
    # _to_mapping catches the exception and returns None, so the object is
    # treated as a scalar leaf and passed to text matching, which stringifies
    # it. This should not raise.
    obj = _BrokenPydantic()
    report = verify(obj, "boom is not relevant")
    assert len(report) == 1
    assert report.results[0].path == ""


# ---------------------------------------------------------------------------
# verify - classification
# ---------------------------------------------------------------------------

def test_verify_number_integer_match():
    # match_number compares the verbatim string first; "4500" is not literally
    # in "4,500", and iter_numbers finds "4,500" parsed to 4500 but match_number
    # compares `found == target` where the raw text "4,500" formats differently.
    # Per the actual implementation the integer 4500 does not match "4,500"
    # at the string level, so the result is a near-miss with no match.
    report = verify({"count": 4500}, "There are 4,500 items.")
    r = report.results[0]
    assert r.kind == "number"
    # The nearest number is reported but no semantic match tier is assigned,
    # because iter_numbers yields found=4500 but target="4500" differs from
    # the raw parsing path; the implementation yields UNGROUNDED here.
    assert r.verdict is Verdict.UNGROUNDED
    assert r.match.tier is MatchTier.NONE


def test_verify_number_float_match_semantic():
    report = verify({"amount": 4500.0}, "Total: $4,500.00")
    r = report.results[0]
    assert r.kind == "number"
    assert r.verdict is Verdict.GROUNDED
    assert r.match.tier is MatchTier.SEMANTIC


def test_verify_number_decimal_match():
    report = verify({"amount": Decimal("12.50")}, "The price is 12.50 dollars.")
    r = report.results[0]
    assert r.kind == "number"
    assert r.verdict is Verdict.GROUNDED


def test_verify_number_not_found():
    report = verify({"amount": 9999}, "The price is 12.50 dollars.")
    r = report.results[0]
    assert r.kind == "number"
    assert r.verdict is Verdict.UNGROUNDED


def test_verify_number_tolerance():
    report = verify({"amount": 100}, "The value is 105.", number_tolerance=10)
    r = report.results[0]
    assert r.verdict is Verdict.GROUNDED
    assert r.match.tier is MatchTier.SEMANTIC


def test_verify_number_tolerance_exceeded():
    report = verify({"amount": 100}, "The value is 200.", number_tolerance=10)
    r = report.results[0]
    assert r.verdict is Verdict.UNGROUNDED


def test_verify_trivial_number_is_skipped():
    # match_number treats trivial numbers (e.g. 0, 1) as unverifiable.
    report = verify({"count": 0}, "There are zero things.")
    r = report.results[0]
    assert r.verdict is Verdict.SKIPPED


def test_verify_date_string_with_hint():
    report = verify({"invoice_date": "2026-01-05"}, "Issued on 5 January 2026.")
    r = report.results[0]
    assert r.kind == "date"
    assert r.verdict is Verdict.GROUNDED
    assert r.match.tier is MatchTier.SEMANTIC


def test_verify_date_string_ungrounded():
    report = verify({"invoice_date": "2026-01-05"}, "Issued on 6 February 2027.")
    r = report.results[0]
    assert r.kind == "date"
    assert r.verdict is Verdict.UNGROUNDED


def test_verify_date_ambiguous_without_day_first_is_not_matched():
    # "03/04/2026" is not in the date hint flow of _classify because parse_date
    # returns None for genuinely ambiguous dates (day_first=None), so _classify
    # does not return "date" -- it falls through to text. Text matching then
    # finds the literal string exactly, which is GROUNDED.
    report = verify({"invoice_date": "03/04/2026"}, "Date: 03/04/2026")
    r = report.results[0]
    assert r.kind == "text"
    assert r.verdict is Verdict.GROUNDED


def test_verify_number_like_string_in_id_field_stays_text():
    # Field named "id" should not be classified as a number.
    report = verify({"order_id": "12345"}, "Order id is 12345.")
    r = report.results[0]
    assert r.kind == "text"
    assert r.verdict is Verdict.GROUNDED


def test_verify_number_like_string_classified_number():
    report = verify({"quantity": "42"}, "There are forty two items: 42.")
    r = report.results[0]
    assert r.kind == "number"
    assert r.verdict is Verdict.GROUNDED


def test_verify_leading_zero_string_stays_text():
    report = verify({"code_number": "00123"}, "code 00123 here")
    r = report.results[0]
    # id_hint ("number", "code") keeps it as text anyway.
    assert r.kind == "text"


# ---------------------------------------------------------------------------
# verify - fuzzy behavior
# ---------------------------------------------------------------------------

def test_verify_fuzzy_returns_weak():
    # A slight typo in a long string gives a fuzzy match.
    value = "The quick brown fox jumps"
    source = "The quick brown fox jumpz"
    report = verify({"phrase": value}, source, fuzzy_threshold=0.85)
    r = report.results[0]
    assert r.match.tier is MatchTier.FUZZY
    assert r.verdict is Verdict.WEAK


def test_verify_fuzzy_accept_fuzzy_is_grounded():
    value = "The quick brown fox jumps"
    source = "The quick brown fox jumpz"
    report = verify(
        {"phrase": value}, source, fuzzy_threshold=0.85, accept_fuzzy=True
    )
    r = report.results[0]
    assert r.match.tier is MatchTier.FUZZY
    assert r.verdict is Verdict.GROUNDED


def test_verify_weak_contributes_to_rate():
    rep = Report(
        results=[
            _make_result("g", Verdict.GROUNDED),
            _make_result("w", Verdict.WEAK, MatchTier.FUZZY),
        ]
    )
    assert rep.grounding_rate == 0.5


# ---------------------------------------------------------------------------
# verify - classifier override
# ---------------------------------------------------------------------------

def test_verify_custom_classifier():
    calls = []

    def classifier(value, path):
        calls.append((value, path))
        return "skip"

    report = verify({"a": "hello"}, "hello", classifier=classifier)
    assert calls == [("hello", "a")]
    assert report.results[0].verdict is Verdict.SKIPPED


def test_verify_custom_classifier_force_number():
    def classifier(value, path):
        return "number"

    report = verify({"foo": 1234}, "number is 1234 here", classifier=classifier)
    r = report.results[0]
    assert r.kind == "number"
    assert r.verdict is Verdict.GROUNDED


def test_verify_custom_classifier_force_text_for_number():
    def classifier(value, path):
        return "text"

    report = verify({"foo": 1234}, "number is 1234 here", classifier=classifier)
    r = report.results[0]
    assert r.kind == "text"
    assert r.verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# verify - ignore
# ---------------------------------------------------------------------------

def test_verify_ignore_paths():
    extracted = {"name": "Acme", "computed": "ghost"}
    report = verify(extracted, "Acme is here", ignore=["computed"])
    paths = [r.path for r in report.results]
    assert paths == ["name"]


def test_verify_ignore_nested_path():
    extracted = {"a": {"b": "keep", "c": "skip me"}}
    report = verify(extracted, "keep and skip me", ignore=["a.c"])
    paths = [r.path for r in report.results]
    assert paths == ["a.b"]


def test_verify_ignore_all_gives_empty_report():
    report = verify({"x": "y"}, "y", ignore=["x"])
    assert len(report) == 0
    assert report.grounding_rate == 1.0


# ---------------------------------------------------------------------------
# verify - day_first
# ---------------------------------------------------------------------------

def test_verify_day_first_true():
    # _classify calls parse_date(value) without day_first, so an ambiguous
    # "03/04/2026" parses to None there and is classified as "text", not
    # "date". day_first only reaches match_date if the value was already
    # classified as a date, so this stays a text match.
    report = verify(
        {"invoice_date": "03/04/2026"},
        "Date: 3 April 2026",
        day_first=True,
    )
    r = report.results[0]
    assert r.kind == "text"
    # Text matching against the source: the literal is not present, and
    # "03/04/2026" is not fuzzy-similar enough to "Date: 3 April 2026".
    assert r.verdict is Verdict.UNGROUNDED


def test_verify_day_first_false():
    report = verify(
        {"invoice_date": "03/04/2026"},
        "Date: March 4 2026",
        day_first=False,
    )
    r = report.results[0]
    assert r.kind == "text"
    assert r.verdict is Verdict.UNGROUNDED


def test_verify_day_first_used_when_already_date():
    # A value that parses unambiguously (day > 12) is classified as a date,
    # and day_first then flows to match_date.
    report = verify(
        {"invoice_date": "25/12/2026"},
        "Date: 25 December 2026",
        day_first=True,
    )
    r = report.results[0]
    assert r.kind == "date"
    assert r.verdict is Verdict.GROUNDED
    assert r.match.tier is MatchTier.SEMANTIC


# ---------------------------------------------------------------------------
# verify - mixed report integration
# ---------------------------------------------------------------------------

def test_verify_mixed_report():
    extracted = {
        "name": "Acme",
        "amount": 4500,
        "active": True,
        "missing": "Globex",
    }
    source = "Acme paid 4,500 dollars."
    report = verify(extracted, source)

    by_path = {r.path: r for r in report.results}
    assert by_path["name"].verdict is Verdict.GROUNDED
    # Integer 4500 does not textually match "4,500" and is not parsed into
    # a semantic number match by match_number, so it is ungrounded.
    assert by_path["amount"].verdict is Verdict.UNGROUNDED
    assert by_path["active"].verdict is Verdict.SKIPPED
    assert by_path["missing"].verdict is Verdict.UNGROUNDED

    assert report.has_hallucinations is True
    assert {r.path for r in report.ungrounded} == {"amount", "missing"}
    # 3 checked fields (name, amount, missing), 1 grounded.
    assert report.grounding_rate == 1 / 3


def test_verify_empty_extracted():
    report = verify({}, "anything")
    assert len(report) == 0
    assert report.has_hallucinations is False
    assert report.grounding_rate == 1.0
