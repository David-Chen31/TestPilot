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
# Verdict
# ---------------------------------------------------------------------------

def test_verdict_is_str_enum():
    assert isinstance(Verdict.GROUNDED, str)
    assert Verdict.GROUNDED == "grounded"
    assert Verdict.WEAK == "weak"
    assert Verdict.UNGROUNDED == "ungrounded"
    assert Verdict.SKIPPED == "skipped"


# ---------------------------------------------------------------------------
# FieldResult
# ---------------------------------------------------------------------------

def test_field_result_is_hallucination():
    fr = FieldResult("a.b", "x", Verdict.UNGROUNDED, Match(MatchTier.NONE))
    assert fr.is_hallucination is True

    for v in (Verdict.GROUNDED, Verdict.WEAK, Verdict.SKIPPED):
        fr = FieldResult("a.b", "x", v, Match(MatchTier.NONE))
        assert fr.is_hallucination is False


def test_field_result_default_kind_is_text():
    fr = FieldResult("a", 1, Verdict.GROUNDED, Match(MatchTier.EXACT))
    assert fr.kind == "text"


def test_field_result_to_dict_basic():
    fr = FieldResult(
        "a.b",
        "hello",
        Verdict.GROUNDED,
        Match(MatchTier.EXACT, 1.0),
        kind="text",
    )
    d = fr.to_dict()
    assert d["path"] == "a.b"
    assert d["value"] == "hello"
    assert d["verdict"] == "grounded"
    assert d["kind"] == "text"
    assert d["match"]["tier"] == "exact"
    assert d["match"]["score"] == 1.0


def test_field_result_to_dict_decimal_value():
    fr = FieldResult("n", Decimal("12.50"), Verdict.GROUNDED, Match(MatchTier.EXACT))
    d = fr.to_dict()
    assert d["value"] == "12.50"


def test_field_result_to_dict_non_jsonable():
    class Weird:
        def __str__(self):
            return "<weird>"

    fr = FieldResult("w", Weird(), Verdict.SKIPPED, Match(MatchTier.NONE))
    d = fr.to_dict()
    assert d["value"] == "<weird>"


def test_field_result_to_dict_none_and_bool():
    fr = FieldResult("w", None, Verdict.SKIPPED, Match(MatchTier.NONE))
    assert fr.to_dict()["value"] is None
    fr = FieldResult("w", True, Verdict.SKIPPED, Match(MatchTier.NONE))
    assert fr.to_dict()["value"] is True


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fr(path, verdict, value="v", tier=MatchTier.EXACT):
    return FieldResult(path, value, verdict, Match(tier))


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
    assert r.get("x") is None
    assert r.summary() == "0/0 fields grounded (100%), 0 weak, 0 ungrounded"


def test_report_iter_and_len():
    results = [
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.UNGROUNDED, tier=MatchTier.NONE),
    ]
    r = Report(results=results)
    assert len(r) == 2
    assert list(r) == results


def test_report_properties():
    r = Report(
        results=[
            _fr("a", Verdict.GROUNDED),
            _fr("b", Verdict.WEAK, tier=MatchTier.FUZZY),
            _fr("c", Verdict.UNGROUNDED, tier=MatchTier.NONE),
            _fr("d", Verdict.SKIPPED, tier=MatchTier.NONE),
        ]
    )
    assert [x.path for x in r.grounded] == ["a"]
    assert [x.path for x in r.weak] == ["b"]
    assert [x.path for x in r.ungrounded] == ["c"]
    assert [x.path for x in r.checked] == ["a", "b", "c"]
    assert r.has_hallucinations is True


def test_report_has_hallucinations_false():
    r = Report(results=[_fr("a", Verdict.GROUNDED)])
    assert r.has_hallucinations is False


def test_report_grounding_rate():
    r = Report(
        results=[
            _fr("a", Verdict.GROUNDED),
            _fr("b", Verdict.GROUNDED),
            _fr("c", Verdict.WEAK, tier=MatchTier.FUZZY),
            _fr("d", Verdict.UNGROUNDED, tier=MatchTier.NONE),
            _fr("e", Verdict.SKIPPED, tier=MatchTier.NONE),
        ]
    )
    # checked = a,b,c,d -> 4; grounded = 2 -> 0.5
    assert r.grounding_rate == 0.5


def test_report_grounding_rate_all_skipped():
    r = Report(results=[_fr("a", Verdict.SKIPPED, tier=MatchTier.NONE)])
    assert r.grounding_rate == 1.0


def test_report_get():
    r = Report(results=[_fr("a", Verdict.GROUNDED), _fr("b", Verdict.WEAK)])
    assert r.get("b").path == "b"
    assert r.get("missing") is None


def test_report_to_dict():
    r = Report(
        results=[
            _fr("a", Verdict.GROUNDED),
            _fr("b", Verdict.WEAK, tier=MatchTier.FUZZY),
            _fr("c", Verdict.UNGROUNDED, tier=MatchTier.NONE),
            _fr("d", Verdict.SKIPPED, tier=MatchTier.NONE),
        ]
    )
    d = r.to_dict()
    assert d["grounding_rate"] == 0.3333
    assert d["fields_checked"] == 3
    assert d["grounded"] == 1
    assert d["weak"] == 1
    assert d["ungrounded"] == 1
    assert len(d["results"]) == 4


def test_report_summary():
    r = Report(
        results=[
            _fr("a", Verdict.GROUNDED),
            _fr("b", Verdict.WEAK, tier=MatchTier.FUZZY),
            _fr("c", Verdict.UNGROUNDED, tier=MatchTier.NONE),
        ]
    )
    assert r.summary() == "1/3 fields grounded (33%), 1 weak, 1 ungrounded"


# ---------------------------------------------------------------------------
# verify: type errors
# ---------------------------------------------------------------------------

def test_verify_source_must_be_string():
    with pytest.raises(TypeError):
        verify({"a": "b"}, 123)


def test_verify_source_none_raises():
    with pytest.raises(TypeError):
        verify({"a": "b"}, None)


# ---------------------------------------------------------------------------
# verify: basic grounded / ungrounded
# ---------------------------------------------------------------------------

def test_verify_simple_grounded():
    report = verify({"name": "Alice"}, "Hello Alice, welcome.")
    assert len(report) == 1
    r = report.results[0]
    assert r.path == "name"
    assert r.value == "Alice"
    assert r.verdict is Verdict.GROUNDED
    assert r.kind == "text"


def test_verify_simple_ungrounded():
    report = verify({"name": "Bob"}, "Hello Alice, welcome.")
    r = report.results[0]
    assert r.path == "name"
    assert r.verdict is Verdict.UNGROUNDED
    assert r.is_hallucination is True


def test_verify_nested_paths():
    extracted = {
        "user": {"name": "Alice", "city": "Paris"},
        "tags": ["alpha", "beta"],
    }
    source = "Alice lives in Paris and likes alpha beta"
    report = verify(extracted, source)
    paths = {r.path: r for r in report}
    assert set(paths) == {"user.name", "user.city", "tags[0]", "tags[1]"}
    assert paths["user.name"].verdict is Verdict.GROUNDED
    assert paths["tags[1]"].verdict is Verdict.GROUNDED


def test_verify_path_prefix_for_root_key():
    report = verify({"k": "v"}, "v")
    assert report.results[0].path == "k"


def test_verify_skips_empty_string():
    report = verify({"a": ""}, "whatever")
    r = report.results[0]
    assert r.verdict is Verdict.SKIPPED
    assert r.kind == "text"


def test_verify_skips_whitespace_string():
    report = verify({"a": "   "}, "whatever")
    assert report.results[0].verdict is Verdict.SKIPPED


def test_verify_skips_none():
    report = verify({"a": None}, "whatever")
    r = report.results[0]
    assert r.verdict is Verdict.SKIPPED
    assert r.kind == "skip"


def test_verify_skips_bool():
    report = verify({"flag": True}, "true")
    r = report.results[0]
    assert r.verdict is Verdict.SKIPPED
    assert r.kind == "skip"


# ---------------------------------------------------------------------------
# verify: numbers
# ---------------------------------------------------------------------------

def test_verify_number_exact():
    report = verify({"amount": 4500}, "Total: 4500 dollars")
    r = report.results[0]
    assert r.kind == "number"
    assert r.verdict is Verdict.GROUNDED


def test_verify_number_semantic_formatting():
    report = verify({"amount": 4500.0}, "Total: $4,500.00")
    r = report.results[0]
    assert r.kind == "number"
    assert r.verdict is Verdict.GROUNDED


def test_verify_number_ungrounded():
    report = verify({"amount": 9999}, "Total: $4,500.00")
    r = report.results[0]
    assert r.verdict is Verdict.UNGROUNDED


def test_verify_number_tolerance():
    report = verify(
        {"amount": 4501}, "Total: 4500", number_tolerance=Decimal("2")
    )
    r = report.results[0]
    assert r.verdict is Verdict.GROUNDED


def test_verify_number_tolerance_float():
    report = verify({"amount": 4501}, "Total: 4500", number_tolerance=2.0)
    assert report.results[0].verdict is Verdict.GROUNDED


def test_verify_number_within_id_hint_stays_text():
    # field name hints identifier -> classified as text
    report = verify({"order_id": "12345"}, "order 12345")
    r = report.results[0]
    assert r.kind == "text"
    assert r.verdict is Verdict.GROUNDED


def test_verify_numeric_string_classified_as_number():
    report = verify({"total": "1234"}, "the total was 1,234")
    r = report.results[0]
    assert r.kind == "number"
    assert r.verdict is Verdict.GROUNDED


def test_verify_leading_zero_string_not_number():
    report = verify({"count": "00123"}, "code 00123")
    r = report.results[0]
    assert r.kind == "text"


# ---------------------------------------------------------------------------
# verify: dates
# ---------------------------------------------------------------------------

def test_verify_date_semantic():
    report = verify(
        {"invoice_date": "2026-01-05"}, "Invoice dated 5 January 2026"
    )
    r = report.results[0]
    assert r.kind == "date"
    assert r.verdict is Verdict.GROUNDED


def test_verify_date_exact():
    report = verify({"due_date": "2026-01-05"}, "Due: 2026-01-05")
    r = report.results[0]
    assert r.kind == "date"
    assert r.verdict is Verdict.GROUNDED


def test_verify_date_ungrounded():
    report = verify({"invoice_date": "2026-01-05"}, "Invoice dated 5 January 2027")
    r = report.results[0]
    assert r.kind == "date"
    assert r.verdict is Verdict.UNGROUNDED


def test_verify_date_day_first():
    report = verify(
        {"invoice_date": "03/04/2026"},
        "dated 3 April 2026",
        day_first=True,
    )
    r = report.results[0]
    assert r.kind == "date"
    assert r.verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# verify: fuzzy / weak
# ---------------------------------------------------------------------------

def test_verify_fuzzy_is_weak_by_default():
    report = verify({"name": "Alicee"}, "Alice lives here")
    r = report.results[0]
    assert r.verdict is Verdict.WEAK
    assert r.match.tier is MatchTier.FUZZY


def test_verify_fuzzy_accept_as_grounded():
    report = verify({"name": "Alicee"}, "Alice lives here", accept_fuzzy=True)
    r = report.results[0]
    assert r.verdict is Verdict.GROUNDED


def test_verify_fuzzy_threshold_respected():
    # A very high threshold should reject the near match.
    report = verify({"name": "Xqzpt"}, "Alice lives here", fuzzy_threshold=0.99)
    r = report.results[0]
    assert r.verdict is Verdict.UNGROUNDED


# ---------------------------------------------------------------------------
# verify: trivial values -> skipped
# ---------------------------------------------------------------------------

def test_verify_trivial_text_skipped():
    report = verify({"x": "a"}, "some source text")
    r = report.results[0]
    assert r.verdict is Verdict.SKIPPED
    assert r.kind == "text"


def test_verify_trivial_number_skipped():
    report = verify({"n": 0}, "some source text")
    r = report.results[0]
    assert r.kind == "number"
    assert r.verdict is Verdict.SKIPPED


def test_verify_not_numeric_skipped():
    # Force a value into number matching that is not numeric by using a
    # custom classifier.
    report = verify(
        {"x": "abc"},
        "some source text",
        classifier=lambda v, p: "number",
    )
    r = report.results[0]
    assert r.kind == "number"
    assert r.verdict is Verdict.SKIPPED


# ---------------------------------------------------------------------------
# verify: ignore
# ---------------------------------------------------------------------------

def test_verify_ignore_paths():
    extracted = {"a": "Alice", "b": "Bob"}
    report = verify(extracted, "Alice only", ignore=["b"])
    paths = {r.path for r in report}
    assert paths == {"a"}


def test_verify_ignore_nested_path():
    extracted = {"user": {"name": "Alice", "secret": "zzz"}}
    report = verify(extracted, "Alice the human", ignore=["user.secret"])
    assert {r.path for r in report} == {"user.name"}


# ---------------------------------------------------------------------------
# verify: classifier override
# ---------------------------------------------------------------------------

def test_verify_custom_classifier():
    calls = []

    def classifier(value, path):
        calls.append((value, path))
        return "skip"

    report = verify({"a": "hello"}, "hello", classifier=classifier)
    assert calls == [("hello", "a")]
    r = report.results[0]
    assert r.verdict is Verdict.SKIPPED
    assert r.kind == "skip"


# ---------------------------------------------------------------------------
# verify: traversal of dataclasses, pydantic-like, tuples, sets
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Person:
    name: str
    age: int


def test_verify_dataclass_input():
    report = verify(Person(name="Alice", age=30), "Alice is 30 years old")
    paths = {r.path: r for r in report}
    assert paths["name"].verdict is Verdict.GROUNDED
    assert paths["age"].verdict is Verdict.GROUNDED


def test_verify_nested_dataclass():
    @dataclasses.dataclass
    class Inner:
        city: str

    @dataclasses.dataclass
    class Outer:
        inner: Inner

    report = verify(Outer(inner=Inner(city="Paris")), "She lives in Paris")
    assert report.results[0].path == "inner.city"
    assert report.results[0].verdict is Verdict.GROUNDED


def test_verify_dataclass_class_not_traversed_as_instance():
    # Passing the dataclass type itself should be treated as a leaf value.
    report = verify(Person, "some source")
    # Not a mapping (is a type), so treated as a scalar leaf with empty path.
    assert len(report) == 1
    assert report.results[0].path == ""


def test_verify_pydantic_like_model():
    class FakeModel:
        def __init__(self, data):
            self._data = data

        def model_dump(self):
            return self._data

    report = verify(FakeModel({"name": "Alice"}), "Alice here")
    assert report.results[0].path == "name"
    assert report.results[0].verdict is Verdict.GROUNDED


def test_verify_pydantic_like_model_dump_raises():
    class BadModel:
        def model_dump(self):
            raise RuntimeError("boom")

    # Falls through to treating it as a scalar leaf.
    report = verify(BadModel(), "some source")
    assert len(report) == 1
    assert report.results[0].path == ""


def test_verify_tuple_and_set():
    report = verify({"items": ("alpha", "beta")}, "alpha beta")
    paths = {r.path for r in report}
    assert paths == {"items[0]", "items[1]"}

    report = verify({"s": {"only"}}, "only")
    assert report.results[0].value == "only"


def test_verify_nested_list_of_dicts():
    extracted = {"rows": [{"v": "one"}, {"v": "two"}]}
    report = verify(extracted, "one two")
    paths = {r.path for r in report}
    assert paths == {"rows[0].v", "rows[1].v"}


# ---------------------------------------------------------------------------
# verify: root is a scalar
# ---------------------------------------------------------------------------

def test_verify_scalar_root():
    report = verify("Alice", "Hello Alice")
    assert len(report) == 1
    r = report.results[0]
    assert r.path == ""
    assert r.verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# verify: report end-to-end properties
# ---------------------------------------------------------------------------

def test_verify_report_aggregates():
    report = verify(
        {"a": "Alice", "b": "ghostvalue", "c": ""},
        "Alice is here",
    )
    assert report.has_hallucinations is True
    assert len(report.ungrounded) == 1
    assert len(report.checked) == 2
    assert report.grounding_rate == 0.5
    d = report.to_dict()
    assert d["fields_checked"] == 2
    assert d["grounded"] == 1
    assert d["ungrounded"] == 1


def test_verify_field_result_kind_stored():
    report = verify({"amount": 42}, "value 42")
    assert report.results[0].kind == "number"
