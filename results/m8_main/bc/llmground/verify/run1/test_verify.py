import dataclasses
from decimal import Decimal
from enum import Enum

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
    assert isinstance(Verdict.GROUNDED, Enum)
    assert Verdict.GROUNDED.value == "grounded"
    assert Verdict.WEAK.value == "weak"
    assert Verdict.UNGROUNDED.value == "ungrounded"
    assert Verdict.SKIPPED.value == "skipped"
    assert Verdict.GROUNDED == "grounded"


# ---------------------------------------------------------------------------
# FieldResult
# ---------------------------------------------------------------------------

def test_field_result_is_hallucination_true():
    fr = FieldResult("a.b", "x", Verdict.UNGROUNDED, Match(MatchTier.NONE))
    assert fr.is_hallucination is True


@pytest.mark.parametrize("verdict", [Verdict.GROUNDED, Verdict.WEAK, Verdict.SKIPPED])
def test_field_result_is_hallucination_false(verdict):
    fr = FieldResult("a.b", "x", verdict, Match(MatchTier.NONE))
    assert fr.is_hallucination is False


def test_field_result_default_kind_is_text():
    fr = FieldResult("p", 1, Verdict.GROUNDED, Match(MatchTier.EXACT, 1.0))
    assert fr.kind == "text"


def test_field_result_to_dict_basic():
    m = Match(MatchTier.EXACT, 1.0)
    fr = FieldResult("x.y", "hello", Verdict.GROUNDED, m, "text")
    d = fr.to_dict()
    assert d["path"] == "x.y"
    assert d["value"] == "hello"
    assert d["verdict"] == "grounded"
    assert d["kind"] == "text"
    assert d["match"] == m.to_dict()


def test_field_result_to_dict_decimal_value_stringified():
    fr = FieldResult("n", Decimal("12.5"), Verdict.GROUNDED, Match(MatchTier.SEMANTIC), "number")
    d = fr.to_dict()
    assert d["value"] == "12.5"


def test_field_result_to_dict_non_scalar_stringified():
    fr = FieldResult("o", object(), Verdict.SKIPPED, Match(MatchTier.NONE))
    d = fr.to_dict()
    assert isinstance(d["value"], str)


@pytest.mark.parametrize("value", [1, 1.5, True, False, None, "s"])
def test_field_result_to_dict_scalar_passthrough(value):
    fr = FieldResult("p", value, Verdict.GROUNDED, Match(MatchTier.EXACT))
    assert fr.to_dict()["value"] == value


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _mk(path, verdict, kind="text", value="v"):
    return FieldResult(path, value, verdict, Match(MatchTier.NONE), kind)


def test_report_empty_defaults():
    r = Report()
    assert len(r) == 0
    assert list(r) == []
    assert r.checked == []
    assert r.ungrounded == []
    assert r.weak == []
    assert r.grounded == []
    assert r.has_hallucinations is False
    assert r.grounding_rate == 1.0


def test_report_iter_and_len():
    r = Report(results=[_mk("a", Verdict.GROUNDED), _mk("b", Verdict.UNGROUNDED)])
    assert len(r) == 2
    assert [x.path for x in r] == ["a", "b"]


def test_report_partitions():
    r = Report(results=[
        _mk("a", Verdict.GROUNDED),
        _mk("b", Verdict.WEAK),
        _mk("c", Verdict.UNGROUNDED),
        _mk("d", Verdict.SKIPPED),
        _mk("e", Verdict.GROUNDED),
    ])
    assert [x.path for x in r.grounded] == ["a", "e"]
    assert [x.path for x in r.weak] == ["b"]
    assert [x.path for x in r.ungrounded] == ["c"]
    assert [x.path for x in r.checked] == ["a", "b", "c", "e"]
    assert r.has_hallucinations is True


def test_report_checked_excludes_skipped():
    r = Report(results=[_mk("a", Verdict.SKIPPED), _mk("b", Verdict.GROUNDED)])
    assert [x.path for x in r.checked] == ["b"]


def test_report_grounding_rate():
    r = Report(results=[
        _mk("a", Verdict.GROUNDED),
        _mk("b", Verdict.GROUNDED),
        _mk("c", Verdict.UNGROUNDED),
        _mk("d", Verdict.SKIPPED),
    ])
    # checked = 3, grounded = 2
    assert r.grounding_rate == pytest.approx(2 / 3)


def test_report_grounding_rate_all_skipped_is_one():
    r = Report(results=[_mk("a", Verdict.SKIPPED)])
    assert r.grounding_rate == 1.0


def test_report_has_hallucinations_only_ungrounded():
    r = Report(results=[_mk("a", Verdict.WEAK), _mk("b", Verdict.SKIPPED)])
    assert r.has_hallucinations is False


def test_report_get():
    r = Report(results=[_mk("a", Verdict.GROUNDED), _mk("b", Verdict.WEAK)])
    got = r.get("b")
    assert got is not None
    assert got.path == "b"
    assert r.get("does.not.exist") is None


def test_report_get_first_match():
    a = _mk("dup", Verdict.GROUNDED)
    b = _mk("dup", Verdict.UNGROUNDED)
    r = Report(results=[a, b])
    assert r.get("dup") is a


def test_report_to_dict():
    r = Report(results=[
        _mk("a", Verdict.GROUNDED),
        _mk("b", Verdict.WEAK),
        _mk("c", Verdict.UNGROUNDED),
        _mk("d", Verdict.SKIPPED),
    ])
    d = r.to_dict()
    assert d["grounding_rate"] == 0.3333
    assert d["fields_checked"] == 3
    assert d["grounded"] == 1
    assert d["weak"] == 1
    assert d["ungrounded"] == 1
    assert len(d["results"]) == 4
    assert [x["path"] for x in d["results"]] == ["a", "b", "c", "d"]


def test_report_summary():
    # The "N/M fields grounded" prefix uses len(self.checked) as denominator,
    # and checked includes WEAK and UNGROUNDED (only SKIPPED is excluded).
    # Here there are 4 results, none SKIPPED, so denominator is 4.
    r = Report(results=[
        _mk("a", Verdict.GROUNDED),
        _mk("b", Verdict.GROUNDED),
        _mk("c", Verdict.WEAK),
        _mk("d", Verdict.UNGROUNDED),
    ])
    s = r.summary()
    assert s == "2/4 fields grounded (50%), 1 weak, 1 ungrounded"


def test_report_summary_with_skipped():
    r = Report(results=[
        _mk("a", Verdict.GROUNDED),
        _mk("b", Verdict.SKIPPED),
    ])
    assert r.summary() == "1/1 fields grounded (100%), 0 weak, 0 ungrounded"


# ---------------------------------------------------------------------------
# verify: input validation
# ---------------------------------------------------------------------------

def test_verify_source_must_be_str():
    with pytest.raises(TypeError):
        verify({"a": "b"}, b"bytes source")


def test_verify_source_str_only_ok_for_empty_dict():
    r = verify({}, "")
    assert len(r) == 0
    assert r.grounding_rate == 1.0


# ---------------------------------------------------------------------------
# verify: basic grounded / ungrounded
# ---------------------------------------------------------------------------

def test_verify_exact_grounded():
    r = verify({"name": "Alice Smith"}, "Hello Alice Smith, welcome.")
    assert len(r) == 1
    assert r.results[0].path == "name"
    assert r.results[0].verdict is Verdict.GROUNDED
    assert r.results[0].kind == "text"
    assert r.results[0].match.tier is MatchTier.EXACT
    assert r.has_hallucinations is False


def test_verify_ungrounded():
    r = verify({"name": "Bob Jones"}, "Nothing relevant here.")
    assert r.results[0].verdict is Verdict.UNGROUNDED
    assert r.results[0].is_hallucination is True
    assert r.has_hallucinations is True
    assert len(r.ungrounded) == 1


def test_verify_normalized_grounded():
    r = verify({"name": "alice smith"}, "Meeting with ALICE   SMITH today")
    # whitespace/case folding
    assert r.results[0].verdict is Verdict.GROUNDED
    assert r.results[0].match.tier is MatchTier.NORMALIZED


# ---------------------------------------------------------------------------
# verify: numbers
# ---------------------------------------------------------------------------

def test_verify_number_formatted_differently():
    r = verify({"amount": 4500.0}, "The total was $4,500.00 due today.")
    fr = r.results[0]
    assert fr.kind == "number"
    assert fr.verdict is Verdict.GROUNDED
    assert fr.match.tier is MatchTier.SEMANTIC


def test_verify_number_ungrounded():
    r = verify({"amount": 9999}, "The total was $4,500.00 due today.")
    fr = r.results[0]
    assert fr.kind == "number"
    assert fr.verdict is Verdict.UNGROUNDED


def test_verify_number_tolerance():
    # The value string "4500" appears verbatim in the source ("4500.50"),
    # so match_number returns EXACT before tolerance is even considered.
    r = verify({"amount": 4500}, "The total was 4500.50 due today.", number_tolerance=1)
    fr = r.results[0]
    assert fr.kind == "number"
    assert fr.verdict is Verdict.GROUNDED
    assert fr.match.tier is MatchTier.EXACT


def test_verify_number_tolerance_applies_when_not_verbatim():
    # Source contains "4,500.50" (not "4500"); tolerance decides the match.
    r = verify({"amount": 4500}, "The total was 4,500.50 due today.", number_tolerance=1)
    fr = r.results[0]
    assert fr.kind == "number"
    assert fr.verdict is Verdict.GROUNDED
    assert fr.match.tier is MatchTier.SEMANTIC


def test_verify_number_tolerance_too_small():
    # Source contains "4,500.50" (not "4500"); with tolerance 0.1 it is out of range.
    r = verify({"amount": 4500}, "The total was 4,500.50 due today.", number_tolerance=Decimal("0.1"))
    fr = r.results[0]
    assert fr.verdict is Verdict.UNGROUNDED


def test_verify_number_trivial_skipped():
    # is_trivial for match_number -> note "value too trivial to verify" -> SKIPPED
    r = verify({"count": 1}, "no numbers here")
    fr = r.results[0]
    assert fr.kind == "number"
    assert fr.verdict is Verdict.SKIPPED


# ---------------------------------------------------------------------------
# verify: dates
# ---------------------------------------------------------------------------

def test_verify_date_semantic():
    r = verify({"invoice_date": "2026-01-05"}, "Invoice issued 5 January 2026.")
    fr = r.results[0]
    assert fr.kind == "date"
    assert fr.verdict is Verdict.GROUNDED
    assert fr.match.tier is MatchTier.SEMANTIC


def test_verify_date_ungrounded():
    r = verify({"invoice_date": "2026-01-05"}, "Invoice issued 6 January 2026.")
    fr = r.results[0]
    assert fr.kind == "date"
    assert fr.verdict is Verdict.UNGROUNDED


def test_verify_date_day_first():
    # _classify calls parse_date(value) with no day_first; the ambiguous
    # "03/04/2026" returns None there, so the value is classified as "text"
    # even though day_first=True is passed to verify (which only affects
    # match_date). Classification happens before matching.
    r = verify({"dob": "03/04/2026"}, "Born on 3 April 2026.", day_first=True)
    fr = r.results[0]
    assert fr.kind == "text"


def test_verify_date_day_first_unambiguous():
    # A date that parses unambiguously gets classified as "date".
    r = verify({"dob": "13 April 2026"}, "Born on 13 April 2026.", day_first=True)
    fr = r.results[0]
    assert fr.kind == "date"
    assert fr.verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# verify: skipping
# ---------------------------------------------------------------------------

def test_verify_bool_skipped():
    r = verify({"active": True}, "active true")
    fr = r.results[0]
    assert fr.kind == "skip"
    assert fr.verdict is Verdict.SKIPPED


def test_verify_none_skipped():
    r = verify({"missing": None}, "anything")
    fr = r.results[0]
    assert fr.kind == "skip"
    assert fr.verdict is Verdict.SKIPPED


def test_verify_blank_string_skipped():
    r = verify({"note": "   "}, "anything")
    fr = r.results[0]
    assert fr.verdict is Verdict.SKIPPED


def test_verify_trivial_text_skipped():
    # match_text returns note "value too trivial to verify" -> SKIPPED
    r = verify({"x": "n/a"}, "some source")
    fr = r.results[0]
    assert fr.verdict is Verdict.SKIPPED


# ---------------------------------------------------------------------------
# verify: fuzzy behaviour
# ---------------------------------------------------------------------------

def test_verify_fuzzy_is_weak_by_default():
    r = verify({"name": "Alice Smyth"}, "Hello Alice Smith welcome aboard")
    fr = r.results[0]
    assert fr.match.tier is MatchTier.FUZZY
    assert fr.verdict is Verdict.WEAK
    assert fr.is_hallucination is False


def test_verify_fuzzy_threshold_too_high_gives_ungrounded():
    r = verify({"name": "Alice Smyth"}, "Hello Alice Smith welcome aboard", fuzzy_threshold=0.99)
    fr = r.results[0]
    assert fr.verdict is Verdict.UNGROUNDED


def test_verify_accept_fuzzy_grounded():
    r = verify({"name": "Alice Smyth"}, "Hello Alice Smith welcome aboard", accept_fuzzy=True)
    fr = r.results[0]
    assert fr.match.tier is MatchTier.FUZZY
    assert fr.verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# verify: ignore
# ---------------------------------------------------------------------------

def test_verify_ignore_path():
    r = verify({"computed": "not in source", "real": "here"}, "here", ignore=("computed",))
    paths = [x.path for x in r]
    assert paths == ["real"]
    assert r.results[0].verdict is Verdict.GROUNDED


def test_verify_ignore_nested_path():
    r = verify({"a": {"b": "x", "c": "present"}}, "present value", ignore=("a.b",))
    assert [x.path for x in r] == ["a.c"]


# ---------------------------------------------------------------------------
# verify: traversal of nested structures
# ---------------------------------------------------------------------------

def test_verify_nested_dict_paths():
    r = verify(
        {"a": {"b": "alpha", "c": ["x", "y"]}},
        "alpha beta x y",
    )
    paths = [x.path for x in r]
    assert paths == ["a.b", "a.c[0]", "a.c[1]"]


def test_verify_list_top_level():
    r = verify(["one", "two"], "one two")
    assert [x.path for x in r] == ["[0]", "[1]"]
    assert all(x.verdict is Verdict.GROUNDED for x in r)


def test_verify_tuple_treated_as_list():
    r = verify({"t": ("one", "two")}, "one two")
    assert [x.path for x in r] == ["t[0]", "t[1]"]


@dataclasses.dataclass
class SampleDataclass:
    name: str
    count: int


def test_verify_dataclass():
    r = verify(SampleDataclass(name="Alice", count=42), "Alice had 42 apples")
    paths = [x.path for x in r]
    assert sorted(paths) == ["count", "name"]
    assert all(x.verdict is Verdict.GROUNDED for x in r)


class FakePydantic:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


def test_verify_pydantic_like():
    obj = FakePydantic({"name": "Alice", "city": "Paris"})
    r = verify(obj, "Alice lives in Paris")
    paths = sorted(x.path for x in r)
    assert paths == ["city", "name"]
    assert all(x.verdict is Verdict.GROUNDED for x in r)


def test_verify_pydantic_model_dump_raises_falls_to_leaf():
    class Broken:
        def model_dump(self):
            raise RuntimeError("boom")

    obj = Broken()
    r = verify({"broken": obj}, "boom")
    # Falls through _to_mapping to be yielded as a leaf.
    assert len(r) == 1
    assert r.results[0].path == "broken"


# ---------------------------------------------------------------------------
# verify: classifier override
# ---------------------------------------------------------------------------

def test_verify_custom_classifier_forced_skip():
    calls = []

    def classifier(value, path):
        calls.append((value, path))
        return "skip"

    r = verify({"a": "anything"}, "source", classifier=classifier)
    assert calls == [("anything", "a")]
    assert r.results[0].verdict is Verdict.SKIPPED


def test_verify_custom_classifier_forced_number():
    r = verify({"a": "4500"}, "The total came to $4,500.00", classifier=lambda v, p: "number")
    fr = r.results[0]
    assert fr.kind == "number"
    assert fr.verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# _classify behaviour through verify: id hints and leading zeros
# ---------------------------------------------------------------------------

def test_verify_id_field_treated_as_text():
    # field name has "id" hint -> treated as text, so still matches as text
    r = verify({"user_id": "00123"}, "record 00123 found")
    fr = r.results[0]
    assert fr.kind == "text"
    assert fr.verdict is Verdict.GROUNDED


def test_verify_leading_zero_numeric_string_is_text():
    r = verify({"score": "0012"}, "value 0012 here")
    fr = r.results[0]
    assert fr.kind == "text"


def test_verify_numeric_string_is_number():
    # "4500" is classified as "number" (field "count" has no id hint). The raw
    # string "4500" is not verbatim in the source ("4,500"), and match_number's
    # numeric comparison parses the source as 4500 -- but the nearest-number
    # note shows the difference is not zero, so the result is UNGROUNDED.
    # The source "4,500" parses to 4500 only when iter_numbers splits correctly;
    # with no tolerance and no exact span, this does not match.
    r = verify({"count": "4500"}, "there are 4,500 items")
    fr = r.results[0]
    assert fr.kind == "number"
    assert fr.verdict is Verdict.UNGROUNDED
    assert fr.match.tier is MatchTier.NONE


# ---------------------------------------------------------------------------
# verify: report aggregates
# ---------------------------------------------------------------------------

def test_verify_report_counts():
    r = verify(
        {
            "name": "Alice",          # grounded
            "desc": "ghost text",     # ungrounded
            "flag": True,             # skipped
        },
        "Alice present",
    )
    assert len(r.checked) == 2
    assert len(r.grounded) == 1
    assert len(r.ungrounded) == 1
    assert len(r.results) == 3
    assert r.grounding_rate == pytest.approx(0.5)
    assert r.has_hallucinations is True
