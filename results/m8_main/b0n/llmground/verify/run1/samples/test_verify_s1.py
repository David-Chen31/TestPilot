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
from llmground.matchers import Span


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Inner:
    city: str
    amount: int


@dataclasses.dataclass
class Outer:
    name: str
    inner: Inner
    tags: tuple


class FakePydantic:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


class BrokenPydantic:
    def model_dump(self):
        raise RuntimeError("boom")


def make_match(tier=MatchTier.EXACT, score=1.0, note="", spans=None):
    return Match(tier, score, list(spans or []), note)


# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------

class TestVerdict:
    def test_members(self):
        assert Verdict.GROUNDED.value == "grounded"
        assert Verdict.WEAK.value == "weak"
        assert Verdict.UNGROUNDED.value == "ungrounded"
        assert Verdict.SKIPPED.value == "skipped"

    def test_is_str(self):
        assert Verdict.GROUNDED == "grounded"
        assert isinstance(Verdict.GROUNDED, str)


# ---------------------------------------------------------------------------
# FieldResult
# ---------------------------------------------------------------------------

class TestFieldResult:
    def test_defaults(self):
        fr = FieldResult("a.b", "x", Verdict.GROUNDED, make_match())
        assert fr.kind == "text"
        assert fr.path == "a.b"
        assert fr.value == "x"

    def test_is_hallucination_true(self):
        fr = FieldResult("p", "v", Verdict.UNGROUNDED, make_match(MatchTier.NONE))
        assert fr.is_hallucination is True

    def test_is_hallucination_false(self):
        fr = FieldResult("p", "v", Verdict.GROUNDED, make_match())
        assert fr.is_hallucination is False

    def test_to_dict_basic(self):
        m = make_match(MatchTier.EXACT, 1.0, spans=[Span(0, 3, "abc")])
        fr = FieldResult("name", "abc", Verdict.GROUNDED, m, kind="text")
        d = fr.to_dict()
        assert d["path"] == "name"
        assert d["value"] == "abc"
        assert d["verdict"] == "grounded"
        assert d["kind"] == "text"
        assert d["match"]["tier"] == "exact"
        assert d["match"]["score"] == 1.0
        assert d["match"]["spans"][0]["start"] == 0

    def test_to_dict_decimal_value(self):
        m = make_match()
        fr = FieldResult("p", Decimal("1.50"), Verdict.GROUNDED, m, kind="number")
        assert fr.to_dict()["value"] == "1.5"

    def test_to_dict_non_jsonable_value(self):
        m = make_match()
        obj = object()
        fr = FieldResult("p", obj, Verdict.GROUNDED, m)
        assert fr.to_dict()["value"] == str(obj)

    def test_to_dict_none_value(self):
        fr = FieldResult("p", None, Verdict.SKIPPED, make_match(MatchTier.NONE))
        assert fr.to_dict()["value"] is None

    def test_to_dict_bool_value(self):
        fr = FieldResult("p", True, Verdict.SKIPPED, make_match(MatchTier.NONE))
        assert fr.to_dict()["value"] is True


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fr(path, verdict, kind="text"):
    tier = MatchTier.NONE if verdict is Verdict.UNGROUNDED else MatchTier.EXACT
    return FieldResult(path, "v", verdict, make_match(tier), kind)


class TestReport:
    def test_empty_defaults(self):
        r = Report()
        assert len(r) == 0
        assert list(r) == []
        assert r.checked == []
        assert r.grounded == []
        assert r.weak == []
        assert r.ungrounded == []
        assert r.has_hallucinations is False
        assert r.grounding_rate == 1.0

    def test_iter_and_len(self):
        r = Report([_fr("a", Verdict.GROUNDED), _fr("b", Verdict.UNGROUNDED)])
        assert len(r) == 2
        assert [x.path for x in r] == ["a", "b"]

    def test_checked_excludes_skipped(self):
        r = Report([
            _fr("a", Verdict.GROUNDED),
            _fr("b", Verdict.SKIPPED),
            _fr("c", Verdict.UNGROUNDED),
        ])
        assert [x.path for x in r.checked] == ["a", "c"]

    def test_ungrounded_property(self):
        r = Report([
            _fr("a", Verdict.GROUNDED),
            _fr("b", Verdict.UNGROUNDED),
            _fr("c", Verdict.UNGROUNDED),
        ])
        assert [x.path for x in r.ungrounded] == ["b", "c"]
        assert r.has_hallucinations is True

    def test_weak_property(self):
        r = Report([_fr("a", Verdict.WEAK), _fr("b", Verdict.GROUNDED)])
        assert [x.path for x in r.weak] == ["a"]

    def test_grounded_property(self):
        r = Report([_fr("a", Verdict.GROUNDED), _fr("b", Verdict.WEAK)])
        assert [x.path for x in r.grounded] == ["a"]

    def test_grounding_rate(self):
        r = Report([
            _fr("a", Verdict.GROUNDED),
            _fr("b", Verdict.SKIPPED),
            _fr("c", Verdict.UNGROUNDED),
            _fr("d", Verdict.GROUNDED),
        ])
        # checked = 3, grounded = 2
        assert r.grounding_rate == pytest.approx(2 / 3)

    def test_grounding_rate_all_skipped(self):
        r = Report([_fr("a", Verdict.SKIPPED), _fr("b", Verdict.SKIPPED)])
        assert r.grounding_rate == 1.0

    def test_get_found(self):
        a = _fr("a.b", Verdict.GROUNDED)
        r = Report([a, _fr("c", Verdict.GROUNDED)])
        assert r.get("a.b") is a

    def test_get_missing(self):
        r = Report([_fr("a", Verdict.GROUNDED)])
        assert r.get("nope") is None

    def test_to_dict(self):
        r = Report([
            _fr("a", Verdict.GROUNDED),
            _fr("b", Verdict.WEAK),
            _fr("c", Verdict.UNGROUNDED),
            _fr("d", Verdict.SKIPPED),
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
            _fr("a", Verdict.GROUNDED),
            _fr("b", Verdict.WEAK),
            _fr("c", Verdict.UNGROUNDED),
            _fr("d", Verdict.SKIPPED),
        ])
        s = r.summary()
        assert "1/3 fields grounded" in s
        assert "33%" in s
        assert "1 weak" in s
        assert "1 ungrounded" in s


# ---------------------------------------------------------------------------
# verify: error handling
# ---------------------------------------------------------------------------

class TestVerifyErrors:
    def test_source_must_be_str(self):
        with pytest.raises(TypeError, match="source must be a string"):
            verify({}, 123)

    def test_source_none(self):
        with pytest.raises(TypeError):
            verify({}, None)


# ---------------------------------------------------------------------------
# verify: skip behavior
# ---------------------------------------------------------------------------

class TestVerifySkips:
    def test_none_is_skipped(self):
        r = verify({"a": None}, "some doc")
        assert len(r) == 1
        assert r.results[0].verdict is Verdict.SKIPPED

    def test_bool_is_skipped(self):
        r = verify({"a": True, "b": False}, "doc")
        assert all(f.verdict is Verdict.SKIPPED for f in r)
        assert len(r.checked) == 0

    def test_empty_string_skipped(self):
        r = verify({"a": ""}, "doc")
        assert r.results[0].verdict is Verdict.SKIPPED

    def test_whitespace_string_skipped(self):
        r = verify({"a": "   "}, "doc")
        assert r.results[0].verdict is Verdict.SKIPPED

    def test_ignore_set(self):
        r = verify({"a": "hello", "b": "hello"}, "hello world", ignore=["a"])
        paths = [f.path for f in r]
        assert paths == ["b"]
        assert r.results[0].verdict is Verdict.GROUNDED

    def test_custom_classifier_skip(self):
        r = verify({"a": "hello"}, "hello", classifier=lambda v, p: "skip")
        assert r.results[0].verdict is Verdict.SKIPPED


# ---------------------------------------------------------------------------
# verify: text matching
# ---------------------------------------------------------------------------

class TestVerifyText:
    def test_exact_text(self):
        r = verify({"name": "Acme"}, "Welcome to Acme Corp.")
        assert len(r) == 1
        f = r.results[0]
        assert f.verdict is Verdict.GROUNDED
        assert f.kind == "text"
        assert f.match.tier is MatchTier.EXACT

    def test_ungrounded_text(self):
        r = verify({"name": "Globex"}, "Welcome to Acme Corp.")
        f = r.results[0]
        assert f.verdict is Verdict.UNGROUNDED
        assert f.is_hallucination is True
        assert r.has_hallucinations is True

    def test_trivial_note_becomes_skipped(self):
        # "a" is trivial per match_text -> note "value too trivial to verify"
        r = verify({"x": "a"}, "long document without a single letter word")
        f = r.results[0]
        assert f.verdict is Verdict.SKIPPED

    def test_normalized_case(self):
        r = verify({"name": "ACME"}, "welcome to acme corp")
        assert r.results[0].verdict is Verdict.GROUNDED
        assert r.results[0].match.tier is MatchTier.NORMALIZED

    def test_fuzzy_weak_by_default(self):
        # very close but not identical string
        r = verify({"name": "Acme Corportion"}, "Welcome to Acme Corporation.")
        f = r.results[0]
        assert f.verdict is Verdict.WEAK
        assert f.match.tier is MatchTier.FUZZY

    def test_fuzzy_accepted(self):
        r = verify(
            {"name": "Acme Corportion"},
            "Welcome to Acme Corporation.",
            accept_fuzzy=True,
        )
        assert r.results[0].verdict is Verdict.GROUNDED

    def test_fuzzy_threshold_high_rejects(self):
        r = verify(
            {"name": "Acme Corportion"},
            "Welcome to Acme Corporation.",
            fuzzy_threshold=0.999,
        )
        assert r.results[0].verdict is Verdict.UNGROUNDED


# ---------------------------------------------------------------------------
# verify: numeric matching
# ---------------------------------------------------------------------------

class TestVerifyNumbers:
    def test_int_exact(self):
        r = verify({"count": 4500}, "There are 4500 items.")
        f = r.results[0]
        assert f.kind == "number"
        assert f.verdict is Verdict.GROUNDED

    def test_int_formatted_differently(self):
        r = verify({"amount": 4500}, "The total is $4,500.00 please.")
        f = r.results[0]
        assert f.verdict is Verdict.GROUNDED
        assert f.match.tier is MatchTier.SEMANTIC

    def test_number_ungrounded(self):
        r = verify({"amount": 9999}, "The total is $4,500.00 please.")
        assert r.results[0].verdict is Verdict.UNGROUNDED

    def test_number_tolerance(self):
        r = verify(
            {"amount": 4501},
            "The total is $4,500.00 please.",
            number_tolerance=2,
        )
        assert r.results[0].verdict is Verdict.GROUNDED

    def test_number_tolerance_fails(self):
        r = verify(
            {"amount": 4600},
            "The total is $4,500.00 please.",
            number_tolerance=2,
        )
        assert r.results[0].verdict is Verdict.UNGROUNDED

    def test_decimal_kind(self):
        r = verify({"amount": Decimal("4500")}, "Total: 4,500")
        assert r.results[0].kind == "number"
        assert r.results[0].verdict is Verdict.GROUNDED

    def test_float_value(self):
        r = verify({"price": 12.5}, "Only 12.50 here")
        assert r.results[0].kind == "number"
        assert r.results[0].verdict is Verdict.GROUNDED

    def test_numeric_string_becomes_number(self):
        r = verify({"amount": "4500"}, "Total is 4,500")
        f = r.results[0]
        assert f.kind == "number"
        assert f.verdict is Verdict.GROUNDED

    def test_leading_zero_id_stays_text(self):
        r = verify({"code": "00123"}, "Reference code 00123 here")
        f = r.results[0]
        # Leading-zero numeric string should not be reclassified as number.
        assert f.kind == "text"
        assert f.verdict is Verdict.GROUNDED

    def test_id_hint_makes_text(self):
        r = verify({"order_id": "12345"}, "order id 12345")
        f = r.results[0]
        assert f.kind == "text"


# ---------------------------------------------------------------------------
# verify: date matching
# ---------------------------------------------------------------------------

class TestVerifyDates:
    def test_date_kind_by_hint(self):
        r = verify({"invoice_date": "2026-01-05"}, "Issued on 5 January 2026.")
        f = r.results[0]
        assert f.kind == "date"
        assert f.verdict is Verdict.GROUNDED

    def test_date_exact(self):
        r = verify({"due_date": "2026-01-05"}, "Due date: 2026-01-05.")
        f = r.results[0]
        assert f.kind == "date"
        assert f.match.tier is MatchTier.EXACT

    def test_date_ungrounded(self):
        r = verify({"invoice_date": "2026-01-05"}, "Issued on 5 January 2025.")
        assert r.results[0].verdict is Verdict.UNGROUNDED

    def test_date_not_parsed_text(self):
        # date hint but value not a date -> text
        r = verify({"invoice_date": "sometime soon"}, "we will invoice sometime soon")
        assert r.results[0].kind == "text"


# ---------------------------------------------------------------------------
# verify: traversal
# ---------------------------------------------------------------------------

class TestVerifyTraversal:
    def test_nested_dict_paths(self):
        r = verify(
            {"a": {"b": {"c": "Value"}}},
            "the Value is here",
        )
        assert r.results[0].path == "a.b.c"

    def test_list_paths(self):
        r = verify({"items": ["alpha", "beta"]}, "alpha and beta")
        paths = [f.path for f in r]
        assert paths == ["items[0]", "items[1]"]
        assert all(f.verdict is Verdict.GROUNDED for f in r)

    def test_tuple_traversal(self):
        r = verify({"t": ("alpha", "beta")}, "alpha and beta")
        assert [f.path for f in r] == ["t[0]", "t[1]"]

    def test_set_traversal(self):
        r = verify({"s": {"solo"}}, "solo here")
        assert r.results[0].path == "s[0]"
        assert r.results[0].verdict is Verdict.GROUNDED

    def test_top_level_list(self):
        r = verify(["alpha", "beta"], "alpha and beta")
        assert [f.path for f in r] == ["[0]", "[1]"]

    def test_dataclass_traversal(self):
        obj = Outer(name="Alice", inner=Inner(city="Paris", amount=100), tags=("x",))
        r = verify(obj, "Alice lives in Paris with 100 things and x marks")
        paths = {f.path for f in r}
        assert "name" in paths
        assert "inner.city" in paths
        assert "inner.amount" in paths
        assert "tags[0]" in paths

    def test_pydantic_like(self):
        obj = FakePydantic({"name": "Alice", "detail": {"city": "Paris"}})
        r = verify(obj, "Alice lives in Paris")
        paths = {f.path for f in r}
        assert paths == {"name", "detail.city"}

    def test_broken_pydantic_falls_through(self):
        obj = BrokenPydantic()
        # _is_pydantic true, model_dump raises, _to_mapping returns None,
        # then the object itself is a leaf -> yielded under "" path.
        r = verify({"x": obj}, "anything")
        assert len(r) == 1
        assert r.results[0].path == "x"

    def test_dataclass_class_not_instance(self):
        # Passing a dataclass type should be treated as a leaf (str form).
        r = verify({"x": Inner}, "Inner")
        assert len(r) == 1
        assert r.results[0].kind == "text"

    def test_mixed_empty_nested(self):
        r = verify({"a": {"b": []}}, "nothing")
        # empty list yields no leaves
        assert len(r) == 0


# ---------------------------------------------------------------------------
# verify: classifier override and full report
# ---------------------------------------------------------------------------

class TestVerifyClassifierOverride:
    def test_override_forces_text(self):
        # Force number-like value to be treated as text
        r = verify({"amount": 4500}, "4500 somewhere", classifier=lambda v, p: "text")
        assert r.results[0].kind == "text"
        assert r.results[0].verdict is Verdict.GROUNDED

    def test_override_forces_number(self):
        r = verify({"foo": "4500"}, "4,500 total", classifier=lambda v, p: "number")
        assert r.results[0].kind == "number"
        assert r.results[0].verdict is Verdict.GROUNDED


class TestVerifyIntegration:
    def test_report_overall(self):
        extracted = {
            "name": "Acme",
            "amount": 4500,
            "notes": "no matching text here",
            "flag": True,
        }
        source = "Payment of $4,500.00 to Acme."
        r = verify(extracted, source)
        assert isinstance(r, Report)
        assert len(r) == 4
        assert r.get("name").verdict is Verdict.GROUNDED
        assert r.get("amount").verdict is Verdict.GROUNDED
        assert r.get("notes").verdict is Verdict.UNGROUNDED
        assert r.get("flag").verdict is Verdict.SKIPPED
        assert r.has_hallucinations is True
        assert r.grounding_rate == pytest.approx(2 / 3)
        d = r.to_dict()
        assert d["fields_checked"] == 3
        assert d["grounded"] == 2
        assert d["ungrounded"] == 1
