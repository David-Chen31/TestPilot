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

class TestVerdict:
    def test_values(self):
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
    def test_is_hallucination_true(self):
        r = FieldResult("p", "v", Verdict.UNGROUNDED, Match(MatchTier.NONE))
        assert r.is_hallucination is True

    def test_is_hallucination_false(self):
        for v in (Verdict.GROUNDED, Verdict.WEAK, Verdict.SKIPPED):
            r = FieldResult("p", "v", v, Match(MatchTier.NONE))
            assert r.is_hallucination is False

    def test_default_kind(self):
        r = FieldResult("p", "v", Verdict.GROUNDED, Match(MatchTier.EXACT))
        assert r.kind == "text"

    def test_to_dict_structure(self):
        r = FieldResult("a.b", "hello", Verdict.GROUNDED, Match(MatchTier.EXACT, 1.0))
        d = r.to_dict()
        assert d["path"] == "a.b"
        assert d["value"] == "hello"
        assert d["verdict"] == "grounded"
        assert d["kind"] == "text"
        assert d["match"]["tier"] == "exact"

    def test_to_dict_decimal_value_is_string(self):
        r = FieldResult("x", Decimal("1.50"), Verdict.GROUNDED, Match(MatchTier.EXACT))
        assert r.to_dict()["value"] == "1.50"

    def test_to_dict_non_jsonable_converted_to_str(self):
        class Custom:
            def __str__(self):
                return "custom-str"

        r = FieldResult("x", Custom(), Verdict.SKIPPED, Match(MatchTier.NONE))
        assert r.to_dict()["value"] == "custom-str"

    def test_to_dict_none_value(self):
        r = FieldResult("x", None, Verdict.SKIPPED, Match(MatchTier.NONE))
        assert r.to_dict()["value"] is None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fr(path, verdict, tier=MatchTier.EXACT, value="v"):
    return FieldResult(path, value, verdict, Match(tier))


class TestReport:
    def test_empty(self):
        r = Report()
        assert len(r) == 0
        assert list(r) == []
        assert r.checked == []
        assert r.ungrounded == []
        assert r.weak == []
        assert r.grounded == []
        assert r.has_hallucinations is False
        # no checked fields -> rate is 1.0
        assert r.grounding_rate == 1.0

    def test_iter_and_len(self):
        a = _fr("a", Verdict.GROUNDED)
        b = _fr("b", Verdict.UNGROUNDED, MatchTier.NONE)
        r = Report([a, b])
        assert len(r) == 2
        assert list(r) == [a, b]

    def test_checked_excludes_skipped(self):
        r = Report([
            _fr("a", Verdict.GROUNDED),
            _fr("b", Verdict.SKIPPED, MatchTier.NONE),
            _fr("c", Verdict.UNGROUNDED, MatchTier.NONE),
        ])
        assert [x.path for x in r.checked] == ["a", "c"]

    def test_ungrounded_and_weak_and_grounded(self):
        r = Report([
            _fr("a", Verdict.GROUNDED),
            _fr("b", Verdict.WEAK, MatchTier.FUZZY),
            _fr("c", Verdict.UNGROUNDED, MatchTier.NONE),
            _fr("d", Verdict.SKIPPED, MatchTier.NONE),
        ])
        assert [x.path for x in r.grounded] == ["a"]
        assert [x.path for x in r.weak] == ["b"]
        assert [x.path for x in r.ungrounded] == ["c"]

    def test_has_hallucinations(self):
        assert Report([_fr("a", Verdict.UNGROUNDED, MatchTier.NONE)]).has_hallucinations
        assert not Report([_fr("a", Verdict.GROUNDED)]).has_hallucinations

    def test_grounding_rate(self):
        r = Report([
            _fr("a", Verdict.GROUNDED),
            _fr("b", Verdict.GROUNDED),
            _fr("c", Verdict.UNGROUNDED, MatchTier.NONE),
            _fr("d", Verdict.SKIPPED, MatchTier.NONE),
        ])
        # checked = a, b, c -> 2/3
        assert r.grounding_rate == pytest.approx(2 / 3)

    def test_grounding_rate_all_skipped(self):
        r = Report([_fr("a", Verdict.SKIPPED, MatchTier.NONE)])
        assert r.grounding_rate == 1.0

    def test_get_found_and_missing(self):
        a = _fr("a", Verdict.GROUNDED)
        r = Report([a])
        assert r.get("a") is a
        assert r.get("missing") is None

    def test_to_dict(self):
        r = Report([
            _fr("a", Verdict.GROUNDED),
            _fr("b", Verdict.GROUNDED),
            _fr("c", Verdict.UNGROUNDED, MatchTier.NONE),
            _fr("d", Verdict.SKIPPED, MatchTier.NONE),
        ])
        d = r.to_dict()
        assert d["grounding_rate"] == round(2 / 3, 4)
        assert d["fields_checked"] == 3
        assert d["grounded"] == 2
        assert d["weak"] == 0
        assert d["ungrounded"] == 1
        assert len(d["results"]) == 4
        assert d["results"][0]["path"] == "a"

    def test_summary(self):
        r = Report([
            _fr("a", Verdict.GROUNDED),
            _fr("c", Verdict.UNGROUNDED, MatchTier.NONE),
        ])
        s = r.summary()
        assert "1/2 fields grounded" in s
        assert "50%" in s
        assert "0 weak" in s
        assert "1 ungrounded" in s


# ---------------------------------------------------------------------------
# verify: type errors
# ---------------------------------------------------------------------------

class TestVerifyErrors:
    def test_source_must_be_str(self):
        with pytest.raises(TypeError):
            verify({}, 123)

    def test_source_must_be_str_none(self):
        with pytest.raises(TypeError):
            verify({}, None)


# ---------------------------------------------------------------------------
# verify: basic behaviors
# ---------------------------------------------------------------------------

class TestVerifyBasic:
    def test_exact_text_grounded(self):
        r = verify({"name": "Acme Corp"}, "Invoice from Acme Corp today")
        assert len(r) == 1
        res = r.results[0]
        assert res.path == "name"
        assert res.value == "Acme Corp"
        assert res.verdict is Verdict.GROUNDED
        assert res.match.tier is MatchTier.EXACT
        assert res.kind == "text"

    def test_ungrounded_text(self):
        r = verify({"name": "Globex"}, "Invoice from Acme Corp today")
        res = r.results[0]
        assert res.verdict is Verdict.UNGROUNDED
        assert res.match.tier is MatchTier.NONE

    def test_fuzzy_is_weak_by_default(self):
        # A string close but not identical to source content.
        source = "The quick brown fox jumps over the lazy dog"
        r = verify({"text": "The quick brown fox jumps over the lazy d0g"}, source)
        res = r.results[0]
        assert res.match.tier is MatchTier.FUZZY
        assert res.verdict is Verdict.WEAK

    def test_fuzzy_accept_fuzzy_grounded(self):
        source = "The quick brown fox jumps over the lazy dog"
        r = verify(
            {"text": "The quick brown fox jumps over the lazy d0g"},
            source,
            accept_fuzzy=True,
        )
        res = r.results[0]
        assert res.match.tier is MatchTier.FUZZY
        assert res.verdict is Verdict.GROUNDED

    def test_fuzzy_threshold_respected(self):
        # Raise the threshold extremely high -> no fuzzy match, ungrounded.
        source = "The quick brown fox jumps over the lazy dog"
        r = verify(
            {"text": "The quick brown fox jumps over the lazy d0g"},
            source,
            fuzzy_threshold=1.1,
        )
        res = r.results[0]
        assert res.match.tier is MatchTier.NONE
        assert res.verdict is Verdict.UNGROUNDED

    def test_empty_string_skipped(self):
        r = verify({"name": "   "}, "anything")
        res = r.results[0]
        assert res.verdict is Verdict.SKIPPED
        assert res.match.tier is MatchTier.NONE

    def test_bool_skipped(self):
        r = verify({"flag": True}, "true in text")
        res = r.results[0]
        assert res.verdict is Verdict.SKIPPED
        assert res.kind == "skip"

    def test_none_skipped(self):
        r = verify({"value": None}, "nothing")
        res = r.results[0]
        assert res.verdict is Verdict.SKIPPED

    def test_trivial_text_skipped(self):
        # Single-char / trivial strings get SKIPPED via match note.
        r = verify({"x": "a"}, "contains a letter")
        res = r.results[0]
        assert res.verdict is Verdict.SKIPPED
        assert res.match.note == "value too trivial to verify"


# ---------------------------------------------------------------------------
# verify: numbers
# ---------------------------------------------------------------------------

class TestVerifyNumbers:
    def test_number_exact_string(self):
        r = verify({"amount": 4500}, "Total: 4500 due")
        res = r.results[0]
        assert res.kind == "number"
        # int -> str "4500" appears verbatim
        assert res.match.tier is MatchTier.EXACT
        assert res.verdict is Verdict.GROUNDED

    def test_number_formatted_differently(self):
        r = verify({"amount": 4500.0}, "Total: $4,500.00 due")
        res = r.results[0]
        # "4500.0" not verbatim, but numeric matches
        assert res.kind == "number"
        assert res.match.tier is MatchTier.SEMANTIC
        assert res.verdict is Verdict.GROUNDED

    def test_number_ungrounded_with_near_note(self):
        r = verify({"amount": 9999}, "Total: 4500.00 due")
        res = r.results[0]
        assert res.verdict is Verdict.UNGROUNDED
        assert res.match.tier is MatchTier.NONE
        assert "nearest number" in res.match.note

    def test_number_tolerance(self):
        r = verify({"amount": 4500}, "Total: 4501 due", number_tolerance=1)
        res = r.results[0]
        assert res.verdict is Verdict.GROUNDED
        assert res.match.tier is MatchTier.SEMANTIC

    def test_number_tolerance_decimal(self):
        r = verify({"amount": Decimal("10.0")}, "Prices: 10.5", number_tolerance=Decimal("0.5"))
        res = r.results[0]
        assert res.verdict is Verdict.GROUNDED

    def test_number_trivial_skipped(self):
        # 0 is trivial -> match note "value too trivial to verify" -> SKIPPED
        r = verify({"count": 0}, "source has 0 as well")
        res = r.results[0]
        assert res.verdict is Verdict.SKIPPED

    def test_decimal_classified_number(self):
        r = verify({"amount": Decimal("1234.56")}, "Total: $1,234.56")
        res = r.results[0]
        assert res.kind == "number"
        assert res.verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# verify: dates
# ---------------------------------------------------------------------------

class TestVerifyDates:
    def test_date_in_date_field_exact(self):
        r = verify({"invoice_date": "2026-01-05"}, "Dated 2026-01-05")
        res = r.results[0]
        assert res.kind == "date"
        assert res.match.tier is MatchTier.EXACT
        assert res.verdict is Verdict.GROUNDED

    def test_date_in_date_field_semantic(self):
        r = verify({"invoice_date": "2026-01-05"}, "Dated 5 January 2026")
        res = r.results[0]
        assert res.kind == "date"
        assert res.match.tier is MatchTier.SEMANTIC
        assert res.verdict is Verdict.GROUNDED

    def test_date_in_date_field_ungrounded(self):
        r = verify({"invoice_date": "2026-01-05"}, "Dated 6 January 2026")
        res = r.results[0]
        assert res.kind == "date"
        assert res.verdict is Verdict.UNGROUNDED

    def test_date_unparseable_not_date(self):
        # A date-ish field name but value not parseable -> text
        r = verify({"due_date": "sometime soon"}, "it will be sometime soon")
        res = r.results[0]
        assert res.kind == "text"

    def test_day_first_option(self):
        r = verify(
            {"dob": "03/04/2026"},
            "Born 3 April 2026",
            day_first=True,
        )
        res = r.results[0]
        assert res.kind == "date"
        assert res.match.tier is MatchTier.SEMANTIC
        assert res.verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# verify: number vs id hints
# ---------------------------------------------------------------------------

class TestVerifyClassificationHints:
    def test_id_field_not_treated_as_number(self):
        r = verify({"order_id": "00123"}, "Order 00123 placed")
        res = r.results[0]
        # id_hint prevents number classification -> text
        assert res.kind == "text"
        assert res.verdict is Verdict.GROUNDED

    def test_bare_numeric_string_is_number(self):
        r = verify({"qty": "42"}, "Quantity is 42 items")
        res = r.results[0]
        assert res.kind == "number"

    def test_leading_zero_numeric_string_not_number(self):
        # e.g. "0007" shouldn't be a number without id hint
        r = verify({"qty": "0007"}, "code 0007")
        res = r.results[0]
        # leading zeros with all-digits -> falls through to text
        assert res.kind == "text"


# ---------------------------------------------------------------------------
# verify: nesting / traversal
# ---------------------------------------------------------------------------

class TestVerifyTraversal:
    def test_nested_dict_paths(self):
        r = verify(
            {"a": {"b": "hello"}},
            "say hello",
        )
        assert r.results[0].path == "a.b"
        assert r.results[0].verdict is Verdict.GROUNDED

    def test_list_paths(self):
        r = verify({"items": ["alpha", "beta"]}, "alpha and beta")
        paths = [x.path for x in r.results]
        assert paths == ["items[0]", "items[1]"]
        assert all(x.verdict is Verdict.GROUNDED for x in r.results)

    def test_tuple_and_set(self):
        r = verify({"vals": ("alpha", "beta")}, "alpha beta")
        assert [x.path for x in r.results] == ["vals[0]", "vals[1]"]
        r2 = verify({"vals": {"zeta"}}, "zeta appears")
        assert len(r2.results) == 1
        assert r2.results[0].value == "zeta"

    def test_dataclass(self):
        @dataclasses.dataclass
        class Inner:
            name: str
            count: int

        @dataclasses.dataclass
        class Outer:
            inner: Inner

        r = verify(Outer(Inner("hello", 7)), "hello count 7")
        paths = [x.path for x in r.results]
        assert paths == ["inner.name", "inner.count"]
        assert all(x.verdict is Verdict.GROUNDED for x in r.results)

    def test_dataclass_type_not_instances(self):
        # A dataclass *class* should be treated as a leaf, not traversed.
        @dataclasses.dataclass
        class Point:
            x: int
            y: int

        r = verify({"p": Point}, "Point class")
        assert len(r.results) == 1
        assert r.results[0].path == "p"

    def test_pydantic_like_model(self):
        class Model:
            def model_dump(self):
                return {"name": "hello", "n": 3}

        r = verify(Model(), "hello number 3")
        paths = [x.path for x in r.results]
        assert paths == ["name", "n"]
        assert all(x.verdict is Verdict.GROUNDED for x in r.results)

    def test_pydantic_like_model_failing_dump_not_traversed(self):
        class Broken:
            def model_dump(self):
                raise RuntimeError("nope")

        r = verify({"broken": Broken()}, "anything")
        # Falls through to leaf; classified as text; likely ungrounded
        assert len(r.results) == 1
        assert r.results[0].path == "broken"

    def test_top_level_list(self):
        r = verify(["alpha", "beta"], "alpha beta")
        assert [x.path for x in r.results] == ["[0]", "[1]"]

    def test_top_level_scalar(self):
        r = verify("hello", "say hello")
        assert len(r.results) == 1
        assert r.results[0].path == ""
        assert r.results[0].verdict is Verdict.GROUNDED


# ---------------------------------------------------------------------------
# verify: ignore & classifier
# ---------------------------------------------------------------------------

class TestVerifyIgnoreAndClassifier:
    def test_ignore_paths(self):
        r = verify(
            {"name": "hello", "computed": "xyz"},
            "hello",
            ignore={"computed"},
        )
        paths = [x.path for x in r.results]
        assert paths == ["name"]

    def test_ignore_iterable(self):
        r = verify(
            {"a": "alpha", "b": "beta"},
            "alpha beta",
            ignore=["a"],
        )
        assert [x.path for x in r.results] == ["b"]

    def test_custom_classifier_forces_text(self):
        calls = []

        def classifier(value, path):
            calls.append((value, path))
            return "text"

        r = verify({"amount": 4500}, "4500", classifier=classifier)
        assert calls == [(4500, "amount")]
        res = r.results[0]
        assert res.kind == "text"
        # text matching against "4500" -> exact
        assert res.match.tier is MatchTier.EXACT
        assert res.verdict is Verdict.GROUNDED

    def test_custom_classifier_skip(self):
        r = verify({"x": "hello"}, "hello", classifier=lambda v, p: "skip")
        assert r.results[0].verdict is Verdict.SKIPPED


# ---------------------------------------------------------------------------
# verify: integration & report aggregate
# ---------------------------------------------------------------------------

class TestVerifyIntegration:
    def test_mixed_report(self):
        extracted = {
            "vendor": "Acme Corp",
            "amount": 4500.0,
            "invoice_date": "2026-01-05",
            "note": "fabricated value that is nowhere",
        }
        source = "Acme Corp billed $4,500.00 on 5 January 2026."
        r = verify(extracted, source)
        by_path = {res.path: res for res in r.results}
        assert by_path["vendor"].verdict is Verdict.GROUNDED
        assert by_path["amount"].verdict is Verdict.GROUNDED
        assert by_path["invoice_date"].verdict is Verdict.GROUNDED
        assert by_path["note"].verdict is Verdict.UNGROUNDED
        assert r.has_hallucinations is True
        assert len(r.ungrounded) == 1
        assert r.grounding_rate == pytest.approx(3 / 4)
        assert r.get("amount") is by_path["amount"]

    def test_skipped_not_counted_in_rate(self):
        extracted = {"flag": True, "name": "Acme"}
        r = verify(extracted, "Acme")
        assert len(r.results) == 2
        assert len(r.checked) == 1
        assert r.grounding_rate == 1.0
        d = r.to_dict()
        assert d["fields_checked"] == 1
        assert d["grounded"] == 1
