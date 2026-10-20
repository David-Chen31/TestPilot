import dataclasses
from decimal import Decimal
from enum import Enum

import pytest

from llmground.matchers import Match, MatchTier, Span
from llmground.verify import (
    FieldResult,
    Report,
    Verdict,
    verify,
)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def test_verdict_is_str_enum():
    assert issubclass(Verdict, str)
    assert issubclass(Verdict, Enum)
    assert Verdict.GROUNDED == "grounded"
    assert Verdict.WEAK.value == "weak"
    assert Verdict.UNGROUNDED.value == "ungrounded"
    assert Verdict.SKIPPED.value == "skipped"


# ---------------------------------------------------------------------------
# FieldResult
# ---------------------------------------------------------------------------

def _match(tier=MatchTier.EXACT, score=1.0, spans=None, note=""):
    return Match(tier, score, spans or [], note)


def test_field_result_is_hallucination_true_for_ungrounded():
    fr = FieldResult("a", "x", Verdict.UNGROUNDED, _match(MatchTier.NONE))
    assert fr.is_hallucination is True


@pytest.mark.parametrize(
    "verdict",
    [Verdict.GROUNDED, Verdict.WEAK, Verdict.SKIPPED],
)
def test_field_result_is_hallucination_false_otherwise(verdict):
    fr = FieldResult("a", "x", verdict, _match())
    assert fr.is_hallucination is False


def test_field_result_default_kind_is_text():
    fr = FieldResult("a", "x", Verdict.GROUNDED, _match())
    assert fr.kind == "text"


def test_field_result_to_dict_serialises_fields():
    span = Span(0, 3, "abc")
    fr = FieldResult(
        "user.name",
        "abc",
        Verdict.GROUNDED,
        Match(MatchTier.EXACT, 1.0, [span]),
        kind="text",
    )
    d = fr.to_dict()
    assert d["path"] == "user.name"
    assert d["value"] == "abc"
    assert d["verdict"] == "grounded"
    assert d["kind"] == "text"
    assert d["match"]["tier"] == "exact"
    assert d["match"]["score"] == 1.0
    assert d["match"]["spans"] == [span.to_dict()]


def test_field_result_to_dict_converts_decimal_to_string():
    fr = FieldResult("amount", Decimal("12.50"), Verdict.GROUNDED, _match())
    assert fr.to_dict()["value"] == "12.50"


def test_field_result_to_dict_converts_other_types_to_string():
    fr = FieldResult("obj", object(), Verdict.SKIPPED, _match())
    assert isinstance(fr.to_dict()["value"], str)


def test_field_result_to_dict_keeps_none_bool_int_float():
    for value in (None, True, 3, 3.5):
        fr = FieldResult("v", value, Verdict.SKIPPED, _match())
        assert fr.to_dict()["value"] == value


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fr(path, verdict, value="x", kind="text"):
    return FieldResult(path, value, verdict, _match(), kind)


def test_report_defaults_empty():
    r = Report()
    assert r.results == []
    assert len(r) == 0
    assert list(r) == []


def test_report_iter_and_len():
    results = [_fr("a", Verdict.GROUNDED), _fr("b", Verdict.UNGROUNDED)]
    r = Report(results)
    assert len(r) == 2
    assert list(r) == results


def test_report_checked_excludes_skipped():
    results = [
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.SKIPPED),
        _fr("c", Verdict.UNGROUNDED),
    ]
    r = Report(results)
    assert [x.path for x in r.checked] == ["a", "c"]


def test_report_ungrounded_weak_grounded_properties():
    results = [
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.WEAK),
        _fr("c", Verdict.UNGROUNDED),
        _fr("d", Verdict.SKIPPED),
    ]
    r = Report(results)
    assert [x.path for x in r.grounded] == ["a"]
    assert [x.path for x in r.weak] == ["b"]
    assert [x.path for x in r.ungrounded] == ["c"]


def test_report_has_hallucinations():
    assert Report([_fr("a", Verdict.GROUNDED)]).has_hallucinations is False
    assert Report([_fr("a", Verdict.UNGROUNDED)]).has_hallucinations is True
    assert Report().has_hallucinations is False


def test_report_grounding_rate_no_checked_returns_one():
    assert Report().grounding_rate == 1.0
    assert Report([_fr("a", Verdict.SKIPPED)]).grounding_rate == 1.0


def test_report_grounding_rate_fraction_of_checked():
    r = Report([
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.GROUNDED),
        _fr("c", Verdict.UNGROUNDED),
        _fr("d", Verdict.WEAK),
        _fr("e", Verdict.SKIPPED),
    ])
    assert r.grounding_rate == pytest.approx(2 / 4)


def test_report_get_finds_result():
    results = [_fr("a", Verdict.GROUNDED), _fr("b", Verdict.UNGROUNDED)]
    r = Report(results)
    assert r.get("a") is results[0]
    assert r.get("b") is results[1]


def test_report_get_missing_returns_none():
    r = Report([_fr("a", Verdict.GROUNDED)])
    assert r.get("zzz") is None


def test_report_to_dict_counts():
    r = Report([
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.GROUNDED),
        _fr("c", Verdict.UNGROUNDED),
        _fr("d", Verdict.WEAK),
        _fr("e", Verdict.SKIPPED),
    ])
    d = r.to_dict()
    assert d["grounding_rate"] == round(2 / 4, 4)
    assert d["fields_checked"] == 4
    assert d["grounded"] == 2
    assert d["weak"] == 1
    assert d["ungrounded"] == 1
    assert len(d["results"]) == 5
    assert d["results"][0]["path"] == "a"


def test_report_summary_text():
    r = Report([
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.UNGROUNDED),
        _fr("c", Verdict.WEAK),
        _fr("d", Verdict.SKIPPED),
    ])
    # 1 grounded of 3 checked (50%), 1 weak, 1 ungrounded
    assert r.summary() == "1/3 fields grounded (50%), 1 weak, 1 ungrounded"


def test_report_summary_empty():
    assert Report().summary() == "0/0 fields grounded (100%), 0 weak, 0 ungrounded"


# ---------------------------------------------------------------------------
# verify: validation
# ---------------------------------------------------------------------------

def test_verify_rejects_non_string_source():
    with pytest.raises(TypeError):
        verify({"a": 1}, 123)
    with pytest.raises(TypeError):
        verify({"a": 1}, None)


# ---------------------------------------------------------------------------
# verify: traversal & matching behavior
# ---------------------------------------------------------------------------

SOURCE = "Invoice from Acme Corp, total $4,500.00, dated 5 January 2026."


def test_verify_exact_text_grounded():
    report = verify({"company": "Acme Corp"}, SOURCE)
    assert len(report) == 1
    r = report.results[0]
    assert r.path == "company"
    assert r.value == "Acme Corp"
    assert r.verdict is Verdict.GROUNDED
    assert r.kind == "text"
    assert r.match.tier is MatchTier.EXACT
    assert report.has_hallucinations is False


def test_verify_ungrounded_text():
    report = verify({"company": "Globex International"}, SOURCE)
    r = report.results[0]
    assert r.verdict is Verdict.UNGROUNDED
    assert r.is_hallucination is True
    assert report.has_hallucinations is True


def test_verify_number_formatted_differently_is_semantic_grounded():
    report = verify({"total": 4500}, SOURCE)
    r = report.results[0]
    assert r.kind == "number"
    assert r.verdict is Verdict.GROUNDED
    assert r.match.tier is MatchTier.SEMANTIC


def test_verify_number_ungrounded():
    report = verify({"total": 9999999}, SOURCE)
    r = report.results[0]
    assert r.verdict is Verdict.UNGROUNDED
    assert r.kind == "number"


def test_verify_number_tolerance():
    report = verify({"total": 4501}, SOURCE, number_tolerance=Decimal("1"))
    r = report.results[0]
    assert r.verdict is Verdict.GROUNDED


def test_verify_date_field_hint_matches_across_format():
    report = verify({"invoice_date": "2026-01-05"}, SOURCE)
    r = report.results[0]
    assert r.kind == "date"
    assert r.verdict is Verdict.GROUNDED
    assert r.match.tier is MatchTier.SEMANTIC


def test_verify_date_field_unmatched():
    report = verify({"invoice_date": "2020-01-05"}, SOURCE)
    r = report.results[0]
    assert r.kind == "date"
    assert r.verdict is Verdict.UNGROUNDED


def test_verify_text_fuzzy_is_weak_by_default():
    source = "Acme Corporation is based here"
    report = verify({"company": "Acme Corporaton"}, source)
    r = report.results[0]
    assert r.match.tier is MatchTier.FUZZY
    assert r.verdict is Verdict.WEAK


def test_verify_fuzzy_accept_fuzzy_is_grounded():
    source = "Acme Corporation is based here"
    report = verify({"company": "Acme Corporaton"}, source, accept_fuzzy=True)
    r = report.results[0]
    assert r.match.tier is MatchTier.FUZZY
    assert r.verdict is Verdict.GROUNDED


def test_verify_fuzzy_threshold_controls_weak():
    source = "Acme Corporation is based here"
    # Very high threshold -> fuzzy match is not reported (NONE -> ungrounded).
    report = verify({"company": "Acme Corporaton"}, source, fuzzy_threshold=0.999)
    r = report.results[0]
    assert r.match.tier is MatchTier.NONE
    assert r.verdict is Verdict.UNGROUNDED


def test_verify_bool_skipped():
    report = verify({"active": True, "inactive": False}, SOURCE)
    assert len(report) == 2
    for r in report.results:
        assert r.kind == "skip"
        assert r.verdict is Verdict.SKIPPED
    assert len(report.checked) == 0


def test_verify_none_skipped():
    report = verify({"note": None}, SOURCE)
    r = report.results[0]
    assert r.kind == "skip"
    assert r.verdict is Verdict.SKIPPED


def test_verify_empty_string_skipped():
    report = verify({"note": "   "}, SOURCE)
    r = report.results[0]
    assert r.verdict is Verdict.SKIPPED


def test_verify_trivial_string_skipped():
    report = verify({"word": "the"}, SOURCE)
    r = report.results[0]
    assert r.match.tier is MatchTier.NONE
    assert r.match.note == "value too trivial to verify"
    assert r.verdict is Verdict.SKIPPED


def test_verify_nested_structure_paths_and_list_indices():
    extracted = {
        "company": "Acme Corp",
        "items": [
            {"label": "Widget", "qty": 2},
            {"label": "Gadget", "qty": 3},
        ],
    }
    source = "Acme Corp Widget Gadget total widgets"
    report = verify(extracted, source)
    paths = {r.path for r in report.results}
    assert paths == {
        "company",
        "items[0].label",
        "items[0].qty",
        "items[1].label",
        "items[1].qty",
    }
    by_path = {r.path: r for r in report.results}
    assert by_path["company"].verdict is Verdict.GROUNDED
    # "Widget"/"Gadget" are in source
    assert by_path["items[0].label"].verdict is Verdict.GROUNDED
    assert by_path["items[1].label"].verdict is Verdict.GROUNDED


def test_verify_ignore_skips_paths():
    extracted = {"keep": "Acme Corp", "drop": "Nothing Here"}
    report = verify(extracted, "Acme Corp", ignore=["drop"])
    assert [r.path for r in report.results] == ["keep"]


def test_verify_dataclass_input():
    @dataclasses.dataclass
    class Doc:
        company: str
        amount: int

    report = verify(Doc(company="Acme Corp", amount=4500), SOURCE)
    by_path = {r.path: r for r in report.results}
    assert by_path["company"].verdict is Verdict.GROUNDED
    assert by_path["amount"].verdict is Verdict.GROUNDED


def test_verify_custom_classifier_overrides_kind():
    def classify(value, path):
        return "skip"

    report = verify({"x": "Acme Corp"}, SOURCE, classifier=classify)
    r = report.results[0]
    assert r.kind == "skip"
    assert r.verdict is Verdict.SKIPPED


def test_verify_custom_classifier_forced_text():
    # Force a number value to be handled as text; match will be UNGROUNDED
    # (no verbatim "12345" in SOURCE).
    def classify(value, path):
        return "text"

    report = verify({"x": 12345}, SOURCE, classifier=classify)
    r = report.results[0]
    assert r.kind == "text"
    assert r.verdict is Verdict.UNGROUNDED


def test_verify_number_id_hint_stays_text():
    # "order_id" suggests an identifier; "00123" should not be parsed as number.
    report = verify({"order_id": "00123"}, "the order 00123 shipped")
    r = report.results[0]
    assert r.kind == "text"
    assert r.verdict is Verdict.GROUNDED


def test_verify_empty_extracted_produces_empty_report():
    report = verify({}, "anything")
    assert len(report) == 0
    assert report.grounding_rate == 1.0
    assert report.summary() == "0/0 fields grounded (100%), 0 weak, 0 ungrounded"


def test_verify_report_to_dict_roundtrip_shapes():
    report = verify({"company": "Acme Corp", "amount": 4500}, SOURCE)
    d = report.to_dict()
    assert d["fields_checked"] == 2
    assert d["grounded"] == 2
    assert d["ungrounded"] == 0
    assert isinstance(d["results"], list)
    assert all("match" in r for r in d["results"])
