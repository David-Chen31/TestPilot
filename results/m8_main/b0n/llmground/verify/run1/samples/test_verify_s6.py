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


SOURCE = (
    "Invoice number INV-001 issued on 5 January 2026. "
    "The customer is Acme Corporation. "
    "Total due: $4,500.00. "
    "Payment reference 2026. "
    "Contact: John Smith, phone 00123."
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

def test_fieldresult_is_hallucination_true():
    fr = FieldResult("a", "x", Verdict.UNGROUNDED, Match(MatchTier.NONE))
    assert fr.is_hallucination is True


@pytest.mark.parametrize("verdict", [Verdict.GROUNDED, Verdict.WEAK, Verdict.SKIPPED])
def test_fieldresult_is_hallucination_false(verdict):
    fr = FieldResult("a", "x", verdict, Match(MatchTier.NONE))
    assert fr.is_hallucination is False


def test_fieldresult_default_kind():
    fr = FieldResult("a", "x", Verdict.GROUNDED, Match(MatchTier.EXACT))
    assert fr.kind == "text"


def test_fieldresult_to_dict_basic():
    m = Match(MatchTier.EXACT, 1.0, [])
    fr = FieldResult("field.one", "hello", Verdict.GROUNDED, m, "text")
    d = fr.to_dict()
    assert d["path"] == "field.one"
    assert d["value"] == "hello"
    assert d["verdict"] == "grounded"
    assert d["kind"] == "text"
    assert d["match"]["tier"] == "exact"
    assert d["match"]["score"] == 1.0


def test_fieldresult_to_dict_decimal():
    m = Match(MatchTier.EXACT, 1.0, [])
    fr = FieldResult("amount", Decimal("4500.00"), Verdict.GROUNDED, m, "number")
    d = fr.to_dict()
    assert d["value"] == "4500.00"


def test_fieldresult_to_dict_none():
    m = Match(MatchTier.NONE)
    fr = FieldResult("x", None, Verdict.SKIPPED, m, "skip")
    d = fr.to_dict()
    assert d["value"] is None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fr(path, verdict, value="v"):
    tier = {
        Verdict.GROUNDED: MatchTier.EXACT,
        Verdict.WEAK: MatchTier.FUZZY,
        Verdict.UNGROUNDED: MatchTier.NONE,
        Verdict.SKIPPED: MatchTier.NONE,
    }[verdict]
    return FieldResult(path, value, verdict, Match(tier))


def test_report_default_empty():
    r = Report()
    assert r.results == []
    assert len(r) == 0
    assert list(r) == []


def test_report_iter_and_len():
    r = Report([_fr("a", Verdict.GROUNDED), _fr("b", Verdict.UNGROUNDED)])
    assert len(r) == 2
    assert [f.path for f in r] == ["a", "b"]


def test_report_checked_excludes_skipped():
    r = Report([
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.SKIPPED),
        _fr("c", Verdict.WEAK),
    ])
    assert [f.path for f in r.checked] == ["a", "c"]


def test_report_ungrounded():
    r = Report([_fr("a", Verdict.UNGROUNDED), _fr("b", Verdict.GROUNDED)])
    assert [f.path for f in r.ungrounded] == ["a"]


def test_report_weak():
    r = Report([_fr("a", Verdict.WEAK), _fr("b", Verdict.GROUNDED)])
    assert [f.path for f in r.weak] == ["a"]


def test_report_grounded():
    r = Report([_fr("a", Verdict.GROUNDED), _fr("b", Verdict.WEAK)])
    assert [f.path for f in r.grounded] == ["a"]


def test_report_has_hallucinations_true():
    r = Report([_fr("a", Verdict.UNGROUNDED)])
    assert r.has_hallucinations is True


def test_report_has_hallucinations_false():
    r = Report([_fr("a", Verdict.GROUNDED), _fr("b", Verdict.SKIPPED)])
    assert r.has_hallucinations is False


def test_report_grounding_rate_empty_is_one():
    assert Report().grounding_rate == 1.0


def test_report_grounding_rate_all_skipped_is_one():
    r = Report([_fr("a", Verdict.SKIPPED)])
    assert r.grounding_rate == 1.0


def test_report_grounding_rate_mixed():
    r = Report([
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.GROUNDED),
        _fr("c", Verdict.WEAK),
        _fr("d", Verdict.UNGROUNDED),
        _fr("e", Verdict.SKIPPED),
    ])
    # 2 grounded out of 4 checked
    assert r.grounding_rate == 0.5


def test_report_get_found():
    r = Report([_fr("a", Verdict.GROUNDED), _fr("b", Verdict.WEAK)])
    assert r.get("b").path == "b"


def test_report_get_missing_returns_none():
    r = Report([_fr("a", Verdict.GROUNDED)])
    assert r.get("nope") is None


def test_report_to_dict_counts_and_rate():
    r = Report([
        _fr("a", Verdict.GROUNDED),
        _fr("b", Verdict.WEAK),
        _fr("c", Verdict.UNGROUNDED),
        _fr("d", Verdict.SKIPPED),
    ])
    d = r.to_dict()
    assert d["fields_checked"] == 3
    assert d["grounded"] == 1
    assert d["weak"] == 1
    assert d["ungrounded"] == 1
    assert d["grounding_rate"] == round(1 / 3, 4)
    assert len(d["results"]) == 4
    assert d["results"][0]["path"] == "a"


def test_report_summary():
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


def test_report_summary_empty():
    s = Report().summary()
    assert "0/0 fields grounded" in s
    assert "100%" in s


# ---------------------------------------------------------------------------
# verify: input validation
# ---------------------------------------------------------------------------

def test_verify_source_must_be_string():
    with pytest.raises(TypeError):
        verify({}, 123)


def test_verify_source_bytes_raises():
    with pytest.raises(TypeError):
        verify({"a": "x"}, b"x")


# ---------------------------------------------------------------------------
# verify: traversal
# ---------------------------------------------------------------------------

def test_verify_flat_dict_paths():
    extracted = {"name": "Acme Corporation", "city": "Atlantis"}
    report = verify(extracted, SOURCE)
    assert report.get("name").verdict is Verdict.GROUNDED
    assert report.get("city").verdict is Verdict.UNGROUNDED


def test_verify_nested_dict_paths():
    extracted = {"customer": {"name": "Acme Corporation"}}
    report = verify(extracted, SOURCE)
    assert report.get("customer.name") is not None
    assert report.get("customer.name").verdict is Verdict.GROUNDED


def test_verify_list_paths():
    extracted = {"names": ["Acme Corporation", "Nobody"]}
    report = verify(extracted, SOURCE)
    assert report.get("names[0]").verdict is Verdict.GROUNDED
    assert report.get("names[1]").verdict is Verdict.UNGROUNDED


def test_verify_tuple_paths():
    extracted = {"vals": ("Acme Corporation",)}
    report = verify(extracted, SOURCE)
    assert report.get("vals[0]").verdict is Verdict.GROUNDED


def test_verify_dataclass():
    @dataclasses.dataclass
    class Customer:
        name: str
        city: str

    report = verify(Customer(name="Acme Corporation", city="Atlantis"), SOURCE)
    assert report.get("name").verdict is Verdict.GROUNDED
    assert report.get("city").verdict is Verdict.UNGROUNDED


def test_verify_dataclass_class_itself_not_treated_as_instance():
    # Passing the dataclass type itself should be walked as a leaf value.
    @dataclasses.dataclass
    class Customer:
        name: str

    report = verify(Customer, SOURCE)
    # The class object becomes a single leaf.
    assert len(report) == 1


def test_verify_pydantic_like():
    class Model:
        def __init__(self, data):
            self._data = data

        def model_dump(self):
            return self._data

    report = verify(Model({"name": "Acme Corporation"}), SOURCE)
    assert report.get("name").verdict is Verdict.GROUNDED


def test_verify_pydantic_model_dump_raises_falls_through():
    class Broken:
        def model_dump(self):
            raise RuntimeError("boom")

        def __str__(self):
            return "broken"

    report = verify({"x": Broken()}, SOURCE)
    # Falls through to being walked as a leaf; str(value) is matched.
    assert report.get("x").kind == "text"
    assert report.get("x").verdict is Verdict.UNGROUNDED


# ---------------------------------------------------------------------------
# verify: classification / kinds
# ---------------------------------------------------------------------------

def test_verify_bool_skipped():
    report = verify({"flag": True}, SOURCE)
    fr = report.get("flag")
    assert fr.verdict is Verdict.SKIPPED
    assert fr.kind == "skip"


def test_verify_none_skipped():
    report = verify({"v": None}, SOURCE)
    assert report.get("v").verdict is Verdict.SKIPPED


def test_verify_empty_string_skipped():
    report = verify({"v": "   "}, SOURCE)
    assert report.get("v").verdict is Verdict.SKIPPED


def test_verify_empty_list_yields_nothing():
    assert len(verify([], SOURCE)) == 0


def test_verify_int_grounded_number():
    report = verify({"qty": 4500}, SOURCE)
    fr = report.get("qty")
    assert fr.kind == "number"
    assert fr.verdict in (Verdict.GROUNDED,)


def test_verify_int_ungrounded_number():
    report = verify({"qty": 9999}, SOURCE)
    assert report.get("qty").kind == "number"
    assert report.get("qty").verdict is Verdict.UNGROUNDED


def test_verify_date_string_classified_as_date():
    report = verify({"invoice_date": "2026-01-05"}, SOURCE)
    fr = report.get("invoice_date")
    assert fr.kind == "date"
    assert fr.verdict is Verdict.GROUNDED


def test_verify_date_string_different_format_semantic():
    report = verify({"invoice_date": "05/01/2026"}, SOURCE)
    fr = report.get("invoice_date")
    assert fr.kind == "date"
    assert fr.verdict is Verdict.GROUNDED


def test_verify_number_like_string_classified_as_number():
    report = verify({"amount": "4500.00"}, SOURCE)
    fr = report.get("amount")
    assert fr.kind == "number"
    assert fr.verdict is Verdict.GROUNDED


def test_verify_id_hint_string_stays_text():
    report = verify({"invoice_id": "00123"}, SOURCE)
    fr = report.get("invoice_id")
    assert fr.kind == "text"


def test_verify_leading_zero_numeric_string_stays_text():
    report = verify({"value": "00123"}, SOURCE)
    fr = report.get("value")
    assert fr.kind == "text"


def test_verify_plain_text_kind():
    report = verify({"name": "Acme Corporation"}, SOURCE)
    assert report.get("name").kind == "text"


# ---------------------------------------------------------------------------
# verify: verdicts from match tiers
# ---------------------------------------------------------------------------

def test_verify_exact_grounded():
    report = verify({"name": "Acme Corporation"}, SOURCE)
    assert report.get("name").verdict is Verdict.GROUNDED


def test_verify_fuzzy_is_weak_by_default():
    # Slightly off spelling yields fuzzy.
    report = verify({"name": "Acme Corporaton"}, SOURCE)
    fr = report.get("name")
    assert fr.verdict is Verdict.WEAK


def test_verify_fuzzy_grounded_with_accept_fuzzy():
    report = verify({"name": "Acme Corporaton"}, SOURCE, accept_fuzzy=True)
    assert report.get("name").verdict is Verdict.GROUNDED


def test_verify_trivial_value_skipped():
    # Very short string is "too trivial to verify" -> SKIPPED.
    report = verify({"x": "a"}, SOURCE)
    assert report.get("x").verdict is Verdict.SKIPPED


def test_verify_boundary_fuzzy_threshold_high_makes_weak_ungrounded():
    # With an extremely high threshold, a fuzzy candidate can't qualify.
    report = verify({"name": "Acme Corporaton"}, SOURCE, fuzzy_threshold=0.999)
    assert report.get("name").verdict is Verdict.UNGROUNDED


# ---------------------------------------------------------------------------
# verify: options
# ---------------------------------------------------------------------------

def test_verify_ignore_path():
    report = verify({"name": "Atlantis"}, SOURCE, ignore=["name"])
    assert report.get("name") is None
    assert len(report) == 0


def test_verify_ignore_nested_path():
    extracted = {"customer": {"name": "Atlantis"}}
    report = verify(extracted, SOURCE, ignore=["customer.name"])
    assert report.get("customer.name") is None


def test_verify_number_tolerance_grounds_close_value():
    extracted = {"amount": 4600}
    default = verify(extracted, SOURCE)
    assert default.get("amount").verdict is Verdict.UNGROUNDED

    tolerant = verify(extracted, SOURCE, number_tolerance=150)
    assert tolerant.get("amount").verdict is Verdict.GROUNDED


def test_verify_number_tolerance_as_float():
    extracted = {"amount": 4600}
    tolerant = verify(extracted, SOURCE, number_tolerance=100.0)
    assert tolerant.get("amount").verdict is Verdict.GROUNDED


def test_verify_custom_classifier():
    def always_skip(value, path):
        return "skip"

    report = verify({"name": "Acme Corporation"}, SOURCE, classifier=always_skip)
    fr = report.get("name")
    assert fr.verdict is Verdict.SKIPPED
    assert fr.kind == "skip"


def test_verify_day_first_disambiguation():
    src = "Date: 03/04/2026."
    extracted = {"date": "03/04/2026"}
    # Exactly as written in source -> exact regardless.
    report = verify(extracted, src, day_first=True)
    assert report.get("date").verdict is Verdict.GROUNDED


def test_verify_date_ambiguous_ungrounded_without_day_first():
    src = "Date: 2026-04-03."
    extracted = {"date": "03/04/2026"}
    report = verify(extracted, src)
    # Ambiguous date, parse_date returns None -> not a recognised date.
    assert report.get("date").verdict is Verdict.UNGROUNDED


def test_verify_skips_when_match_note_too_trivial():
    # A number of 0 is trivial per is_trivial in matchers -> SKIPPED.
    report = verify({"qty": 0}, SOURCE)
    assert report.get("qty").verdict is Verdict.SKIPPED


# ---------------------------------------------------------------------------
# verify: end to end report
# ---------------------------------------------------------------------------

def test_verify_full_report_consistency():
    extracted = {
        "customer": {"name": "Acme Corporation", "city": "Atlantis"},
        "amount": 4500,
        "flag": True,
    }
    report = verify(extracted, SOURCE)
    assert len(report) == 4
    assert report.has_hallucinations is True
    assert [f.path for f in report.ungrounded] == ["customer.city"]
    assert report.get("flag").verdict is Verdict.SKIPPED
    assert report.checked  # excludes the skipped bool
    d = report.to_dict()
    assert d["fields_checked"] == len(report.checked)
