import asyncio
import json
import pytest

from booth.core import (
    ACCEPTED,
    AMBIGUOUS,
    BLOCKED,
    COMPARE_FAILED,
    EMPTY_ANSWER,
    EVIDENCE_DISAGREES,
    INVALID_SCORE,
    NO_EVIDENCE,
    REPAIRED,
    UNCERTAIN,
    Attempt,
    BoothRejected,
    BoothResult,
    acheck,
    check,
    check_with_evidence,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_json(answer="42", confidence=0.9, ambiguous=None, interpretations=None,
              chosen=None, include_answer=True, include_confidence=True):
    obj = {}
    if include_answer:
        obj["answer"] = answer
    if include_confidence:
        obj["confidence"] = confidence
    if ambiguous is not None:
        obj["ambiguous"] = ambiguous
    if interpretations is not None:
        obj["interpretations"] = interpretations
    if chosen is not None:
        obj["chosen_interpretation"] = chosen
    return json.dumps(obj)


def scripted(responses):
    """call_fn returning successive items; list items may be callables or values."""
    it = iter(responses)

    def fn(prompt):
        item = next(it)
        if callable(item) and not isinstance(item, str):
            return item(prompt)
        return item

    return fn


# ---------------------------------------------------------------------------
# Attempt / BoothResult dataclasses
# ---------------------------------------------------------------------------

class TestAttempt:
    def test_defaults(self):
        a = Attempt(raw_text="x", answer="a", confidence=0.5, parse_ok=True)
        assert a.error is None
        assert a.ambiguous is False
        assert a.interpretations == []
        assert a.chosen_interpretation is None
        assert a.passed_validation is True
        assert a.validation_error is None
        assert a.parsed is None
        assert a.call_failed is False

    def test_interpretations_factory_independent(self):
        a = Attempt(raw_text="", answer=None, confidence=None, parse_ok=False)
        b = Attempt(raw_text="", answer=None, confidence=None, parse_ok=False)
        a.interpretations.append("x")
        assert b.interpretations == []


class TestBoothResultProperties:
    def test_n_attempts(self):
        r = BoothResult(answer="a", status=ACCEPTED)
        assert r.n_attempts == 0
        r.attempts.append(Attempt(raw_text="", answer="a", confidence=0.9, parse_ok=True))
        assert r.n_attempts == 1

    @pytest.mark.parametrize("status", [ACCEPTED, REPAIRED])
    def test_ok_true(self, status):
        assert BoothResult(answer="a", status=status).ok is True

    @pytest.mark.parametrize("status", [UNCERTAIN, BLOCKED, AMBIGUOUS])
    def test_ok_false(self, status):
        assert BoothResult(answer="a", status=status).ok is False

    def test_all_parse_failed_empty(self):
        assert BoothResult(answer=None, status=UNCERTAIN).all_parse_failed is False

    def test_all_parse_failed_all_false(self):
        r = BoothResult(answer=None, status=UNCERTAIN, attempts=[
            Attempt(raw_text="", answer=None, confidence=None, parse_ok=False),
            Attempt(raw_text="", answer=None, confidence=None, parse_ok=False),
        ])
        assert r.all_parse_failed is True

    def test_all_parse_failed_mixed(self):
        r = BoothResult(answer=None, status=UNCERTAIN, attempts=[
            Attempt(raw_text="", answer=None, confidence=None, parse_ok=False),
            Attempt(raw_text="", answer="a", confidence=0.5, parse_ok=True),
        ])
        assert r.all_parse_failed is False

    def test_call_failed_empty(self):
        assert BoothResult(answer=None, status=UNCERTAIN).call_failed is False

    def test_call_failed_all(self):
        r = BoothResult(answer=None, status=UNCERTAIN, attempts=[
            Attempt(raw_text="", answer=None, confidence=None, parse_ok=False, call_failed=True),
        ])
        assert r.call_failed is True

    def test_call_failed_mixed(self):
        r = BoothResult(answer=None, status=UNCERTAIN, attempts=[
            Attempt(raw_text="", answer=None, confidence=None, parse_ok=False, call_failed=True),
            Attempt(raw_text="", answer="a", confidence=0.9, parse_ok=True),
        ])
        assert r.call_failed is False

    @pytest.mark.parametrize("reason", [COMPARE_FAILED, INVALID_SCORE])
    def test_checker_failed_true(self, reason):
        assert BoothResult(answer="a", status=UNCERTAIN, reason=reason).checker_failed is True

    @pytest.mark.parametrize("reason", [None, EMPTY_ANSWER, NO_EVIDENCE, EVIDENCE_DISAGREES])
    def test_checker_failed_false(self, reason):
        assert BoothResult(answer="a", status=UNCERTAIN, reason=reason).checker_failed is False

    def test_method_ambiguous(self):
        assert BoothResult(answer="a", status=AMBIGUOUS).method == "ambiguity"

    def test_method_no_attempts(self):
        assert BoothResult(answer="a", status=ACCEPTED).method == "evidence"

    def test_method_parse_failure(self):
        r = BoothResult(answer=None, status=UNCERTAIN, attempts=[
            Attempt(raw_text="", answer=None, confidence=None, parse_ok=False),
        ])
        assert r.method == "parse_failure"

    def test_method_validation(self):
        r = BoothResult(answer="a", status=UNCERTAIN, attempts=[
            Attempt(raw_text="", answer="a", confidence=0.9, parse_ok=True,
                    passed_validation=False),
        ])
        assert r.method == "validation"

    def test_method_confidence(self):
        r = BoothResult(answer="a", status=ACCEPTED, attempts=[
            Attempt(raw_text="", answer="a", confidence=0.9, parse_ok=True),
        ])
        assert r.method == "confidence"

    def test_unwrap_ok(self):
        assert BoothResult(answer="answer", status=ACCEPTED).unwrap() == "answer"

    def test_unwrap_not_ok_raises(self):
        r = BoothResult(answer="answer", status=UNCERTAIN)
        with pytest.raises(BoothRejected) as exc_info:
            r.unwrap()
        assert exc_info.value.result is r

    def test_unwrap_none_answer_raises(self):
        r = BoothResult(answer=None, status=ACCEPTED)
        with pytest.raises(BoothRejected):
            r.unwrap()

    def test_unwrap_or_ok(self):
        assert BoothResult(answer="a", status=ACCEPTED).unwrap_or("d") == "a"

    def test_unwrap_or_default(self):
        assert BoothResult(answer="a", status=UNCERTAIN).unwrap_or("d") == "d"

    def test_to_dict(self):
        a = Attempt(raw_text="raw", answer="a", confidence=0.9, parse_ok=True)
        r = BoothResult(
            answer="a", status=ACCEPTED, confidence=0.9, attempts=[a],
            ambiguous=False, interpretations=["i"], evidence_agreement=1.0,
            parsed={"k": "v"}, reason=None, detail="d",
        )
        d = r.to_dict()
        assert d["answer"] == "a"
        assert d["status"] == ACCEPTED
        assert d["confidence"] == 0.9
        assert d["n_attempts"] == 1
        assert d["ok"] is True
        assert d["all_parse_failed"] is False
        assert d["call_failed"] is False
        assert d["method"] == "confidence"
        assert d["reason"] is None
        assert d["detail"] == "d"
        assert d["checker_failed"] is False
        assert d["interpretations"] == ["i"]
        assert d["evidence_agreement"] == 1.0
        assert d["parsed"] == {"k": "v"}
        assert len(d["attempts"]) == 1
        assert d["attempts"][0]["answer"] == "a"
        assert d["attempts"][0]["raw_text"] == "raw"

    def test_to_dict_empty(self):
        d = BoothResult(answer=None, status=UNCERTAIN).to_dict()
        assert d["attempts"] == []
        assert d["n_attempts"] == 0
        assert d["ok"] is False


class TestBoothRejected:
    def test_message_omits_answer(self):
        r = BoothResult(answer="secret", status=UNCERTAIN)
        exc = BoothRejected(r)
        assert "secret" not in str(exc)
        assert "UNCERTAIN" in str(exc)
        assert exc.result is r
        assert isinstance(exc, Exception)


# ---------------------------------------------------------------------------
# check_with_evidence
# ---------------------------------------------------------------------------

class TestCheckWithEvidence:
    def test_accepted_bool_true(self):
        r = check_with_evidence("a", ["e"], lambda a, e: True)
        assert r.status == ACCEPTED
        assert r.answer == "a"
        assert r.confidence == 1.0
        assert r.evidence_agreement == 1.0
        assert r.reason is None
        assert r.detail is None
        assert r.ok is True

    def test_blocked_bool_false(self):
        r = check_with_evidence("a", ["e"], lambda a, e: False)
        assert r.status == BLOCKED
        assert r.confidence == 0.0
        assert r.reason == EVIDENCE_DISAGREES
        assert r.detail == "compare_fn returned False"
        assert r.ok is False

    def test_float_above_threshold(self):
        r = check_with_evidence("a", ["e"], lambda a, e: 0.85, evidence_threshold=0.7)
        assert r.status == ACCEPTED
        assert r.confidence == 0.85
        assert r.evidence_agreement == 0.85
        assert r.reason is None

    def test_float_equal_threshold(self):
        r = check_with_evidence("a", ["e"], lambda a, e: 0.7, evidence_threshold=0.7)
        assert r.status == ACCEPTED

    def test_float_below_threshold(self):
        r = check_with_evidence("a", ["e"], lambda a, e: 0.6, evidence_threshold=0.7)
        assert r.status == BLOCKED
        assert r.confidence == 0.6
        assert r.evidence_agreement == 0.6
        assert r.reason == EVIDENCE_DISAGREES
        assert "below evidence_threshold" in r.detail
        assert "0.6" in r.detail

    def test_int_score_coerced(self):
        r = check_with_evidence("a", ["e"], lambda a, e: 1)
        assert r.status == ACCEPTED
        assert r.confidence == 1.0

    def test_string_numeric_score_coerced(self):
        r = check_with_evidence("a", ["e"], lambda a, e: "0.9")
        assert r.status == ACCEPTED
        assert r.confidence == 0.9

    def test_empty_answer(self):
        r = check_with_evidence("", ["e"], lambda a, e: True)
        assert r.status == UNCERTAIN
        assert r.answer is None
        assert r.reason == EMPTY_ANSWER
        assert r.detail == "answer was empty or whitespace-only"

    def test_whitespace_answer(self):
        r = check_with_evidence("   ", ["e"], lambda a, e: True)
        assert r.status == UNCERTAIN
        assert r.reason == EMPTY_ANSWER

    def test_empty_evidence_list(self):
        r = check_with_evidence("a", [], lambda a, e: True)
        assert r.status == UNCERTAIN
        assert r.answer == "a"
        assert r.reason == NO_EVIDENCE
        assert r.detail == "evidence sequence was empty"

    def test_empty_evidence_string(self):
        r = check_with_evidence("a", "", lambda a, e: True)
        assert r.reason == NO_EVIDENCE

    def test_answer_not_str_raises(self):
        with pytest.raises(TypeError):
            check_with_evidence(123, ["e"], lambda a, e: True)

    def test_compare_fn_raises(self):
        def bad(a, e):
            raise RuntimeError("boom")

        r = check_with_evidence("a", ["e"], bad)
        assert r.status == UNCERTAIN
        assert r.answer == "a"
        assert r.reason == COMPARE_FAILED
        assert r.detail.startswith("RuntimeError: boom")

    def test_compare_fn_raises_truncated(self):
        def bad(a, e):
            raise RuntimeError("x" * 500)

        r = check_with_evidence("a", ["e"], bad)
        assert len(r.detail) < 300

    def test_non_numeric_score(self):
        r = check_with_evidence("a", ["e"], lambda a, e: object())
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE
        assert r.detail.startswith("compare_fn returned a non-numeric value")

    def test_score_out_of_range(self):
        r = check_with_evidence("a", ["e"], lambda a, e: 1.5)
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE
        assert "outside the [0.0, 1.0] range" in r.detail

    def test_score_negative(self):
        r = check_with_evidence("a", ["e"], lambda a, e: -0.1)
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE

    def test_async_compare_fn_raises(self):
        async def cmp(a, e):
            return True

        with pytest.raises(TypeError):
            check_with_evidence("a", ["e"], cmp)

    def test_evidence_threshold_bool_raises(self):
        with pytest.raises(TypeError):
            check_with_evidence("a", ["e"], lambda a, e: True, evidence_threshold=True)

    def test_evidence_threshold_str_raises(self):
        with pytest.raises(TypeError):
            check_with_evidence("a", ["e"], lambda a, e: True, evidence_threshold="0.5")

    def test_evidence_threshold_out_of_range(self):
        with pytest.raises(ValueError):
            check_with_evidence("a", ["e"], lambda a, e: True, evidence_threshold=1.5)

    def test_evidence_threshold_negative(self):
        with pytest.raises(ValueError):
            check_with_evidence("a", ["e"], lambda a, e: True, evidence_threshold=-0.1)

    def test_passes_evidence_to_compare_fn(self):
        captured = {}

        def cmp(answer, evidence):
            captured["answer"] = answer
            captured["evidence"] = evidence
            return True

        check_with_evidence("my answer", ["e1", "e2"], cmp)
        assert captured["answer"] == "my answer"
        assert captured["evidence"] == ["e1", "e2"]


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

class TestCheckBasic:
    def test_accepted_first_try(self):
        fn = scripted([make_json("42", 0.9)])
        r = check(fn, "prompt")
        assert r.status == ACCEPTED
        assert r.answer == "42"
        assert r.confidence == 0.9
        assert r.n_attempts == 1
        assert r.method == "confidence"
        assert r.attempts[0].parse_ok is True

    def test_prompt_has_suffix(self):
        captured = {}

        def fn(prompt):
            captured["prompt"] = prompt
            return make_json()

        check(fn, "hello")
        assert captured["prompt"].startswith("hello")
        assert "confidence" in captured["prompt"]

    def test_prompt_stripped(self):
        captured = {}

        def fn(prompt):
            captured["prompt"] = prompt
            return make_json()

        check(fn, "hello   \n\n")
        assert captured["prompt"].startswith("hello\n")

    def test_retry_then_repaired(self):
        fn = scripted([make_json("a", 0.3), make_json("b", 0.9)])
        r = check(fn, "prompt", max_retries=1)
        assert r.status == REPAIRED
        assert r.answer == "b"
        assert r.n_attempts == 2

    def test_uncertain_exhausted(self):
        fn = scripted([make_json("a", 0.3), make_json("b", 0.4)])
        r = check(fn, "prompt", max_retries=1)
        assert r.status == UNCERTAIN
        assert r.answer == "b"
        assert r.confidence == 0.4
        assert r.n_attempts == 2
        assert r.method == "confidence"

    def test_max_retries_zero(self):
        fn = scripted([make_json("a", 0.3)])
        r = check(fn, "prompt", max_retries=0)
        assert r.status == UNCERTAIN
        assert r.n_attempts == 1

    def test_threshold_accepted(self):
        fn = scripted([make_json("a", 0.5)])
        r = check(fn, "prompt", threshold=0.5)
        assert r.status == ACCEPTED

    def test_threshold_below(self):
        fn = scripted([make_json("a", 0.4)])
        r = check(fn, "prompt", threshold=0.5, max_retries=0)
        assert r.status == UNCERTAIN


class TestCheckAmbiguous:
    def test_ambiguous_short_circuits(self):
        fn = scripted([make_json("a", 0.9, ambiguous=True,
                                 interpretations=["x", "y"],
                                 chosen="x")])
        r = check(fn, "prompt", max_retries=2)
        assert r.status == AMBIGUOUS
        assert r.ambiguous is True
        assert r.interpretations == ["x", "y"]
        assert r.answer == "a"
        assert r.n_attempts == 1
        assert r.method == "ambiguity"

    def test_ambiguous_string_true(self):
        fn = scripted([make_json("a", 0.9, ambiguous="TRUE")])
        r = check(fn, "prompt")
        assert r.status == AMBIGUOUS

    def test_ambiguous_blank_answer_ok(self):
        fn = scripted([make_json("", 0.9, ambiguous=True, interpretations=["x"])])
        r = check(fn, "prompt")
        assert r.status == AMBIGUOUS
        assert r.answer == ""

    def test_blank_answer_not_ambiguous_fails_parse(self):
        fn = scripted([make_json("  ", 0.9, ambiguous=False)])
        r = check(fn, "prompt", max_retries=0)
        # blank non-ambiguous answer -> parse failure -> uncertain
        assert r.status == UNCERTAIN
        assert r.attempts[0].parse_ok is False
        assert r.all_parse_failed is True


class TestCheckParseFailure:
    def test_parse_failure_then_success(self):
        fn = scripted(["garbage", make_json("a", 0.9)])
        r = check(fn, "prompt", max_retries=1)
        assert r.status == REPAIRED
        assert r.answer == "a"
        assert r.attempts[0].parse_ok is False

    def test_all_parse_failed(self):
        fn = scripted(["garbage1", "garbage2"])
        r = check(fn, "prompt", max_retries=1)
        assert r.status == UNCERTAIN
        assert r.all_parse_failed is True
        assert r.method == "parse_failure"
        assert r.answer is None

    def test_missing_keys(self):
        fn = scripted([json.dumps({"answer": "a"})])
        r = check(fn, "prompt", max_retries=0)
        assert r.attempts[0].parse_ok is False
        assert "missing the required" in r.attempts[0].error

    def test_parse_retry_prompt(self):
        prompts = []

        def fn(prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                return "garbage"
            return make_json("a", 0.9)

        check(fn, "prompt", max_retries=1)
        assert "could not be parsed" in prompts[1]

    def test_markdown_fenced_json(self):
        fenced = "
