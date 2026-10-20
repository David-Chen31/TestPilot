import asyncio
from dataclasses import asdict

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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

GOOD_JSON = '{"answer": "42", "confidence": 0.9}'
LOW_JSON = '{"answer": "42", "confidence": 0.5}'


def make_attempt(**kwargs):
    defaults = dict(raw_text="raw", answer="a", confidence=0.9, parse_ok=True)
    defaults.update(kwargs)
    return Attempt(**defaults)


# --------------------------------------------------------------------------- #
# Attempt dataclass
# --------------------------------------------------------------------------- #

def test_attempt_defaults():
    a = Attempt(raw_text="raw", answer=None, confidence=None, parse_ok=False)
    assert a.error is None
    assert a.ambiguous is False
    assert a.interpretations == []
    assert a.chosen_interpretation is None
    assert a.passed_validation is True
    assert a.validation_error is None
    assert a.parsed is None
    assert a.call_failed is False


def test_attempt_interpretations_are_independent():
    a = Attempt(raw_text="r", answer="x", confidence=0.5, parse_ok=True)
    b = Attempt(raw_text="r", answer="x", confidence=0.5, parse_ok=True)
    a.interpretations.append("foo")
    assert b.interpretations == []


# --------------------------------------------------------------------------- #
# BoothResult properties
# --------------------------------------------------------------------------- #

def test_boothresult_n_attempts():
    r = BoothResult(answer="x", status=ACCEPTED)
    assert r.n_attempts == 0
    r.attempts = [make_attempt(), make_attempt()]
    assert r.n_attempts == 2


def test_boothresult_ok():
    assert BoothResult(answer="x", status=ACCEPTED).ok is True
    assert BoothResult(answer="x", status=REPAIRED).ok is True
    assert BoothResult(answer="x", status=UNCERTAIN).ok is False
    assert BoothResult(answer="x", status=BLOCKED).ok is False
    assert BoothResult(answer="x", status=AMBIGUOUS).ok is False


def test_boothresult_all_parse_failed():
    assert BoothResult(answer=None, status=UNCERTAIN).all_parse_failed is False
    r = BoothResult(
        answer=None,
        status=UNCERTAIN,
        attempts=[make_attempt(parse_ok=False), make_attempt(parse_ok=False)],
    )
    assert r.all_parse_failed is True
    r2 = BoothResult(
        answer=None,
        status=UNCERTAIN,
        attempts=[make_attempt(parse_ok=False), make_attempt(parse_ok=True)],
    )
    assert r2.all_parse_failed is False


def test_boothresult_call_failed():
    assert BoothResult(answer=None, status=UNCERTAIN).call_failed is False
    r = BoothResult(
        answer=None,
        status=UNCERTAIN,
        attempts=[
            make_attempt(parse_ok=False, call_failed=True),
            make_attempt(parse_ok=False, call_failed=True),
        ],
    )
    assert r.call_failed is True
    r2 = BoothResult(
        answer=None,
        status=UNCERTAIN,
        attempts=[
            make_attempt(parse_ok=False, call_failed=True),
            make_attempt(parse_ok=False, call_failed=False),
        ],
    )
    assert r2.call_failed is False


def test_boothresult_checker_failed():
    assert BoothResult(answer="x", status=UNCERTAIN, reason=COMPARE_FAILED).checker_failed is True
    assert BoothResult(answer="x", status=UNCERTAIN, reason=INVALID_SCORE).checker_failed is True
    assert BoothResult(answer="x", status=UNCERTAIN, reason=NO_EVIDENCE).checker_failed is False
    assert BoothResult(answer="x", status=UNCERTAIN).checker_failed is False


def test_boothresult_method_ambiguity():
    r = BoothResult(answer="x", status=AMBIGUOUS, attempts=[make_attempt()])
    assert r.method == "ambiguity"


def test_boothresult_method_evidence_when_no_attempts():
    assert BoothResult(answer="x", status=UNCERTAIN).method == "evidence"


def test_boothresult_method_parse_failure():
    r = BoothResult(
        answer=None,
        status=UNCERTAIN,
        attempts=[make_attempt(parse_ok=False)],
    )
    assert r.method == "parse_failure"


def test_boothresult_method_validation():
    r = BoothResult(
        answer=None,
        status=UNCERTAIN,
        attempts=[make_attempt(passed_validation=False)],
    )
    assert r.method == "validation"


def test_boothresult_method_confidence():
    r = BoothResult(
        answer="x",
        status=ACCEPTED,
        attempts=[make_attempt(passed_validation=True)],
    )
    assert r.method == "confidence"


# --------------------------------------------------------------------------- #
# BoothResult.unwrap / unwrap_or
# --------------------------------------------------------------------------- #

def test_unwrap_ok():
    r = BoothResult(answer="hello", status=ACCEPTED)
    assert r.unwrap() == "hello"


def test_unwrap_repaired():
    r = BoothResult(answer="hello", status=REPAIRED)
    assert r.unwrap() == "hello"


def test_unwrap_not_ok_raises():
    r = BoothResult(answer="hello", status=UNCERTAIN)
    with pytest.raises(BoothRejected) as exc_info:
        r.unwrap()
    assert exc_info.value.result is r


def test_unwrap_ok_but_answer_none_raises():
    r = BoothResult(answer=None, status=ACCEPTED)
    with pytest.raises(BoothRejected):
        r.unwrap()


def test_unwrap_or_returns_default_on_failure():
    r = BoothResult(answer="hello", status=UNCERTAIN)
    assert r.unwrap_or("default") == "default"


def test_unwrap_or_returns_answer_on_success():
    r = BoothResult(answer="hello", status=ACCEPTED)
    assert r.unwrap_or("default") == "hello"


def test_boothrejected_message():
    r = BoothResult(answer="secret-answer", status=UNCERTAIN)
    exc = BoothRejected(r)
    assert "secret-answer" not in str(exc)
    assert r.status in str(exc)
    assert r.method in str(exc)


# --------------------------------------------------------------------------- #
# to_dict
# --------------------------------------------------------------------------- #

def test_to_dict_keys_and_values():
    a = make_attempt(raw_text="r", answer="a", confidence=0.9, parse_ok=True)
    r = BoothResult(
        answer="a",
        status=ACCEPTED,
        confidence=0.9,
        attempts=[a],
        ambiguous=False,
        interpretations=["x"],
        evidence_agreement=0.8,
        parsed={"answer": "a"},
        reason=None,
        detail="d",
    )
    d = r.to_dict()
    assert d["answer"] == "a"
    assert d["status"] == ACCEPTED
    assert d["confidence"] == 0.9
    assert d["attempts"] == [asdict(a)]
    assert d["ambiguous"] is False
    assert d["interpretations"] == ["x"]
    assert d["evidence_agreement"] == 0.8
    assert d["parsed"] == {"answer": "a"}
    assert d["n_attempts"] == 1
    assert d["ok"] is True
    assert d["all_parse_failed"] is False
    assert d["call_failed"] is False
    assert d["method"] == "confidence"
    assert d["reason"] is None
    assert d["detail"] == "d"
    assert d["checker_failed"] is False


# --------------------------------------------------------------------------- #
# check_with_evidence
# --------------------------------------------------------------------------- #

def test_evidence_empty_answer():
    r = check_with_evidence("   ", ["ev"], lambda a, e: True)
    assert r.status == UNCERTAIN
    assert r.reason == EMPTY_ANSWER
    assert r.answer is None


def test_evidence_no_evidence():
    r = check_with_evidence("ans", [], lambda a, e: True)
    assert r.status == UNCERTAIN
    assert r.reason == NO_EVIDENCE
    assert r.answer == "ans"


def test_evidence_compare_failed():
    def boom(a, e):
        raise RuntimeError("nope")

    r = check_with_evidence("ans", ["ev"], boom)
    assert r.status == UNCERTAIN
    assert r.reason == COMPARE_FAILED
    assert "RuntimeError" in r.detail
    assert r.answer == "ans"


def test_evidence_compare_async_raises():
    async def coro(a, e):
        return True

    with pytest.raises(TypeError):
        check_with_evidence("ans", ["ev"], coro)


def test_evidence_bool_true():
    r = check_with_evidence("ans", ["ev"], lambda a, e: True)
    assert r.status == ACCEPTED
    assert r.confidence == 1.0
    assert r.evidence_agreement == 1.0
    assert r.reason is None
    assert r.detail is None


def test_evidence_bool_false():
    r = check_with_evidence("ans", ["ev"], lambda a, e: False)
    assert r.status == BLOCKED
    assert r.confidence == 0.0
    assert r.evidence_agreement == 0.0
    assert r.reason == EVIDENCE_DISAGREES
    assert r.detail == "compare_fn returned False"


def test_evidence_score_above_threshold():
    r = check_with_evidence("ans", ["ev"], lambda a, e: 0.9, evidence_threshold=0.7)
    assert r.status == ACCEPTED
    assert r.confidence == 0.9
    assert r.evidence_agreement == 0.9
    assert r.reason is None


def test_evidence_score_equal_threshold():
    r = check_with_evidence("ans", ["ev"], lambda a, e: 0.5, evidence_threshold=0.5)
    assert r.status == ACCEPTED


def test_evidence_score_below_threshold():
    r = check_with_evidence("ans", ["ev"], lambda a, e: 0.3, evidence_threshold=0.7)
    assert r.status == BLOCKED
    assert r.confidence == 0.3
    assert r.reason == EVIDENCE_DISAGREES
    assert "below evidence_threshold" in r.detail


def test_evidence_invalid_score_type():
    r = check_with_evidence("ans", ["ev"], lambda a, e: "garbage")
    assert r.status == UNCERTAIN
    assert r.reason == INVALID_SCORE
    assert "non-numeric" in r.detail


def test_evidence_score_out_of_range():
    r = check_with_evidence("ans", ["ev"], lambda a, e: 1.5)
    assert r.status == UNCERTAIN
    assert r.reason == INVALID_SCORE
    assert "outside the [0.0, 1.0] range" in r.detail


def test_evidence_threshold_bool_raises():
    with pytest.raises(TypeError):
        check_with_evidence("ans", ["ev"], lambda a, e: True, evidence_threshold=True)


def test_evidence_threshold_non_number_raises():
    with pytest.raises(TypeError):
        check_with_evidence("ans", ["ev"], lambda a, e: True, evidence_threshold="x")


def test_evidence_threshold_out_of_range_raises():
    with pytest.raises(ValueError):
        check_with_evidence("ans", ["ev"], lambda a, e: True, evidence_threshold=2.0)


def test_evidence_answer_not_str_raises():
    with pytest.raises(TypeError):
        check_with_evidence(123, ["ev"], lambda a, e: True)


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #

def test_check_arg_prompt_type():
    with pytest.raises(TypeError):
        check(lambda p: GOOD_JSON, 123)


def test_check_arg_call_fn_not_callable():
    with pytest.raises(TypeError):
        check("not callable", "prompt")


def test_check_threshold_out_of_range():
    with pytest.raises(ValueError):
        check(lambda p: GOOD_JSON, "prompt", threshold=1.5)


def test_check_max_retries_not_int():
    with pytest.raises(TypeError):
        check(lambda p: GOOD_JSON, "prompt", max_retries=1.5)


def test_check_max_retries_negative():
    with pytest.raises(ValueError):
        check(lambda p: GOOD_JSON, "prompt", max_retries=-1)


def test_check_async_call_fn_raises():
    async def async_fn(p):
        return GOOD_JSON

    with pytest.raises(TypeError):
        check(async_fn, "prompt")


def test_check_on_attempt_not_callable():
    with pytest.raises(TypeError):
        check(lambda p: GOOD_JSON, "prompt", on_attempt=123)


def test_check_async_on_attempt_raises():
    async def cb(i, a):
        pass

    with pytest.raises(TypeError):
        check(lambda p: GOOD_JSON, "prompt", on_attempt=cb)


def test_check_accepted_first_try():
    calls = []

    def fn(p):
        calls.append(p)
        return GOOD_JSON

    r = check(fn, "What is the answer?")
    assert r.status == ACCEPTED
    assert r.answer == "42"
    assert r.confidence == 0.9
    assert r.n_attempts == 1
    assert r.method == "confidence"
    assert len(calls) == 1
    assert calls[0].startswith("What is the answer?")


def test_check_parse_failure_then_success_repaired():
    responses = iter([LOW_JSON, GOOD_JSON])

    r = check(lambda p: next(responses), "prompt", threshold=0.7)
    assert r.status == REPAIRED
    assert r.answer == "42"
    assert r.n_attempts == 2
    assert r.method == "confidence"
    assert r.attempts[0].parse_ok is True


def test_check_parse_failure_returns_uncertain_on_max_retries_zero():
    r = check(lambda p: "not json", "prompt", max_retries=0)
    assert r.status == UNCERTAIN
    assert r.answer is None
    assert r.n_attempts == 1
    assert r.all_parse_failed is True
    assert r.method == "parse_failure"


def test_check_call_error_branch():
    r = check(lambda p: (_ for _ in ()).throw(RuntimeError("boom")), "prompt", max_retries=0)
    assert r.status == UNCERTAIN
    assert r.attempts[0].call_failed is True
    assert r.attempts[0].parse_ok is False
    assert r.call_failed is True
    assert "boom" in r.attempts[0].error


def test_check_non_str_return_call_failed():
    r = check(lambda p: 12345, "prompt", max_retries=0)
    assert r.status == UNCERTAIN
    assert r.attempts[0].call_failed is True
    assert "expected str" in r.attempts[0].error
    assert "int" in r.attempts[0].error


def test_check_low_confidence_returns_uncertain():
    r = check(lambda p: LOW_JSON, "prompt", threshold=0.9, max_retries=0)
    assert r.status == UNCERTAIN
    assert r.answer == "42"
    assert r.confidence == 0.5


def test_check_ambiguous_returns_immediately():
    ambiguous_json = (
        '{"answer": "42", "confidence": 0.9, "ambiguous": true, '
        '"interpretations": ["i1", "i2"], "chosen_interpretation": "i1"}'
    )
    r = check(lambda p: ambiguous_json, "prompt")
    assert r.status == AMBIGUOUS
    assert r.ambiguous is True
    assert r.interpretations == ["i1", "i2"]
    assert r.n_attempts == 1
    assert r.method == "ambiguity"


def test_check_on_attempt_called_each_attempt():
    seen = []
    responses = iter([LOW_JSON, GOOD_JSON])

    def cb(i, attempt):
        seen.append((i, attempt.answer))

    check(lambda p: next(responses), "prompt", threshold=0.7, on_attempt=cb)
    assert seen == [(0, "42"), (1, "42")]


def test_check_validator_failure_leads_to_retry():
    seen = []

    def validator(ans):
        return ans == "ok", "must be ok"

    responses = iter(
        [
            '{"answer": "bad", "confidence": 0.9}',
            '{"answer": "ok", "confidence": 0.9}',
        ]
    )

    def cb(i, attempt):
        seen.append(attempt.passed_validation)

    r = check(lambda p: next(responses), "prompt", on_attempt=cb, validator=validator)
    assert r.status == REPAIRED
    assert r.answer == "ok"
    assert seen == [False, True]


def test_check_validator_raising_marks_validation_failure():
    def validator(ans):
        raise ValueError("bad")

    r = check(lambda p: GOOD_JSON, "prompt", max_retries=0, validator=validator)
    assert r.status == UNCERTAIN
    assert r.attempts[0].passed_validation is False
    assert "ValueError" in r.attempts[0].validation_error
    assert r.method == "validation"


def test_check_not_called_when_prompt_invalid():
    calls = []

    def fn(p):
        calls.append(p)
        return GOOD_JSON

    with pytest.raises(TypeError):
        check(fn, 123)
    assert calls == []


# --------------------------------------------------------------------------- #
# acheck
# --------------------------------------------------------------------------- #

def test_acheck_requires_async_call_fn():
    with pytest.raises(TypeError):
        asyncio.run(acheck(lambda p: GOOD_JSON, "prompt"))


def test_acheck_arg_validation():
    async def fn(p):
        return GOOD_JSON

    with pytest.raises(TypeError):
        asyncio.run(acheck(fn, 123))
    with pytest.raises(ValueError):
        asyncio.run(acheck(fn, "prompt", threshold=-1.0))
    with pytest.raises(TypeError):
        asyncio.run(acheck(fn, "prompt", max_retries=1.5))


def test_acheck_accepted():
    async def fn(p):
        return GOOD_JSON

    r = asyncio.run(acheck(fn, "prompt"))
    assert r.status == ACCEPTED
    assert r.answer == "42"


def test_acheck_sync_on_attempt():
    seen = []

    async def fn(p):
        return GOOD_JSON

    def cb(i, attempt):
        seen.append(i)

    asyncio.run(acheck(fn, "prompt", on_attempt=cb))
    assert seen == [0]


def test_acheck_async_on_attempt():
    seen = []

    async def fn(p):
        return GOOD_JSON

    async def cb(i, attempt):
        seen.append(i)

    asyncio.run(acheck(fn, "prompt", on_attempt=cb))
    assert seen == [0]


def test_acheck_call_error():
    async def fn(p):
        raise RuntimeError("boom")

    r = asyncio.run(acheck(fn, "prompt", max_retries=0))
    assert r.status == UNCERTAIN
    assert r.call_failed is True


def test_acheck_repair_flow():
    responses = iter([LOW_JSON, GOOD_JSON])

    async def fn(p):
        return next(responses)

    r = asyncio.run(acheck(fn, "prompt", threshold=0.7))
    assert r.status == REPAIRED
    assert r.n_attempts == 2


def test_acheck_validator():
    async def fn(p):
        return GOOD_JSON

    def validator(ans):
        return False

    r = asyncio.run(acheck(fn, "prompt", max_retries=0, validator=validator))
    assert r.attempts[0].passed_validation is False
    assert r.status == UNCERTAIN


def test_acheck_on_attempt_not_callable():
    async def fn(p):
        return GOOD_JSON

    with pytest.raises(TypeError):
        asyncio.run(acheck(fn, "prompt", on_attempt=123))
