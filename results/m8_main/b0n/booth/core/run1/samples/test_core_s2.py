import asyncio

import pytest

from booth.core import (
    ACCEPTED,
    AMBIGUOUS,
    BLOCKED,
    REPAIRED,
    UNCERTAIN,
    EMPTY_ANSWER,
    NO_EVIDENCE,
    EVIDENCE_DISAGREES,
    COMPARE_FAILED,
    INVALID_SCORE,
    Attempt,
    BoothRejected,
    BoothResult,
    check,
    acheck,
    check_with_evidence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _good_response(answer="Paris", confidence=0.9):
    return (
        '{"ambiguous": false, "interpretations": [], '
        '"chosen_interpretation": null, '
        f'"answer": "{answer}", "confidence": {confidence}}}'
    )


def _mk_attempt(parse_ok=True, passed_validation=True, call_failed=False, **kw):
    return Attempt(
        raw_text=kw.pop("raw_text", ""),
        answer=kw.pop("answer", "x" if parse_ok else None),
        confidence=kw.pop("confidence", 0.9 if parse_ok else None),
        parse_ok=parse_ok,
        passed_validation=passed_validation,
        call_failed=call_failed,
        **kw,
    )


# ---------------------------------------------------------------------------
# Attempt dataclass
# ---------------------------------------------------------------------------

def test_attempt_defaults():
    a = Attempt(raw_text="raw", answer="a", confidence=0.5, parse_ok=True)
    assert a.raw_text == "raw"
    assert a.answer == "a"
    assert a.confidence == 0.5
    assert a.parse_ok is True
    assert a.error is None
    assert a.ambiguous is False
    assert a.interpretations == []
    assert a.chosen_interpretation is None
    assert a.passed_validation is True
    assert a.validation_error is None
    assert a.parsed is None
    assert a.call_failed is False


def test_attempt_interpretations_not_shared():
    a = Attempt(raw_text="", answer=None, confidence=None, parse_ok=False)
    b = Attempt(raw_text="", answer=None, confidence=None, parse_ok=False)
    a.interpretations.append("x")
    assert b.interpretations == []


# ---------------------------------------------------------------------------
# BoothResult properties
# ---------------------------------------------------------------------------

def test_n_attempts_and_ok():
    r = BoothResult(answer="x", status=ACCEPTED)
    assert r.n_attempts == 0
    assert r.ok is True
    assert BoothResult(answer="x", status=REPAIRED).ok is True
    assert BoothResult(answer="x", status=UNCERTAIN).ok is False
    assert BoothResult(answer="x", status=BLOCKED).ok is False
    assert BoothResult(answer="x", status=AMBIGUOUS).ok is False


def test_all_parse_failed():
    r = BoothResult(answer=None, status=UNCERTAIN)
    assert r.all_parse_failed is False  # no attempts

    r.attempts.append(_mk_attempt(parse_ok=False))
    assert r.all_parse_failed is True

    r.attempts.append(_mk_attempt(parse_ok=True))
    assert r.all_parse_failed is False


def test_call_failed_property():
    r = BoothResult(answer=None, status=UNCERTAIN)
    assert r.call_failed is False

    r.attempts.append(_mk_attempt(parse_ok=False, call_failed=True))
    assert r.call_failed is True

    r.attempts.append(_mk_attempt(parse_ok=False, call_failed=False))
    assert r.call_failed is False


def test_checker_failed_property():
    assert BoothResult(answer="x", status=UNCERTAIN, reason=COMPARE_FAILED).checker_failed is True
    assert BoothResult(answer="x", status=UNCERTAIN, reason=INVALID_SCORE).checker_failed is True
    assert BoothResult(answer="x", status=UNCERTAIN, reason=NO_EVIDENCE).checker_failed is False
    assert BoothResult(answer="x", status=UNCERTAIN).checker_failed is False


def test_method_ambiguity():
    r = BoothResult(answer="x", status=AMBIGUOUS)
    assert r.method == "ambiguity"


def test_method_evidence_when_no_attempts():
    r = BoothResult(answer="x", status=UNCERTAIN)
    assert r.method == "evidence"


def test_method_parse_failure():
    r = BoothResult(answer=None, status=UNCERTAIN)
    r.attempts.append(_mk_attempt(parse_ok=False))
    assert r.method == "parse_failure"


def test_method_validation():
    r = BoothResult(answer="x", status=UNCERTAIN)
    r.attempts.append(_mk_attempt(parse_ok=True, passed_validation=False))
    assert r.method == "validation"


def test_method_confidence():
    r = BoothResult(answer="x", status=UNCERTAIN)
    r.attempts.append(_mk_attempt(parse_ok=True, passed_validation=True))
    assert r.method == "confidence"


def test_unwrap_ok():
    r = BoothResult(answer="hi", status=ACCEPTED)
    assert r.unwrap() == "hi"


def test_unwrap_not_ok_raises():
    r = BoothResult(answer="hi", status=UNCERTAIN)
    with pytest.raises(BoothRejected) as ei:
        r.unwrap()
    assert ei.value.result is r
    assert "UNCERTAIN" in str(ei.value)


def test_unwrap_ok_but_answer_none_raises():
    r = BoothResult(answer=None, status=ACCEPTED)
    with pytest.raises(BoothRejected):
        r.unwrap()


def test_unwrap_or():
    assert BoothResult(answer="hi", status=ACCEPTED).unwrap_or("d") == "hi"
    assert BoothResult(answer="hi", status=UNCERTAIN).unwrap_or("d") == "d"


def test_booth_rejected_message_omits_answer():
    r = BoothResult(answer="secret-answer", status=UNCERTAIN, method_override=None) if False else \
        BoothResult(answer="secret-answer", status=UNCERTAIN)
    r.attempts.append(_mk_attempt(parse_ok=True))
    exc = BoothRejected(r)
    assert "secret-answer" not in str(exc)
    assert exc.result is r


def test_to_dict():
    a = _mk_attempt(parse_ok=True)
    r = BoothResult(answer="x", status=ACCEPTED, confidence=0.9, attempts=[a])
    d = r.to_dict()
    assert d["answer"] == "x"
    assert d["status"] == ACCEPTED
    assert d["confidence"] == 0.9
    assert d["n_attempts"] == 1
    assert d["ok"] is True
    assert d["method"] == "confidence"
    assert isinstance(d["attempts"], list)
    assert d["attempts"][0]["answer"] == "x"
    assert "all_parse_failed" in d
    assert "call_failed" in d
    assert "checker_failed" in d


# ---------------------------------------------------------------------------
# check() basic flows
# ---------------------------------------------------------------------------

def test_check_accept_first_attempt():
    calls = []

    def call_fn(prompt):
        calls.append(prompt)
        return _good_response()

    r = check(call_fn, "Q?")
    assert r.status == ACCEPTED
    assert r.answer == "Paris"
    assert r.confidence == 0.9
    assert r.n_attempts == 1
    assert r.ok is True
    assert len(calls) == 1
    assert "Q?" in calls[0]


def test_check_repaired_on_retry():
    responses = [
        # low confidence -> retry
        '{"answer": "Londn", "confidence": 0.2}',
        '{"answer": "London", "confidence": 0.95}',
    ]

    def call_fn(prompt):
        return responses.pop(0)

    r = check(call_fn, "Q?", max_retries=1)
    assert r.status == REPAIRED
    assert r.answer == "London"
    assert r.n_attempts == 2


def test_check_uncertain_when_all_low_confidence():
    def call_fn(prompt):
        return '{"answer": "guess", "confidence": 0.1}'

    r = check(call_fn, "Q?", max_retries=1)
    assert r.status == UNCERTAIN
    assert r.answer == "guess"
    assert r.confidence == 0.1
    assert r.n_attempts == 2


def test_check_parse_failure_then_uncertain():
    def call_fn(prompt):
        return "not json at all"

    r = check(call_fn, "Q?", max_retries=1)
    assert r.status == UNCERTAIN
    assert r.all_parse_failed is True
    assert r.answer is None
    assert r.n_attempts == 2
    assert r.method == "parse_failure"


def test_check_parse_failure_then_success():
    responses = [
        "garbage",
        _good_response(),
    ]

    def call_fn(prompt):
        return responses.pop(0)

    r = check(call_fn, "Q?", max_retries=1)
    assert r.status == REPAIRED
    assert r.answer == "Paris"


def test_check_call_fn_raises_captured():
    def call_fn(prompt):
        raise RuntimeError("boom")

    r = check(call_fn, "Q?", max_retries=0)
    assert r.status == UNCERTAIN
    assert r.call_failed is True
    assert r.n_attempts == 1
    assert r.attempts[0].call_failed is True
    assert "boom" in r.attempts[0].error


def test_check_call_fn_returns_non_str():
    def call_fn(prompt):
        return 123

    r = check(call_fn, "Q?", max_retries=0)
    assert r.status == UNCERTAIN
    assert r.call_failed is True
    assert r.attempts[0].call_failed is True


def test_check_call_failed_then_recovers():
    calls = {"n": 0}

    def call_fn(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("x")
        return _good_response()

    r = check(call_fn, "Q?", max_retries=1)
    assert r.status == REPAIRED
    assert r.n_attempts == 2
    assert r.attempts[0].call_failed is True


def test_check_ambiguous_returns_immediately():
    resp = (
        '{"ambiguous": true, "interpretations": ["A", "B"], '
        '"chosen_interpretation": "A", "answer": "x", "confidence": 0.9}'
    )

    def call_fn(prompt):
        return resp

    r = check(call_fn, "Q?", max_retries=2)
    assert r.status == AMBIGUOUS
    assert r.ambiguous is True
    assert r.interpretations == ["A", "B"]
    assert r.n_attempts == 1
    assert r.method == "ambiguity"


def test_check_ambiguous_skips_validator():
    called = []

    def validator(answer):
        called.append(answer)
        return False

    resp = (
        '{"ambiguous": true, "interpretations": ["A", "B"], '
        '"chosen_interpretation": "A", "answer": "x", "confidence": 0.9}'
    )

    def call_fn(prompt):
        return resp

    r = check(call_fn, "Q?", validator=validator)
    assert r.status == AMBIGUOUS
    assert called == []


# ---------------------------------------------------------------------------
# check() validator behavior
# ---------------------------------------------------------------------------

def test_check_validator_pass():
    def validator(answer):
        return True

    r = check(lambda p: _good_response(), "Q?", validator=validator)
    assert r.status == ACCEPTED


def test_check_validator_fail_then_pass():
    responses = [_good_response("bad"), _good_response("good")]
    answers = []

    def validator(answer):
        answers.append(answer)
        return answer == "good"

    def call_fn(prompt):
        return responses.pop(0)

    r = check(call_fn, "Q?", validator=validator, max_retries=1)
    assert r.status == REPAIRED
    assert r.answer == "good"
    assert answers == ["bad", "good"]


def test_check_validator_all_fail_uncertain():
    def validator(answer):
        return False

    r = check(lambda p: _good_response(), "Q?", validator=validator, max_retries=1)
    assert r.status == UNCERTAIN
    assert r.n_attempts == 2
    assert r.attempts[-1].passed_validation is False
    assert r.method == "validation"


def test_check_validator_raises_is_treated_as_failure():
    def validator(answer):
        raise ValueError("nope")

    r = check(lambda p: _good_response(), "Q?", validator=validator, max_retries=0)
    assert r.status == UNCERTAIN
    assert r.attempts[0].passed_validation is False
    assert "ValueError" in r.attempts[0].validation_error


def test_check_validator_tuple_message():
    def validator(answer):
        return (False, "custom message")

    r = check(lambda p: _good_response(), "Q?", validator=validator, max_retries=0)
    assert r.attempts[0].validation_error == "custom message"


# ---------------------------------------------------------------------------
# check() argument validation
# ---------------------------------------------------------------------------

def test_check_prompt_not_str():
    with pytest.raises(TypeError):
        check(lambda p: "", 123)


def test_check_call_fn_not_callable():
    with pytest.raises(TypeError):
        check("notcallable", "Q?")


def test_check_threshold_out_of_range():
    with pytest.raises(ValueError):
        check(lambda p: "", "Q?", threshold=1.5)
    with pytest.raises(ValueError):
        check(lambda p: "", "Q?", threshold=-0.1)


def test_check_max_retries_not_int():
    with pytest.raises(TypeError):
        check(lambda p: "", "Q?", max_retries=1.0)


def test_check_max_retries_negative():
    with pytest.raises(ValueError):
        check(lambda p: "", "Q?", max_retries=-1)


def test_check_async_call_fn_rejected():
    async def call_fn(prompt):
        return "x"

    with pytest.raises(TypeError):
        check(call_fn, "Q?")


def test_check_async_on_attempt_rejected():
    async def on_attempt(i, attempt):
        pass

    with pytest.raises(TypeError):
        check(lambda p: _good_response(), "Q?", on_attempt=on_attempt)


def test_check_non_callable_on_attempt_rejected():
    with pytest.raises(TypeError):
        check(lambda p: _good_response(), "Q?", on_attempt=5)


def test_check_on_attempt_invoked():
    seen = []

    def on_attempt(i, attempt):
        seen.append((i, attempt.parse_ok))

    check(lambda p: _good_response(), "Q?", on_attempt=on_attempt)
    assert seen == [(0, True)]


# ---------------------------------------------------------------------------
# acheck()
# ---------------------------------------------------------------------------

def test_acheck_accept_first_attempt():
    async def call_fn(prompt):
        return _good_response()

    r = asyncio.run(acheck(call_fn, "Q?"))
    assert r.status == ACCEPTED
    assert r.answer == "Paris"


def test_acheck_repaired():
    responses = ['{"answer": "x", "confidence": 0.1}', _good_response()]

    async def call_fn(prompt):
        return responses.pop(0)

    r = asyncio.run(acheck(call_fn, "Q?", max_retries=1))
    assert r.status == REPAIRED


def test_acheck_sync_call_fn_rejected():
    def call_fn(prompt):
        return _good_response()

    with pytest.raises(TypeError):
        asyncio.run(acheck(call_fn, "Q?"))


def test_acheck_call_fn_raises():
    async def call_fn(prompt):
        raise RuntimeError("boom")

    r = asyncio.run(acheck(call_fn, "Q?", max_retries=0))
    assert r.status == UNCERTAIN
    assert r.call_failed is True


def test_acheck_on_attempt_async_invoked():
    seen = []

    async def on_attempt(i, attempt):
        seen.append(i)

    async def call_fn(prompt):
        return _good_response()

    asyncio.run(acheck(call_fn, "Q?", on_attempt=on_attempt))
    assert seen == [0]


def test_acheck_on_attempt_sync_invoked():
    seen = []

    def on_attempt(i, attempt):
        seen.append(i)

    async def call_fn(prompt):
        return _good_response()

    asyncio.run(acheck(call_fn, "Q?", on_attempt=on_attempt))
    assert seen == [0]


def test_acheck_non_callable_on_attempt_rejected():
    async def call_fn(prompt):
        return _good_response()

    with pytest.raises(TypeError):
        asyncio.run(acheck(call_fn, "Q?", on_attempt="bad"))


def test_acheck_validator_applied():
    answers = []

    def validator(answer):
        answers.append(answer)
        return True

    async def call_fn(prompt):
        return _good_response()

    r = asyncio.run(acheck(call_fn, "Q?", validator=validator))
    assert r.status == ACCEPTED
    assert answers == ["Paris"]


# ---------------------------------------------------------------------------
# check_with_evidence()
# ---------------------------------------------------------------------------

def test_evidence_accept_bool_true():
    r = check_with_evidence("a", ["e"], lambda ans, ev: True)
    assert r.status == ACCEPTED
    assert r.confidence == 1.0
    assert r.evidence_agreement == 1.0
    assert r.reason is None


def test_evidence_blocked_bool_false():
    r = check_with_evidence("a", ["e"], lambda ans, ev: False)
    assert r.status == BLOCKED
    assert r.confidence == 0.0
    assert r.evidence_agreement == 0.0
    assert r.reason == EVIDENCE_DISAGREES
    assert r.detail == "compare_fn returned False"


def test_evidence_float_above_threshold():
    r = check_with_evidence("a", ["e"], lambda ans, ev: 0.8, evidence_threshold=0.5)
    assert r.status == ACCEPTED
    assert r.confidence == 0.8
    assert r.evidence_agreement == 0.8


def test_evidence_float_below_threshold():
    r = check_with_evidence("a", ["e"], lambda ans, ev: 0.3, evidence_threshold=0.5)
    assert r.status == BLOCKED
    assert r.reason == EVIDENCE_DISAGREES
    assert "0.3" in r.detail
    assert "0.5" in r.detail


def test_evidence_empty_answer():
    r = check_with_evidence("   ", ["e"], lambda ans, ev: True)
    assert r.status == UNCERTAIN
    assert r.answer is None
    assert r.reason == EMPTY_ANSWER


def test_evidence_none_answer_allowed():
    r = check_with_evidence(None, ["e"], lambda ans, ev: True)
    assert r.status == UNCERTAIN
    assert r.reason == EMPTY_ANSWER


def test_evidence_non_str_answer():
    with pytest.raises(TypeError):
        check_with_evidence(123, ["e"], lambda ans, ev: True)


def test_evidence_empty_evidence():
    r = check_with_evidence("a", [], lambda ans, ev: True)
    assert r.status == UNCERTAIN
    assert r.reason == NO_EVIDENCE
    assert r.answer == "a"


def test_evidence_compare_fn_raises():
    def compare(ans, ev):
        raise ValueError("bad")

    r = check_with_evidence("a", ["e"], compare)
    assert r.status == UNCERTAIN
    assert r.reason == COMPARE_FAILED
    assert "ValueError" in r.detail


def test_evidence_async_compare_fn_raises():
    async def compare(ans, ev):
        return True

    with pytest.raises(TypeError):
        check_with_evidence("a", ["e"], compare)


def test_evidence_invalid_non_numeric_score():
    r = check_with_evidence("a", ["e"], lambda ans, ev: "not-a-number")
    assert r.status == UNCERTAIN
    assert r.reason == INVALID_SCORE


def test_evidence_score_out_of_range():
    r = check_with_evidence("a", ["e"], lambda ans, ev: 1.5)
    assert r.status == UNCERTAIN
    assert r.reason == INVALID_SCORE


def test_evidence_threshold_bool_rejected():
    with pytest.raises(TypeError):
        check_with_evidence("a", ["e"], lambda ans, ev: True, evidence_threshold=True)


def test_evidence_threshold_string_rejected():
    with pytest.raises(TypeError):
        check_with_evidence("a", ["e"], lambda ans, ev: True, evidence_threshold="0.5")


def test_evidence_threshold_out_of_range():
    with pytest.raises(ValueError):
        check_with_evidence("a", ["e"], lambda ans, ev: True, evidence_threshold=2.0)


def test_evidence_compare_fn_receives_args():
    received = {}

    def compare(ans, ev):
        received["ans"] = ans
        received["ev"] = ev
        return True

    check_with_evidence("hello", ["e1", "e2"], compare)
    assert received["ans"] == "hello"
    assert received["ev"] == ["e1", "e2"]


def test_evidence_bool_ignores_threshold():
    r = check_with_evidence(
        "a", ["e"], lambda ans, ev: True, evidence_threshold=0.99
    )
    assert r.status == ACCEPTED
