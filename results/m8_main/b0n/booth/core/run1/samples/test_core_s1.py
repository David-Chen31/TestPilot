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

def _json_response(answer="42", confidence=0.9, **extra):
    obj = {"answer": answer, "confidence": confidence}
    obj.update(extra)
    return json.dumps(obj)


def _make_call_fn(responses):
    """Return a call_fn that yields responses in order; list items may be
    exceptions (raised), str (returned) or other objects (returned)."""
    calls = []

    def call_fn(prompt):
        calls.append(prompt)
        item = responses[len(calls) - 1]
        if isinstance(item, BaseException):
            raise item
        return item

    call_fn.calls = calls
    return call_fn


# ---------------------------------------------------------------------------
# Attempt / BoothResult / BoothRejected
# ---------------------------------------------------------------------------

def test_attempt_defaults():
    a = Attempt(raw_text="x", answer="a", confidence=0.5, parse_ok=True)
    assert a.answer == "a"
    assert a.error is None
    assert a.ambiguous is False
    assert a.interpretations == []
    assert a.chosen_interpretation is None
    assert a.passed_validation is True
    assert a.validation_error is None
    assert a.parsed is None
    assert a.call_failed is False


def test_booth_result_ok_and_n_attempts():
    r = BoothResult(answer="a", status=ACCEPTED)
    assert r.ok is True
    assert r.n_attempts == 0
    r2 = BoothResult(answer="a", status=REPAIRED)
    assert r2.ok is True
    r3 = BoothResult(answer=None, status=UNCERTAIN)
    assert r3.ok is False
    r4 = BoothResult(answer=None, status=BLOCKED)
    assert r4.ok is False


def test_booth_result_all_parse_failed_and_call_failed():
    empty = BoothResult(answer=None, status=UNCERTAIN)
    assert empty.all_parse_failed is False
    assert empty.call_failed is False

    a1 = Attempt(raw_text="", answer=None, confidence=None, parse_ok=False)
    r = BoothResult(answer=None, status=UNCERTAIN, attempts=[a1])
    assert r.all_parse_failed is True
    assert r.call_failed is False

    a2 = Attempt(raw_text="", answer=None, confidence=None, parse_ok=False, call_failed=True)
    r2 = BoothResult(answer=None, status=UNCERTAIN, attempts=[a2])
    assert r2.call_failed is True

    a_ok = Attempt(raw_text="", answer="x", confidence=0.9, parse_ok=True)
    r3 = BoothResult(answer="x", status=ACCEPTED, attempts=[a_ok])
    assert r3.all_parse_failed is False
    assert r3.call_failed is False


def test_booth_result_checker_failed():
    assert BoothResult(answer=None, status=UNCERTAIN, reason=COMPARE_FAILED).checker_failed is True
    assert BoothResult(answer=None, status=UNCERTAIN, reason=INVALID_SCORE).checker_failed is True
    assert BoothResult(answer=None, status=UNCERTAIN, reason=EMPTY_ANSWER).checker_failed is False
    assert BoothResult(answer=None, status=UNCERTAIN).checker_failed is False


def test_booth_result_method():
    assert BoothResult(answer=None, status=AMBIGUOUS).method == "ambiguity"
    assert BoothResult(answer=None, status=UNCERTAIN).method == "evidence"

    failed_parse = Attempt(raw_text="", answer=None, confidence=None, parse_ok=False)
    r = BoothResult(answer=None, status=UNCERTAIN, attempts=[failed_parse])
    assert r.method == "parse_failure"

    val_fail = Attempt(raw_text="", answer="a", confidence=0.9, parse_ok=True,
                       passed_validation=False)
    r2 = BoothResult(answer="a", status=UNCERTAIN, attempts=[val_fail])
    assert r2.method == "validation"

    ok = Attempt(raw_text="", answer="a", confidence=0.9, parse_ok=True)
    r3 = BoothResult(answer="a", status=ACCEPTED, attempts=[ok])
    assert r3.method == "confidence"


def test_booth_result_unwrap_success():
    r = BoothResult(answer="hi", status=ACCEPTED)
    assert r.unwrap() == "hi"


def test_booth_result_unwrap_none_answer_raises():
    r = BoothResult(answer=None, status=ACCEPTED)
    with pytest.raises(BoothRejected):
        r.unwrap()


def test_booth_result_unwrap_rejected_raises_with_result():
    r = BoothResult(answer="bad", status=BLOCKED)
    with pytest.raises(BoothRejected) as exc_info:
        r.unwrap()
    assert exc_info.value.result is r
    msg = str(exc_info.value)
    assert "bad" not in msg
    assert BLOCKED in msg


def test_booth_result_unwrap_or():
    ok = BoothResult(answer="hi", status=ACCEPTED)
    assert ok.unwrap_or("default") == "hi"
    bad = BoothResult(answer="x", status=BLOCKED)
    assert bad.unwrap_or("default") == "default"


def test_booth_result_to_dict():
    a = Attempt(raw_text="raw", answer="a", confidence=0.8, parse_ok=True)
    r = BoothResult(
        answer="a",
        status=ACCEPTED,
        confidence=0.8,
        attempts=[a],
        ambiguous=False,
        interpretations=["x"],
        evidence_agreement=0.5,
        parsed={"answer": "a"},
        reason=None,
        detail="d",
    )
    d = r.to_dict()
    assert d["answer"] == "a"
    assert d["status"] == ACCEPTED
    assert d["confidence"] == 0.8
    assert d["n_attempts"] == 1
    assert d["ok"] is True
    assert d["all_parse_failed"] is False
    assert d["call_failed"] is False
    assert d["method"] == "confidence"
    assert d["interpretations"] == ["x"]
    assert d["evidence_agreement"] == 0.5
    assert d["parsed"] == {"answer": "a"}
    assert d["reason"] is None
    assert d["detail"] == "d"
    assert d["checker_failed"] is False
    assert len(d["attempts"]) == 1
    assert d["attempts"][0]["answer"] == "a"
    assert d["attempts"][0]["raw_text"] == "raw"


# ---------------------------------------------------------------------------
# check_with_evidence
# ---------------------------------------------------------------------------

def test_evidence_accepts_true():
    r = check_with_evidence("ans", ["e1"], lambda a, ev: True)
    assert r.status == ACCEPTED
    assert r.confidence == 1.0
    assert r.evidence_agreement == 1.0
    assert r.reason is None
    assert r.detail is None


def test_evidence_blocks_false():
    r = check_with_evidence("ans", ["e1"], lambda a, ev: False)
    assert r.status == BLOCKED
    assert r.confidence == 0.0
    assert r.reason == EVIDENCE_DISAGREES
    assert r.detail == "compare_fn returned False"


def test_evidence_float_above_threshold():
    r = check_with_evidence("ans", ["e"], lambda a, ev: 0.8, evidence_threshold=0.7)
    assert r.status == ACCEPTED
    assert r.confidence == 0.8
    assert r.evidence_agreement == 0.8
    assert r.reason is None


def test_evidence_float_at_threshold_passes():
    r = check_with_evidence("ans", ["e"], lambda a, ev: 0.7, evidence_threshold=0.7)
    assert r.status == ACCEPTED


def test_evidence_float_below_threshold_blocks():
    r = check_with_evidence("ans", ["e"], lambda a, ev: 0.3, evidence_threshold=0.7)
    assert r.status == BLOCKED
    assert r.reason == EVIDENCE_DISAGREES
    assert "0.3" in r.detail
    assert "0.7" in r.detail


def test_evidence_bool_ignores_threshold():
    r = check_with_evidence("ans", ["e"], lambda a, ev: True, evidence_threshold=0.0)
    assert r.status == ACCEPTED


def test_evidence_empty_answer():
    r = check_with_evidence("   ", ["e"], lambda a, ev: True)
    assert r.status == UNCERTAIN
    assert r.reason == EMPTY_ANSWER
    assert r.answer is None


def test_evidence_empty_evidence():
    r = check_with_evidence("ans", [], lambda a, ev: True)
    assert r.status == UNCERTAIN
    assert r.reason == NO_EVIDENCE
    assert r.answer == "ans"


def test_evidence_compare_fn_raises():
    def fn(a, ev):
        raise RuntimeError("boom")

    r = check_with_evidence("ans", ["e"], fn)
    assert r.status == UNCERTAIN
    assert r.reason == COMPARE_FAILED
    assert "RuntimeError" in r.detail
    assert "boom" in r.detail


def test_evidence_compare_fn_non_numeric():
    r = check_with_evidence("ans", ["e"], lambda a, ev: "nope")
    assert r.status == UNCERTAIN
    assert r.reason == INVALID_SCORE
    assert "nope" in r.detail


def test_evidence_compare_fn_out_of_range():
    r = check_with_evidence("ans", ["e"], lambda a, ev: 1.5)
    assert r.status == UNCERTAIN
    assert r.reason == INVALID_SCORE
    assert "1.5" in r.detail


def test_evidence_answer_type_error():
    with pytest.raises(TypeError):
        check_with_evidence(123, ["e"], lambda a, ev: True)


def test_evidence_threshold_bool_rejected():
    with pytest.raises(TypeError):
        check_with_evidence("a", ["e"], lambda a, ev: True, evidence_threshold=True)


def test_evidence_threshold_type_error():
    with pytest.raises(TypeError):
        check_with_evidence("a", ["e"], lambda a, ev: True, evidence_threshold="0.5")


def test_evidence_threshold_out_of_range():
    with pytest.raises(ValueError):
        check_with_evidence("a", ["e"], lambda a, ev: True, evidence_threshold=2.0)


def test_evidence_async_compare_fn_raises():
    async def fn(a, ev):
        return True

    with pytest.raises(TypeError):
        check_with_evidence("a", ["e"], fn)


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def test_check_accepts_first_attempt():
    call_fn = _make_call_fn([_json_response("Paris", 0.95)])
    r = check(call_fn, "capital of France?")
    assert r.status == ACCEPTED
    assert r.answer == "Paris"
    assert r.confidence == 0.95
    assert r.n_attempts == 1
    assert len(call_fn.calls) == 1
    # prompt should have the confidence suffix appended
    assert "confidence" in call_fn.calls[0]


def test_check_repairs_on_retry():
    responses = [_json_response("bad", 0.2), _json_response("good", 0.9)]
    call_fn = _make_call_fn(responses)
    r = check(call_fn, "q")
    assert r.status == REPAIRED
    assert r.answer == "good"
    assert r.n_attempts == 2


def test_check_uncertain_after_retries():
    responses = [_json_response("a", 0.1), _json_response("b", 0.2)]
    call_fn = _make_call_fn(responses)
    r = check(call_fn, "q")
    assert r.status == UNCERTAIN
    assert r.answer == "b"
    assert r.confidence == 0.2
    assert r.n_attempts == 2


def test_check_zero_retries():
    call_fn = _make_call_fn([_json_response("a", 0.1)])
    r = check(call_fn, "q", max_retries=0)
    assert r.status == UNCERTAIN
    assert r.n_attempts == 1


def test_check_ambiguous_short_circuits():
    resp = json.dumps({
        "answer": "x",
        "confidence": 0.3,
        "ambiguous": True,
        "interpretations": ["r1", "r2"],
        "chosen_interpretation": "r1",
    })
    call_fn = _make_call_fn([resp])
    r = check(call_fn, "q")
    assert r.status == AMBIGUOUS
    assert r.ambiguous is True
    assert r.interpretations == ["r1", "r2"]
    assert r.answer == "x"


def test_check_parse_failure_then_success():
    responses = ["not json", _json_response("ok", 0.9)]
    call_fn = _make_call_fn(responses)
    r = check(call_fn, "q")
    assert r.status == REPAIRED
    assert r.n_attempts == 2


def test_check_all_parse_failed():
    call_fn = _make_call_fn(["nope", "still nope"])
    r = check(call_fn, "q")
    assert r.status == UNCERTAIN
    assert r.all_parse_failed is True
    assert r.method == "parse_failure"
    assert r.answer is None


def test_check_call_fn_raises():
    call_fn = _make_call_fn([RuntimeError("boom"), RuntimeError("boom2")])
    r = check(call_fn, "q")
    assert r.status == UNCERTAIN
    assert r.call_failed is True
    assert r.method == "parse_failure"
    assert "boom" in r.attempts[0].raw_text


def test_check_call_fn_returns_non_str():
    call_fn = _make_call_fn([123, _json_response("ok", 0.9)])
    r = check(call_fn, "q")
    assert r.attempts[0].call_failed is True
    assert "int" in r.attempts[0].error
    assert r.status == REPAIRED
    assert r.answer == "ok"


def test_check_on_attempt_callback():
    seen = []

    def cb(i, attempt):
        seen.append((i, attempt.parse_ok))

    call_fn = _make_call_fn([_json_response("a", 0.9)])
    check(call_fn, "q", on_attempt=cb)
    assert seen == [(0, True)]


def test_check_on_attempt_not_callable():
    call_fn = _make_call_fn([_json_response("a", 0.9)])
    with pytest.raises(TypeError):
        check(call_fn, "q", on_attempt=42)


def test_check_async_call_fn_rejected():
    async def fn(p):
        return "x"

    with pytest.raises(TypeError):
        check(fn, "q")


def test_check_async_on_attempt_rejected():
    async def cb(i, a):
        pass

    call_fn = _make_call_fn([_json_response("a", 0.9)])
    with pytest.raises(TypeError):
        check(call_fn, "q", on_attempt=cb)


def test_check_prompt_type_error():
    with pytest.raises(TypeError):
        check(lambda p: "x", 123)


def test_check_call_fn_not_callable():
    with pytest.raises(TypeError):
        check("not callable", "q")


def test_check_threshold_out_of_range():
    with pytest.raises(ValueError):
        check(lambda p: "x", "q", threshold=1.5)


def test_check_max_retries_type_error():
    with pytest.raises(TypeError):
        check(lambda p: "x", "q", max_retries=1.5)


def test_check_max_retries_negative():
    with pytest.raises(ValueError):
        check(lambda p: "x", "q", max_retries=-1)


def test_check_validator_pass():
    call_fn = _make_call_fn([_json_response("42", 0.9)])
    r = check(call_fn, "q", validator=lambda a: a == "42")
    assert r.status == ACCEPTED
    assert r.attempts[0].passed_validation is True


def test_check_validator_fail_triggers_retry():
    responses = [_json_response("bad", 0.9), _json_response("good", 0.9)]
    call_fn = _make_call_fn(responses)
    r = check(call_fn, "q", validator=lambda a: a == "good")
    assert r.status == REPAIRED
    assert r.attempts[0].passed_validation is False
    assert r.attempts[0].validation_error == "Validator returned False"
    assert "validation" in call_fn.calls[1]


def test_check_validator_tuple_form():
    call_fn = _make_call_fn([_json_response("bad", 0.9), _json_response("ok", 0.9)])
    r = check(call_fn, "q", validator=lambda a: (a == "ok", "must be ok"))
    assert r.attempts[0].validation_error == "must be ok"
    assert r.status == REPAIRED


def test_check_validator_raises():
    def val(a):
        raise ValueError("nope")

    call_fn = _make_call_fn([_json_response("a", 0.9), _json_response("a", 0.9)])
    r = check(call_fn, "q", validator=val)
    assert r.attempts[0].passed_validation is False
    assert r.attempts[0].validation_error == "Validator raised ValueError: nope"


def test_check_validator_invalid_return_type():
    call_fn = _make_call_fn([_json_response("a", 0.9)])
    r = check(call_fn, "q", validator=lambda a: 42)
    assert r.attempts[0].passed_validation is False
    assert "invalid type" in r.attempts[0].validation_error


def test_check_validator_async_rejected():
    async def val(a):
        return True

    call_fn = _make_call_fn([_json_response("a", 0.9)])
    r = check(call_fn, "q", validator=val)
    assert r.attempts[0].passed_validation is False
    assert "coroutine" in r.attempts[0].validation_error


def test_check_ambiguous_skips_validator():
    resp = json.dumps({"answer": "x", "confidence": 0.9, "ambiguous": True})
    call_fn = _make_call_fn([resp])
    called = []

    def val(a):
        called.append(a)
        return False

    r = check(call_fn, "q", validator=val)
    assert r.status == AMBIGUOUS
    assert called == []


# ---------------------------------------------------------------------------
# acheck
# ---------------------------------------------------------------------------

def test_acheck_accepts_first_attempt():
    calls = []

    async def call_fn(prompt):
        calls.append(prompt)
        return _json_response("Paris", 0.95)

    r = asyncio.run(acheck(call_fn, "q"))
    assert r.status == ACCEPTED
    assert r.answer == "Paris"
    assert len(calls) == 1


def test_acheck_repairs_on_retry():
    responses = [_json_response("a", 0.1), _json_response("b", 0.9)]
    calls = []

    async def call_fn(prompt):
        calls.append(prompt)
        return responses[len(calls) - 1]

    r = asyncio.run(acheck(call_fn, "q"))
    assert r.status == REPAIRED
    assert r.answer == "b"


def test_acheck_sync_call_fn_rejected():
    def fn(p):
        return "x"

    with pytest.raises(TypeError):
        asyncio.run(acheck(fn, "q"))


def test_acheck_call_fn_raises():
    async def call_fn(p):
        raise RuntimeError("boom")

    r = asyncio.run(acheck(call_fn, "q", max_retries=0))
    assert r.status == UNCERTAIN
    assert r.call_failed is True


def test_acheck_sync_on_attempt():
    seen = []

    def cb(i, attempt):
        seen.append(i)

    async def call_fn(p):
        return _json_response("a", 0.9)

    asyncio.run(acheck(call_fn, "q", on_attempt=cb))
    assert seen == [0]


def test_acheck_async_on_attempt():
    seen = []

    async def cb(i, attempt):
        seen.append(i)

    async def call_fn(p):
        return _json_response("a", 0.9)

    asyncio.run(acheck(call_fn, "q", on_attempt=cb))
    assert seen == [0]


def test_acheck_on_attempt_not_callable():
    async def call_fn(p):
        return _json_response("a", 0.9)

    with pytest.raises(TypeError):
        asyncio.run(acheck(call_fn, "q", on_attempt=5))


def test_acheck_threshold_out_of_range():
    async def call_fn(p):
        return _json_response("a", 0.9)

    with pytest.raises(ValueError):
        asyncio.run(acheck(call_fn, "q", threshold=2.0))


def test_acheck_validator():
    responses = [_json_response("bad", 0.9), _json_response("good", 0.9)]
    calls = []

    async def call_fn(p):
        calls.append(p)
        return responses[len(calls) - 1]

    r = asyncio.run(acheck(call_fn, "q", validator=lambda a: a == "good"))
    assert r.status == REPAIRED
    assert r.attempts[0].passed_validation is False


def test_acheck_ambiguous():
    async def call_fn(p):
        return json.dumps({"answer": "x", "confidence": 0.5, "ambiguous": True})

    r = asyncio.run(acheck(call_fn, "q"))
    assert r.status == AMBIGUOUS
    assert r.ambiguous is True


def test_acheck_prompt_type_error():
    async def call_fn(p):
        return "x"

    with pytest.raises(TypeError):
        asyncio.run(acheck(call_fn, 123))


def test_acheck_object_with_async_call():
    class Caller:
        async def __call__(self, prompt):
            return _json_response("fromobj", 0.9)

    r = asyncio.run(acheck(Caller(), "q"))
    assert r.status == ACCEPTED
    assert r.answer == "fromobj"
