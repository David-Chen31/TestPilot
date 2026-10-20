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

def _response(**overrides):
    base = {
        "ambiguous": False,
        "interpretations": [],
        "chosen_interpretation": None,
        "answer": "42",
        "confidence": 0.9,
    }
    base.update(overrides)
    return json.dumps(base)


def _make_attempt(parse_ok=True, confidence=0.9, answer="42", **kw):
    return Attempt(
        raw_text="x",
        answer=answer,
        confidence=confidence,
        parse_ok=parse_ok,
        **kw,
    )


# ---------------------------------------------------------------------------
# Attempt
# ---------------------------------------------------------------------------

def test_attempt_defaults():
    a = Attempt(raw_text="r", answer="a", confidence=0.5, parse_ok=True)
    assert a.error is None
    assert a.ambiguous is False
    assert a.interpretations == []
    assert a.chosen_interpretation is None
    assert a.passed_validation is True
    assert a.validation_error is None
    assert a.parsed is None
    assert a.call_failed is False


def test_attempt_interpretations_are_independent():
    a = Attempt(raw_text="r", answer="a", confidence=0.5, parse_ok=True)
    b = Attempt(raw_text="r", answer="a", confidence=0.5, parse_ok=True)
    a.interpretations.append("x")
    assert b.interpretations == []


# ---------------------------------------------------------------------------
# BoothResult
# ---------------------------------------------------------------------------

def test_booth_result_n_attempts_and_ok():
    r = BoothResult(answer="a", status=ACCEPTED)
    assert r.n_attempts == 0
    assert r.ok is True
    assert BoothResult(answer="a", status=REPAIRED).ok is True
    assert BoothResult(answer="a", status=UNCERTAIN).ok is False
    assert BoothResult(answer="a", status=BLOCKED).ok is False
    assert BoothResult(answer="a", status=AMBIGUOUS).ok is False


def test_all_parse_failed_property():
    assert BoothResult(answer=None, status=UNCERTAIN).all_parse_failed is False
    all_bad = BoothResult(
        answer=None, status=UNCERTAIN,
        attempts=[_make_attempt(parse_ok=False), _make_attempt(parse_ok=False)],
    )
    assert all_bad.all_parse_failed is True
    mixed = BoothResult(
        answer=None, status=UNCERTAIN,
        attempts=[_make_attempt(parse_ok=False), _make_attempt(parse_ok=True)],
    )
    assert mixed.all_parse_failed is False


def test_call_failed_property():
    assert BoothResult(answer=None, status=UNCERTAIN).call_failed is False
    r = BoothResult(
        answer=None, status=UNCERTAIN,
        attempts=[
            _make_attempt(parse_ok=False, call_failed=True),
            _make_attempt(parse_ok=False, call_failed=True),
        ],
    )
    assert r.call_failed is True
    r2 = BoothResult(
        answer=None, status=UNCERTAIN,
        attempts=[_make_attempt(parse_ok=False, call_failed=True),
                  _make_attempt(parse_ok=False)],
    )
    assert r2.call_failed is False


def test_checker_failed_property():
    assert BoothResult(answer=None, status=UNCERTAIN, reason=COMPARE_FAILED).checker_failed
    assert BoothResult(answer=None, status=UNCERTAIN, reason=INVALID_SCORE).checker_failed
    assert not BoothResult(answer=None, status=UNCERTAIN, reason=NO_EVIDENCE).checker_failed
    assert not BoothResult(answer=None, status=UNCERTAIN).checker_failed


def test_method_property_all_branches():
    assert BoothResult(answer="a", status=AMBIGUOUS).method == "ambiguity"
    assert BoothResult(answer="a", status=UNCERTAIN).method == "evidence"
    parse_fail = BoothResult(
        answer=None, status=UNCERTAIN,
        attempts=[_make_attempt(parse_ok=False)],
    )
    assert parse_fail.method == "parse_failure"
    validation = BoothResult(
        answer=None, status=UNCERTAIN,
        attempts=[_make_attempt(passed_validation=False)],
    )
    assert validation.method == "validation"
    confidence = BoothResult(
        answer="a", status=ACCEPTED,
        attempts=[_make_attempt(passed_validation=True)],
    )
    assert confidence.method == "confidence"


def test_unwrap_ok():
    r = BoothResult(answer="hello", status=ACCEPTED)
    assert r.unwrap() == "hello"


def test_unwrap_raises_on_not_ok():
    r = BoothResult(answer="nope", status=UNCERTAIN)
    with pytest.raises(BoothRejected) as excinfo:
        r.unwrap()
    assert excinfo.value.result is r
    assert "UNCERTAIN" in str(excinfo.value)
    assert "nope" not in str(excinfo.value)


def test_unwrap_raises_when_answer_none_even_if_ok():
    r = BoothResult(answer=None, status=ACCEPTED)
    with pytest.raises(BoothRejected):
        r.unwrap()


def test_unwrap_or():
    assert BoothResult(answer="hello", status=ACCEPTED).unwrap_or("d") == "hello"
    assert BoothResult(answer="x", status=UNCERTAIN).unwrap_or("d") == "d"
    assert BoothResult(answer=None, status=ACCEPTED).unwrap_or("d") == "d"


def test_to_dict():
    attempt = _make_attempt()
    r = BoothResult(
        answer="42", status=ACCEPTED, confidence=0.9, attempts=[attempt],
        ambiguous=False, interpretations=[], evidence_agreement=None,
        parsed={"x": 1}, reason=None, detail=None,
    )
    d = r.to_dict()
    assert d["answer"] == "42"
    assert d["status"] == ACCEPTED
    assert d["confidence"] == 0.9
    assert d["n_attempts"] == 1
    assert d["ok"] is True
    assert d["all_parse_failed"] is False
    assert d["call_failed"] is False
    assert d["method"] == "confidence"
    assert d["checker_failed"] is False
    assert d["parsed"] == {"x": 1}
    assert d["attempts"][0]["answer"] == "42"


# ---------------------------------------------------------------------------
# check_with_evidence
# ---------------------------------------------------------------------------

def test_evidence_arg_validation():
    with pytest.raises(TypeError):
        check_with_evidence("a", ["x"], lambda a, e: True, evidence_threshold=True)
    with pytest.raises(TypeError):
        check_with_evidence("a", ["x"], lambda a, e: True, evidence_threshold="0.5")
    with pytest.raises(ValueError):
        check_with_evidence("a", ["x"], lambda a, e: True, evidence_threshold=1.5)
    with pytest.raises(ValueError):
        check_with_evidence("a", ["x"], lambda a, e: True, evidence_threshold=-0.1)


def test_evidence_answer_type():
    with pytest.raises(TypeError):
        check_with_evidence(123, ["x"], lambda a, e: True)


def test_evidence_empty_answer():
    r = check_with_evidence("   ", ["x"], lambda a, e: True)
    assert r.status == UNCERTAIN
    assert r.answer is None
    assert r.reason == EMPTY_ANSWER


def test_evidence_none_answer_is_empty():
    r = check_with_evidence(None, ["x"], lambda a, e: True)
    assert r.status == UNCERTAIN
    assert r.reason == EMPTY_ANSWER


def test_evidence_empty_sequence():
    r = check_with_evidence("a", [], lambda a, e: True)
    assert r.status == UNCERTAIN
    assert r.answer == "a"
    assert r.reason == NO_EVIDENCE


def test_evidence_compare_fn_raises():
    def boom(a, e):
        raise RuntimeError("kaboom")

    r = check_with_evidence("a", ["x"], boom)
    assert r.status == UNCERTAIN
    assert r.reason == COMPARE_FAILED
    assert "RuntimeError" in r.detail
    assert "kaboom" in r.detail


def test_evidence_async_compare_fn_raises_typeerror():
    async def cmp(a, e):
        return True

    with pytest.raises(TypeError):
        check_with_evidence("a", ["x"], cmp)


def test_evidence_bool_true():
    r = check_with_evidence("a", ["x"], lambda a, e: True)
    assert r.status == ACCEPTED
    assert r.confidence == 1.0
    assert r.evidence_agreement == 1.0
    assert r.reason is None
    assert r.detail is None


def test_evidence_bool_false():
    r = check_with_evidence("a", ["x"], lambda a, e: False)
    assert r.status == BLOCKED
    assert r.confidence == 0.0
    assert r.evidence_agreement == 0.0
    assert r.reason == EVIDENCE_DISAGREES
    assert r.detail == "compare_fn returned False"


def test_evidence_float_above_threshold():
    r = check_with_evidence("a", ["x"], lambda a, e: 0.8)
    assert r.status == ACCEPTED
    assert r.confidence == 0.8
    assert r.evidence_agreement == 0.8


def test_evidence_float_below_threshold():
    r = check_with_evidence("a", ["x"], lambda a, e: 0.5)
    assert r.status == BLOCKED
    assert r.evidence_agreement == 0.5
    assert r.reason == EVIDENCE_DISAGREES
    assert "below evidence_threshold" in r.detail


def test_evidence_float_exactly_at_threshold():
    r = check_with_evidence("a", ["x"], lambda a, e: 0.7)
    assert r.status == ACCEPTED


def test_evidence_non_numeric_score():
    r = check_with_evidence("a", ["x"], lambda a, e: "nope")
    assert r.status == UNCERTAIN
    assert r.reason == INVALID_SCORE
    assert "non-numeric" in r.detail


def test_evidence_score_out_of_range():
    r = check_with_evidence("a", ["x"], lambda a, e: 1.5)
    assert r.status == UNCERTAIN
    assert r.reason == INVALID_SCORE
    assert "outside" in r.detail


def test_evidence_custom_threshold():
    r = check_with_evidence("a", ["x"], lambda a, e: 0.6, evidence_threshold=0.5)
    assert r.status == ACCEPTED


def test_evidence_passes_answer_and_evidence():
    seen = {}

    def cmp(a, e):
        seen["a"] = a
        seen["e"] = e
        return True

    ev = ("p1", "p2")
    check_with_evidence("ans", ev, cmp)
    assert seen["a"] == "ans"
    assert seen["e"] is ev


# ---------------------------------------------------------------------------
# check()
# ---------------------------------------------------------------------------

def test_check_arg_validation():
    with pytest.raises(TypeError):
        check(lambda p: "", 123)
    with pytest.raises(TypeError):
        check("not-callable", "prompt")
    with pytest.raises(ValueError):
        check(lambda p: "", "prompt", threshold=1.5)
    with pytest.raises(TypeError):
        check(lambda p: "", "prompt", max_retries=1.5)
    with pytest.raises(ValueError):
        check(lambda p: "", "prompt", max_retries=-1)
    with pytest.raises(TypeError):
        check(lambda p: "", "prompt", on_attempt="nope")


def test_check_rejects_async_call_fn():
    async def acall(p):
        return _response()

    with pytest.raises(TypeError):
        check(acall, "prompt")


def test_check_rejects_async_on_attempt():
    async def cb(i, a):
        pass

    with pytest.raises(TypeError):
        check(lambda p: _response(), "prompt", on_attempt=cb)


def test_check_accepts_object_with_async_call():
    class Caller:
        async def __call__(self, p):
            return _response()

    with pytest.raises(TypeError):
        check(Caller(), "prompt")


def test_check_accepted_first_try():
    calls = []

    def call_fn(p):
        calls.append(p)
        return _response(answer="42", confidence=0.9)

    r = check(call_fn, "What?")
    assert r.status == ACCEPTED
    assert r.answer == "42"
    assert r.confidence == 0.9
    assert r.n_attempts == 1
    assert r.method == "confidence"
    assert len(calls) == 1
    # prompt should include the confidence suffix
    assert "confidence" in calls[0]
    assert calls[0].startswith("What?")


def test_check_ambiguous_short_circuits_no_retry():
    calls = []

    def call_fn(p):
        calls.append(p)
        return _response(
            ambiguous=True,
            interpretations=["reading A", "reading B"],
            chosen_interpretation="reading A",
            answer="x",
            confidence=0.4,
        )

    r = check(call_fn, "Q", max_retries=3)
    assert r.status == AMBIGUOUS
    assert r.ambiguous is True
    assert r.interpretations == ["reading A", "reading B"]
    assert r.method == "ambiguity"
    assert len(calls) == 1


def test_check_retry_then_repaired():
    responses = [_response(confidence=0.3), _response(confidence=0.95, answer="43")]
    idx = {"i": 0}
    prompts = []

    def call_fn(p):
        prompts.append(p)
        resp = responses[idx["i"]]
        idx["i"] += 1
        return resp

    r = check(call_fn, "Q", max_retries=1)
    assert r.status == REPAIRED
    assert r.answer == "43"
    assert r.n_attempts == 2
    # second prompt is a retry prompt referencing prior answer
    assert "previous attempt" in prompts[1].lower() or "previous" in prompts[1].lower()
    assert "0.3" in prompts[1]


def test_check_retry_parse_failure_prompt():
    responses = ["not json at all", _response(confidence=0.9)]
    idx = {"i": 0}
    prompts = []

    def call_fn(p):
        prompts.append(p)
        resp = responses[idx["i"]]
        idx["i"] += 1
        return resp

    r = check(call_fn, "Q", max_retries=1)
    assert r.status == REPAIRED
    assert "could not be parsed" in prompts[1]


def test_check_uncertain_after_exhaust_retries():
    def call_fn(p):
        return _response(confidence=0.1)

    r = check(call_fn, "Q", max_retries=2)
    assert r.status == UNCERTAIN
    assert r.n_attempts == 3
    assert r.answer == "42"  # last parseable answer
    assert r.confidence == 0.1


def test_check_uncertain_no_attempts_when_max_retries_zero_parse_fail():
    r = check(lambda p: "garbage", "Q", max_retries=0)
    assert r.status == UNCERTAIN
    assert r.n_attempts == 1
    assert r.answer is None
    assert r.all_parse_failed is True


def test_check_call_fn_raises():
    def call_fn(p):
        raise ValueError("oops")

    r = check(call_fn, "Q", max_retries=0)
    assert r.status == UNCERTAIN
    assert r.call_failed is True
    assert r.method == "parse_failure"
    assert r.n_attempts == 1


def test_check_call_fn_returns_non_str():
    r = check(lambda p: 123, "Q", max_retries=0)
    assert r.status == UNCERTAIN
    assert r.call_failed is True
    assert "int" in r.attempts[0].error


def test_check_validator_pass():
    def validator(ans):
        return ans == "42"

    r = check(lambda p: _response(answer="42", confidence=0.9), "Q", validator=validator)
    assert r.status == ACCEPTED
    assert r.attempts[0].passed_validation is True
    assert r.attempts[0].validation_error is None


def test_check_validator_fail_then_retry():
    responses = [_response(answer="bad", confidence=0.9), _response(answer="good", confidence=0.9)]
    idx = {"i": 0}
    prompts = []

    def call_fn(p):
        prompts.append(p)
        resp = responses[idx["i"]]
        idx["i"] += 1
        return resp

    def validator(ans):
        return (ans == "good", "must equal good")

    r = check(call_fn, "Q", max_retries=1, validator=validator)
    assert r.status == REPAIRED
    assert r.answer == "good"
    assert r.attempts[0].passed_validation is False
    assert "failed validation" in prompts[1]
    assert "must equal good" in prompts[1]


def test_check_validator_fail_final_is_uncertain_method_validation():
    def validator(ans):
        return False

    r = check(lambda p: _response(), "Q", max_retries=0, validator=validator)
    assert r.status == UNCERTAIN
    assert r.method == "validation"
    assert r.attempts[0].passed_validation is False
    assert r.attempts[0].validation_error == "Validator returned False"


def test_check_validator_raises():
    def validator(ans):
        raise RuntimeError("bad validator")

    r = check(lambda p: _response(), "Q", max_retries=0, validator=validator)
    a = r.attempts[0]
    assert a.passed_validation is False
    assert "Validator raised RuntimeError: bad validator" == a.validation_error


def test_check_validator_returns_async_coroutine():
    async def validator(ans):
        return True

    r = check(lambda p: _response(), "Q", max_retries=0, validator=validator)
    a = r.attempts[0]
    assert a.passed_validation is False
    assert "coroutine" in a.validation_error


def test_check_validator_invalid_type():
    def validator(ans):
        return "not a bool"

    r = check(lambda p: _response(), "Q", max_retries=0, validator=validator)
    a = r.attempts[0]
    assert a.passed_validation is False
    assert "invalid type" in a.validation_error


def test_check_on_attempt_callback_called_each_attempt():
    seen = []

    def on_attempt(i, attempt):
        seen.append((i, attempt.parse_ok))

    responses = ["bad", _response(confidence=0.9)]
    idx = {"i": 0}

    def call_fn(p):
        resp = responses[idx["i"]]
        idx["i"] += 1
        return resp

    r = check(call_fn, "Q", max_retries=1, on_attempt=on_attempt)
    assert seen == [(0, False), (1, True)]
    assert r.status == REPAIRED


def test_check_ambiguous_skips_validation():
    called = {"v": 0}

    def validator(ans):
        called["v"] += 1
        return False

    def call_fn(p):
        return _response(ambiguous=True, interpretations=["a"], answer="x")

    r = check(call_fn, "Q", validator=validator)
    assert r.status == AMBIGUOUS
    assert called["v"] == 0


def test_check_parse_candidate_json_embedded_in_text():
    def call_fn(p):
        return "Here is my answer:\n" + _response(answer="embedded", confidence=0.9)

    r = check(call_fn, "Q")
    assert r.status == ACCEPTED
    assert r.answer == "embedded"


def test_check_parse_answer_non_string_scalar_stringified():
    def call_fn(p):
        return json.dumps({"answer": 42, "confidence": 0.9})

    r = check(call_fn, "Q")
    assert r.status == ACCEPTED
    assert r.answer == "42"


# ---------------------------------------------------------------------------
# acheck()
# ---------------------------------------------------------------------------

def test_acheck_requires_async_call_fn():
    with pytest.raises(TypeError):
        asyncio.run(acheck(lambda p: _response(), "Q"))


def test_acheck_arg_validation():
    async def acall(p):
        return _response()

    with pytest.raises(TypeError):
        asyncio.run(acheck(acall, 123))

    async def run_bad_threshold():
        await acheck(acall, "Q", threshold=2.0)

    with pytest.raises(ValueError):
        asyncio.run(run_bad_threshold())


def test_acheck_accepted():
    async def acall(p):
        return _response(answer="ok", confidence=0.9)

    r = asyncio.run(acheck(acall, "Q"))
    assert r.status == ACCEPTED
    assert r.answer == "ok"


def test_acheck_retry_with_sync_on_attempt():
    responses = [_response(confidence=0.1), _response(confidence=0.9)]
    idx = {"i": 0}
    seen = []

    async def acall(p):
        resp = responses[idx["i"]]
        idx["i"] += 1
        return resp

    def on_attempt(i, attempt):
        seen.append(i)

    r = asyncio.run(acheck(acall, "Q", max_retries=1, on_attempt=on_attempt))
    assert r.status == REPAIRED
    assert seen == [0, 1]


def test_acheck_retry_with_async_on_attempt():
    responses = [_response(confidence=0.1), _response(confidence=0.9)]
    idx = {"i": 0}
    seen = []

    async def acall(p):
        resp = responses[idx["i"]]
        idx["i"] += 1
        return resp

    async def on_attempt(i, attempt):
        seen.append(i)

    r = asyncio.run(acheck(acall, "Q", max_retries=1, on_attempt=on_attempt))
    assert r.status == REPAIRED
    assert seen == [0, 1]


def test_acheck_call_raises():
    async def acall(p):
        raise RuntimeError("boom")

    r = asyncio.run(acheck(acall, "Q", max_retries=0))
    assert r.status == UNCERTAIN
    assert r.call_failed is True


def test_acheck_non_str_return():
    async def acall(p):
        return None

    r = asyncio.run(acheck(acall, "Q", max_retries=0))
    assert r.call_failed is True
    assert "NoneType" in r.attempts[0].error


def test_acheck_ambiguous():
    async def acall(p):
        return _response(ambiguous=True, interpretations=["x", "y"], answer="z")

    r = asyncio.run(acheck(acall, "Q"))
    assert r.status == AMBIGUOUS
    assert r.interpretations == ["x", "y"]


def test_acheck_uncertain():
    async def acall(p):
        return _response(confidence=0.1)

    r = asyncio.run(acheck(acall, "Q", max_retries=0))
    assert r.status == UNCERTAIN
    assert r.answer == "42"


def test_acheck_object_with_async_call():
    class Caller:
        async def __call__(self, p):
            return _response(answer="fromcaller", confidence=0.9)

    r = asyncio.run(acheck(Caller(), "Q"))
    assert r.status == ACCEPTED
    assert r.answer == "fromcaller"
