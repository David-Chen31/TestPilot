import asyncio

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


def valid_json(answer="42", confidence=0.9, **extra):
    obj = {"answer": answer, "confidence": confidence}
    obj.update(extra)
    import json

    return json.dumps(obj)


# ---------------------------------------------------------------------------
# Attempt
# ---------------------------------------------------------------------------


def test_attempt_defaults():
    a = Attempt(raw_text="x", answer="a", confidence=0.5, parse_ok=True)
    assert a.error is None
    assert a.ambiguous is False
    assert a.interpretations == []
    assert a.chosen_interpretation is None
    assert a.passed_validation is True
    assert a.validation_error is None
    assert a.parsed is None
    assert a.call_failed is False


def test_attempt_independent_default_lists():
    a = Attempt(raw_text="x", answer="a", confidence=0.5, parse_ok=True)
    b = Attempt(raw_text="y", answer="b", confidence=0.5, parse_ok=True)
    a.interpretations.append("z")
    assert b.interpretations == []


# ---------------------------------------------------------------------------
# BoothResult
# ---------------------------------------------------------------------------


def test_booth_result_basic_properties():
    r = BoothResult(answer="a", status=ACCEPTED)
    assert r.n_attempts == 0
    assert r.ok is True
    assert r.all_parse_failed is False
    assert r.call_failed is False
    assert r.checker_failed is False
    assert r.method == "evidence"


def test_booth_result_ok_statuses():
    assert BoothResult(answer="a", status=ACCEPTED).ok is True
    assert BoothResult(answer="a", status=REPAIRED).ok is True
    assert BoothResult(answer="a", status=UNCERTAIN).ok is False
    assert BoothResult(answer="a", status=BLOCKED).ok is False
    assert BoothResult(answer="a", status=AMBIGUOUS).ok is False


def test_booth_result_all_parse_failed():
    attempts = [
        Attempt(raw_text="x", answer=None, confidence=None, parse_ok=False),
        Attempt(raw_text="y", answer=None, confidence=None, parse_ok=False),
    ]
    r = BoothResult(answer=None, status=UNCERTAIN, attempts=attempts)
    assert r.all_parse_failed is True

    attempts.append(Attempt(raw_text="z", answer="a", confidence=0.5, parse_ok=True))
    assert r.all_parse_failed is False


def test_booth_result_parse_failed_empty_attempts():
    r = BoothResult(answer=None, status=UNCERTAIN)
    assert r.all_parse_failed is False
    assert r.call_failed is False


def test_booth_result_call_failed_property():
    attempts = [
        Attempt(raw_text="x", answer=None, confidence=None, parse_ok=False, call_failed=True),
        Attempt(raw_text="y", answer=None, confidence=None, parse_ok=False, call_failed=True),
    ]
    r = BoothResult(answer=None, status=UNCERTAIN, attempts=attempts)
    assert r.call_failed is True

    attempts[1].call_failed = False
    assert r.call_failed is False


def test_booth_result_checker_failed():
    assert BoothResult(answer=None, status=UNCERTAIN, reason=COMPARE_FAILED).checker_failed
    assert BoothResult(answer=None, status=UNCERTAIN, reason=INVALID_SCORE).checker_failed
    assert not BoothResult(answer=None, status=UNCERTAIN, reason=UNCERTAIN).checker_failed
    assert not BoothResult(answer=None, status=UNCERTAIN).checker_failed


def test_booth_result_method_ambiguity():
    r = BoothResult(
        answer="a",
        status=AMBIGUOUS,
        attempts=[
            Attempt(raw_text="x", answer="a", confidence=0.5, parse_ok=True, ambiguous=True)
        ],
    )
    assert r.method == "ambiguity"


def test_booth_result_method_evidence():
    r = BoothResult(answer="a", status=ACCEPTED)
    assert r.method == "evidence"


def test_booth_result_method_parse_failure():
    attempts = [Attempt(raw_text="x", answer=None, confidence=None, parse_ok=False)]
    r = BoothResult(answer=None, status=UNCERTAIN, attempts=attempts)
    assert r.method == "parse_failure"


def test_booth_result_method_validation():
    attempts = [
        Attempt(
            raw_text="x",
            answer="a",
            confidence=0.5,
            parse_ok=True,
            passed_validation=False,
            validation_error="nope",
        )
    ]
    r = BoothResult(answer=None, status=UNCERTAIN, attempts=attempts)
    assert r.method == "validation"


def test_booth_result_method_confidence():
    attempts = [Attempt(raw_text="x", answer="a", confidence=0.5, parse_ok=True)]
    r = BoothResult(answer="a", status=UNCERTAIN, attempts=attempts)
    assert r.method == "confidence"


def test_booth_result_unwrap_ok():
    r = BoothResult(answer="hello", status=ACCEPTED)
    assert r.unwrap() == "hello"


def test_booth_result_unwrap_rejected():
    r = BoothResult(answer="hello", status=UNCERTAIN)
    with pytest.raises(BoothRejected) as excinfo:
        r.unwrap()
    assert excinfo.value.result is r
    assert "hello" not in str(excinfo.value)


def test_booth_result_unwrap_none_answer_even_if_ok():
    r = BoothResult(answer=None, status=ACCEPTED)
    with pytest.raises(BoothRejected) as excinfo:
        r.unwrap()
    assert excinfo.value.result is r


def test_booth_result_unwrap_or():
    assert BoothResult(answer="hi", status=ACCEPTED).unwrap_or("default") == "hi"
    assert BoothResult(answer="hi", status=UNCERTAIN).unwrap_or("default") == "default"
    assert BoothResult(answer=None, status=ACCEPTED).unwrap_or("default") == "default"


def test_booth_result_to_dict():
    attempts = [Attempt(raw_text="x", answer="a", confidence=0.9, parse_ok=True)]
    r = BoothResult(
        answer="a",
        status=ACCEPTED,
        confidence=0.9,
        attempts=attempts,
        parsed={"answer": "a"},
        detail="d",
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
    assert d["checker_failed"] is False
    assert d["detail"] == "d"
    assert d["parsed"] == {"answer": "a"}
    assert d["attempts"][0]["answer"] == "a"
    assert d["attempts"][0]["parse_ok"] is True


# ---------------------------------------------------------------------------
# check_with_evidence
# ---------------------------------------------------------------------------


def test_evidence_empty_answer():
    r = check_with_evidence("", ["e"], lambda a, ev: True)
    assert r.status == UNCERTAIN
    assert r.answer is None
    assert r.reason == EMPTY_ANSWER


def test_evidence_whitespace_answer():
    r = check_with_evidence("   ", ["e"], lambda a, ev: True)
    assert r.status == UNCERTAIN
    assert r.reason == EMPTY_ANSWER


def test_evidence_non_str_answer():
    with pytest.raises(TypeError):
        check_with_evidence(123, ["e"], lambda a, ev: True)


def test_evidence_empty_sequence():
    r = check_with_evidence("42", [], lambda a, ev: True)
    assert r.status == UNCERTAIN
    assert r.answer == "42"
    assert r.reason == NO_EVIDENCE


def test_evidence_compare_raises():
    def bad(a, ev):
        raise RuntimeError("boom")

    r = check_with_evidence("42", ["e"], bad)
    assert r.status == UNCERTAIN
    assert r.reason == COMPARE_FAILED
    assert r.detail == "RuntimeError: boom"


def test_evidence_async_compare_raises_type_error():
    async def coro(a, ev):
        return True

    with pytest.raises(TypeError):
        check_with_evidence("42", ["e"], coro)


def test_evidence_bool_true():
    r = check_with_evidence("42", ["e"], lambda a, ev: True)
    assert r.status == ACCEPTED
    assert r.confidence == 1.0
    assert r.evidence_agreement == 1.0
    assert r.reason is None
    assert r.detail is None


def test_evidence_bool_false():
    r = check_with_evidence("42", ["e"], lambda a, ev: False)
    assert r.status == BLOCKED
    assert r.confidence == 0.0
    assert r.evidence_agreement == 0.0
    assert r.reason == EVIDENCE_DISAGREES
    assert r.detail == "compare_fn returned False"


def test_evidence_float_passing():
    r = check_with_evidence("42", ["e"], lambda a, ev: 0.9, evidence_threshold=0.7)
    assert r.status == ACCEPTED
    assert r.confidence == 0.9
    assert r.reason is None


def test_evidence_float_below_threshold():
    r = check_with_evidence("42", ["e"], lambda a, ev: 0.5, evidence_threshold=0.7)
    assert r.status == BLOCKED
    assert r.confidence == 0.5
    assert r.reason == EVIDENCE_DISAGREES
    assert "0.5" in r.detail


def test_evidence_float_at_threshold():
    r = check_with_evidence("42", ["e"], lambda a, ev: 0.7, evidence_threshold=0.7)
    assert r.status == ACCEPTED


def test_evidence_non_numeric_score():
    r = check_with_evidence("42", ["e"], lambda a, ev: "not a number")
    assert r.status == UNCERTAIN
    assert r.reason == INVALID_SCORE


def test_evidence_out_of_range_score():
    r = check_with_evidence("42", ["e"], lambda a, ev: 1.5)
    assert r.status == UNCERTAIN
    assert r.reason == INVALID_SCORE


def test_evidence_out_of_range_negative():
    r = check_with_evidence("42", ["e"], lambda a, ev: -0.1)
    assert r.status == UNCERTAIN
    assert r.reason == INVALID_SCORE


def test_evidence_threshold_bool_rejected():
    with pytest.raises(TypeError):
        check_with_evidence("42", ["e"], lambda a, ev: True, evidence_threshold=True)


def test_evidence_threshold_non_number():
    with pytest.raises(TypeError):
        check_with_evidence("42", ["e"], lambda a, ev: True, evidence_threshold="x")


def test_evidence_threshold_out_of_range():
    with pytest.raises(ValueError):
        check_with_evidence("42", ["e"], lambda a, ev: True, evidence_threshold=1.5)


# ---------------------------------------------------------------------------
# check()
# ---------------------------------------------------------------------------


def test_check_args_invalid_prompt():
    with pytest.raises(TypeError):
        check(lambda p: valid_json(), 123)


def test_check_args_invalid_call_fn():
    with pytest.raises(TypeError):
        check("not callable", "prompt")


def test_check_threshold_out_of_range():
    with pytest.raises(ValueError):
        check(lambda p: valid_json(), "prompt", threshold=1.5)


def test_check_max_retries_negative():
    with pytest.raises(ValueError):
        check(lambda p: valid_json(), "prompt", max_retries=-1)


def test_check_max_retries_non_int():
    with pytest.raises(TypeError):
        check(lambda p: valid_json(), "prompt", max_retries=1.0)


def test_check_async_call_rejected():
    async def call(p):
        return valid_json()

    with pytest.raises(TypeError):
        check(call, "prompt")


def test_check_on_attempt_not_callable():
    with pytest.raises(TypeError):
        check(lambda p: valid_json(), "prompt", on_attempt="not callable")


def test_check_async_on_attempt_rejected():
    async def on_attempt(i, a):
        pass

    with pytest.raises(TypeError):
        check(lambda p: valid_json(), "prompt", on_attempt=on_attempt)


def test_check_success_first_attempt():
    calls = []

    def call(p):
        calls.append(p)
        return valid_json("Paris", 0.95)

    r = check(call, "Capital of France?")
    assert r.status == ACCEPTED
    assert r.answer == "Paris"
    assert r.confidence == 0.95
    assert r.n_attempts == 1
    assert len(calls) == 1
    # prompt has confidence suffix appended
    assert "confidence" in calls[0]


def test_check_retry_then_success_repaired():
    responses = iter([valid_json("wrong", 0.1), valid_json("right", 0.9)])

    def call(p):
        return next(responses)

    r = check(call, "q")
    assert r.status == REPAIRED
    assert r.answer == "right"
    assert r.n_attempts == 2


def test_check_uncertain_after_retries():
    r = check(lambda p: valid_json("x", 0.1), "q", max_retries=1)
    assert r.status == UNCERTAIN
    assert r.answer == "x"
    assert r.confidence == 0.1
    assert r.n_attempts == 2


def test_check_parse_failure_retry_and_final():
    r = check(lambda p: "no json at all", "q", max_retries=0)
    assert r.status == UNCERTAIN
    assert r.answer is None
    assert r.all_parse_failed is True
    assert r.method == "parse_failure"


def test_check_call_fn_raises():
    def call(p):
        raise RuntimeError("oops")

    r = check(call, "q", max_retries=0)
    assert r.status == UNCERTAIN
    assert r.call_failed is True
    assert r.attempts[0].error == "oops"


def test_check_call_fn_returns_non_str():
    r = check(lambda p: 123, "q", max_retries=0)
    assert r.status == UNCERTAIN
    assert r.call_failed is True
    assert "int" in r.attempts[0].error


def test_check_ambiguous_result():
    resp = valid_json(
        "answer text",
        0.5,
        ambiguous=True,
        interpretations=["a", "b"],
        chosen_interpretation="a",
    )
    r = check(lambda p: resp, "q")
    assert r.status == AMBIGUOUS
    assert r.ambiguous is True
    assert r.interpretations == ["a", "b"]
    assert r.method == "ambiguity"


def test_check_validator_pass():
    seen = []

    def validator(answer):
        seen.append(answer)
        return True

    r = check(lambda p: valid_json("ok", 0.9), "q", validator=validator)
    assert r.status == ACCEPTED
    assert seen == ["ok"]


def test_check_validator_fail_triggers_retry_prompt():
    prompts = []
    responses = iter([valid_json("bad", 0.9), valid_json("good", 0.9)])

    def call(p):
        prompts.append(p)
        return next(responses)

    def validator(answer):
        if answer == "good":
            return True
        return (False, "must be good")

    r = check(call, "q", validator=validator)
    assert r.status == REPAIRED
    assert r.answer == "good"
    assert "failed validation" in prompts[1]
    assert "must be good" in prompts[1]


def test_check_validator_fail_all_attempts_validation_method():
    r = check(
        lambda p: valid_json("bad", 0.9),
        "q",
        max_retries=0,
        validator=lambda a: False,
    )
    assert r.status == UNCERTAIN
    assert r.method == "validation"


def test_check_on_attempt_called_for_each_attempt():
    seen = []

    def on_attempt(i, attempt):
        seen.append((i, attempt.parse_ok))

    r = check(lambda p: valid_json("a", 0.9), "q", on_attempt=on_attempt)
    assert r.status == ACCEPTED
    assert seen == [(0, True)]


def test_check_on_attempt_sees_failures():
    seen = []
    responses = iter(["garbage", valid_json("a", 0.9)])

    def call(p):
        return next(responses)

    def on_attempt(i, attempt):
        seen.append((i, attempt.parse_ok))

    check(call, "q", on_attempt=on_attempt)
    assert seen == [(0, False), (1, True)]


def test_check_retry_prompt_includes_previous_answer():
    prompts = []
    responses = iter([valid_json("first", 0.1), valid_json("second", 0.9)])

    def call(p):
        prompts.append(p)
        return next(responses)

    check(call, "q")
    assert '"first"' in prompts[1]


def test_check_parse_failure_prompt_mentions_parse():
    prompts = []
    responses = iter(["garbage", valid_json("a", 0.9)])

    def call(p):
        prompts.append(p)
        return next(responses)

    check(call, "q")
    assert "could not be parsed" in prompts[1]


def test_check_call_error_prompt_is_plain():
    prompts = []
    count = {"n": 0}

    def call(p):
        prompts.append(p)
        count["n"] += 1
        if count["n"] == 1:
            raise RuntimeError("fail")
        return valid_json("a", 0.9)

    r = check(call, "q")
    assert r.status == REPAIRED
    # second prompt should just have the original + suffix, no "previous attempt" text
    assert "On a previous attempt" not in prompts[1]


# ---------------------------------------------------------------------------
# acheck()
# ---------------------------------------------------------------------------


def test_acheck_requires_async_call():
    with pytest.raises(TypeError):
        asyncio.run(acheck(lambda p: valid_json(), "q"))


def test_acheck_invalid_prompt():
    async def call(p):
        return valid_json()

    with pytest.raises(TypeError):
        asyncio.run(acheck(call, 123))


def test_acheck_success():
    async def call(p):
        return valid_json("Paris", 0.95)

    r = asyncio.run(acheck(call, "Capital of France?"))
    assert r.status == ACCEPTED
    assert r.answer == "Paris"
    assert r.n_attempts == 1


def test_acheck_retry_then_success():
    responses = iter([valid_json("no", 0.1), valid_json("yes", 0.9)])

    async def call(p):
        return next(responses)

    r = asyncio.run(acheck(call, "q"))
    assert r.status == REPAIRED
    assert r.answer == "yes"


def test_acheck_uncertain():
    async def call(p):
        return valid_json("x", 0.1)

    r = asyncio.run(acheck(call, "q", max_retries=0))
    assert r.status == UNCERTAIN
    assert r.answer == "x"


def test_acheck_call_error():
    async def call(p):
        raise RuntimeError("oops")

    r = asyncio.run(acheck(call, "q", max_retries=0))
    assert r.status == UNCERTAIN
    assert r.call_failed is True


def test_acheck_non_str_return():
    async def call(p):
        return 42

    r = asyncio.run(acheck(call, "q", max_retries=0))
    assert r.call_failed is True


def test_acheck_sync_on_attempt():
    seen = []

    def on_attempt(i, attempt):
        seen.append((i, attempt.parse_ok))

    async def call(p):
        return valid_json("a", 0.9)

    asyncio.run(acheck(call, "q", on_attempt=on_attempt))
    assert seen == [(0, True)]


def test_acheck_async_on_attempt():
    seen = []

    async def on_attempt(i, attempt):
        seen.append((i, attempt.parse_ok))

    async def call(p):
        return valid_json("a", 0.9)

    asyncio.run(acheck(call, "q", on_attempt=on_attempt))
    assert seen == [(0, True)]


def test_acheck_on_attempt_not_callable():
    async def call(p):
        return valid_json()

    with pytest.raises(TypeError):
        asyncio.run(acheck(call, "q", on_attempt="nope"))


def test_acheck_with_validator():
    async def call(p):
        return valid_json("a", 0.9)

    r = asyncio.run(acheck(call, "q", validator=lambda a: a == "a"))
    assert r.status == ACCEPTED


def test_acheck_invalid_threshold():
    async def call(p):
        return valid_json()

    with pytest.raises(ValueError):
        asyncio.run(acheck(call, "q", threshold=-0.1))


def test_acheck_invalid_max_retries():
    async def call(p):
        return valid_json()

    with pytest.raises(TypeError):
        asyncio.run(acheck(call, "q", max_retries=1.5))
