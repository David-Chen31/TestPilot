import asyncio
import json
import pytest

from booth.core import (
    ACCEPTED,
    REPAIRED,
    AMBIGUOUS,
    BLOCKED,
    UNCERTAIN,
    EMPTY_ANSWER,
    NO_EVIDENCE,
    EVIDENCE_DISAGREES,
    COMPARE_FAILED,
    INVALID_SCORE,
    Attempt,
    BoothRejected,
    BoothResult,
    check_with_evidence,
    check,
    acheck,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Run an async coroutine without relying on pytest-asyncio."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def make_json(answer="A", confidence=0.9, **extra):
    obj = {"answer": answer, "confidence": confidence}
    obj.update(extra)
    return json.dumps(obj)


def fake_call(responses):
    """call_fn returns successive responses (or raises if an Exception is given)."""
    it = iter(responses)
    calls = []

    def fn(prompt):
        calls.append(prompt)
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item

    fn.calls = calls
    return fn


# ---------------------------------------------------------------------------
# Attempt / BoothResult basic dataclass behaviour
# ---------------------------------------------------------------------------

def test_attempt_defaults():
    a = Attempt(raw_text="x", answer="A", confidence=0.5, parse_ok=True)
    assert a.error is None
    assert a.ambiguous is False
    assert a.interpretations == []
    assert a.chosen_interpretation is None
    assert a.passed_validation is True
    assert a.validation_error is None
    assert a.parsed is None
    assert a.call_failed is False


def test_attempt_interpretations_independent():
    a1 = Attempt(raw_text="x", answer="A", confidence=0.5, parse_ok=True)
    a2 = Attempt(raw_text="x", answer="A", confidence=0.5, parse_ok=True)
    a1.interpretations.append("foo")
    assert a2.interpretations == []


def test_booth_result_n_attempts_and_ok():
    r = BoothResult(answer="A", status=ACCEPTED)
    assert r.n_attempts == 0
    assert r.ok is True
    assert BoothResult(answer="A", status=REPAIRED).ok is True
    assert BoothResult(answer="A", status=UNCERTAIN).ok is False
    assert BoothResult(answer="A", status=BLOCKED).ok is False
    assert BoothResult(answer="A", status=AMBIGUOUS).ok is False


def test_booth_result_all_parse_failed():
    assert BoothResult(answer=None, status=UNCERTAIN).all_parse_failed is False
    ok = Attempt(raw_text="x", answer="A", confidence=0.5, parse_ok=True)
    bad = Attempt(raw_text="x", answer=None, confidence=None, parse_ok=False)
    assert BoothResult(answer=None, status=UNCERTAIN, attempts=[ok]).all_parse_failed is False
    assert BoothResult(answer=None, status=UNCERTAIN, attempts=[bad]).all_parse_failed is True
    assert BoothResult(answer=None, status=UNCERTAIN, attempts=[bad, bad]).all_parse_failed is True


def test_booth_result_call_failed():
    assert BoothResult(answer=None, status=UNCERTAIN).call_failed is False
    bad = Attempt(raw_text="x", answer=None, confidence=None, parse_ok=False, call_failed=True)
    ok = Attempt(raw_text="x", answer="A", confidence=0.5, parse_ok=True)
    assert BoothResult(answer=None, status=UNCERTAIN, attempts=[bad]).call_failed is True
    assert BoothResult(answer=None, status=UNCERTAIN, attempts=[bad, ok]).call_failed is False


def test_booth_result_checker_failed():
    assert BoothResult(answer="A", status=UNCERTAIN, reason=COMPARE_FAILED).checker_failed is True
    assert BoothResult(answer="A", status=UNCERTAIN, reason=INVALID_SCORE).checker_failed is True
    assert BoothResult(answer="A", status=UNCERTAIN, reason=NO_EVIDENCE).checker_failed is False
    assert BoothResult(answer="A", status=ACCEPTED).checker_failed is False


def test_booth_result_method():
    assert BoothResult(answer="A", status=AMBIGUOUS).method == "ambiguity"
    assert BoothResult(answer="A", status=UNCERTAIN).method == "evidence"

    ok = Attempt(raw_text="x", answer="A", confidence=0.9, parse_ok=True)
    bad = Attempt(raw_text="x", answer=None, confidence=None, parse_ok=False)
    failed_val = Attempt(
        raw_text="x", answer="A", confidence=0.9, parse_ok=True,
        passed_validation=False,
    )

    assert BoothResult(answer="A", status=UNCERTAIN, attempts=[ok]).method == "confidence"
    assert BoothResult(answer=None, status=UNCERTAIN, attempts=[bad]).method == "parse_failure"
    assert BoothResult(
        answer="A", status=UNCERTAIN, attempts=[ok, failed_val]
    ).method == "validation"


def test_unwrap_ok():
    r = BoothResult(answer="the answer", status=ACCEPTED)
    assert r.unwrap() == "the answer"


def test_unwrap_not_ok_raises():
    r = BoothResult(answer="bad", status=UNCERTAIN)
    with pytest.raises(BoothRejected) as exc_info:
        r.unwrap()
    assert exc_info.value.result is r


def test_unwrap_answer_none_raises():
    r = BoothResult(answer=None, status=ACCEPTED)
    with pytest.raises(BoothRejected):
        r.unwrap()


def test_unwrap_or():
    assert BoothResult(answer="A", status=ACCEPTED).unwrap_or("d") == "A"
    assert BoothResult(answer="A", status=UNCERTAIN).unwrap_or("d") == "d"
    assert BoothResult(answer=None, status=ACCEPTED).unwrap_or("d") == "d"


def test_booth_rejected_message_hides_answer():
    r = BoothResult(answer="secret_text", status=UNCERTAIN)
    exc = BoothRejected(r)
    assert "secret_text" not in str(exc)
    assert "status=UNCERTAIN" in str(exc)
    assert exc.result is r


def test_to_dict_structure():
    ok = Attempt(raw_text="raw", answer="A", confidence=0.9, parse_ok=True, parsed={"answer": "A"})
    r = BoothResult(answer="A", status=ACCEPTED, confidence=0.9, attempts=[ok])
    d = r.to_dict()
    assert d["answer"] == "A"
    assert d["status"] == ACCEPTED
    assert d["confidence"] == 0.9
    assert d["n_attempts"] == 1
    assert d["ok"] is True
    assert d["all_parse_failed"] is False
    assert d["call_failed"] is False
    assert d["method"] == "confidence"
    assert d["reason"] is None
    assert d["detail"] is None
    assert d["checker_failed"] is False
    assert d["attempts"][0]["answer"] == "A"
    assert d["attempts"][0]["parse_ok"] is True
    assert isinstance(d["attempts"], list)
    assert "n_attempts" not in d["attempts"][0]


# ---------------------------------------------------------------------------
# check_with_evidence
# ---------------------------------------------------------------------------

def test_evidence_empty_answer():
    def cmp(a, e):
        raise AssertionError("should not be called")

    r = check_with_evidence("", ["x"], cmp)
    assert r.status == UNCERTAIN
    assert r.answer is None
    assert r.reason == EMPTY_ANSWER
    assert r.confidence is None


def test_evidence_whitespace_answer():
    r = check_with_evidence("   ", ["x"], lambda a, e: True)
    assert r.status == UNCERTAIN
    assert r.reason == EMPTY_ANSWER


def test_evidence_no_evidence():
    r = check_with_evidence("ans", [], lambda a, e: True)
    assert r.status == UNCERTAIN
    assert r.reason == NO_EVIDENCE
    assert r.answer == "ans"


def test_evidence_compare_raises():
    def cmp(a, e):
        raise RuntimeError("boom")

    r = check_with_evidence("ans", ["e"], cmp)
    assert r.status == UNCERTAIN
    assert r.reason == COMPARE_FAILED
    assert r.answer == "ans"
    assert "RuntimeError" in r.detail
    assert "boom" in r.detail


def test_evidence_bool_true():
    r = check_with_evidence("ans", ["e"], lambda a, e: True)
    assert r.status == ACCEPTED
    assert r.confidence == 1.0
    assert r.evidence_agreement == 1.0
    assert r.reason is None
    assert r.detail is None


def test_evidence_bool_false():
    r = check_with_evidence("ans", ["e"], lambda a, e: False)
    assert r.status == BLOCKED
    assert r.confidence == 0.0
    assert r.evidence_agreement == 0.0
    assert r.reason == EVIDENCE_DISAGREES
    assert r.detail == "compare_fn returned False"


def test_evidence_float_above_threshold():
    r = check_with_evidence("ans", ["e"], lambda a, e: 0.85, evidence_threshold=0.7)
    assert r.status == ACCEPTED
    assert r.confidence == 0.85
    assert r.evidence_agreement == 0.85
    assert r.reason is None


def test_evidence_float_below_threshold():
    r = check_with_evidence("ans", ["e"], lambda a, e: 0.5, evidence_threshold=0.7)
    assert r.status == BLOCKED
    assert r.confidence == 0.5
    assert r.reason == EVIDENCE_DISAGREES
    assert "0.5" in r.detail
    assert "0.7" in r.detail


def test_evidence_float_exactly_threshold():
    r = check_with_evidence("ans", ["e"], lambda a, e: 0.7, evidence_threshold=0.7)
    assert r.status == ACCEPTED
    assert r.confidence == 0.7


def test_evidence_non_numeric_score():
    r = check_with_evidence("ans", ["e"], lambda a, e: "nope")
    assert r.status == UNCERTAIN
    assert r.reason == INVALID_SCORE
    assert r.confidence is None


def test_evidence_out_of_range_score():
    r = check_with_evidence("ans", ["e"], lambda a, e: 1.5)
    assert r.status == UNCERTAIN
    assert r.reason == INVALID_SCORE
    assert "1.5" in r.detail


def test_evidence_int_score():
    r = check_with_evidence("ans", ["e"], lambda a, e: 1)
    assert r.status == ACCEPTED
    assert r.confidence == 1.0


def test_evidence_answer_not_str():
    with pytest.raises(TypeError):
        check_with_evidence(123, ["e"], lambda a, e: True)


def test_evidence_threshold_bool_rejected():
    with pytest.raises(TypeError):
        check_with_evidence("ans", ["e"], lambda a, e: True, evidence_threshold=True)


def test_evidence_threshold_non_number():
    with pytest.raises(TypeError):
        check_with_evidence("ans", ["e"], lambda a, e: True, evidence_threshold="x")


def test_evidence_threshold_out_of_range():
    with pytest.raises(ValueError):
        check_with_evidence("ans", ["e"], lambda a, e: True, evidence_threshold=2.0)


def test_evidence_async_compare_fn():
    async def cmp(a, e):
        return True

    with pytest.raises(TypeError):
        check_with_evidence("ans", ["e"], cmp)


# ---------------------------------------------------------------------------
# check() argument validation
# ---------------------------------------------------------------------------

def test_check_prompt_not_str():
    with pytest.raises(TypeError):
        check(lambda p: "", 123)


def test_check_call_fn_not_callable():
    with pytest.raises(TypeError):
        check("notcallable", "prompt")


def test_check_threshold_out_of_range():
    with pytest.raises(ValueError):
        check(lambda p: "", "prompt", threshold=1.5)


def test_check_max_retries_not_int():
    with pytest.raises(TypeError):
        check(lambda p: "", "prompt", max_retries=1.5)


def test_check_max_retries_negative():
    with pytest.raises(ValueError):
        check(lambda p: "", "prompt", max_retries=-1)


def test_check_on_attempt_not_callable():
    with pytest.raises(TypeError):
        check(lambda p: "", "prompt", on_attempt="nope")


def test_check_async_call_fn_rejected():
    async def acall(p):
        return ""

    with pytest.raises(TypeError):
        check(acall, "prompt")


def test_check_async_on_attempt_rejected():
    async def cb(i, a):
        pass

    with pytest.raises(TypeError):
        check(lambda p: make_json(), "prompt", on_attempt=cb)


# ---------------------------------------------------------------------------
# check() behaviour
# ---------------------------------------------------------------------------

def test_check_accepted_first_try():
    fn = fake_call([make_json("A", 0.9)])
    r = check(fn, "prompt")
    assert r.status == ACCEPTED
    assert r.answer == "A"
    assert r.confidence == 0.9
    assert r.n_attempts == 1
    assert r.method == "confidence"
    assert len(fn.calls) == 1
    assert "confidence" in fn.calls[0]


def test_check_repaired_after_low_confidence():
    fn = fake_call([make_json("wrong", 0.2), make_json("right", 0.95)])
    r = check(fn, "prompt", threshold=0.7, max_retries=1)
    assert r.status == REPAIRED
    assert r.answer == "right"
    assert r.n_attempts == 2
    assert len(fn.calls) == 2
    # second prompt should reference prior attempt
    assert "wrong" in fn.calls[1]


def test_check_uncertain_after_exhausting_retries():
    fn = fake_call([make_json("a", 0.1), make_json("b", 0.2)])
    r = check(fn, "prompt", threshold=0.7, max_retries=1)
    assert r.status == UNCERTAIN
    assert r.answer == "b"  # last parsed attempt
    assert r.confidence == 0.2
    assert r.n_attempts == 2


def test_check_ambiguous_returns_immediately():
    payload = json.dumps({
        "ambiguous": True,
        "interpretations": ["x", "y"],
        "chosen_interpretation": "x",
        "answer": "A",
        "confidence": 0.9,
    })
    fn = fake_call([payload])
    r = check(fn, "prompt")
    assert r.status == AMBIGUOUS
    assert r.ambiguous is True
    assert r.interpretations == ["x", "y"]
    assert r.method == "ambiguity"
    assert r.n_attempts == 1


def test_check_call_fn_raises_then_succeeds():
    fn = fake_call([RuntimeError("boom"), make_json("A", 0.9)])
    r = check(fn, "prompt", max_retries=1)
    assert r.status == REPAIRED
    assert r.answer == "A"
    assert r.attempts[0].call_failed is True
    assert "boom" in r.attempts[0].error
    # retry prompt should be the plain prompt (no mention of prior)
    assert "Reconsider carefully" not in fn.calls[1]


def test_check_all_call_failures():
    fn = fake_call([RuntimeError("a"), RuntimeError("b")])
    r = check(fn, "prompt", max_retries=1)
    assert r.status == UNCERTAIN
    assert r.answer is None
    assert r.call_failed is True
    assert r.method == "parse_failure"
    assert r.all_parse_failed is True


def test_check_non_str_return_treated_as_call_failure():
    fn = fake_call([123, make_json("A", 0.9)])
    r = check(fn, "prompt", max_retries=1)
    assert r.status == REPAIRED
    assert r.attempts[0].call_failed is True
    assert "int" in r.attempts[0].error


def test_check_parse_failure_then_success():
    fn = fake_call(["garbage", make_json("A", 0.9)])
    r = check(fn, "prompt", max_retries=1)
    assert r.status == REPAIRED
    assert r.attempts[0].parse_ok is False
    assert "could not be parsed" in fn.calls[1]


def test_check_on_attempt_called_per_attempt():
    seen = []
    fn = fake_call([make_json("a", 0.1), make_json("b", 0.9)])

    def cb(i, attempt):
        seen.append((i, attempt.answer))

    r = check(fn, "prompt", max_retries=1, on_attempt=cb)
    assert r.status == REPAIRED
    assert seen == [(0, "a"), (1, "b")]


def test_check_validator_failure_triggers_retry():
    fn = fake_call([make_json("bad", 0.99), make_json("good", 0.99)])

    def validator(ans):
        return ans == "good"

    r = check(fn, "prompt", max_retries=1, validator=validator)
    assert r.status == REPAIRED
    assert r.answer == "good"
    assert r.attempts[0].passed_validation is False
    assert "validation" in fn.calls[1].lower()


def test_check_validator_fails_all_attempts():
    fn = fake_call([make_json("bad", 0.99)])

    def validator(ans):
        return False

    r = check(fn, "prompt", max_retries=0, validator=validator)
    assert r.status == UNCERTAIN
    assert r.attempts[-1].passed_validation is False
    assert r.method == "validation"


def test_check_validator_exception():
    fn = fake_call([make_json("bad", 0.99)])

    def validator(ans):
        raise ValueError("nope")

    r = check(fn, "prompt", max_retries=0, validator=validator)
    assert r.status == UNCERTAIN
    a = r.attempts[0]
    assert a.passed_validation is False
    assert "ValueError" in a.validation_error


# ---------------------------------------------------------------------------
# acheck()
# ---------------------------------------------------------------------------

def test_acheck_accepted():
    async def acall(prompt):
        return make_json("A", 0.9)

    r = run(acheck(acall, "prompt"))
    assert r.status == ACCEPTED
    assert r.answer == "A"


def test_acheck_non_async_fn_rejected():
    def sync_call(p):
        return ""

    with pytest.raises(TypeError):
        run(acheck(sync_call, "prompt"))


def test_acheck_repaired_with_async_on_attempt():
    calls = []

    async def acall(prompt):
        if len(calls) == 0:
            calls.append(prompt)
            return make_json("a", 0.1)
        calls.append(prompt)
        return make_json("b", 0.95)

    seen = []

    async def cb(i, attempt):
        seen.append((i, attempt.answer))

    r = run(acheck(acall, "prompt", max_retries=1, on_attempt=cb))
    assert r.status == REPAIRED
    assert r.answer == "b"
    assert seen == [(0, "a"), (1, "b")]


def test_acheck_sync_on_attempt():
    async def acall(prompt):
        return make_json("A", 0.9)

    seen = []

    def cb(i, attempt):
        seen.append((i, attempt.answer))

    r = run(acheck(acall, "prompt", on_attempt=cb))
    assert r.status == ACCEPTED
    assert seen == [(0, "A")]


def test_acheck_call_fn_raises():
    async def acall(prompt):
        raise RuntimeError("boom")

    r = run(acheck(acall, "prompt", max_retries=0))
    assert r.status == UNCERTAIN
    assert r.call_failed is True


def test_acheck_ambiguous():
    payload = json.dumps({
        "ambiguous": True,
        "interpretations": ["p", "q"],
        "chosen_interpretation": "p",
        "answer": "A",
        "confidence": 0.95,
    })

    async def acall(prompt):
        return payload

    r = run(acheck(acall, "prompt"))
    assert r.status == AMBIGUOUS
    assert r.interpretations == ["p", "q"]
