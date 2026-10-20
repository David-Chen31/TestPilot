import asyncio
import inspect
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
    DEFAULT_THRESHOLD,
    DEFAULT_MAX_RETRIES,
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

def make_response(answer="42", confidence=0.9, ambiguous=None, interpretations=None,
                  chosen=None):
    obj = {"answer": answer, "confidence": confidence}
    if ambiguous is not None:
        obj["ambiguous"] = ambiguous
    if interpretations is not None:
        obj["interpretations"] = interpretations
    if chosen is not None:
        obj["chosen_interpretation"] = chosen
    return json.dumps(obj)


def static_call_fn(text, n=None):
    """Returns a call_fn that always returns `text`."""
    calls = []

    def fn(prompt):
        calls.append(prompt)
        return text

    fn.calls = calls
    return fn


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Attempt
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

    def test_interpretations_independent(self):
        a1 = Attempt(raw_text="x", answer="a", confidence=0.5, parse_ok=True)
        a2 = Attempt(raw_text="x", answer="a", confidence=0.5, parse_ok=True)
        a1.interpretations.append("one")
        assert a2.interpretations == []


# ---------------------------------------------------------------------------
# BoothResult
# ---------------------------------------------------------------------------

class TestBoothResult:
    def test_n_attempts(self):
        r = BoothResult(answer="a", status=ACCEPTED)
        assert r.n_attempts == 0
        r2 = BoothResult(answer="a", status=ACCEPTED,
                         attempts=[Attempt("x", "a", 0.9, True)])
        assert r2.n_attempts == 1

    def test_ok_statuses(self):
        assert BoothResult(answer="a", status=ACCEPTED).ok is True
        assert BoothResult(answer="a", status=REPAIRED).ok is True
        for st in (AMBIGUOUS, BLOCKED, UNCERTAIN):
            assert BoothResult(answer="a", status=st).ok is False

    def test_all_parse_failed(self):
        assert BoothResult(answer=None, status=UNCERTAIN).all_parse_failed is False
        ok = Attempt("x", "a", 0.9, True)
        f1 = Attempt("x", None, None, False)
        assert BoothResult(answer=None, status=UNCERTAIN,
                           attempts=[f1]).all_parse_failed is True
        assert BoothResult(answer=None, status=UNCERTAIN,
                           attempts=[f1, ok]).all_parse_failed is False

    def test_call_failed(self):
        assert BoothResult(answer=None, status=UNCERTAIN).call_failed is False
        cf = Attempt("x", None, None, False, call_failed=True)
        ok = Attempt("x", "a", 0.9, True)
        assert BoothResult(answer=None, status=UNCERTAIN,
                           attempts=[cf]).call_failed is True
        assert BoothResult(answer=None, status=UNCERTAIN,
                           attempts=[cf, ok]).call_failed is False

    def test_checker_failed(self):
        assert BoothResult(answer="a", status=UNCERTAIN,
                           reason=COMPARE_FAILED).checker_failed is True
        assert BoothResult(answer="a", status=UNCERTAIN,
                           reason=INVALID_SCORE).checker_failed is True
        assert BoothResult(answer="a", status=UNCERTAIN,
                           reason=NO_EVIDENCE).checker_failed is False
        assert BoothResult(answer="a", status=ACCEPTED).checker_failed is False

    def test_method_ambiguous(self):
        r = BoothResult(answer="a", status=AMBIGUOUS)
        assert r.method == "ambiguity"

    def test_method_no_attempts(self):
        r = BoothResult(answer="a", status=UNCERTAIN, reason=EMPTY_ANSWER)
        assert r.method == "evidence"

    def test_method_parse_failure(self):
        f = Attempt("x", None, None, False)
        r = BoothResult(answer=None, status=UNCERTAIN, attempts=[f])
        assert r.method == "parse_failure"

    def test_method_validation(self):
        ok = Attempt("x", "a", 0.9, True, passed_validation=False)
        r = BoothResult(answer="a", status=UNCERTAIN, attempts=[ok])
        assert r.method == "validation"

    def test_method_confidence(self):
        ok = Attempt("x", "a", 0.5, True)
        r = BoothResult(answer="a", status=UNCERTAIN, attempts=[ok])
        assert r.method == "confidence"

    def test_unwrap_ok(self):
        assert BoothResult(answer="hello", status=ACCEPTED).unwrap() == "hello"
        assert BoothResult(answer="hello", status=REPAIRED).unwrap() == "hello"

    def test_unwrap_rejected(self):
        r = BoothResult(answer="bad", status=BLOCKED)
        with pytest.raises(BoothRejected) as exc_info:
            r.unwrap()
        assert exc_info.value.result is r

    def test_unwrap_none_answer(self):
        r = BoothResult(answer=None, status=ACCEPTED)
        with pytest.raises(BoothRejected):
            r.unwrap()

    def test_unwrap_or(self):
        assert BoothResult(answer="hello", status=ACCEPTED).unwrap_or("d") == "hello"
        assert BoothResult(answer="bad", status=BLOCKED).unwrap_or("d") == "d"

    def test_to_dict(self):
        a = Attempt(raw_text="raw", answer="a", confidence=0.9, parse_ok=True,
                    parsed={"answer": "a"})
        r = BoothResult(answer="a", status=ACCEPTED, confidence=0.9,
                        attempts=[a], parsed={"answer": "a"})
        d = r.to_dict()
        assert d["answer"] == "a"
        assert d["status"] == ACCEPTED
        assert d["confidence"] == 0.9
        assert d["n_attempts"] == 1
        assert d["ok"] is True
        assert d["method"] == "confidence"
        assert d["attempts"][0]["raw_text"] == "raw"
        assert d["attempts"][0]["answer"] == "a"
        assert d["all_parse_failed"] is False
        assert d["call_failed"] is False
        assert d["checker_failed"] is False
        assert d["reason"] is None
        assert d["detail"] is None
        json.dumps(d)  # must be JSON-serializable


class TestBoothRejected:
    def test_message_omits_answer(self):
        r = BoothResult(answer="SECRET", status=BLOCKED)
        exc = BoothRejected(r)
        assert "SECRET" not in str(exc)
        assert BLOCKED in str(exc)
        assert exc.result is r


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

class TestCheckSuccess:
    def test_single_accept(self):
        fn = static_call_fn(make_response("42", 0.95))
        r = check(fn, "What is the answer?")
        assert r.status == ACCEPTED
        assert r.answer == "42"
        assert r.confidence == 0.95
        assert r.method == "confidence"
        assert len(fn.calls) == 1
        assert fn.calls[0].startswith("What is the answer?")
        assert "confidence" in fn.calls[0]

    def test_prompt_stripped(self):
        fn = static_call_fn(make_response())
        check(fn, "prompt\n\n")
        assert fn.calls[0].startswith("prompt\n")

    def test_confidence_boundary_at_threshold(self):
        fn = static_call_fn(make_response("x", 0.7))
        r = check(fn, "p", threshold=0.7)
        assert r.status == ACCEPTED

    def test_low_confidence_retry_repaired(self):
        responses = [make_response("bad", 0.1), make_response("good", 0.9)]
        calls = []

        def fn(p):
            calls.append(p)
            return responses[len(calls) - 1]

        r = check(fn, "p", max_retries=1)
        assert r.status == REPAIRED
        assert r.answer == "good"
        assert len(calls) == 2
        # Retry prompt should mention previous answer
        assert "bad" in calls[1]

    def test_max_retries_zero_uncertain(self):
        fn = static_call_fn(make_response("x", 0.1))
        r = check(fn, "p", max_retries=0)
        assert r.status == UNCERTAIN
        assert r.answer == "x"
        assert r.confidence == 0.1
        assert len(fn.calls) == 1

    def test_all_low_confidence_uncertain(self):
        fn = static_call_fn(make_response("x", 0.1))
        r = check(fn, "p", max_retries=2)
        assert r.status == UNCERTAIN
        assert len(fn.calls) == 3
        assert r.answer == "x"


class TestCheckCallFnErrors:
    def test_call_fn_raises(self):
        def fn(p):
            raise RuntimeError("boom")

        r = check(fn, "p", max_retries=0)
        assert r.status == UNCERTAIN
        assert r.call_failed is True
        assert r.attempts[0].error == "boom"
        assert r.attempts[0].call_failed is True
        assert r.attempts[0].raw_text == "RuntimeError: boom"

    def test_call_fn_returns_non_str(self):
        def fn(p):
            return 123

        r = check(fn, "p", max_retries=0)
        assert r.status == UNCERTAIN
        assert r.attempts[0].call_failed is True
        assert "expected str" in r.attempts[0].error
        assert r.attempts[0].raw_text == "123"

    def test_call_fn_returns_none(self):
        def fn(p):
            return None

        r = check(fn, "p", max_retries=0)
        assert r.attempts[0].call_failed is True
        assert r.attempts[0].raw_text == "None"

    def test_call_fn_error_then_success(self):
        attempts = []

        def fn(p):
            attempts.append(p)
            if len(attempts) == 1:
                raise RuntimeError("temp")
            return make_response("ok", 0.95)

        r = check(fn, "p", max_retries=1)
        assert r.status == REPAIRED
        assert r.answer == "ok"
        # Retry prompt on call failure is just the plain prompt again
        assert "temp" not in attempts[1]


class TestCheckParseErrors:
    def test_garbage_response(self):
        fn = static_call_fn("not json at all")
        r = check(fn, "p", max_retries=0)
        assert r.status == UNCERTAIN
        assert r.all_parse_failed is True
        assert r.method == "parse_failure"
        assert r.answer is None

    def test_missing_keys(self):
        fn = static_call_fn('{"foo": "bar"}')
        r = check(fn, "p", max_retries=0)
        assert r.status == UNCERTAIN
        assert "missing" in r.attempts[0].error

    def test_confidence_out_of_range(self):
        fn = static_call_fn('{"answer": "a", "confidence": 2.0}')
        r = check(fn, "p", max_retries=0)
        assert r.attempts[0].parse_ok is False
        assert "range" in r.attempts[0].error

    def test_confidence_boolean_rejected(self):
        fn = static_call_fn('{"answer": "a", "confidence": true}')
        r = check(fn, "p", max_retries=0)
        assert r.attempts[0].parse_ok is False
        assert "boolean" in r.attempts[0].error

    def test_answer_dict_rejected(self):
        fn = static_call_fn('{"answer": {"x": 1}, "confidence": 0.9}')
        r = check(fn, "p", max_retries=0)
        assert r.attempts[0].parse_ok is False
        assert "dict/list" in r.attempts[0].error

    def test_answer_scalar_stringified(self):
        fn = static_call_fn('{"answer": 42, "confidence": 0.9}')
        r = check(fn, "p")
        assert r.status == ACCEPTED
        assert r.answer == "42"

    def test_empty_answer_rejected(self):
        fn = static_call_fn('{"answer": "   ", "confidence": 0.9}')
        r = check(fn, "p", max_retries=0)
        assert r.attempts[0].parse_ok is False
        assert "empty" in r.attempts[0].error

    def test_parse_failure_then_success(self):
        responses = ["nope", make_response("good", 0.95)]
        calls = []

        def fn(p):
            calls.append(p)
            return responses[len(calls) - 1]

        r = check(fn, "p", max_retries=1)
        assert r.status == REPAIRED
        assert r.answer == "good"
        assert "could not be parsed" in calls[1]

    def test_extracts_json_from_surrounding_text(self):
        fn = static_call_fn('Here you go: {"answer": "a", "confidence": 0.9} done')
        r = check(fn, "p")
        assert r.status == ACCEPTED
        assert r.answer == "a"

    def test_braces_inside_string(self):
        text = 'prefix {"answer": "a {weird} b", "confidence": 0.9}'
        fn = static_call_fn(text)
        r = check(fn, "p")
        assert r.status == ACCEPTED
        assert r.answer == "a {weird} b"


class TestCheckAmbiguity:
    def test_ambiguous_result(self):
        resp = make_response("paris", 0.9, ambiguous=True,
                             interpretations=["Paris, France", "Paris, Texas"],
                             chosen="Paris, France")
        fn = static_call_fn(resp)
        r = check(fn, "p")
        assert r.status == AMBIGUOUS
        assert r.ambiguous is True
        assert r.interpretations == ["Paris, France", "Paris, Texas"]
        assert r.answer == "paris"
        assert r.method == "ambiguity"
        assert len(fn.calls) == 1

    def test_ambiguous_string_true(self):
        resp = make_response("x", 0.9, ambiguous="TRUE")
        r = check(static_call_fn(resp), "p")
        assert r.status == AMBIGUOUS

    def test_ambiguous_string_false(self):
        resp = make_response("x", 0.9, ambiguous="false")
        r = check(static_call_fn(resp), "p")
        assert r.status == ACCEPTED

    def test_ambiguous_invalid_value(self):
        resp = '{"answer": "x", "confidence": 0.9, "ambiguous": "maybe"}'
        r = check(static_call_fn(resp), "p", max_retries=0)
        assert r.attempts[0].parse_ok is False
        assert "ambiguous" in r.attempts[0].error

    def test_blank_answer_allowed_when_ambiguous(self):
        resp = '{"answer": "", "confidence": 0.9, "ambiguous": true, "interpretations": ["a", "b"]}'
        r = check(static_call_fn(resp), "p")
        assert r.status == AMBIGUOUS

    def test_chosen_interpretation_falsy_kept(self):
        resp = '{"answer": "x", "confidence": 0.9, "ambiguous": true, "chosen_interpretation": 0}'
        r = check(static_call_fn(resp), "p")
        assert r.attempts[0].chosen_interpretation == "0"

    def test_interpretations_non_list_coerced(self):
        resp = '{"answer": "x", "confidence": 0.9, "ambiguous": true, "interpretations": "single"}'
        r = check(static_call_fn(resp), "p")
        assert r.attempts[0].interpretations == []


class TestCheckValidation:
    def test_validator_pass(self):
        fn = static_call_fn(make_response("ok", 0.9))
        r = check(fn, "p", validator=lambda a: True)
        assert r.status == ACCEPTED
        assert r.attempts[0].passed_validation is True

    def test_validator_fails_then_passes(self):
        responses = [make_response("bad", 0.9), make_response("good", 0.9)]
        calls = []

        def fn(p):
            calls.append(p)
            return responses[len(calls) - 1]

        r = check(fn, "p", validator=lambda a: a == "good", max_retries=1)
        assert r.status == REPAIRED
        assert r.answer == "good"
        assert "failed validation" in calls[1]

    def test_validator_tuple_false(self):
        fn = static_call_fn(make_response("x", 0.9))
        r = check(fn, "p", validator=lambda a: (False, "nope"), max_retries=0)
        assert r.status == UNCERTAIN
        assert r.method == "validation"
        assert r.attempts[0].passed_validation is False
        assert r.attempts[0].validation_error == "nope"

    def test_validator_tuple_false_none_message(self):
        fn = static_call_fn(make_response("x", 0.9))
        r = check(fn, "p", validator=lambda a: (False, None), max_retries=0)
        assert r.attempts[0].passed_validation is False
        assert "no message" in r.attempts[0].validation_error

    def test_validator_list_form(self):
        fn = static_call_fn(make_response("x", 0.9))
        r = check(fn, "p", validator=lambda a: [True, "ok"])
        assert r.status == ACCEPTED

    def test_validator_raises(self):
        def bad(a):
            raise ValueError("boom")

        fn = static_call_fn(make_response("x", 0.9))
        r = check(fn, "p", validator=bad, max_retries=0)
        assert r.attempts[0].passed_validation is False
        assert "ValueError: boom" in r.attempts[0].validation_error

    def test_validator_returns_invalid_type(self):
        fn = static_call_fn(make_response("x", 0.9))
        r = check(fn, "p", validator=lambda a: "yes", max_retries=0)
        assert r.attempts[0].passed_validation is False
        assert "invalid type" in r.attempts[0].validation_error

    def test_validator_returns_coroutine(self):
        async def coro_validator(a):
            return True

        fn = static_call_fn(make_response("x", 0.9))
        r = check(fn, "p", validator=coro_validator, max_retries=0)
        assert r.attempts[0].passed_validation is False
        assert "coroutine" in r.attempts[0].validation_error

    def test_validator_not_run_when_ambiguous(self):
        called = []

        def validator(a):
            called.append(a)
            return False

        resp = make_response("x", 0.9, ambiguous=True)
        r = check(static_call_fn(resp), "p", validator=validator)
        assert r.status == AMBIGUOUS
        assert called == []


class TestCheckArgumentValidation:
    def test_prompt_not_str(self):
        with pytest.raises(TypeError):
            check(lambda p: "x", 123)

    def test_call_fn_not_callable(self):
        with pytest.raises(TypeError):
            check("not callable", "p")

    def test_threshold_out_of_range(self):
        with pytest.raises(ValueError):
            check(lambda p: "x", "p", threshold=1.5)
        with pytest.raises(ValueError):
            check(lambda p: "x", "p", threshold=-0.1)

    def test_max_retries_not_int(self):
        with pytest.raises(TypeError):
            check(lambda p: "x", "p", max_retries=1.5)

    def test_max_retries_negative(self):
        with pytest.raises(ValueError):
            check(lambda p: "x", "p", max_retries=-1)

    def test_async_call_fn_rejected(self):
        async def fn(p):
            return "x"

        with pytest.raises(TypeError, match="acheck"):
            check(fn, "p")

    def test_async_callable_object_rejected(self):
        class AsyncCallable:
            async def __call__(self, p):
                return "x"

        with pytest.raises(TypeError):
            check(AsyncCallable(), "p")

    def test_on_attempt_not_callable(self):
        with pytest.raises(TypeError):
            check(lambda p: "x", "p", on_attempt=5)

    def test_async_on_attempt_rejected(self):
        async def cb(i, a):
            pass

        with pytest.raises(TypeError, match="async on_attempt"):
            check(lambda p: "x", "p", on_attempt=cb)


class TestCheckOnAttempt:
    def test_on_attempt_called(self):
        seen = []
        fn = static_call_fn(make_response("a", 0.9))
        check(fn, "p", on_attempt=lambda i, a: seen.append((i, a)))
        assert len(seen) == 1
        assert seen[0][0] == 0
        assert isinstance(seen[0][1], Attempt)
        assert seen[0][1].answer == "a"

    def test_on_attempt_multiple(self):
        responses = [make_response("a", 0.1), make_response("b", 0.9)]
        calls = []
        seen = []

        def fn(p):
            calls.append(p)
            return responses[len(calls) - 1]

        check(fn, "p", max_retries=1, on_attempt=lambda i, a: seen.append(i))
        assert seen == [0, 1]


# ---------------------------------------------------------------------------
# acheck
# ---------------------------------------------------------------------------

class TestAcheck:
    def test_accept(self):
        async def fn(p):
            return make_response("a", 0.9)

        r = run(acheck(fn, "p"))
        assert r.status == ACCEPTED
        assert r.answer == "a"

    def test_retry(self):
        responses = [make_response("a", 0.1), make_response("b", 0.9)]
        calls = []

        async def fn(p):
            calls.append(p)
            return responses[len(calls) - 1]

        r = run(acheck(fn, "p", max_retries=1))
        assert r.status == REPAIRED
        assert r.answer == "b"

    def test_async_call_fn_rejected_for_sync(self):
        with pytest.raises(TypeError, match="async call_fn"):
            run(acheck(lambda p: "x", "p"))

    def test_call_error(self):
        async def fn(p):
            raise RuntimeError("boom")

        r = run(acheck(fn, "p", max_retries=0))
        assert r.status == UNCERTAIN
        assert r.call_failed is True

    def test_sync_on_attempt(self):
        seen = []

        async def fn(p):
            return make_response("a", 0.9)

        r = run(acheck(fn, "p", on_attempt=lambda i, a: seen.append(i)))
        assert r.status == ACCEPTED
        assert seen == [0]

    def test_async_on_attempt(self):
        seen = []

        async def cb(i, a):
            seen.append(i)

        async def fn(p):
            return make_response("a", 0.9)

        r = run(acheck(fn, "p", on_attempt=cb))
        assert r.status == ACCEPTED
        assert seen == [0]

    def test_async_callable_object(self):
        class AsyncCallable:
            async def __call__(self, p):
                return make_response("a", 0.9)

        r = run(acheck(AsyncCallable(), "p"))
        assert r.status == ACCEPTED

    def test_argument_validation(self):
        async def fn(p):
            return "x"

        with pytest.raises(TypeError):
            run(acheck(fn, 123))
        with pytest.raises(ValueError):
            run(acheck(fn, "p", threshold=2.0))
        with pytest.raises(TypeError):
            run(acheck(fn, "p", max_retries="x"))


# ---------------------------------------------------------------------------
# check_with_evidence
# ---------------------------------------------------------------------------

class TestCheckWithEvidence:
    def test_accepted_above_threshold(self):
        r = check_with_evidence("a", ["e1", "e2"], lambda a, ev: 0.9)
        assert r.status == ACCEPTED
        assert r.confidence == 0.9
        assert r.evidence_agreement == 0.9
        assert r.reason is None
        assert r.detail is None
        assert r.method == "evidence"
        assert r.ok is True

    def test_blocked_below_threshold(self):
        r = check_with_evidence("a", ["e"], lambda a, ev: 0.2)
        assert r.status == BLOCKED
        assert r.reason == EVIDENCE_DISAGREES
        assert r.evidence_agreement == 0.2
        assert r.confidence == 0.2
        assert "below" in r.detail

    def test_threshold_boundary(self):
        r = check_with_evidence("a", ["e"], lambda a, ev: 0.7, evidence_threshold=0.7)
        assert r.status == ACCEPTED

    def test_bool_true(self):
        r = check_with_evidence("a", ["e"], lambda a, ev: True)
        assert r.status == ACCEPTED
        assert r.confidence == 1.0
        assert r.evidence_agreement == 1.0

    def test_bool_false(self):
        r = check_with_evidence("a", ["e"], lambda a, ev: False)
        assert r.status == BLOCKED
        assert r.confidence == 0.0
        assert r.reason == EVIDENCE_DISAGREES
        assert r.detail == "compare_fn returned False"

    def test_bool_ignores_threshold(self):
        r = check_with_evidence("a", ["e"], lambda a, ev: True, evidence_threshold=1.0)
        assert r.status == ACCEPTED

    def test_numeric_string_score(self):
        r = check_with_evidence("a", ["e"], lambda a, ev: "0.9")
        assert r.status == ACCEPTED
        assert r.confidence == 0.9

    def test_int_score(self):
        r = check_with_evidence("a", ["e"], lambda a, ev: 1)
        assert r.status == ACCEPTED
        assert r.confidence == 1.0

    def test_empty_answer(self):
        r = check_with_evidence("", ["e"], lambda a, ev: 1.0)
        assert r.status == UNCERTAIN
        assert r.reason == EMPTY_ANSWER
        assert r.answer is None

    def test_whitespace_answer(self):
        r = check_with_evidence("   ", ["e"], lambda a, ev: 1.0)
        assert r.status == UNCERTAIN
        assert r.reason == EMPTY_ANSWER

    def test_none_answer(self):
        r = check_with_evidence(None, ["e"], lambda a, ev: 1.0)
        assert r.status == UNCERTAIN
        assert r.reason == EMPTY_ANSWER

    def test_empty_evidence(self):
        r = check_with_evidence("a", [], lambda a, ev: 1.0)
        assert r.status == UNCERTAIN
        assert r.reason == NO_EVIDENCE
        assert r.answer == "a"

    def test_answer_not_str(self):
        with pytest.raises(TypeError):
            check_with_evidence(123, ["e"], lambda a, ev: 1.0)

    def test_compare_fn_raises(self):
        def fn(a, ev):
            raise ValueError("nope")

        r = check_with_evidence("a", ["e"], fn)
        assert r.status == UNCERTAIN
        assert r.reason == COMPARE_FAILED
        assert "ValueError" in r.detail
        assert r.checker_failed is True

    def test_compare_fn_detail_truncated(self):
        def fn(a, ev):
            raise ValueError("x" * 500)

        r = check_with_evidence("a", ["e"], fn)
        # detail is "ValueError: " + first 200 chars
        assert len(r.detail) == len("ValueError: ") + 200

    def test_async_compare_fn_raises(self):
        async def fn(a, ev):
            return 1.0

        with pytest.raises(TypeError, match="synchronous"):
            check_with_evidence("a", ["e"], fn)

    def test_non_numeric_score(self):
        r = check_with_evidence("a", ["e"], lambda a, ev: "abc")
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE
        assert "non-numeric" in r.detail
        assert r.checker_failed is True

    def test_none_score(self):
        r = check_with_evidence("a", ["e"], lambda a, ev: None)
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE

    def test_out_of_range_score(self):
        r = check_with_evidence("a", ["e"], lambda a, ev: 2.5)
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE
        assert "range" in r.detail

    def test_negative_score(self):
        r = check_with_evidence("a", ["e"], lambda a, ev: -0.5)
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE

    def test_evidence_threshold_not_real_number(self):
        with pytest.raises(TypeError):
            check_with_evidence("a", ["e"], lambda a, ev: 1.0, evidence_threshold=True)
        with pytest.raises(TypeError):
            check_with_evidence("a", ["e"], lambda a, ev: 1.0, evidence_threshold="0.5")

    def test_evidence_threshold_out_of_range(self):
        with pytest.raises(ValueError):
            check_with_evidence("a", ["e"], lambda a, ev: 1.0, evidence_threshold=2.0)
        with pytest.raises(ValueError):
            check_with_evidence("a", ["e"], lambda a, ev: 1.0, evidence_threshold=-0.1)

    def test_evidence_passed_to_compare_fn(self):
        received = {}

        def fn(a, ev):
            received["answer"] = a
            received["evidence"] = ev
            return 1.0

        check_with_evidence("answer", ["x", "y"], fn)
        assert received["answer"] == "answer"
        assert received["evidence"] == ["x", "y"]
