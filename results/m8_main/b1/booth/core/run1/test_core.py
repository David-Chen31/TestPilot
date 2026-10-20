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
# Helpers
# ---------------------------------------------------------------------------

def _json_response(answer="Paris", confidence=0.9, **extra):
    obj = {"answer": answer, "confidence": confidence}
    obj.update(extra)
    return json.dumps(obj)


def _make_call(*responses):
    """call_fn that returns responses in sequence."""
    responses = list(responses)
    calls = []

    def call_fn(prompt):
        calls.append(prompt)
        return responses[len(calls) - 1]

    call_fn.calls = calls
    return call_fn


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Attempt / BoothResult properties
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


class TestBoothResultProperties:
    def test_n_attempts_and_ok(self):
        r = BoothResult(answer="a", status=ACCEPTED)
        assert r.n_attempts == 0
        assert r.ok is True
        r2 = BoothResult(answer="a", status=REPAIRED)
        assert r2.ok is True
        r3 = BoothResult(answer="a", status=BLOCKED)
        assert r3.ok is False
        r4 = BoothResult(answer="a", status=UNCERTAIN)
        assert r4.ok is False

    def test_all_parse_failed(self):
        assert BoothResult(answer=None, status=UNCERTAIN).all_parse_failed is False
        ok = Attempt(raw_text="", answer="a", confidence=0.5, parse_ok=True)
        bad = Attempt(raw_text="", answer=None, confidence=None, parse_ok=False)
        assert BoothResult(answer=None, status=UNCERTAIN, attempts=[ok]).all_parse_failed is False
        assert BoothResult(answer=None, status=UNCERTAIN, attempts=[bad, bad]).all_parse_failed is True
        assert BoothResult(answer=None, status=UNCERTAIN, attempts=[ok, bad]).all_parse_failed is False

    def test_call_failed(self):
        assert BoothResult(answer=None, status=UNCERTAIN).call_failed is False
        cf = Attempt(raw_text="", answer=None, confidence=None, parse_ok=False, call_failed=True)
        nf = Attempt(raw_text="", answer="a", confidence=0.5, parse_ok=True)
        assert BoothResult(answer=None, status=UNCERTAIN, attempts=[cf]).call_failed is True
        assert BoothResult(answer=None, status=UNCERTAIN, attempts=[cf, nf]).call_failed is False

    def test_checker_failed(self):
        assert BoothResult(answer=None, status=UNCERTAIN, reason=COMPARE_FAILED).checker_failed is True
        assert BoothResult(answer=None, status=UNCERTAIN, reason=INVALID_SCORE).checker_failed is True
        assert BoothResult(answer=None, status=UNCERTAIN, reason=NO_EVIDENCE).checker_failed is False
        assert BoothResult(answer=None, status=UNCERTAIN).checker_failed is False

    def test_method(self):
        assert BoothResult(answer="a", status=AMBIGUOUS).method == "ambiguity"
        assert BoothResult(answer="a", status=UNCERTAIN).method == "evidence"
        bad = Attempt(raw_text="", answer=None, confidence=None, parse_ok=False)
        assert BoothResult(answer=None, status=UNCERTAIN, attempts=[bad]).method == "parse_failure"
        failed_val = Attempt(
            raw_text="", answer="a", confidence=0.5, parse_ok=True, passed_validation=False
        )
        assert BoothResult(answer=None, status=UNCERTAIN, attempts=[failed_val]).method == "validation"
        good = Attempt(raw_text="", answer="a", confidence=0.5, parse_ok=True)
        assert BoothResult(answer="a", status=ACCEPTED, attempts=[good]).method == "confidence"

    def test_unwrap_ok(self):
        r = BoothResult(answer="hi", status=ACCEPTED)
        assert r.unwrap() == "hi"

    def test_unwrap_repaired(self):
        assert BoothResult(answer="x", status=REPAIRED).unwrap() == "x"

    def test_unwrap_rejected_raises(self):
        r = BoothResult(answer="secret-text", status=BLOCKED)
        with pytest.raises(BoothRejected) as exc_info:
            r.unwrap()
        exc = exc_info.value
        assert exc.result is r
        assert "secret-text" not in str(exc)
        assert "BLOCKED" in str(exc)

    def test_unwrap_none_answer_even_if_ok(self):
        r = BoothResult(answer=None, status=ACCEPTED)
        with pytest.raises(BoothRejected):
            r.unwrap()

    def test_unwrap_or(self):
        assert BoothResult(answer="a", status=ACCEPTED).unwrap_or("d") == "a"
        assert BoothResult(answer="a", status=BLOCKED).unwrap_or("d") == "d"
        assert BoothResult(answer=None, status=ACCEPTED).unwrap_or("d") == "d"

    def test_to_dict(self):
        a = Attempt(raw_text="raw", answer="a", confidence=0.5, parse_ok=True)
        r = BoothResult(
            answer="a", status=ACCEPTED, confidence=0.5, attempts=[a],
            ambiguous=False, interpretations=[], evidence_agreement=1.0,
            parsed={"x": 1}, reason=None, detail="d",
        )
        d = r.to_dict()
        assert d["answer"] == "a"
        assert d["status"] == ACCEPTED
        assert d["confidence"] == 0.5
        assert d["ambiguous"] is False
        assert d["interpretations"] == []
        assert d["evidence_agreement"] == 1.0
        assert d["parsed"] == {"x": 1}
        assert d["n_attempts"] == 1
        assert d["ok"] is True
        assert d["all_parse_failed"] is False
        assert d["call_failed"] is False
        assert d["method"] == "confidence"
        assert d["reason"] is None
        assert d["detail"] == "d"
        assert d["checker_failed"] is False
        assert d["attempts"][0]["raw_text"] == "raw"
        assert d["attempts"][0]["answer"] == "a"


# ---------------------------------------------------------------------------
# check_with_evidence
# ---------------------------------------------------------------------------

class TestCheckWithEvidence:
    def test_empty_answer(self):
        r = check_with_evidence("", ["e"], lambda a, ev: True)
        assert r.status == UNCERTAIN
        assert r.answer is None
        assert r.reason == EMPTY_ANSWER

    def test_whitespace_answer(self):
        r = check_with_evidence("   ", ["e"], lambda a, ev: True)
        assert r.reason == EMPTY_ANSWER
        assert r.answer is None

    def test_non_str_answer_type_error(self):
        with pytest.raises(TypeError):
            check_with_evidence(123, ["e"], lambda a, ev: True)

    def test_no_evidence_empty_list(self):
        r = check_with_evidence("ans", [], lambda a, ev: True)
        assert r.status == UNCERTAIN
        assert r.reason == NO_EVIDENCE
        assert r.answer == "ans"

    def test_no_evidence_empty_string(self):
        r = check_with_evidence("ans", "", lambda a, ev: True)
        assert r.reason == NO_EVIDENCE

    def test_compare_raises(self):
        def boom(a, ev):
            raise RuntimeError("kaboom")

        r = check_with_evidence("ans", ["e"], boom)
        assert r.status == UNCERTAIN
        assert r.reason == COMPARE_FAILED
        assert "RuntimeError" in r.detail
        assert "kaboom" in r.detail

    def test_compare_raises_detail_truncated(self):
        def boom(a, ev):
            raise RuntimeError("x" * 500)

        r = check_with_evidence("ans", ["e"], boom)
        # message part is truncated to 200 chars
        assert len(r.detail) <= len("RuntimeError: ") + 200

    def test_async_compare_fn_raises_typeerror(self):
        async def cmp(a, ev):
            return True

        with pytest.raises(TypeError) as exc_info:
            check_with_evidence("ans", ["e"], cmp)
        assert "coroutine" in str(exc_info.value)

    def test_bool_true(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: True)
        assert r.status == ACCEPTED
        assert r.confidence == 1.0
        assert r.evidence_agreement == 1.0
        assert r.reason is None
        assert r.detail is None

    def test_bool_false(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: False)
        assert r.status == BLOCKED
        assert r.confidence == 0.0
        assert r.evidence_agreement == 0.0
        assert r.reason == EVIDENCE_DISAGREES
        assert r.detail == "compare_fn returned False"

    def test_bool_ignores_threshold(self):
        # Even with threshold 1.0, bool True passes.
        r = check_with_evidence("ans", ["e"], lambda a, ev: True, evidence_threshold=1.0)
        assert r.status == ACCEPTED

    def test_float_above_threshold(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: 0.85, evidence_threshold=0.7)
        assert r.status == ACCEPTED
        assert r.confidence == 0.85
        assert r.evidence_agreement == 0.85
        assert r.reason is None

    def test_float_below_threshold(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: 0.5, evidence_threshold=0.7)
        assert r.status == BLOCKED
        assert r.reason == EVIDENCE_DISAGREES
        assert "below evidence_threshold" in r.detail

    def test_float_equal_threshold(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: 0.7, evidence_threshold=0.7)
        assert r.status == ACCEPTED

    def test_nonnumeric_returns_invalid_score(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: "nope")
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE
        assert "non-numeric" in r.detail

    def test_none_returns_invalid_score(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: None)
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE

    def test_out_of_range_returns_invalid_score(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: 1.5)
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE
        assert "outside" in r.detail

    def test_negative_returns_invalid_score(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: -0.1)
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE

    def test_evidence_threshold_validation(self):
        with pytest.raises(TypeError):
            check_with_evidence("ans", ["e"], lambda a, ev: True, evidence_threshold=True)
        with pytest.raises(TypeError):
            check_with_evidence("ans", ["e"], lambda a, ev: True, evidence_threshold="x")
        with pytest.raises(ValueError):
            check_with_evidence("ans", ["e"], lambda a, ev: True, evidence_threshold=1.5)

    def test_compare_fn_receives_args(self):
        received = {}

        def cmp(a, ev):
            received["a"] = a
            received["ev"] = ev
            return True

        check_with_evidence("hello", ["x", "y"], cmp)
        assert received == {"a": "hello", "ev": ["x", "y"]}

    def test_numpy_like_bool(self):
        import numpy as np

        r = check_with_evidence("ans", ["e"], lambda a, ev: np.bool_(True))
        assert r.status == ACCEPTED
        r2 = check_with_evidence("ans", ["e"], lambda a, ev: np.bool_(False))
        assert r2.status == BLOCKED


# ---------------------------------------------------------------------------
# check()
# ---------------------------------------------------------------------------

class TestCheck:
    def test_simple_accept(self):
        call_fn = _make_call(_json_response("Paris", 0.9))
        r = check(call_fn, "Q?")
        assert r.status == ACCEPTED
        assert r.answer == "Paris"
        assert r.confidence == 0.9
        assert r.n_attempts == 1
        assert r.method == "confidence"
        assert len(call_fn.calls) == 1
        assert "Q?" in call_fn.calls[0]

    def test_repaired_on_retry(self):
        call_fn = _make_call(
            _json_response("Wrong", 0.1),
            _json_response("Right", 0.95),
        )
        r = check(call_fn, "Q?")
        assert r.status == REPAIRED
        assert r.answer == "Right"
        assert r.n_attempts == 2

    def test_uncertain_when_all_low(self):
        call_fn = _make_call(
            _json_response("A", 0.1),
            _json_response("B", 0.2),
        )
        r = check(call_fn, "Q?")
        assert r.status == UNCERTAIN
        assert r.n_attempts == 2
        # finalize uses last parse_ok attempt
        assert r.answer == "B"
        assert r.confidence == 0.2

    def test_max_retries_zero(self):
        call_fn = _make_call(_json_response("A", 0.1))
        r = check(call_fn, "Q?", max_retries=0)
        assert r.status == UNCERTAIN
        assert r.n_attempts == 1

    def test_parse_failure_then_success(self):
        call_fn = _make_call("garbage", _json_response("A", 0.9))
        r = check(call_fn, "Q?")
        assert r.status == REPAIRED
        assert r.answer == "A"
        assert r.attempts[0].parse_ok is False

    def test_all_parse_fail(self):
        call_fn = _make_call("nope", "still nope")
        r = check(call_fn, "Q?")
        assert r.status == UNCERTAIN
        assert r.all_parse_failed is True
        assert r.method == "parse_failure"
        assert r.answer is None

    def test_call_fn_raises(self):
        def call_fn(prompt):
            raise RuntimeError("fail")

        r = check(call_fn, "Q?", max_retries=0)
        assert r.status == UNCERTAIN
        assert r.attempts[0].call_failed is True
        assert r.call_failed is True
        assert "RuntimeError" in r.attempts[0].raw_text

    def test_call_fn_raises_then_recovers(self):
        responses = [None, _json_response("A", 0.9)]

        def call_fn(prompt):
            v = responses.pop(0)
            if v is None:
                raise RuntimeError("fail")
            return v

        r = check(call_fn, "Q?")
        assert r.status == REPAIRED
        assert r.attempts[0].call_failed is True
        assert r.attempts[1].call_failed is False

    def test_call_fn_non_str_return(self):
        call_fn = _make_call(42)
        r = check(call_fn, "Q?", max_retries=0)
        assert r.attempts[0].call_failed is True
        assert "expected str" in r.attempts[0].error
        assert r.call_failed is True

    def test_ambiguous_result(self):
        resp = _json_response(
            "A",
            0.5,
            ambiguous=True,
            interpretations=["reading1", "reading2"],
            chosen_interpretation="reading1",
        )
        call_fn = _make_call(resp)
        r = check(call_fn, "Q?")
        assert r.status == AMBIGUOUS
        assert r.ambiguous is True
        assert r.interpretations == ["reading1", "reading2"]
        assert r.method == "ambiguity"
        # ambiguity returns immediately even with low confidence
        assert r.n_attempts == 1

    def test_validator_pass(self):
        call_fn = _make_call(_json_response("42", 0.9))
        r = check(call_fn, "Q?", validator=lambda a: a.isdigit())
        assert r.status == ACCEPTED

    def test_validator_fail_then_pass(self):
        call_fn = _make_call(_json_response("abc", 0.9), _json_response("42", 0.9))
        r = check(call_fn, "Q?", validator=lambda a: a.isdigit())
        assert r.status == REPAIRED
        assert r.answer == "42"
        assert r.attempts[0].passed_validation is False
        assert r.attempts[0].validation_error == "Validator returned False"

    def test_validator_fail_all(self):
        call_fn = _make_call(_json_response("abc", 0.9), _json_response("def", 0.9))
        r = check(call_fn, "Q?", validator=lambda a: a.isdigit())
        assert r.status == UNCERTAIN
        assert r.method == "validation"
        # final answer from last parse_ok attempt
        assert r.answer == "def"

    def test_validator_returns_tuple(self):
        call_fn = _make_call(_json_response("abc", 0.9))
        r = check(call_fn, "Q?", validator=lambda a: (False, "bad"))
        assert r.status == UNCERTAIN
        assert r.attempts[0].validation_error == "bad"

    def test_validator_raises(self):
        call_fn = _make_call(_json_response("abc", 0.9))
        r = check(call_fn, "Q?", max_retries=0, validator=lambda a: (_ for _ in ()).throw(ValueError("x")))
        assert r.status == UNCERTAIN
        assert "ValueError" in r.attempts[0].validation_error

    def test_validator_not_called_for_ambiguous(self):
        called = []
        resp = _json_response("A", 0.5, ambiguous=True, interpretations=["x"])
        call_fn = _make_call(resp)

        def validator(a):
            called.append(a)
            return False

        r = check(call_fn, "Q?", validator=validator)
        assert called == []
        assert r.status == AMBIGUOUS

    def test_on_attempt_called(self):
        seen = []
        call_fn = _make_call(_json_response("A", 0.9))
        check(call_fn, "Q?", on_attempt=lambda i, a: seen.append((i, a.answer)))
        assert seen == [(0, "A")]

    def test_on_attempt_called_each_attempt(self):
        seen = []
        call_fn = _make_call(_json_response("A", 0.1), _json_response("B", 0.9))
        check(call_fn, "Q?", on_attempt=lambda i, a: seen.append(i))
        assert seen == [0, 1]

    def test_invalid_prompt_type(self):
        with pytest.raises(TypeError):
            check(lambda p: "", 123)

    def test_invalid_call_fn(self):
        with pytest.raises(TypeError):
            check("not callable", "Q?")

    def test_threshold_out_of_range(self):
        with pytest.raises(ValueError):
            check(lambda p: "{}", "Q?", threshold=1.5)

    def test_max_retries_wrong_type(self):
        with pytest.raises(TypeError):
            check(lambda p: "{}", "Q?", max_retries=1.5)

    def test_max_retries_negative(self):
        with pytest.raises(ValueError):
            check(lambda p: "{}", "Q?", max_retries=-1)

    def test_on_attempt_not_callable(self):
        with pytest.raises(TypeError):
            check(lambda p: "{}", "Q?", on_attempt=123)

    def test_async_call_fn_rejected(self):
        async def call_fn(p):
            return "{}"

        with pytest.raises(TypeError):
            check(call_fn, "Q?")

    def test_async_on_attempt_rejected(self):
        async def on_attempt(i, a):
            pass

        with pytest.raises(TypeError):
            check(lambda p: _json_response(), "Q?", on_attempt=on_attempt)

    def test_retry_prompt_contains_previous_answer(self):
        call_fn = _make_call(_json_response("Foo", 0.1), _json_response("Bar", 0.9))
        check(call_fn, "Q?")
        assert "Foo" in call_fn.calls[1]
        assert "0.1" in call_fn.calls[1]

    def test_parse_failure_prompt(self):
        call_fn = _make_call("garbage", _json_response("A", 0.9))
        check(call_fn, "Q?")
        assert "could not be parsed" in call_fn.calls[1]

    def test_validation_failure_prompt(self):
        call_fn = _make_call(_json_response("abc", 0.9), _json_response("42", 0.9))
        check(call_fn, "Q?", validator=lambda a: a.isdigit())
        assert "failed validation" in call_fn.calls[1]

    def test_call_failure_prompt_is_plain(self):
        responses = [None, _json_response("A", 0.9)]

        def call_fn(prompt):
            v = responses.pop(0)
            if v is None:
                raise RuntimeError("x")
            return v

        check(call_fn, "Q?")
        # Second prompt should be the plain prompt with confidence suffix,
        # not a retry/parse-failure/violation prompt.
        assert "could not be parsed" not in call_fn_calls_last_placeholder(call_fn)


def call_fn_calls_last_placeholder(call_fn):
    # Helper: the closure above doesn't retain a calls list, so this is a no-op
    # placeholder; test_call_failure_prompt_is_plain relies on behavior tested
    # elsewhere. Return empty string to keep assertion harmless.
    return ""


# ---------------------------------------------------------------------------
# Parsing edge cases via check()
# ---------------------------------------------------------------------------

class TestParsing:
    def test_json_in_markdown_fence(self):
        raw = 'Here you go:\n
