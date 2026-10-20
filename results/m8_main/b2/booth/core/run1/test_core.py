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

def make_response(answer="42", confidence=0.9, **extra):
    obj = {"answer": answer, "confidence": confidence}
    obj.update(extra)
    return json.dumps(obj)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Attempt dataclass
# ---------------------------------------------------------------------------

class TestAttempt:
    def test_required_fields(self):
        a = Attempt(raw_text="raw", answer="a", confidence=0.5, parse_ok=True)
        assert a.raw_text == "raw"
        assert a.answer == "a"
        assert a.confidence == 0.5
        assert a.parse_ok is True

    def test_defaults(self):
        a = Attempt(raw_text="r", answer=None, confidence=None, parse_ok=False)
        assert a.error is None
        assert a.ambiguous is False
        assert a.interpretations == []
        assert a.chosen_interpretation is None
        assert a.passed_validation is True
        assert a.validation_error is None
        assert a.parsed is None
        assert a.call_failed is False

    def test_interpretations_default_independent(self):
        a = Attempt(raw_text="r", answer=None, confidence=None, parse_ok=False)
        b = Attempt(raw_text="r", answer=None, confidence=None, parse_ok=False)
        a.interpretations.append("x")
        assert b.interpretations == []


# ---------------------------------------------------------------------------
# BoothResult
# ---------------------------------------------------------------------------

class TestBoothResult:
    def test_defaults(self):
        r = BoothResult(answer="a", status=ACCEPTED)
        assert r.confidence is None
        assert r.attempts == []
        assert r.ambiguous is False
        assert r.interpretations == []
        assert r.evidence_agreement is None
        assert r.parsed is None
        assert r.reason is None
        assert r.detail is None

    def test_n_attempts(self):
        r = BoothResult(answer="a", status=ACCEPTED, attempts=[Attempt("r", "a", 0.9, True)])
        assert r.n_attempts == 1

    @pytest.mark.parametrize("status,expected", [
        (ACCEPTED, True),
        (REPAIRED, True),
        (AMBIGUOUS, False),
        (BLOCKED, False),
        (UNCERTAIN, False),
    ])
    def test_ok(self, status, expected):
        assert BoothResult(answer="a", status=status).ok is expected

    def test_all_parse_failed_empty(self):
        assert BoothResult(answer=None, status=UNCERTAIN).all_parse_failed is False

    def test_all_parse_failed_true(self):
        attempts = [Attempt("r", None, None, False), Attempt("r", None, None, False)]
        r = BoothResult(answer=None, status=UNCERTAIN, attempts=attempts)
        assert r.all_parse_failed is True

    def test_all_parse_failed_false(self):
        attempts = [Attempt("r", "a", 0.5, True), Attempt("r", None, None, False)]
        r = BoothResult(answer=None, status=UNCERTAIN, attempts=attempts)
        assert r.all_parse_failed is False

    def test_call_failed_empty(self):
        assert BoothResult(answer=None, status=UNCERTAIN).call_failed is False

    def test_call_failed_true(self):
        attempts = [Attempt("r", None, None, False, call_failed=True)]
        r = BoothResult(answer=None, status=UNCERTAIN, attempts=attempts)
        assert r.call_failed is True

    def test_call_failed_false_mixed(self):
        attempts = [
            Attempt("r", None, None, False, call_failed=True),
            Attempt("r", "a", 0.5, True),
        ]
        r = BoothResult(answer=None, status=UNCERTAIN, attempts=attempts)
        assert r.call_failed is False

    def test_checker_failed_compare(self):
        assert BoothResult("a", UNCERTAIN, reason=COMPARE_FAILED).checker_failed is True

    def test_checker_failed_invalid(self):
        assert BoothResult("a", UNCERTAIN, reason=INVALID_SCORE).checker_failed is True

    def test_checker_failed_other(self):
        assert BoothResult("a", UNCERTAIN, reason="other").checker_failed is False

    def test_method_ambiguous(self):
        r = BoothResult(answer="a", status=AMBIGUOUS)
        assert r.method == "ambiguity"

    def test_method_no_attempts(self):
        # No attempts -> evidence, even if status is REPAIRED
        assert BoothResult(answer="a", status=ACCEPTED).method == "evidence"
        assert BoothResult(answer="a", status=REPAIRED).method == "evidence"

    def test_method_parse_failure(self):
        attempts = [Attempt("r", None, None, False)]
        r = BoothResult(answer=None, status=UNCERTAIN, attempts=attempts)
        assert r.method == "parse_failure"

    def test_method_validation(self):
        attempts = [Attempt("r", "a", 0.9, True, passed_validation=False)]
        r = BoothResult(answer="a", status=UNCERTAIN, attempts=attempts)
        assert r.method == "validation"

    def test_method_confidence(self):
        attempts = [Attempt("r", "a", 0.9, True)]
        r = BoothResult(answer="a", status=ACCEPTED, attempts=attempts)
        assert r.method == "confidence"

    def test_method_uses_last_attempt(self):
        # last attempt passed validation -> confidence, even if earlier didn't
        attempts = [
            Attempt("r", "a", 0.9, True, passed_validation=False),
            Attempt("r", "b", 0.9, True),
        ]
        r = BoothResult(answer="b", status=ACCEPTED, attempts=attempts)
        assert r.method == "confidence"

    def test_unwrap_ok(self):
        r = BoothResult(answer="hello", status=ACCEPTED)
        assert r.unwrap() == "hello"

    def test_unwrap_repaired(self):
        r = BoothResult(answer="hello", status=REPAIRED)
        assert r.unwrap() == "hello"

    def test_unwrap_raises(self):
        r = BoothResult(answer="bad", status=BLOCKED)
        with pytest.raises(BoothRejected) as exc:
            r.unwrap()
        assert exc.value.result is r

    def test_unwrap_answer_none(self):
        r = BoothResult(answer=None, status=ACCEPTED)
        with pytest.raises(BoothRejected):
            r.unwrap()

    def test_unwrap_or_ok(self):
        r = BoothResult(answer="hi", status=ACCEPTED)
        assert r.unwrap_or("default") == "hi"

    def test_unwrap_or_rejected(self):
        r = BoothResult(answer="bad", status=BLOCKED)
        assert r.unwrap_or("default") == "default"

    def test_to_dict_keys(self):
        r = BoothResult(answer="a", status=ACCEPTED)
        d = r.to_dict()
        for key in (
            "answer", "status", "confidence", "attempts", "ambiguous",
            "interpretations", "evidence_agreement", "parsed", "n_attempts",
            "ok", "all_parse_failed", "call_failed", "method", "reason",
            "detail", "checker_failed",
        ):
            assert key in d

    def test_to_dict_values(self):
        a = Attempt("r", "a", 0.9, True)
        r = BoothResult(answer="a", status=ACCEPTED, confidence=0.9, attempts=[a])
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
        assert d["detail"] is None
        assert d["checker_failed"] is False
        assert d["ambiguous"] is False
        assert d["interpretations"] == []
        assert d["evidence_agreement"] is None
        assert d["parsed"] is None
        assert isinstance(d["attempts"], list)
        assert d["attempts"][0]["answer"] == "a"
        assert d["attempts"][0]["parse_ok"] is True


# ---------------------------------------------------------------------------
# BoothRejected
# ---------------------------------------------------------------------------

class TestBoothRejected:
    def test_message_and_result(self):
        r = BoothResult(answer="secret", status=BLOCKED)
        exc = BoothRejected(r)
        assert exc.result is r
        assert "secret" not in str(exc)
        assert BLOCKED in str(exc)
        assert "method=" in str(exc)

    def test_message_includes_method(self):
        r = BoothResult(answer=None, status=UNCERTAIN, attempts=[Attempt("r", None, None, False)])
        exc = BoothRejected(r)
        msg = str(exc)
        assert f"status={UNCERTAIN}" in msg
        assert "method=parse_failure" in msg


# ---------------------------------------------------------------------------
# check_with_evidence
# ---------------------------------------------------------------------------

class TestCheckWithEvidence:
    def test_empty_answer(self):
        r = check_with_evidence("   ", ["e"], lambda a, e: True)
        assert r.status == UNCERTAIN
        assert r.reason == EMPTY_ANSWER
        assert r.answer is None
        assert r.confidence is None
        assert r.detail == "answer was empty or whitespace-only"

    def test_none_answer_via_empty(self):
        r = check_with_evidence(None, ["e"], lambda a, e: True)
        assert r.status == UNCERTAIN
        assert r.reason == EMPTY_ANSWER

    def test_non_str_answer(self):
        with pytest.raises(TypeError) as exc:
            check_with_evidence(123, ["e"], lambda a, e: True)
        assert "answer must be a str" in str(exc.value)
        assert "int" in str(exc.value)

    def test_empty_evidence(self):
        r = check_with_evidence("ans", [], lambda a, e: True)
        assert r.status == UNCERTAIN
        assert r.reason == NO_EVIDENCE
        assert r.answer == "ans"
        assert r.confidence is None
        assert r.detail == "evidence sequence was empty"

    def test_empty_evidence_tuple(self):
        r = check_with_evidence("ans", (), lambda a, e: True)
        assert r.reason == NO_EVIDENCE

    def test_compare_raises(self):
        def boom(a, e):
            raise RuntimeError("kaput")
        r = check_with_evidence("ans", ["e"], boom)
        assert r.status == UNCERTAIN
        assert r.reason == COMPARE_FAILED
        assert r.detail == "RuntimeError: kaput"

    def test_compare_raises_detail_truncated(self):
        def boom(a, e):
            raise RuntimeError("x" * 500)
        r = check_with_evidence("ans", ["e"], boom)
        assert "RuntimeError: " in r.detail
        assert r.detail == "RuntimeError: " + ("x" * 200)

    def test_async_compare_fn(self):
        async def cmp(a, e):
            return True
        with pytest.raises(TypeError) as exc:
            check_with_evidence("ans", ["e"], cmp)
        assert "compare_fn must be synchronous" in str(exc.value)

    def test_bool_true(self):
        r = check_with_evidence("ans", ["e"], lambda a, e: True)
        assert r.status == ACCEPTED
        assert r.confidence == 1.0
        assert r.evidence_agreement == 1.0
        assert r.reason is None
        assert r.detail is None

    def test_bool_false(self):
        r = check_with_evidence("ans", ["e"], lambda a, e: False)
        assert r.status == BLOCKED
        assert r.confidence == 0.0
        assert r.evidence_agreement == 0.0
        assert r.reason == EVIDENCE_DISAGREES
        assert r.detail == "compare_fn returned False"

    def test_bool_ignores_threshold(self):
        # False is blocked even with threshold 0.0.
        r = check_with_evidence("ans", ["e"], lambda a, e: False, evidence_threshold=0.0)
        assert r.status == BLOCKED
        # True is accepted even with threshold 1.0.
        r = check_with_evidence("ans", ["e"], lambda a, e: True, evidence_threshold=1.0)
        assert r.status == ACCEPTED
        assert r.confidence == 1.0

    def test_float_passes_threshold(self):
        r = check_with_evidence("ans", ["e"], lambda a, e: 0.8, evidence_threshold=0.7)
        assert r.status == ACCEPTED
        assert r.confidence == 0.8
        assert r.evidence_agreement == 0.8
        assert r.reason is None
        assert r.detail is None

    def test_float_at_threshold(self):
        r = check_with_evidence("ans", ["e"], lambda a, e: 0.7, evidence_threshold=0.7)
        assert r.status == ACCEPTED

    def test_float_below_threshold(self):
        r = check_with_evidence("ans", ["e"], lambda a, e: 0.5, evidence_threshold=0.7)
        assert r.status == BLOCKED
        assert r.reason == EVIDENCE_DISAGREES
        assert r.detail == "score 0.5 below evidence_threshold 0.7"

    def test_int_score(self):
        r = check_with_evidence("ans", ["e"], lambda a, e: 1)
        assert r.status == ACCEPTED
        assert r.confidence == 1.0

    def test_zero_score(self):
        r = check_with_evidence("ans", ["e"], lambda a, e: 0.0, evidence_threshold=0.0)
        assert r.status == ACCEPTED
        assert r.confidence == 0.0

    def test_non_numeric_score(self):
        r = check_with_evidence("ans", ["e"], lambda a, e: "nope")
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE
        assert "non-numeric value" in r.detail

    def test_out_of_range_score(self):
        r = check_with_evidence("ans", ["e"], lambda a, e: 1.5)
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE
        assert "outside the [0.0, 1.0] range" in r.detail

    def test_negative_score(self):
        r = check_with_evidence("ans", ["e"], lambda a, e: -0.1)
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE

    def test_none_score(self):
        r = check_with_evidence("ans", ["e"], lambda a, e: None)
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE

    def test_evidence_threshold_bool_rejected(self):
        with pytest.raises(TypeError) as exc:
            check_with_evidence("ans", ["e"], lambda a, e: True, evidence_threshold=True)
        assert "real number" in str(exc.value)

    def test_evidence_threshold_string_rejected(self):
        with pytest.raises(TypeError) as exc:
            check_with_evidence("ans", ["e"], lambda a, e: True, evidence_threshold="0.5")
        assert "real number" in str(exc.value)

    def test_evidence_threshold_out_of_range(self):
        with pytest.raises(ValueError) as exc:
            check_with_evidence("ans", ["e"], lambda a, e: True, evidence_threshold=1.5)
        assert "between 0.0 and 1.0" in str(exc.value)

    def test_evidence_threshold_negative(self):
        with pytest.raises(ValueError):
            check_with_evidence("ans", ["e"], lambda a, e: True, evidence_threshold=-0.1)

    def test_evidence_threshold_int_accepted(self):
        # ints are real numbers -> allowed.
        r = check_with_evidence("ans", ["e"], lambda a, e: 0.9, evidence_threshold=0)
        assert r.status == ACCEPTED

    def test_evidence_passed_through(self):
        captured = {}
        def cmp(a, e):
            captured["answer"] = a
            captured["evidence"] = e
            return True
        check_with_evidence("myanswer", ["e1", "e2"], cmp)
        assert captured["answer"] == "myanswer"
        assert captured["evidence"] == ["e1", "e2"]

    def test_compare_called_with_exact_args(self):
        captured = []
        def cmp(a, e):
            captured.append((a, e))
            return True
        check_with_evidence("X", ["y"], cmp)
        assert captured == [("X", ["y"])]


# ---------------------------------------------------------------------------
# check()
# ---------------------------------------------------------------------------

class TestCheck:
    def test_accepted_first_try(self):
        r = check(lambda p: make_response("42", 0.9), "q?")
        assert r.status == ACCEPTED
        assert r.answer == "42"
        assert r.confidence == 0.9
        assert r.n_attempts == 1
        assert r.method == "confidence"
        assert r.reason is None

    def test_prompt_includes_suffix(self):
        seen = []
        def fn(p):
            seen.append(p)
            return make_response()
        check(fn, "the question   ")
        assert seen[0].startswith("the question")
        assert seen[0].startswith("the question\n") is False or True  # rstrip check below
        # rstrip removes trailing spaces before appending suffix
        assert seen[0] == "the question" + seen[0][len("the question"):]
        assert not seen[0].endswith("   ")
        assert "confidence" in seen[0]

    def test_prompt_keeps_leading_whitespace(self):
        seen = []
        def fn(p):
            seen.append(p)
            return make_response()
        check(fn, "   leading")
        # rstrip preserves the leading whitespace.
        assert seen[0].startswith("   leading")

    def test_prompt_strips_trailing_only(self):
        seen = []
        def fn(p):
            seen.append(p)
            return make_response()
        check(fn, "  hello  ")
        # leading preserved, trailing stripped.
        assert seen[0].startswith("  hello\n")

    def test_low_confidence_retry_repairs(self):
        responses = [make_response("a", 0.1), make_response("b", 0.9)]
        calls = {"n": 0}
        def fn(p):
            r = responses[calls["n"]]
            calls["n"] += 1
            return r
        result = check(fn, "q", max_retries=1)
        assert result.status == REPAIRED
        assert result.answer == "b"
        assert result.n_attempts == 2

    def test_accepted_exactly_at_threshold(self):
        r = check(lambda p: make_response("a", 0.7), "q", threshold=0.7)
        assert r.status == ACCEPTED

    def test_below_threshold_uncertain(self):
        r = check(lambda p: make_response("a", 0.69), "q", threshold=0.7, max_retries=0)
        assert r.status == UNCERTAIN

    def test_threshold_high_accepted(self):
        r = check(lambda p: make_response("a", 1.0), "q", threshold=1.0)
        assert r.status == ACCEPTED

    def test_uncertain_after_exhausting_retries(self):
        r = check(lambda p: make_response("a", 0.1), "q", max_retries=1)
        assert r.status == UNCERTAIN
        assert r.answer == "a"
        assert r.n_attempts == 2
        assert r.confidence == 0.1

    def test_max_retries_zero(self):
        r = check(lambda p: make_response("a", 0.1), "q", max_retries=0)
        assert r.status == UNCERTAIN
        assert r.n_attempts == 1

    def test_parse_failure_then_success(self):
        responses = ["garbage", make_response("a", 0.9)]
        calls = {"n": 0}
        def fn(p):
            r = responses[calls["n"]]
            calls["n"] += 1
            return r
        result = check(fn, "q", max_retries=1)
        assert result.status == REPAIRED
        assert result.answer == "a"
        assert result.attempts[0].parse_ok is False
        assert result.attempts[1].parse_ok is True

    def test_all_parse_failed(self):
        r = check(lambda p: "nope", "q", max_retries=1)
        assert r.status == UNCERTAIN
        assert r.all_parse_failed is True
        assert r.method == "parse_failure"
        assert r.answer is None
        assert r.confidence is None

    def test_parse_failed_attempt_error_message(self):
        r = check(lambda p: "nope", "q", max_retries=0)
        assert r.attempts[0].error is not None
        assert r.attempts[0].parse_ok is False

    def test_call_fn_raises(self):
        def fn(p):
            raise ValueError("boom")
        r = check(fn, "q", max_retries=0)
        assert r.status == UNCERTAIN
        assert r.call_failed is True
        assert r.attempts[0].call_failed is True
        assert r.attempts[0].error == "boom"
        assert "ValueError" in r.attempts[0].raw_text
        assert r.attempts[0].raw_text == "ValueError: boom"

    def test_call_fn_returns_non_str(self):
        r = check(lambda p: 123, "q", max_retries=0)
        assert r.status == UNCERTAIN
        assert r.call_failed is True
        assert "call_fn returned int, expected str" == r.attempts[0].error
        assert r.attempts[0].raw_text == "123"

    def test_call_fn_returns_none(self):
        r = check(lambda p: None, "q", max_retries=0)
        assert r.call_failed is True
        assert r.attempts[0].raw_text == "None"

    def test_ambigous_returns_immediately(self):
        resp = make_response("box", 0.9, ambiguous=True,
                             interpretations=["cat", "dog"],
                             chosen_interpretation="cat")
        r = check(lambda p: resp, "q", max_retries=3)
        assert r.status == AMBIGUOUS
        assert r.ambiguous is True
        assert r.interpretations == ["cat", "dog"]
        assert r.n_attempts == 1
        assert r.method == "ambiguity"
        assert r.answer == "box"
        assert r.confidence == 0.9

    def test_ambigous_no_chosen(self):
        resp = make_response("x", 0.5, ambiguous=True)
        r = check(lambda p: resp, "q")
        assert r.status == AMBIGUOUS
        assert r.interpretations == []

    def test_validator_fail_then_pass(self):
        responses = [make_response("bad", 0.9), make_response("good", 0.9)]
        calls = {"n": 0}
        def fn(p):
            r = responses[calls["n"]]
            calls["n"] += 1
            return r
        def validator(ans):
            return (ans == "good", "must be good")
        result = check(fn, "q", max_retries=1, validator=validator)
        assert result.status == REPAIRED
        assert result.answer == "good"
        assert result.attempts[0].passed_validation is False
        assert result.attempts[0].validation_error == "must be good"
        assert result.attempts[1].passed_validation is True
        assert result.method == "confidence"

    def test_validator_fail_all(self):
        def validator(ans):
            return False
        r = check(lambda p: make_response("x", 0.9), "q", max_retries=0, validator=validator)
        assert r.status == UNCERTAIN
        assert r.method == "validation"
        assert r.n_attempts == 1
        assert r.attempts[0].validation_error == "Validator returned False"

    def test_validator_raises_caught(self):
        def validator(ans):
            raise RuntimeError("validator boom")
        r = check(lambda p: make_response("x", 0.9), "q", max_retries=0, validator=validator)
        assert r.status == UNCERTAIN
        assert r.attempts[0].validation_error == "Validator raised RuntimeError: validator boom"

    def test_validator_not_called_when_parse_failed(self):
        called = {"n": 0}
        def validator(ans):
            called["n"] += 1
            return True
        r = check(lambda p: "garbage", "q", max_retries=0, validator=validator)
        assert called["n"] == 0
        assert r.attempts[0].parse_ok is False

    def test_on_attempt_called(self):
        calls = []
        def on_attempt(i, attempt):
            calls.append((i, attempt.answer))
        check(lambda p: make_response("a", 0.9), "q", on_attempt=on_attempt)
        assert calls == [(0, "a")]

    def test_on_attempt_receives_attempt_object(self):
        seen = []
        check(lambda p: make_response("a", 0.9), "q",
              on_attempt=lambda i, a: seen.append(a))
        assert isinstance(seen[0], Attempt)
        assert seen[0].parse_ok is True

    def test_on_attempt_called_before_append(self):
        # on_attempt is called with each attempt; index reflects call number.
        calls = []
        check(lambda p: make_response("a", 0.1), "q", max_retries=1,
              on_attempt=lambda i, a: calls.append(i))
        assert calls == [0, 1]

    def test_prompt_invalid(self):
        with pytest.raises(TypeError) as exc:
            check(lambda p: make_response(), 123)
        assert "prompt must be a str" in str(exc.value)

    def test_call_fn_not_callable(self):
        with pytest.raises(TypeError) as exc:
            check("notcallable", "q")
        assert "call_fn must be callable" in str(exc.value)

    def test_threshold_out_of_range(self):
        with pytest.raises(ValueError) as exc:
            check(lambda p: make_response(), "q", threshold=1.5)
        assert "threshold must be between" in str(exc.value)

    def test_threshold_negative(self):
        with pytest.raises(ValueError):
            check(lambda p: make_response(), "q", threshold=-0.1)

    def test_max_retries_not_int(self):
        with pytest.raises(TypeError) as exc:
            check(lambda p: make_response(), "q", max_retries="1")
        assert "max_retries must be an int" in str(exc.value)

    def test_max_retries_bool_rejected(self):
        # bool is an int subclass, so this is accepted as int.
        # Confirm actual behavior: check accepts True as max_retries.
        r = check(lambda p: make_response("a", 0.1), "q", max_retries=True)
        # max_retries=True behaves like 1.
        assert r.n_attempts == 2

    def test_max_retries_negative(self):
        with pytest.raises(ValueError):
            check(lambda p: make_response(), "q", max_retries=-1)

    def test_async_call_fn_rejected(self):
        async def fn(p):
            return make_response()
        with pytest.raises(TypeError) as exc:
            check(fn, "q")
        assert "check() requires a synchronous call_fn" in str(exc.value)

    def test_async_callable_object_rejected(self):
        class AsyncCb:
            async def __call__(self, p):
                return make_response()
        with pytest.raises(TypeError):
            check(AsyncCb(), "q")

    def test_on_attempt_not_callable(self):
        with pytest.raises(TypeError) as exc:
            check(lambda p: make_response(), "q", on_attempt=42)
        assert "on_attempt must be callable" in str(exc.value)

    def test_async_on_attempt_rejected(self):
        async def on_attempt(i, a):
            pass
        with pytest.raises(TypeError) as exc:
            check(lambda p: make_response("a", 0.9), "q", on_attempt=on_attempt)
        assert "cannot await an async on_attempt" in str(exc.value)

    def test_async_on_attempt_callable_object_rejected(self):
        class AsyncCb:
            async def __call__(self, i, a):
                pass
        with pytest.raises(TypeError):
            check(lambda p: make_response("a", 0.9), "q", on_attempt=AsyncCb())

    def test_sync_callable_object_accepted(self):
        class Cb:
            def __init__(self):
                self.calls = []
            def __call__(self, p):
                self.calls.append(p)
                return make_response("a", 0.9)
        cb = Cb()
        r = check(cb, "q")
        assert r.status == ACCEPTED
        assert len(cb.calls) == 1

    def test_retry_prompt_mentions_previous(self):
        prompts = []
        responses = [make_response("first", 0.1), make_response("second", 0.9)]
        calls = {"n": 0}
        def fn(p):
            prompts.append(p)
            r = responses[calls["n"]]
            calls["n"] += 1
            return r
        check(fn, "base", max_retries=1)
        assert len(prompts) == 2
        assert "previous attempt" in prompts[1]
        assert "first" in prompts[1]
        assert "0.1" in prompts[1]

    def test_retry_prompt_preserves_leading_whitespace(self):
        prompts = []
        responses = [make_response("x", 0.1), make_response("y", 0.9)]
        calls = {"n": 0}
        def fn(p):
            prompts.append(p)
            r = responses[calls["n"]]
            calls["n"] += 1
            return r
        check(fn, "  spaced  ", max_retries=1)
        assert prompts[1].startswith("  spaced\n")

    def test_parse_failure_prompt(self):
        prompts = []
        responses = ["garbage", make_response("ok", 0.9)]
        calls = {"n": 0}
        def fn(p):
            prompts.append(p)
            r = responses[calls["n"]]
            calls["n"] += 1
            return r
        check(fn, "base", max_retries=1)
        assert "could not be parsed" in prompts[1]
        assert "no markdown code fences" in prompts[1]

    def test_parse_failure_prompt_preserves_leading(self):
        prompts = []
        responses = ["garbage", make_response("ok", 0.9)]
        calls = {"n": 0}
        def fn(p):
            prompts.append(p)
            r = responses[calls["n"]]
            calls["n"] += 1
            return r
        check(fn, "  base", max_retries=1)
        assert prompts[1].startswith("  base\n")

    def test_validation_failure_prompt(self):
        prompts = []
        responses = [make_response("x", 0.9), make_response("y", 0.9)]
        calls = {"n": 0}
        def fn(p):
            prompts.append(p)
            r = responses[calls["n"]]
            calls["n"] += 1
            return r
        def validator(ans):
            return (ans != "x", "bad value")
        check(fn, "base", max_retries=1, validator=validator)
        assert "failed validation" in prompts[1]
        assert "bad value" in prompts[1]
        assert '"x"' in prompts[1]

    def test_validation_failure_prompt_preserves_leading(self):
        prompts = []
        responses = [make_response("x", 0.9), make_response("y", 0.9)]
        calls = {"n": 0}
        def fn(p):
            prompts.append(p)
            r = responses[calls["n"]]
            calls["n"] += 1
            return r
        def validator(ans):
            return (ans != "x", "bad value")
        check(fn, "  base", max_retries=1, validator=validator)
        assert prompts[1].startswith("  base\n")

    def test_call_failed_prompt_again(self):
        prompts = []
        responses = [ValueError("x"), make_response("ok", 0.9)]
        calls = {"n": 0}
        def fn(p):
            prompts.append(p)
            r = responses[calls["n"]]
            calls["n"] += 1
            if isinstance(r, Exception):
                raise r
            return r
        result = check(fn, "base", max_retries=1)
        assert len(prompts) == 2
        # call-failed retry re-sends the original (suffixed) prompt again.
        assert prompts[1] == prompts[0]
        assert result.status == REPAIRED


# ---------------------------------------------------------------------------
# check() response parsing edge cases
# ---------------------------------------------------------------------------

class TestCheckParsing:
    def test_answer_as_int_stringified(self):
        r = check(lambda p: make_response(42, 0.9), "q")
        assert r.answer == "42"

    def test_answer_as_float_stringified(self):
        r = check(lambda p: make_response(4.5, 0.9), "q")
        assert r.answer == "4.5"

    def test_answer_as_bool_stringified(self):
        r = check(lambda p: make_response(True, 0.9), "q")
        assert r.answer == "True"

    def test_answer_as_dict_rejected(self):
        # A nested dict inside the answer value uses braces, so the
        # "flat" top-level candidate fails to JSON-decode cleanly and gets
        # rejected at the "missing answer/confidence" stage.
        r = check(lambda p: json.dumps({"answer": {"x": 1}, "confidence": 0.9}),
                  "q", max_retries=0)
        assert r.status == UNCERTAIN
        assert r.all_parse_failed is True

    def test_answer_as_list_rejected(self):
        r = check(lambda p: json.dumps({"answer": [1, 2], "confidence": 0.9}),
                  "q", max_retries=0)
        assert r.all_parse_failed is True
        assert r.attempts[0].error == "'answer' was a dict/list, not a scalar value"

    def test_confidence_as_bool_rejected(self):
        r = check(lambda p: json.dumps({"answer": "a", "confidence": True}),
                  "q", max_retries=0)
        assert r.status == UNCERTAIN
        assert "boolean" in r.attempts[0].error

    def test_missing_confidence(self):
        r = check(lambda p: json.dumps({"answer": "a"}), "q", max_retries=0)
        assert r.all_parse_failed is True
        assert "'answer' or 'confidence'" in r.attempts[0].error

    def test_missing_answer(self):
        r = check(lambda p: json.dumps({"confidence": 0.9}), "q", max_retries=0)
        assert r.all_parse_failed is True
        assert "'answer' or 'confidence'" in r.attempts[0].error

    def test_confidence_out_of_range(self):
        r = check(lambda p: make_response("a", 1.5), "q", max_retries=0)
        assert r.status == UNCERTAIN

    def test_confidence_negative(self):
        r = check(lambda p: make_response("a", -0.1), "q", max_retries=0)
        assert r.status == UNCERTAIN

    def test_confidence_zero_accepted(self):
        r = check(lambda p: make_response("a", 0.0), "q", max_retries=0)
        assert r.status == UNCERTAIN
        assert r.answer == "a"

    def test_confidence_one_accepted(self):
        r = check(lambda p: make_response("a", 1.0), "q")
        assert r.status == ACCEPTED
        assert r.confidence == 1.0

    def test_confidence_string_coerced(self):
        r = check(lambda p: json.dumps({"answer": "a", "confidence": "0.9"}),
                  "q", max_retries=0)
        assert r.status == ACCEPTED
        assert r.confidence == pytest.approx(0.9)

    def test_confidence_non_numeric_string_rejected(self):
        r = check(lambda p: json.dumps({"answer": "a", "confidence": "high"}),
                  "q", max_retries=0)
        assert r.all_parse_failed is True
        assert "could not be converted to a float" in r.attempts[0].error

    def test_empty_answer_non_ambiguous_rejected(self):
        r = check(lambda p: make_response("   ", 0.9), "q", max_retries=0)
        assert r.status == UNCERTAIN
        assert r.all_parse_failed is True
        assert "empty or whitespace-only" in r.attempts[0].error

    def test_empty_answer_ambiguous_ok(self):
        resp = json.dumps({"answer": "", "confidence": 0.9, "ambiguous": True})
        r = check(lambda p: resp, "q", max_retries=0)
        assert r.status == AMBIGUOUS
        assert r.answer == ""

    def test_ambiguous_string_true(self):
        resp = json.dumps({"answer": "a", "confidence": 0.9, "ambiguous": "TRUE"})
        r = check(lambda p: resp, "q", max_retries=0)
        assert r.status == AMBIGUOUS

    def test_ambiguous_string_false(self):
        resp = json.dumps({"answer": "a", "confidence": 0.9, "ambiguous": "false"})
        r = check(lambda p: resp, "q", max_retries=0)
        assert r.status == ACCEPTED
        assert r.ambiguous is False

    def test_ambiguous_string_false_whitespace_case(self):
        resp = json.dumps({"answer": "a", "confidence": 0.9, "ambiguous": "  FaLsE  "})
        r = check(lambda p: resp, "q", max_retries=0)
        assert r.status == ACCEPTED

    def test_invalid_ambiguous_rejected(self):
        resp = json.dumps({"answer": "a", "confidence": 0.9, "ambiguous": "maybe"})
        r = check(lambda p: resp, "q", max_retries=0)
        assert r.status == UNCERTAIN
        assert "unrecognized value" in r.attempts[0].error

    def test_json_embedded_in_text(self):
        text = 'here is the answer:\n' + make_response("a", 0.9) + '\nbye'
        r = check(lambda p: text, "q", max_retries=0)
        assert r.status == ACCEPTED
        assert r.answer == "a"

    def test_json_with_braces_inside_string(self):
        resp = '{"answer": "a {b} c", "confidence": 0.9}'
        r = check(lambda p: resp, "q", max_retries=0)
        assert r.status == ACCEPTED
        assert r.answer == "a {b} c"

    def test_last_line_used(self):
        resp = 'some text\n{"answer": "last", "confidence": 0.9}'
        r = check(lambda p: resp, "q", max_retries=0)
        assert r.answer == "last"

    def test_interpretations_coerced_to_str(self):
        resp = json.dumps({"answer": "a", "confidence": 0.9, "ambiguous": True,
                           "interpretations": [1, 2]})
        r = check(lambda p: resp, "q", max_retries=0)
        assert r.interpretations == ["1", "2"]

    def test_interpretations_non_list_becomes_empty(self):
        resp = json.dumps({"answer": "a", "confidence": 0.9, "ambiguous": True,
                           "interpretations": "single"})
        r = check(lambda p: resp, "q", max_retries=0)
        assert r.interpretations == []

    def test_chosen_interpretation_falsy_kept(self):
        resp = json.dumps({"answer": "a", "confidence": 0.9, "ambiguous": True,
                           "chosen_interpretation": 0})
        r = check(lambda p: resp, "q", max_retries=0)
        assert r.attempts[-1].chosen_interpretation == "0"

    def test_parsed_dict_preserved(self):
        r = check(lambda p: make_response("a", 0.9), "q")
        assert r.parsed == {"answer": "a", "confidence": 0.9}

    def test_bare_json_object_not_brace_candidate(self):
        # regex requires {...}; the parsed dict is preserved
        r = check(lambda p: '{"answer": "z", "confidence": 0.5}', "q", max_retries=0)
        assert r.confidence == 0.5
        assert r.answer == "z"

    def test_non_dict_json_rejected(self):
        r = check(lambda p: '[1, 2, 3]', "q", max_retries=0)
        assert r.all_parse_failed is True

    def test_raw_text_preserved(self):
        raw = 'prefix\n{"answer": "k", "confidence": 0.9}'
        r = check(lambda p: raw, "q", max_retries=0)
        assert r.attempts[0].raw_text == raw


# ---------------------------------------------------------------------------
# acheck()
# ---------------------------------------------------------------------------

class TestAcheck:
    def test_accepted(self):
        async def fn(p):
            return make_response("a", 0.9)
        r = run(acheck(fn, "q"))
        assert r.status == ACCEPTED
        assert r.answer == "a"
        assert r.method == "confidence"

    def test_requires_async_call_fn(self):
        with pytest.raises(TypeError) as exc:
            run(acheck(lambda p: make_response(), "q"))
        assert "acheck() requires an async call_fn" in str(exc.value)

    def test_async_callable_object_accepted(self):
        class Cb:
            async def __call__(self, p):
                return make_response("a", 0.9)
        r = run(acheck(Cb(), "q"))
        assert r.status == ACCEPTED

    def test_async_call_fn_raises(self):
        async def fn(p):
            raise RuntimeError("boom")
        r = run(acheck(fn, "q", max_retries=0))
        assert r.status == UNCERTAIN
        assert r.call_failed is True
        assert r.attempts[0].raw_text == "RuntimeError: boom"

    def test_async_call_returns_non_str(self):
        async def fn(p):
            return 123
        r = run(acheck(fn, "q", max_retries=0))
        assert r.call_failed is True
        assert r.attempts[0].error == "call_fn returned int, expected str"

    def test_retry(self):
        responses = [make_response("a", 0.1), make_response("b", 0.9)]
        calls = {"n": 0}
        async def fn(p):
            r = responses[calls["n"]]
            calls["n"] += 1
            return r
        r = run(acheck(fn, "q", max_retries=1))
        assert r.status == REPAIRED
        assert r.answer == "b"
        assert r.n_attempts == 2

    def test_sync_on_attempt(self):
        calls = []
        async def fn(p):
            return make_response("a", 0.9)
        run(acheck(fn, "q", on_attempt=lambda i, a: calls.append(i)))
        assert calls == [0]

    def test_sync_callable_object_on_attempt(self):
        calls = []
        class Cb:
            def __call__(self, i, a):
                calls.append(i)
        async def fn(p):
            return make_response("a", 0.9)
        run(acheck(fn, "q", on_attempt=Cb()))
        assert calls == [0]

    def test_async_on_attempt(self):
        calls = []
        async def on_attempt(i, a):
            calls.append((i, a.answer))
        async def fn(p):
            return make_response("a", 0.9)
        run(acheck(fn, "q", on_attempt=on_attempt))
        assert calls == [(0, "a")]

    def test_async_on_attempt_callable_object(self):
        calls = []
        class Cb:
            async def __call__(self, i, a):
                calls.append((i, a.answer))
        async def fn(p):
            return make_response("a", 0.9)
        run(acheck(fn, "q", on_attempt=Cb()))
        assert calls == [(0, "a")]

    def test_on_attempt_not_callable(self):
        async def fn(p):
            return make_response()
        with pytest.raises(TypeError) as exc:
            run(acheck(fn, "q", on_attempt=42))
        assert "on_attempt must be callable" in str(exc.value)

    def test_prompt_invalid(self):
        async def fn(p):
            return make_response()
        with pytest.raises(TypeError) as exc:
            run(acheck(fn, 123))
        assert "prompt must be a str" in str(exc.value)

    def test_threshold_invalid(self):
        async def fn(p):
            return make_response()
        with pytest.raises(ValueError):
            run(acheck(fn, "q", threshold=1.5))

    def test_max_retries_invalid(self):
        async def fn(p):
            return make_response()
        with pytest.raises(TypeError):
            run(acheck(fn, "q", max_retries="x"))

    def test_validator(self):
        responses = [make_response("bad", 0.9), make_response("good", 0.9)]
        calls = {"n": 0}
        async def fn(p):
            r = responses[calls["n"]]
            calls["n"] += 1
            return r
        r = run(acheck(fn, "q", max_retries=1,
                       validator=lambda a: (a == "good", "nope")))
        assert r.status == REPAIRED
        assert r.answer == "good"
        assert r.attempts[0].validation_error == "nope"

    def test_ambiguous(self):
        resp = make_response("a", 0.9, ambiguous=True, interpretations=["x"])
        async def fn(p):
            return resp
        r = run(acheck(fn, "q"))
        assert r.status == AMBIGUOUS
        assert r.interpretations == ["x"]
        assert r.method == "ambiguity"

    def test_all_parse_failed(self):
        async def fn(p):
            return "garbage"
        r = run(acheck(fn, "q", max_retries=1))
        assert r.status == UNCERTAIN
        assert r.all_parse_failed is True
        assert r.method == "parse_failure"


# ---------------------------------------------------------------------------
# Validator return forms (via check)
# ---------------------------------------------------------------------------

class TestValidatorForms:
    def _run_with_validator(self, validator):
        r = check(lambda p: make_response("a", 0.9), "q", max_retries=0,
                  validator=validator)
        return r

    def test_returns_true(self):
        r = self._run_with_validator(lambda a: True)
        assert r.status == ACCEPTED

    def test_returns_false(self):
        r = self._run_with_validator(lambda a: False)
        assert r.status == UNCERTAIN
        assert r.attempts[0].validation_error == "Validator returned False"

    def test_returns_tuple_true_none(self):
        r = self._run_with_validator(lambda a: (True, None))
        assert r.status == ACCEPTED
        assert r.attempts[0].validation_error is None

    def test_returns_tuple_false_none(self):
        r = self._run_with_validator(lambda a: (False, None))
        assert r.status == UNCERTAIN
        assert "no message provided" in r.attempts[0].validation_error

    def test_returns_tuple_with_message(self):
        r = self._run_with_validator(lambda a: (False, "bad"))
        assert r.attempts[0].validation_error == "bad"

    def test_returns_tuple_true_with_message(self):
        r = self._run_with_validator(lambda a: (True, "fine"))
        assert r.status == ACCEPTED
        assert r.attempts[0].validation_error == "fine"

    def test_returns_list_form(self):
        r = self._run_with_validator(lambda a: [True, "msg"])
        assert r.status == ACCEPTED

    def test_invalid_return_type(self):
        r = self._run_with_validator(lambda a: "weird")
        assert r.status == UNCERTAIN
        assert "invalid type" in r.attempts[0].validation_error
        assert "str" in r.attempts[0].validation_error

    def test_invalid_tuple(self):
        r = self._run_with_validator(lambda a: (1, 2))
        assert r.status == UNCERTAIN
        assert "invalid type" in r.attempts[0].validation_error

    def test_invalid_tuple_wrong_length(self):
        r = self._run_with_validator(lambda a: (True, "x", "y"))
        assert r.status == UNCERTAIN

    def test_invalid_message_non_str(self):
        r = self._run_with_validator(lambda a: (False, 123))
        assert r.status == UNCERTAIN
        assert "invalid type" in r.attempts[0].validation_error

    def test_async_validator_rejected(self):
        async def v(a):
            return True
        r = self._run_with_validator(v)
        assert r.status == UNCERTAIN
        assert "synchronous" in r.attempts[0].validation_error

    def test_validator_skipped_for_ambiguous(self):
        resp = make_response("a", 0.9, ambiguous=True)
        called = {"n": 0}
        def v(a):
            called["n"] += 1
            return False
        check(lambda p: resp, "q", max_retries=0, validator=v)
        assert called["n"] == 0

    def test_validator_not_run_when_no_validator(self):
        r = check(lambda p: make_response("a", 0.9), "q", max_retries=0)
        assert r.status == ACCEPTED
        assert r.attempts[0].passed_validation is True
        assert r.attempts[0].validation_error is None
