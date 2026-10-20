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
    _CONFIDENCE_SUFFIX,
    Attempt,
    BoothRejected,
    BoothResult,
    check,
    acheck,
    check_with_evidence,
    _coerce_ambiguous,
    _is_async_callable,
    _parse_response,
    _run_validator,
    _build_prompt,
    _is_empty_evidence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_response(answer="paris", confidence=0.9, **extra):
    obj = {"answer": answer, "confidence": confidence}
    obj.update(extra)
    return json.dumps(obj)


def make_attempt(**kwargs):
    base = dict(raw_text="", answer="a", confidence=0.9, parse_ok=True)
    base.update(kwargs)
    return Attempt(**base)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _coerce_ambiguous
# ---------------------------------------------------------------------------

class TestCoerceAmbiguous:
    def test_bool_true(self):
        assert _coerce_ambiguous(True) is True

    def test_bool_false(self):
        assert _coerce_ambiguous(False) is False

    @pytest.mark.parametrize("val", ["true", "TRUE", " True ", "TrUe"])
    def test_string_true(self, val):
        assert _coerce_ambiguous(val) is True

    @pytest.mark.parametrize("val", ["false", "FALSE", " False ", "FaLsE"])
    def test_string_false(self, val):
        assert _coerce_ambiguous(val) is False

    @pytest.mark.parametrize("val", ["yes", "1", "", None, 1, 0.0, []])
    def test_other_returns_none(self, val):
        assert _coerce_ambiguous(val) is None


# ---------------------------------------------------------------------------
# _is_async_callable
# ---------------------------------------------------------------------------

class TestIsAsyncCallable:
    def test_plain_function(self):
        assert _is_async_callable(lambda x: x) is False

    def test_async_def(self):
        async def fn(x):
            return x
        assert _is_async_callable(fn) is True

    def test_sync_callable_object(self):
        class C:
            def __call__(self, x):
                return x
        assert _is_async_callable(C()) is False

    def test_async_callable_object(self):
        class C:
            async def __call__(self, x):
                return x
        assert _is_async_callable(C()) is True


# ---------------------------------------------------------------------------
# Attempt dataclass
# ---------------------------------------------------------------------------

class TestAttempt:
    def test_defaults(self):
        a = Attempt(raw_text="r", answer="x", confidence=0.5, parse_ok=True)
        assert a.error is None
        assert a.ambiguous is False
        assert a.interpretations == []
        assert a.chosen_interpretation is None
        assert a.passed_validation is True
        assert a.validation_error is None
        assert a.parsed is None
        assert a.call_failed is False

    def test_interpretations_not_shared(self):
        a = Attempt(raw_text="", answer=None, confidence=None, parse_ok=False)
        b = Attempt(raw_text="", answer=None, confidence=None, parse_ok=False)
        a.interpretations.append("x")
        assert b.interpretations == []


# ---------------------------------------------------------------------------
# BoothResult
# ---------------------------------------------------------------------------

class TestBoothResult:
    def test_ok_accepted(self):
        assert BoothResult(answer="a", status=ACCEPTED).ok is True

    def test_ok_repaired(self):
        assert BoothResult(answer="a", status=REPAIRED).ok is True

    @pytest.mark.parametrize("status", [AMBIGUOUS, BLOCKED, UNCERTAIN])
    def test_not_ok(self, status):
        assert BoothResult(answer="a", status=status).ok is False

    def test_n_attempts(self):
        r = BoothResult(answer="a", status=ACCEPTED, attempts=[make_attempt()])
        assert r.n_attempts == 1
        assert BoothResult(answer=None, status=UNCERTAIN).n_attempts == 0

    def test_all_parse_failed_empty(self):
        assert BoothResult(answer=None, status=UNCERTAIN).all_parse_failed is False

    def test_all_parse_failed_true(self):
        r = BoothResult(answer=None, status=UNCERTAIN, attempts=[
            make_attempt(parse_ok=False), make_attempt(parse_ok=False)
        ])
        assert r.all_parse_failed is True

    def test_all_parse_failed_partial(self):
        r = BoothResult(answer=None, status=UNCERTAIN, attempts=[
            make_attempt(parse_ok=False), make_attempt(parse_ok=True)
        ])
        assert r.all_parse_failed is False

    def test_call_failed_empty(self):
        assert BoothResult(answer=None, status=UNCERTAIN).call_failed is False

    def test_call_failed_true(self):
        r = BoothResult(answer=None, status=UNCERTAIN, attempts=[
            make_attempt(parse_ok=False, call_failed=True)
        ])
        assert r.call_failed is True

    def test_call_failed_mixed(self):
        r = BoothResult(answer=None, status=UNCERTAIN, attempts=[
            make_attempt(parse_ok=False, call_failed=True),
            make_attempt(parse_ok=False, call_failed=False),
        ])
        assert r.call_failed is False

    def test_checker_failed_compare(self):
        assert BoothResult(answer="a", status=UNCERTAIN, reason=COMPARE_FAILED).checker_failed is True

    def test_checker_failed_invalid(self):
        assert BoothResult(answer="a", status=UNCERTAIN, reason=INVALID_SCORE).checker_failed is True

    def test_checker_failed_other(self):
        assert BoothResult(answer="a", status=UNCERTAIN, reason=NO_EVIDENCE).checker_failed is False
        assert BoothResult(answer="a", status=UNCERTAIN).checker_failed is False

    def test_method_ambiguity(self):
        assert BoothResult(answer="a", status=AMBIGUOUS).method == "ambiguity"

    def test_method_evidence_no_attempts(self):
        assert BoothResult(answer="a", status=UNCERTAIN).method == "evidence"

    def test_method_parse_failure(self):
        r = BoothResult(answer=None, status=UNCERTAIN, attempts=[make_attempt(parse_ok=False)])
        assert r.method == "parse_failure"

    def test_method_validation(self):
        r = BoothResult(answer="a", status=UNCERTAIN, attempts=[
            make_attempt(passed_validation=False)
        ])
        assert r.method == "validation"

    def test_method_confidence(self):
        r = BoothResult(answer="a", status=ACCEPTED, attempts=[make_attempt()])
        assert r.method == "confidence"

    def test_unwrap_ok(self):
        r = BoothResult(answer="a", status=ACCEPTED)
        assert r.unwrap() == "a"

    def test_unwrap_not_ok_raises(self):
        r = BoothResult(answer="a", status=BLOCKED)
        with pytest.raises(BoothRejected) as ei:
            r.unwrap()
        assert ei.value.result is r

    def test_unwrap_ok_but_answer_none_raises(self):
        r = BoothResult(answer=None, status=ACCEPTED)
        with pytest.raises(BoothRejected):
            r.unwrap()

    def test_unwrap_or_ok(self):
        assert BoothResult(answer="a", status=ACCEPTED).unwrap_or("d") == "a"

    def test_unwrap_or_default(self):
        assert BoothResult(answer="a", status=BLOCKED).unwrap_or("d") == "d"

    def test_to_dict_keys(self):
        r = BoothResult(answer="a", status=ACCEPTED, confidence=0.9)
        d = r.to_dict()
        assert set(d.keys()) == {
            "answer", "status", "confidence", "attempts", "ambiguous",
            "interpretations", "evidence_agreement", "parsed", "n_attempts",
            "ok", "all_parse_failed", "call_failed", "method", "reason",
            "detail", "checker_failed",
        }
        assert d["answer"] == "a"
        assert d["status"] == ACCEPTED
        assert d["ok"] is True
        assert d["n_attempts"] == 0
        assert d["method"] == "evidence"
        assert d["attempts"] == []

    def test_to_dict_attempts_as_dicts(self):
        r = BoothResult(answer="a", status=ACCEPTED, attempts=[make_attempt()])
        d = r.to_dict()
        assert isinstance(d["attempts"][0], dict)
        assert d["attempts"][0]["answer"] == "a"


class TestBoothRejected:
    def test_message_hides_answer(self):
        r = BoothResult(answer="secret-answer", status=BLOCKED, method="evidence")
        exc = BoothRejected(r)
        assert "secret-answer" not in str(exc)
        assert "BLOCKED" in str(exc)
        assert exc.result is r


# ---------------------------------------------------------------------------
# _build_prompt / _is_empty_evidence
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_appends_suffix(self):
        p = _build_prompt("hello")
        assert p == "hello" + _CONFIDENCE_SUFFIX

    def test_strips_trailing_whitespace(self):
        p = _build_prompt("hello   \n")
        assert p.startswith("hello" + _CONFIDENCE_SUFFIX)


class TestIsEmptyEvidence:
    def test_empty_list(self):
        assert _is_empty_evidence([]) is True

    def test_empty_string(self):
        assert _is_empty_evidence("") is True

    def test_nonempty(self):
        assert _is_empty_evidence(["a"]) is False


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_simple_json(self):
        a = _parse_response(make_response())
        assert a.parse_ok is True
        assert a.answer == "paris"
        assert a.confidence == 0.9
        assert a.ambiguous is False
        assert a.error is None

    def test_with_surrounding_text(self):
        text = "Here you go:\n" + make_response()
        a = _parse_response(text)
        assert a.parse_ok is True
        assert a.answer == "paris"

    def test_json_with_braces_in_string(self):
        text = 'prefix {"answer": "a{b}c", "confidence": 0.5}'
        a = _parse_response(text)
        assert a.parse_ok is True
        assert a.answer == "a{b}c"

    def test_missing_confidence(self):
        a = _parse_response('{"answer": "x"}')
        assert a.parse_ok is False
        assert "missing" in a.error

    def test_answer_dict_rejected(self):
        a = _parse_response(json.dumps({"answer": {"a": 1}, "confidence": 0.5}))
        assert a.parse_ok is False
        assert "dict/list" in a.error

    def test_answer_list_rejected(self):
        a = _parse_response(json.dumps({"answer": [1, 2], "confidence": 0.5}))
        assert a.parse_ok is False
        assert "dict/list" in a.error

    def test_answer_scalar_stringified(self):
        a = _parse_response(json.dumps({"answer": 42, "confidence": 0.5}))
        assert a.parse_ok is True
        assert a.answer == "42"

    def test_confidence_bool_rejected(self):
        a = _parse_response(json.dumps({"answer": "x", "confidence": True}))
        assert a.parse_ok is False
        assert "boolean" in a.error

    def test_confidence_string_float(self):
        a = _parse_response('{"answer": "x", "confidence": "0.4"}')
        assert a.parse_ok is True
        assert a.confidence == 0.4

    def test_confidence_bad_string(self):
        a = _parse_response('{"answer": "x", "confidence": "abc"}')
        assert a.parse_ok is False
        assert "could not be converted" in a.error

    def test_confidence_out_of_range_high(self):
        a = _parse_response(json.dumps({"answer": "x", "confidence": 1.5}))
        assert a.parse_ok is False
        assert "range" in a.error

    def test_confidence_out_of_range_low(self):
        a = _parse_response(json.dumps({"answer": "x", "confidence": -0.1}))
        assert a.parse_ok is False
        assert "range" in a.error

    def test_ambiguous_string_true(self):
        a = _parse_response('{"answer": "x", "confidence": 0.5, "ambiguous": "true"}')
        assert a.parse_ok is True
        assert a.ambiguous is True

    def test_ambiguous_invalid(self):
        a = _parse_response('{"answer": "x", "confidence": 0.5, "ambiguous": "maybe"}')
        assert a.parse_ok is False
        assert "ambiguous" in a.error

    def test_empty_answer_not_ambiguous(self):
        a = _parse_response('{"answer": "   ", "confidence": 0.5}')
        assert a.parse_ok is False
        assert "empty" in a.error

    def test_empty_answer_ambiguous_ok(self):
        a = _parse_response(
            '{"answer": "", "confidence": 0.5, "ambiguous": true, '
            '"interpretations": ["a", "b"]}'
        )
        assert a.parse_ok is True
        assert a.ambiguous is True
        assert a.interpretations == ["a", "b"]

    def test_interpretations_non_list(self):
        a = _parse_response(
            '{"answer": "x", "confidence": 0.5, "ambiguous": false, '
            '"interpretations": "not a list"}'
        )
        assert a.parse_ok is True
        assert a.interpretations == []

    def test_chosen_interpretation_zero_preserved(self):
        a = _parse_response(
            '{"answer": "x", "confidence": 0.5, "chosen_interpretation": 0}'
        )
        assert a.chosen_interpretation == "0"

    def test_chosen_interpretation_null(self):
        a = _parse_response(
            '{"answer": "x", "confidence": 0.5, "chosen_interpretation": null}'
        )
        assert a.chosen_interpretation is None

    def test_no_json(self):
        a = _parse_response("just plain text")
        assert a.parse_ok is False
        assert a.answer is None
        assert a.confidence is None
        assert a.error is not None

    def test_empty_string(self):
        a = _parse_response("")
        assert a.parse_ok is False


# ---------------------------------------------------------------------------
# _run_validator
# ---------------------------------------------------------------------------

class TestRunValidator:
    def test_none_validator(self):
        assert _run_validator(None, "x") == (True, None)

    def test_none_answer(self):
        assert _run_validator(lambda x: False, None) == (True, None)

    def test_bool_true(self):
        assert _run_validator(lambda x: True, "x") == (True, None)

    def test_bool_false(self):
        passed, err = _run_validator(lambda x: False, "x")
        assert passed is False
        assert err == "Validator returned False"

    def test_tuple_pass(self):
        assert _run_validator(lambda x: (True, "ok"), "x") == (True, "ok")

    def test_tuple_fail(self):
        assert _run_validator(lambda x: (False, "bad"), "x") == (False, "bad")

    def test_tuple_fail_no_message(self):
        passed, err = _run_validator(lambda x: (False, None), "x")
        assert passed is False
        assert "no message" in err

    def test_tuple_pass_no_message(self):
        assert _run_validator(lambda x: (True, None), "x") == (True, None)

    def test_list_form(self):
        assert _run_validator(lambda x: [True, "ok"], "x") == (True, "ok")

    def test_validator_raises(self):
        def bad(x):
            raise RuntimeError("boom")
        passed, err = _run_validator(bad, "x")
        assert passed is False
        assert "RuntimeError" in err
        assert "boom" in err

    def test_invalid_return_type(self):
        passed, err = _run_validator(lambda x: "yes", "x")
        assert passed is False
        assert "invalid type" in err

    def test_invalid_tuple_length(self):
        passed, err = _run_validator(lambda x: (True,), "x")
        assert passed is False
        assert "invalid type" in err

    def test_invalid_tuple_message_type(self):
        passed, err = _run_validator(lambda x: (True, 123), "x")
        assert passed is False
        assert "invalid type" in err

    def test_coroutine_validator_rejected(self):
        async def coro(x):
            return True
        passed, err = _run_validator(coro, "x")
        assert passed is False
        assert "synchronous" in err


# ---------------------------------------------------------------------------
# check_with_evidence
# ---------------------------------------------------------------------------

class TestCheckWithEvidence:
    def test_accepted_bool_true(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: True)
        assert r.status == ACCEPTED
        assert r.confidence == 1.0
        assert r.evidence_agreement == 1.0
        assert r.reason is None
        assert r.answer == "ans"

    def test_blocked_bool_false(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: False)
        assert r.status == BLOCKED
        assert r.confidence == 0.0
        assert r.evidence_agreement == 0.0
        assert r.reason == EVIDENCE_DISAGREES
        assert r.detail == "compare_fn returned False"

    def test_float_above_threshold(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: 0.9)
        assert r.status == ACCEPTED
        assert r.confidence == 0.9
        assert r.evidence_agreement == 0.9

    def test_float_below_threshold(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: 0.3)
        assert r.status == BLOCKED
        assert r.reason == EVIDENCE_DISAGREES
        assert "below evidence_threshold" in r.detail

    def test_float_at_threshold(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: 0.5, evidence_threshold=0.5)
        assert r.status == ACCEPTED

    def test_custom_threshold(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: 0.6, evidence_threshold=0.5)
        assert r.status == ACCEPTED

    def test_compare_fn_raises(self):
        def bad(a, ev):
            raise ValueError("nope")
        r = check_with_evidence("ans", ["e"], bad)
        assert r.status == UNCERTAIN
        assert r.reason == COMPARE_FAILED
        assert "ValueError" in r.detail
        assert "nope" in r.detail

    def test_compare_fn_returns_string(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: "yes")
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE

    def test_compare_fn_returns_out_of_range(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: 1.5)
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE
        assert "range" in r.detail

    def test_compare_fn_returns_none(self):
        r = check_with_evidence("ans", ["e"], lambda a, ev: None)
        assert r.status == UNCERTAIN
        assert r.reason == INVALID_SCORE

    def test_empty_answer(self):
        r = check_with_evidence("", ["e"], lambda a, ev: True)
        assert r.status == UNCERTAIN
        assert r.reason == EMPTY_ANSWER
        assert r.answer is None

    def test_whitespace_answer(self):
        r = check_with_evidence("   ", ["e"], lambda a, ev: True)
        assert r.reason == EMPTY_ANSWER

    def test_empty_evidence(self):
        r = check_with_evidence("ans", [], lambda a, ev: True)
        assert r.status == UNCERTAIN
        assert r.reason == NO_EVIDENCE
        assert r.answer == "ans"

    def test_evidence_passed_through(self):
        captured = {}

        def cmp(a, ev):
            captured["a"] = a
            captured["ev"] = ev
            return True

        check_with_evidence("ans", ["x", "y"], cmp)
        assert captured["a"] == "ans"
        assert captured["ev"] == ["x", "y"]

    def test_async_compare_raises(self):
        from booth.core import _is_boolish  # noqa: F401
        async def cmp(a, ev):
            return True
        with pytest.raises(TypeError) as ei:
            check_with_evidence("ans", ["e"], cmp)
        assert "synchronous" in str(ei.value)

    def test_non_str_answer_type_error(self):
        with pytest.raises(TypeError) as ei:
            check_with_evidence(123, ["e"], lambda a, ev: True)
        assert "answer must be a str" in str(ei.value)

    def test_evidence_threshold_bool_type_error(self):
        with pytest.raises(TypeError):
            check_with_evidence("ans", ["e"], lambda a, ev: True, evidence_threshold=True)

    def test_evidence_threshold_non_number(self):
        with pytest.raises(TypeError):
            check_with_evidence("ans", ["e"], lambda a, ev: True, evidence_threshold="0.5")

    def test_evidence_threshold_out_of_range(self):
        with pytest.raises(ValueError):
            check_with_evidence("ans", ["e"], lambda a, ev: True, evidence_threshold=1.5)


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

class TestCheck:
    def test_accepted_first_try(self):
        calls = []

        def call_fn(p):
            calls.append(p)
            return make_response()

        r = check(call_fn, "q")
        assert r.status == ACCEPTED
        assert r.ok is True
        assert r.answer == "paris"
        assert r.confidence == 0.9
        assert r.n_attempts == 1
        assert len(calls) == 1
        assert calls[0] == _build_prompt("q")
        assert r.reason is None

    def test_parse_failure_then_success(self):
        responses = iter(["garbage", make_response()])
        prompts = []

        def call_fn(p):
            prompts.append(p)
            return next(responses)

        r = check(call_fn, "q", max_retries=1)
        assert r.status == REPAIRED
        assert r.n_attempts == 2
        assert "could not be parsed" in prompts[1]

    def test_all_parse_failures(self):
        r = check(lambda p: "bad", "q", max_retries=1)
        assert r.status == UNCERTAIN
        assert r.method == "parse_failure"
        assert r.all_parse_failed is True
        assert r.answer is None

    def test_low_confidence_retried_then_uncertain(self):
        responses = iter([
            make_response(confidence=0.1),
            make_response(confidence=0.2),
        ])
        r = check(lambda p: next(responses), "q", max_retries=1, threshold=0.7)
        assert r.status == UNCERTAIN
        assert r.n_attempts == 2
        # last ok attempt's answer is preserved
        assert r.answer == "paris"
        assert r.confidence == 0.2

    def test_low_confidence_then_high_repaired(self):
        responses = iter([
            make_response(confidence=0.1),
            make_response(confidence=0.95),
        ])
        prompts = []

        def call_fn(p):
            prompts.append(p)
            return next(responses)

        r = check(call_fn, "q", max_retries=1)
        assert r.status == REPAIRED
        assert r.confidence == 0.95
        assert "Reconsider" in prompts[1]

    def test_ambiguous_result(self):
        resp = json.dumps({
            "answer": "a", "confidence": 0.9, "ambiguous": True,
            "interpretations": ["x", "y"], "chosen_interpretation": "x",
        })
        r = check(lambda p: resp, "q")
        assert r.status == AMBIGUOUS
        assert r.method == "ambiguity"
        assert r.ambiguous is True
        assert r.interpretations == ["x", "y"]

    def test_max_retries_zero(self):
        r = check(lambda p: make_response(confidence=0.1), "q", max_retries=0)
        assert r.status == UNCERTAIN
        assert r.n_attempts == 1

    def test_call_fn_raises(self):
        def bad(p):
            raise RuntimeError("fail")

        r = check(bad, "q", max_retries=1)
        assert r.status == UNCERTAIN
        assert r.call_failed is True
        assert r.n_attempts == 2
        assert all(a.call_failed for a in r.attempts)

    def test_call_fn_returns_non_str(self):
        r = check(lambda p: 123, "q", max_retries=0)
        assert r.status == UNCERTAIN
        assert r.call_failed is True
        assert "expected str" in r.attempts[0].error

    def test_on_attempt_called(self):
        seen = []

        def on_attempt(i, a):
            seen.append((i, a.parse_ok))

        check(lambda p: make_response(), "q", on_attempt=on_attempt)
        assert seen == [(0, True)]

    def test_on_attempt_called_each_attempt(self):
        responses = iter(["bad", make_response()])
        seen = []
        check(lambda p: next(responses), "q", max_retries=1,
              on_attempt=lambda i, a: seen.append(i))
        assert seen == [0, 1]

    def test_threshold_out_of_range(self):
        with pytest.raises(ValueError):
            check(lambda p: make_response(), "q", threshold=1.5)

    def test_max_retries_negative(self):
        with pytest.raises(ValueError):
            check(lambda p: make_response(), "q", max_retries=-1)

    def test_max_retries_non_int(self):
        with pytest.raises(TypeError):
            check(lambda p: make_response(), "q", max_retries=1.0)

    def test_prompt_not_str(self):
        with pytest.raises(TypeError):
            check(lambda p: make_response(), 123)

    def test_call_fn_not_callable(self):
        with pytest.raises(TypeError):
            check("not callable", "q")

    def test_async_call_fn_rejected(self):
        async def fn(p):
            return make_response()
        with pytest.raises(TypeError) as ei:
            check(fn, "q")
        assert "synchronous" in str(ei.value)

    def test_async_on_attempt_rejected(self):
        async def cb(i, a):
            pass
        with pytest.raises(TypeError) as ei:
            check(lambda p: make_response(), "q", on_attempt=cb)
        assert "cannot await" in str(ei.value)

    def test_non_callable_on_attempt(self):
        with pytest.raises(TypeError):
            check(lambda p: make_response(), "q", on_attempt=42)

    def test_validator_failure_retried(self):
        responses = iter([make_response(answer="bad"), make_response(answer="good")])
        prompts = []

        def call_fn(p):
            prompts.append(p)
            return next(responses)

        r = check(call_fn, "q", max_retries=1,
                  validator=lambda a: (a == "good", "answer must be 'good'"))
        assert r.status == REPAIRED
        assert r.answer == "good"
        assert "failed validation" in prompts[1]

    def test_validator_failure_exhausted(self):
        r = check(lambda p: make_response(answer="bad"), "q", max_retries=0,
                  validator=lambda a: False)
        assert r.status == UNCERTAIN
        assert r.method == "validation"
        assert r.attempts[0].passed_validation is False

    def test_validator_not_run_when_ambiguous(self):
        called = []

        def validator(a):
            called.append(a)
            return True

        resp = json.dumps({"answer": "a", "confidence": 0.9, "ambiguous": True,
                           "interpretations": ["x"]})
        r = check(lambda p: resp, "q", validator=validator)
        assert r.status == AMBIGUOUS
        assert called == []

    def test_retry_on_call_failure_uses_plain_prompt(self):
        calls = []

        def call_fn(p):
            calls.append(p)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return make_response()

        r = check(call_fn, "q", max_retries=1)
        assert r.status == REPAIRED
        assert calls[1] == _build_prompt("q")


# ---------------------------------------------------------------------------
# acheck
# ---------------------------------------------------------------------------

class TestAcheck:
    def test_accepted(self):
        async def fn(p):
            return make_response()
        r = run(acheck(fn, "q"))
        assert r.status == ACCEPTED
        assert r.answer == "paris"

    def test_parse_failure_then_ok(self):
        responses = iter(["bad", make_response()])

        async def fn(p):
            return next(responses)

        r = run(acheck(fn, "q", max_retries=1))
        assert r.status == REPAIRED
        assert r.n_attempts == 2

    def test_call_fn_raises(self):
        async def fn(p):
            raise RuntimeError("boom")

        r = run(acheck(fn, "q", max_retries=0))
        assert r.status == UNCERTAIN
        assert r.call_failed is True

    def test_sync_on_attempt(self):
        seen = []

        async def fn(p):
            return make_response()

        r = run(acheck(fn, "q", on_attempt=lambda i, a: seen.append(i)))
        assert seen == [0]
        assert r.status == ACCEPTED

    def test_async_on_attempt(self):
        seen = []

        async def fn(p):
            return make_response()

        async def cb(i, a):
            seen.append(i)

        run(acheck(fn, "q", on_attempt=cb))
        assert seen == [0]

    def test_sync_call_fn_rejected(self):
        with pytest.raises(TypeError) as ei:
            run(acheck(lambda p: make_response(), "q"))
        assert "async call_fn" in str(ei.value)

    def test_non_callable_call_fn(self):
        with pytest.raises(TypeError):
            run(acheck("nope", "q"))

    def test_threshold_invalid(self):
        async def fn(p):
            return make_response()
        with pytest.raises(ValueError):
            run(acheck(fn, "q", threshold=-0.1))

    def test_non_callable_on_attempt(self):
        async def fn(p):
            return make_response()
        with pytest.raises(TypeError):
            run(acheck(fn, "q", on_attempt=42))

    def test_validator(self):
        async def fn(p):
            return make_response(answer="bad")

        r = run(acheck(fn, "q", max_retries=0, validator=lambda a: False))
        assert r.status == UNCERTAIN
        assert r.method == "validation"

    def test_async_callable_object(self):
        class C:
            async def __call__(self, p):
                return make_response()

        r = run(acheck(C(), "q"))
        assert r.status == ACCEPTED
