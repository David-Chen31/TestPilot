from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any

import pytest

from backon._conditions import (
    RetryCondition,
    Stop,
    retry_all,
    retry_always,
    retry_any,
    retry_if_exception,
    retry_if_exception_cause_type,
    retry_if_exception_message,
    retry_if_exception_type,
    retry_if_not_exception_message,
    retry_if_not_exception_type,
    retry_if_not_result,
    retry_if_result,
    retry_never,
    retry_unless_exception_type,
    stop_after_attempt,
    stop_after_delay,
    stop_all,
    stop_any,
    stop_before_delay,
    stop_never,
    stop_when_event_set,
)
from backon._state import RetryState


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

@dataclass
class FakeAttempt:
    """Minimal stand-in for backon's Attempt."""

    value: Any = None
    exception: BaseException | None = None
    wait: float = 0.0


def make_state(
    *,
    tries: int = 0,
    elapsed: float = 0.0,
    value: Any = None,
    exception: BaseException | None = None,
    wait: float = 0.0,
    has_outcome: bool = False,
) -> RetryState:
    """Build a RetryState with a fake outcome when needed."""
    state = RetryState(tries=tries, elapsed=elapsed)
    if has_outcome or value is not None or exception is not None or wait:
        state.outcome = FakeAttempt(value=value, exception=exception, wait=wait)
    return state


# --------------------------------------------------------------------------- #
# Base classes
# --------------------------------------------------------------------------- #

class TestStopBase:
    def test_base_call_raises(self):
        with pytest.raises(NotImplementedError):
            Stop()(make_state())

    def test_or_returns_stop_any(self):
        a = stop_after_attempt(1)
        b = stop_after_attempt(2)
        combined = a | b
        assert isinstance(combined, stop_any)
        assert combined.stops == (a, b)

    def test_and_returns_stop_all(self):
        a = stop_after_attempt(1)
        b = stop_after_attempt(2)
        combined = a & b
        assert isinstance(combined, stop_all)
        assert combined.stops == (a, b)


class TestRetryConditionBase:
    def test_base_call_raises(self):
        with pytest.raises(NotImplementedError):
            RetryCondition()(make_state())

    def test_or_returns_retry_any(self):
        a = retry_always()
        b = retry_never()
        combined = a | b
        assert isinstance(combined, retry_any)
        assert combined.conditions == (a, b)

    def test_and_returns_retry_all(self):
        a = retry_always()
        b = retry_never()
        combined = a & b
        assert isinstance(combined, retry_all)
        assert combined.conditions == (a, b)


# --------------------------------------------------------------------------- #
# Stop conditions
# --------------------------------------------------------------------------- #

class TestStopAfterAttempt:
    def test_stores_max_attempts(self):
        s = stop_after_attempt(5)
        assert s.max_attempts == 5

    def test_true_at_and_above(self):
        assert stop_after_attempt(3)(make_state(tries=3)) is True
        assert stop_after_attempt(3)(make_state(tries=10)) is True

    def test_false_below(self):
        assert stop_after_attempt(3)(make_state(tries=2)) is False
        assert stop_after_attempt(3)(make_state(tries=0)) is False

    def test_zero_stops_immediately(self):
        assert stop_after_attempt(0)(make_state(tries=0)) is True


class TestStopAfterDelay:
    def test_stores_max_delay(self):
        s = stop_after_delay(4.5)
        assert s.max_delay == 4.5

    def test_true_at_and_above(self):
        assert stop_after_delay(2.0)(make_state(elapsed=2.0)) is True
        assert stop_after_delay(2.0)(make_state(elapsed=3.5)) is True

    def test_false_below(self):
        assert stop_after_delay(2.0)(make_state(elapsed=1.999)) is False


class TestStopBeforeDelay:
    def test_stores_max_delay(self):
        s = stop_before_delay(1.5)
        assert s.max_delay == 1.5

    def test_no_outcome_uses_only_elapsed(self):
        assert stop_before_delay(1.0)(make_state(elapsed=1.0)) is True
        assert stop_before_delay(1.0)(make_state(elapsed=0.5)) is False

    def test_outcome_wait_included(self):
        state = make_state(elapsed=0.5, wait=0.5, value=1)
        assert stop_before_delay(1.0)(state) is True

    def test_outcome_without_wait_attribute(self):
        class NoWait:
            pass

        state = make_state(elapsed=0.5, has_outcome=True)
        state.outcome = NoWait()
        assert stop_before_delay(1.0)(state) is False
        assert stop_before_delay(0.5)(state) is True

    def test_empty_outcome_is_falsy(self):
        # outcome present but falsy -> treated as no wait
        state = make_state(elapsed=0.5)
        state.outcome = FakeAttempt(value=0)
        # FakeAttempt is truthy by default; use explicit falsy object
        state.outcome = None
        assert stop_before_delay(1.0)(state) is False


class TestStopAll:
    def test_stores_stops(self):
        a = stop_after_attempt(1)
        b = stop_after_attempt(2)
        s = stop_all(a, b)
        assert s.stops == (a, b)

    def test_all_true(self):
        s = stop_all(stop_after_attempt(1), stop_after_delay(0.0))
        assert s(make_state(tries=1, elapsed=0.0)) is True

    def test_one_false(self):
        s = stop_all(stop_after_attempt(1), stop_after_attempt(5))
        assert s(make_state(tries=1)) is False

    def test_empty_is_true(self):
        assert stop_all()(make_state()) is True

    def test_short_circuits(self):
        calls = []

        def make(name, result):
            def fn(state):
                calls.append(name)
                return result

            return fn

        s = stop_all(make("a", False), make("b", True))
        assert s(make_state()) is False
        assert calls == ["a"]


class TestStopAny:
    def test_stores_stops(self):
        a = stop_after_attempt(1)
        b = stop_after_attempt(2)
        s = stop_any(a, b)
        assert s.stops == (a, b)

    def test_any_true(self):
        s = stop_any(stop_after_attempt(5), stop_after_attempt(1))
        assert s(make_state(tries=1)) is True

    def test_all_false(self):
        s = stop_any(stop_after_attempt(5), stop_after_attempt(6))
        assert s(make_state(tries=1)) is False

    def test_empty_is_false(self):
        assert stop_any()(make_state()) is False

    def test_short_circuits(self):
        calls = []

        def make(name, result):
            def fn(state):
                calls.append(name)
                return result

            return fn

        s = stop_any(make("a", True), make("b", True))
        assert s(make_state()) is True
        assert calls == ["a"]


class TestStopNever:
    def test_always_false(self):
        assert stop_never()(make_state()) is False
        assert stop_never()(make_state(tries=100, elapsed=1e9)) is False


class TestStopWhenEventSet:
    def test_stores_event(self):
        event = threading.Event()
        s = stop_when_event_set(event)
        assert s.event is event

    def test_unset(self):
        event = threading.Event()
        assert stop_when_event_set(event)(make_state()) is False

    def test_set(self):
        event = threading.Event()
        event.set()
        assert stop_when_event_set(event)(make_state()) is True


# --------------------------------------------------------------------------- #
# retry_if_exception_type
# --------------------------------------------------------------------------- #

class TestRetryIfExceptionType:
    def test_single_type_stored_as_tuple(self):
        c = retry_if_exception_type(ValueError)
        assert c.exc_types == (ValueError,)

    def test_sequence_stored_as_tuple(self):
        c = retry_if_exception_type([ValueError, TypeError])
        assert c.exc_types == (ValueError, TypeError)

    def test_no_outcome(self):
        assert retry_if_exception_type(ValueError)(make_state()) is False

    def test_no_exception(self):
        state = make_state(value=1)
        state.outcome = FakeAttempt(value=1, exception=None)
        assert retry_if_exception_type(ValueError)(state) is False

    def test_matching_exception(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("x"))
        assert retry_if_exception_type(ValueError)(state) is True

    def test_non_matching_exception(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=TypeError("x"))
        assert retry_if_exception_type(ValueError)(state) is False

    def test_subclass_matches(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("x"))
        assert retry_if_exception_type(Exception)(state) is True


# --------------------------------------------------------------------------- #
# retry_if_exception
# --------------------------------------------------------------------------- #

class TestRetryIfException:
    def test_stores_predicate(self):
        def pred(e):
            return True

        c = retry_if_exception(pred)
        assert c.predicate is pred

    def test_no_outcome(self):
        assert retry_if_exception(lambda e: True)(make_state()) is False

    def test_no_exception(self):
        state = make_state()
        state.outcome = FakeAttempt(value=1, exception=None)
        assert retry_if_exception(lambda e: True)(state) is False

    def test_bool_result(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("x"))
        assert retry_if_exception(lambda e: True)(state) is True
        assert retry_if_exception(lambda e: False)(state) is False

    def test_receives_exception(self):
        seen = []
        state = make_state()
        exc = ValueError("boom")
        state.outcome = FakeAttempt(exception=exc)

        c = retry_if_exception(lambda e: seen.append(e) or True)
        assert c(state) is True
        assert seen == [exc]

    def test_numeric_result_converted_to_float(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("x"))
        assert retry_if_exception(lambda e: 5)(state) == 5.0
        assert isinstance(retry_if_exception(lambda e: 5)(state), float)

    def test_bool_is_not_converted(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("x"))
        result = retry_if_exception(lambda e: True)(state)
        assert result is True

    def test_truthy_non_numeric_returns_bool(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("x"))
        # e.g. a list result -> bool(list)
        assert retry_if_exception(lambda e: [1, 2])(state) is True
        assert retry_if_exception(lambda e: [])(state) is False


# --------------------------------------------------------------------------- #
# retry_if_exception_message
# --------------------------------------------------------------------------- #

class TestRetryIfExceptionMessage:
    def test_stores_args(self):
        c = retry_if_exception_message("abc", match="re")
        assert c.message == "abc"
        assert c.match == "re"

    def test_no_outcome(self):
        assert retry_if_exception_message("x")(make_state()) is False

    def test_no_exception(self):
        state = make_state()
        state.outcome = FakeAttempt(value=1, exception=None)
        assert retry_if_exception_message("x")(state) is False

    def test_substring_match(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("hello world"))
        assert retry_if_exception_message("world")(state) is True
        assert retry_if_exception_message("nope")(state) is False

    def test_regex_match(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("code 123"))
        c = retry_if_exception_message(r"\d+", match="re")
        assert c(state) is True
        c2 = retry_if_exception_message(r"^\d+$", match="re")
        assert c2(state) is False

    def test_match_other_value_uses_substring(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("abc"))
        assert retry_if_exception_message("b", match="other")(state) is True


# --------------------------------------------------------------------------- #
# retry_if_exception_cause_type
# --------------------------------------------------------------------------- #

class TestRetryIfExceptionCauseType:
    def test_single_type_stored(self):
        c = retry_if_exception_cause_type(ValueError)
        assert c.exc_types == (ValueError,)

    def test_sequence_stored(self):
        c = retry_if_exception_cause_type([ValueError, TypeError])
        assert c.exc_types == (ValueError, TypeError)

    def test_no_outcome(self):
        assert retry_if_exception_cause_type(ValueError)(make_state()) is False

    def test_no_exception(self):
        state = make_state()
        state.outcome = FakeAttempt(value=1, exception=None)
        assert retry_if_exception_cause_type(ValueError)(state) is False

    def test_direct_cause(self):
        try:
            try:
                raise ValueError("inner")
            except ValueError as inner:
                raise RuntimeError("outer") from inner
        except RuntimeError as outer:
            state = make_state()
            state.outcome = FakeAttempt(exception=outer)
            assert retry_if_exception_cause_type(ValueError)(state) is True
            assert retry_if_exception_cause_type(TypeError)(state) is False

    def test_context_fallback(self):
        exc = RuntimeError("outer")
        ctx = ValueError("inner")
        exc.__context__ = ctx
        state = make_state()
        state.outcome = FakeAttempt(exception=exc)
        assert retry_if_exception_cause_type(ValueError)(state) is True

    def test_nested_chain(self):
        try:
            try:
                try:
                    raise KeyError("deep")
                except KeyError as deepest:
                    raise ValueError("mid") from deepest
            except ValueError as mid:
                raise RuntimeError("top") from mid
        except RuntimeError as top:
            state = make_state()
            state.outcome = FakeAttempt(exception=top)
            assert retry_if_exception_cause_type(KeyError)(state) is True

    def test_no_cause_returns_false(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("x"))
        assert retry_if_exception_cause_type(ValueError)(state) is False


# --------------------------------------------------------------------------- #
# retry_if_result / retry_if_not_result
# --------------------------------------------------------------------------- #

class TestRetryIfResult:
    def test_stores_predicate(self):
        def pred(v):
            return True

        c = retry_if_result(pred)
        assert c.predicate is pred

    def test_no_outcome(self):
        assert retry_if_result(lambda v: True)(make_state()) is False

    def test_exception_outcome_returns_false(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("x"))
        assert retry_if_result(lambda v: True)(state) is False

    def test_predicate_true(self):
        state = make_state()
        state.outcome = FakeAttempt(value=42)
        assert retry_if_result(lambda v: v == 42)(state) is True
        assert retry_if_result(lambda v: v == 0)(state) is False

    def test_predicate_receives_value(self):
        seen = []
        state = make_state()
        state.outcome = FakeAttempt(value="abc")
        retry_if_result(lambda v: seen.append(v) or False)(state)
        assert seen == ["abc"]


class TestRetryIfNotResult:
    def test_stores_predicate(self):
        def pred(v):
            return True

        c = retry_if_not_result(pred)
        assert c.predicate is pred

    def test_no_outcome(self):
        assert retry_if_not_result(lambda v: True)(make_state()) is False

    def test_exception_outcome_returns_false(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("x"))
        assert retry_if_not_result(lambda v: True)(state) is False

    def test_negates_predicate(self):
        state = make_state()
        state.outcome = FakeAttempt(value=42)
        assert retry_if_not_result(lambda v: v == 42)(state) is False
        assert retry_if_not_result(lambda v: v == 0)(state) is True


# --------------------------------------------------------------------------- #
# retry_all / retry_any
# --------------------------------------------------------------------------- #

class TestRetryAll:
    def test_stores_conditions(self):
        a = retry_always()
        b = retry_never()
        c = retry_all(a, b)
        assert c.conditions == (a, b)

    def test_empty_is_true(self):
        assert retry_all()(make_state()) is True

    def test_all_true(self):
        assert retry_all(retry_always(), retry_always())(make_state()) is True

    def test_one_false(self):
        assert retry_all(retry_always(), retry_never())(make_state()) is False

    def test_short_circuits_on_false(self):
        calls = []

        def make(name, result):
            def fn(state):
                calls.append(name)
                return result

            return fn

        c = retry_all(make("a", False), make("b", True))
        assert c(make_state()) is False
        assert calls == ["a"]

    def test_numeric_results_do_not_cause_false(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("x"))

        # retry_if_exception returns 0.0 (numeric) - should be treated as "continue"
        numeric = retry_if_exception(lambda e: 0)
        result = retry_all(numeric, retry_always())(state)
        assert result is True


class TestRetryAny:
    def test_stores_conditions(self):
        a = retry_always()
        b = retry_never()
        c = retry_any(a, b)
        assert c.conditions == (a, b)

    def test_empty_is_false(self):
        assert retry_any()(make_state()) is False

    def test_one_true(self):
        assert retry_any(retry_never(), retry_always())(make_state()) is True

    def test_all_false(self):
        assert retry_any(retry_never(), retry_never())(make_state()) is False

    def test_short_circuits_on_true(self):
        calls = []

        def make(name, result):
            def fn(state):
                calls.append(name)
                return result

            return fn

        c = retry_any(make("a", True), make("b", True))
        assert c(make_state()) is True
        assert calls == ["a"]

    def test_numeric_result_returned_as_float(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("x"))

        numeric = retry_if_exception(lambda e: 3)
        result = retry_any(retry_never(), numeric, retry_always())(state)
        assert result == 3.0
        assert isinstance(result, float)

    def test_bool_true_returned_as_true(self):
        result = retry_any(retry_never(), retry_always())(make_state())
        assert result is True


# --------------------------------------------------------------------------- #
# retry_always / retry_never
# --------------------------------------------------------------------------- #

class TestRetryAlwaysNever:
    def test_retry_always(self):
        assert retry_always()(make_state()) is True

    def test_retry_never(self):
        assert retry_never()(make_state()) is False


# --------------------------------------------------------------------------- #
# retry_if_not_exception_type / retry_unless_exception_type
# --------------------------------------------------------------------------- #

class TestRetryIfNotExceptionType:
    def test_single_type_stored(self):
        c = retry_if_not_exception_type(ValueError)
        assert c.exc_types == (ValueError,)

    def test_sequence_stored(self):
        c = retry_if_not_exception_type([ValueError, TypeError])
        assert c.exc_types == (ValueError, TypeError)

    def test_no_outcome(self):
        assert retry_if_not_exception_type(ValueError)(make_state()) is False

    def test_no_exception(self):
        state = make_state()
        state.outcome = FakeAttempt(value=1, exception=None)
        assert retry_if_not_exception_type(ValueError)(state) is False

    def test_not_matching(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=TypeError("x"))
        assert retry_if_not_exception_type(ValueError)(state) is True

    def test_matching(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("x"))
        assert retry_if_not_exception_type(ValueError)(state) is False

    def test_subclass_matches(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("x"))
        assert retry_if_not_exception_type(Exception)(state) is False


class TestRetryUnlessExceptionType:
    def test_subclass_of_not_exception_type(self):
        assert issubclass(retry_unless_exception_type, retry_if_not_exception_type)

    def test_behavior(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=TypeError("x"))
        assert retry_unless_exception_type(ValueError)(state) is True

        state2 = make_state()
        state2.outcome = FakeAttempt(exception=ValueError("x"))
        assert retry_unless_exception_type(ValueError)(state2) is False


# --------------------------------------------------------------------------- #
# retry_if_not_exception_message
# --------------------------------------------------------------------------- #

class TestRetryIfNotExceptionMessage:
    def test_stores_args(self):
        c = retry_if_not_exception_message("abc", regex=True)
        assert c.match == "abc"
        assert c.regex is True

    def test_default_regex_false(self):
        c = retry_if_not_exception_message("abc")
        assert c.regex is False

    def test_no_outcome(self):
        assert retry_if_not_exception_message("x")(make_state()) is False

    def test_no_exception(self):
        state = make_state()
        state.outcome = FakeAttempt(value=1, exception=None)
        assert retry_if_not_exception_message("x")(state) is False

    def test_substring_not_present(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("hello world"))
        assert retry_if_not_exception_message("nope")(state) is True
        assert retry_if_not_exception_message("world")(state) is False

    def test_regex(self):
        state = make_state()
        state.outcome = FakeAttempt(exception=ValueError("code 123"))
        assert retry_if_not_exception_message(r"\d+", regex=True)(state) is False
        assert retry_if_not_exception_message(r"[a-z]+$", regex=True)(state) is True
