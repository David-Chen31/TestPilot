from __future__ import annotations

import re
import threading

import pytest

from backon._conditions import (
    Stop,
    RetryCondition,
    stop_after_attempt,
    stop_after_delay,
    stop_before_delay,
    stop_all,
    stop_any,
    stop_never,
    stop_when_event_set,
    retry_if_exception_type,
    retry_if_exception,
    retry_if_exception_message,
    retry_if_result,
    retry_if_not_result,
    retry_all,
    retry_any,
    retry_always,
    retry_never,
    retry_if_not_exception_type,
    retry_unless_exception_type,
    retry_if_not_exception_message,
    retry_if_exception_cause_type,
)
from backon._state import Attempt, RetryState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(**kwargs) -> RetryState:
    return RetryState(**kwargs)


def make_exc_outcome(exc: BaseException) -> Attempt:
    return Attempt(value=None, exception=exc)


def make_value_outcome(value) -> Attempt:
    return Attempt(value=value, exception=None)


class FakeWaitOutcome:
    """Mimics an Attempt-like object exposing only a .wait attribute."""

    def __init__(self, wait):
        self.wait = wait


# ---------------------------------------------------------------------------
# Base class behavior
# ---------------------------------------------------------------------------

class TestStopBase:
    def test_call_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Stop()(make_state())

    def test_or_returns_stop_any(self):
        a = stop_never()
        b = stop_never()
        combined = a | b
        assert isinstance(combined, stop_any)
        assert combined.stops == (a, b)

    def test_and_returns_stop_all(self):
        a = stop_never()
        b = stop_never()
        combined = a & b
        assert isinstance(combined, stop_all)
        assert combined.stops == (a, b)


class TestRetryConditionBase:
    def test_call_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            RetryCondition()(make_state())

    def test_or_returns_retry_any(self):
        a = retry_never()
        b = retry_never()
        combined = a | b
        assert isinstance(combined, retry_any)
        assert combined.conditions == (a, b)

    def test_and_returns_retry_all(self):
        a = retry_never()
        b = retry_never()
        combined = a & b
        assert isinstance(combined, retry_all)
        assert combined.conditions == (a, b)


# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------

class TestStopAfterAttempt:
    def test_stores_max_attempts(self):
        c = stop_after_attempt(5)
        assert c.max_attempts == 5

    def test_false_below_threshold(self):
        c = stop_after_attempt(3)
        assert c(make_state(tries=0)) is False
        assert c(make_state(tries=2)) is False

    def test_true_at_threshold(self):
        c = stop_after_attempt(3)
        assert c(make_state(tries=3)) is True

    def test_true_above_threshold(self):
        c = stop_after_attempt(3)
        assert c(make_state(tries=10)) is True

    def test_zero_attempts_always_true(self):
        c = stop_after_attempt(0)
        assert c(make_state(tries=0)) is True


class TestStopAfterDelay:
    def test_stores_max_delay(self):
        c = stop_after_delay(2.5)
        assert c.max_delay == 2.5

    def test_false_below(self):
        c = stop_after_delay(10.0)
        assert c(make_state(elapsed=9.999)) is False

    def test_true_at_threshold(self):
        c = stop_after_delay(10.0)
        assert c(make_state(elapsed=10.0)) is True

    def test_true_above(self):
        c = stop_after_delay(10.0)
        assert c(make_state(elapsed=11.0)) is True


class TestStopBeforeDelay:
    def test_stores_max_delay(self):
        c = stop_before_delay(1.5)
        assert c.max_delay == 1.5

    def test_no_outcome_uses_zero_wait(self):
        c = stop_before_delay(5.0)
        assert c(make_state(elapsed=4.0, outcome=None)) is False
        assert c(make_state(elapsed=5.0, outcome=None)) is True

    def test_outcome_with_wait_added(self):
        c = stop_before_delay(5.0)
        state = make_state(elapsed=3.0, outcome=FakeWaitOutcome(1.0))
        assert c(state) is False  # 4.0 < 5.0

    def test_outcome_with_wait_reaches_threshold(self):
        c = stop_before_delay(5.0)
        state = make_state(elapsed=3.0, outcome=FakeWaitOutcome(2.0))
        assert c(state) is True  # 5.0 >= 5.0

    def test_outcome_without_wait_attribute_defaults_zero(self):
        c = stop_before_delay(5.0)
        state = make_state(elapsed=5.0, outcome=object())
        assert c(state) is True

    def test_falsy_outcome_uses_zero_wait(self):
        c = stop_before_delay(5.0)
        # outcome that is falsy -> 0.0 wait
        state = make_state(elapsed=5.0, outcome=FakeWaitOutcome(100.0))
        # FakeWaitOutcome instance is truthy; use empty container instead
        class EmptyList(list):
            wait = 100.0

        state2 = make_state(elapsed=5.0, outcome=EmptyList())
        assert c(state2) is True  # falsy -> wait 0.0, elapsed 5.0 >= 5.0


class TestStopAll:
    def test_empty_is_true(self):
        assert stop_all()(make_state()) is True

    def test_all_true(self):
        c = stop_all(stop_after_attempt(1), stop_after_delay(0.0))
        assert c(make_state(tries=5, elapsed=0.0)) is True

    def test_one_false(self):
        c = stop_all(stop_after_attempt(10), stop_after_delay(0.0))
        assert c(make_state(tries=0, elapsed=0.0)) is False

    def test_short_circuits(self):
        calls = []

        class Recording(Stop):
            def __init__(self, name, result):
                self.name = name
                self.result = result

            def __call__(self, state):
                calls.append(self.name)
                return self.result

        c = stop_all(Recording("a", False), Recording("b", True))
        assert c(make_state()) is False
        assert calls == ["a"]


class TestStopAny:
    def test_empty_is_false(self):
        assert stop_any()(make_state()) is False

    def test_one_true(self):
        c = stop_any(stop_after_attempt(0), stop_after_delay(100.0))
        assert c(make_state(tries=0)) is True

    def test_all_false(self):
        c = stop_any(stop_after_attempt(10), stop_after_delay(100.0))
        assert c(make_state(tries=0, elapsed=0.0)) is False

    def test_short_circuits(self):
        calls = []

        class Recording(Stop):
            def __init__(self, name, result):
                self.name = name
                self.result = result

            def __call__(self, state):
                calls.append(self.name)
                return self.result

        c = stop_any(Recording("a", True), Recording("b", False))
        assert c(make_state()) is True
        assert calls == ["a"]


class TestStopNever:
    def test_always_false(self):
        c = stop_never()
        assert c(make_state()) is False
        assert c(make_state(tries=1000, elapsed=1e9)) is False


class TestStopWhenEventSet:
    def test_stores_event(self):
        ev = threading.Event()
        c = stop_when_event_set(ev)
        assert c.event is ev

    def test_false_when_not_set(self):
        ev = threading.Event()
        c = stop_when_event_set(ev)
        assert c(make_state()) is False

    def test_true_when_set(self):
        ev = threading.Event()
        ev.set()
        c = stop_when_event_set(ev)
        assert c(make_state()) is True


# ---------------------------------------------------------------------------
# Retry conditions
# ---------------------------------------------------------------------------

class TestRetryIfExceptionType:
    def test_single_type_wrapped_in_tuple(self):
        c = retry_if_exception_type(ValueError)
        assert c.exc_types == (ValueError,)

    def test_sequence_converted_to_tuple(self):
        c = retry_if_exception_type([ValueError, TypeError])
        assert c.exc_types == (ValueError, TypeError)

    def test_no_outcome_false(self):
        c = retry_if_exception_type(ValueError)
        assert c(make_state(outcome=None)) is False

    def test_matching_exception_true(self):
        c = retry_if_exception_type(ValueError)
        assert c(make_state(outcome=make_exc_outcome(ValueError()))) is True

    def test_subclass_matches(self):
        c = retry_if_exception_type(Exception)
        assert c(make_state(outcome=make_exc_outcome(ValueError()))) is True

    def test_non_matching_exception_false(self):
        c = retry_if_exception_type(ValueError)
        assert c(make_state(outcome=make_exc_outcome(TypeError()))) is False

    def test_value_outcome_false(self):
        c = retry_if_exception_type(ValueError)
        assert c(make_state(outcome=make_value_outcome(1))) is False

    def test_multiple_types(self):
        c = retry_if_exception_type((ValueError, TypeError))
        assert c(make_state(outcome=make_exc_outcome(TypeError()))) is True


class TestRetryIfException:
    def test_no_outcome_false(self):
        c = retry_if_exception(lambda e: True)
        assert c(make_state(outcome=None)) is False

    def test_value_outcome_false(self):
        c = retry_if_exception(lambda e: True)
        assert c(make_state(outcome=make_value_outcome(1))) is False

    def test_predicate_true(self):
        c = retry_if_exception(lambda e: isinstance(e, ValueError))
        assert c(make_state(outcome=make_exc_outcome(ValueError()))) is True

    def test_predicate_false(self):
        c = retry_if_exception(lambda e: isinstance(e, ValueError))
        assert c(make_state(outcome=make_exc_outcome(TypeError()))) is False

    def test_numeric_result_converted_to_float(self):
        c = retry_if_exception(lambda e: 3)
        result = c(make_state(outcome=make_exc_outcome(ValueError())))
        assert result == 3.0
        assert isinstance(result, float)

    def test_float_result_preserved(self):
        c = retry_if_exception(lambda e: 2.5)
        result = c(make_state(outcome=make_exc_outcome(ValueError())))
        assert result == 2.5
        assert isinstance(result, float)

    def test_bool_true_not_converted(self):
        c = retry_if_exception(lambda e: True)
        result = c(make_state(outcome=make_exc_outcome(ValueError())))
        assert result is True

    def test_bool_false_not_converted(self):
        c = retry_if_exception(lambda e: False)
        result = c(make_state(outcome=make_exc_outcome(ValueError())))
        assert result is False

    def test_non_numeric_result_coerced_to_bool(self):
        c = retry_if_exception(lambda e: "yes")
        result = c(make_state(outcome=make_exc_outcome(ValueError())))
        assert result is True

        c2 = retry_if_exception(lambda e: "")
        result2 = c2(make_state(outcome=make_exc_outcome(ValueError())))
        assert result2 is False


class TestRetryIfExceptionMessage:
    def test_stores_attributes(self):
        c = retry_if_exception_message("boom")
        assert c.message == "boom"
        assert c.match is None

    def test_no_outcome_false(self):
        c = retry_if_exception_message("boom")
        assert c(make_state(outcome=None)) is False

    def test_value_outcome_false(self):
        c = retry_if_exception_message("boom")
        assert c(make_state(outcome=make_value_outcome(1))) is False

    def test_substring_match_true(self):
        c = retry_if_exception_message("boom")
        assert c(make_state(outcome=make_exc_outcome(ValueError("kaboom!")))) is True

    def test_substring_no_match_false(self):
        c = retry_if_exception_message("boom")
        assert c(make_state(outcome=make_exc_outcome(ValueError("nothing")))) is False

    def test_regex_match_true(self):
        c = retry_if_exception_message(r"bo+m", match="re")
        assert c(make_state(outcome=make_exc_outcome(ValueError("boom")))) is True

    def test_regex_no_match_false(self):
        c = retry_if_exception_message(r"bo+m", match="re")
        assert c(make_state(outcome=make_exc_outcome(ValueError("bang")))) is False

    def test_regex_uses_search(self):
        c = retry_if_exception_message(r"\d+", match="re")
        assert c(make_state(outcome=make_exc_outcome(ValueError("error 42")))) is True


class TestRetryIfResult:
    def test_stores_predicate(self):
        pred = lambda v: v > 0
        c = retry_if_result(pred)
        assert c.predicate is pred

    def test_no_outcome_false(self):
        c = retry_if_result(lambda v: True)
        assert c(make_state(outcome=None)) is False

    def test_exception_outcome_false(self):
        c = retry_if_result(lambda v: True)
        assert c(make_state(outcome=make_exc_outcome(ValueError()))) is False

    def test_predicate_true(self):
        c = retry_if_result(lambda v: v == 1)
        assert c(make_state(outcome=make_value_outcome(1))) is True

    def test_predicate_false(self):
        c = retry_if_result(lambda v: v == 1)
        assert c(make_state(outcome=make_value_outcome(2))) is False

    def test_predicate_receives_value(self):
        seen = []

        def pred(v):
            seen.append(v)
            return False

        c = retry_if_result(pred)
        c(make_state(outcome=make_value_outcome("payload")))
        assert seen == ["payload"]


class TestRetryIfNotResult:
    def test_stores_predicate(self):
        pred = lambda v: v > 0
        c = retry_if_not_result(pred)
        assert c.predicate is pred

    def test_no_outcome_false(self):
        c = retry_if_not_result(lambda v: True)
        assert c(make_state(outcome=None)) is False

    def test_exception_outcome_false(self):
        c = retry_if_not_result(lambda v: True)
        assert c(make_state(outcome=make_exc_outcome(ValueError()))) is False

    def test_predicate_true_negated_to_false(self):
        c = retry_if_not_result(lambda v: v == 1)
        assert c(make_state(outcome=make_value_outcome(1))) is False

    def test_predicate_false_negated_to_true(self):
        c = retry_if_not_result(lambda v: v == 1)
        assert c(make_state(outcome=make_value_outcome(2))) is True


class TestRetryAll:
    def test_stores_conditions(self):
        a = retry_always()
        b = retry_never()
        c = retry_all(a, b)
        assert c.conditions == (a, b)

    def test_empty_is_true(self):
        assert retry_all()(make_state()) is True

    def test_all_true(self):
        c = retry_all(retry_always(), retry_always())
        assert c(make_state()) is True

    def test_one_false(self):
        c = retry_all(retry_always(), retry_never())
        assert c(make_state()) is False

    def test_numeric_result_does_not_short_circuit(self):
        # A numeric result is not a bool, so retry_all continues.
        calls = []

        class Numeric(RetryCondition):
            def __init__(self, name, value):
                self.name = name
                self.value = value

            def __call__(self, state):
                calls.append(self.name)
                return self.value

        c = retry_all(Numeric("a", 2.0), Numeric("b", True))
        result = c(make_state())
        assert result is True
        assert calls == ["a", "b"]

    def test_short_circuits_on_false(self):
        calls = []

        class Recording(RetryCondition):
            def __init__(self, name, result):
                self.name = name
                self.result = result

            def __call__(self, state):
                calls.append(self.name)
                return self.result

        c = retry_all(Recording("a", False), Recording("b", True))
        assert c(make_state()) is False
        assert calls == ["a"]


class TestRetryAny:
    def test_stores_conditions(self):
        a = retry_always()
        b = retry_never()
        c = retry_any(a, b)
        assert c.conditions == (a, b)

    def test_empty_is_false(self):
        assert retry_any()(make_state()) is False

    def test_all_false(self):
        c = retry_any(retry_never(), retry_never())
        assert c(make_state()) is False

    def test_one_true(self):
        c = retry_any(retry_never(), retry_always())
        assert c(make_state()) is True

    def test_numeric_result_returned_as_float(self):
        c = retry_any(retry_if_exception(lambda e: 4))
        result = c(make_state(outcome=make_exc_outcome(ValueError())))
        assert result == 4.0
        assert isinstance(result, float)

    def test_bool_true_short_circuits(self):
        calls = []

        class Recording(RetryCondition):
            def __init__(self, name, result):
                self.name = name
                self.result = result

            def __call__(self, state):
                calls.append(self.name)
                return self.result

        c = retry_any(Recording("a", True), Recording("b", True))
        assert c(make_state()) is True
        assert calls == ["a"]

    def test_numeric_after_false(self):
        class Numeric(RetryCondition):
            def __call__(self, state):
                return 1.5

        c = retry_any(retry_never(), Numeric())
        result = c(make_state())
        assert result == 1.5
        assert isinstance(result, float)


class TestRetryAlwaysNever:
    def test_always_true(self):
        assert retry_always()(make_state()) is True
        assert retry_always()(make_state(tries=100)) is True

    def test_never_false(self):
        assert retry_never()(make_state()) is False
        assert retry_never()(make_state(tries=100)) is False


class TestRetryIfNotExceptionType:
    def test_single_type_wrapped(self):
        c = retry_if_not_exception_type(ValueError)
        assert c.exc_types == (ValueError,)

    def test_sequence_converted(self):
        c = retry_if_not_exception_type([ValueError, TypeError])
        assert c.exc_types == (ValueError, TypeError)

    def test_no_outcome_false(self):
        c = retry_if_not_exception_type(ValueError)
        assert c(make_state(outcome=None)) is False

    def test_value_outcome_false(self):
        c = retry_if_not_exception_type(ValueError)
        assert c(make_state(outcome=make_value_outcome(1))) is False

    def test_matching_type_returns_false(self):
        c = retry_if_not_exception_type(ValueError)
        assert c(make_state(outcome=make_exc_outcome(ValueError()))) is False

    def test_non_matching_type_returns_true(self):
        c = retry_if_not_exception_type(ValueError)
        assert c(make_state(outcome=make_exc_outcome(TypeError()))) is True


class TestRetryUnlessExceptionType:
    def test_is_subclass(self):
        assert issubclass(retry_unless_exception_type, retry_if_not_exception_type)

    def test_behavior_matches(self):
        c = retry_unless_exception_type(ValueError)
        assert c.exc_types == (ValueError,)
        assert c(make_state(outcome=make_exc_outcome(ValueError()))) is False
        assert c(make_state(outcome=make_exc_outcome(TypeError()))) is True
        assert c(make_state(outcome=None)) is False


class TestRetryIfNotExceptionMessage:
    def test_stores_attributes(self):
        c = retry_if_not_exception_message("boom")
        assert c.match == "boom"
        assert c.regex is False

    def test_no_outcome_false(self):
        c = retry_if_not_exception_message("boom")
        assert c(make_state(outcome=None)) is False

    def test_value_outcome_false(self):
        c = retry_if_not_exception_message("boom")
        assert c(make_state(outcome=make_value_outcome(1))) is False

    def test_substring_present_returns_false(self):
        c = retry_if_not_exception_message("boom")
        assert c(make_state(outcome=make_exc_outcome(ValueError("kaboom")))) is False

    def test_substring_absent_returns_true(self):
        c = retry_if_not_exception_message("boom")
        assert c(make_state(outcome=make_exc_outcome(ValueError("bang")))) is True

    def test_regex_present_returns_false(self):
        c = retry_if_not_exception_message(r"bo+m", regex=True)
        assert c(make_state(outcome=make_exc_outcome(ValueError("boom")))) is False

    def test_regex_absent_returns_true(self):
        c = retry_if_not_exception_message(r"bo+m", regex=True)
        assert c(make_state(outcome=make_exc_outcome(ValueError("bang")))) is True

    def test_regex_keyword_only(self):
        with pytest.raises(TypeError):
            retry_if_not_exception_message("x", True)  # type: ignore[misc]


class TestRetryIfExceptionCauseType:
    def test_single_type_wrapped(self):
        c = retry_if_exception_cause_type(ValueError)
        assert c.exc_types == (ValueError,)

    def test_sequence_converted(self):
        c = retry_if_exception_cause_type([ValueError, TypeError])
        assert c.exc_types == (ValueError, TypeError)

    def test_no_outcome_false(self):
        c = retry_if_exception_cause_type(ValueError)
        assert c(make_state(outcome=None)) is False

    def test_value_outcome_false(self):
        c = retry_if_exception_cause_type(ValueError)
        assert c(make_state(outcome=make_value_outcome(1))) is False

    def test_no_cause_false(self):
        c = retry_if_exception_cause_type(ValueError)
        assert c(make_state(outcome=make_exc_outcome(RuntimeError("x")))) is False

    def test_direct_cause_true(self):
        try:
            try:
                raise ValueError("inner")
            except ValueError as inner:
                raise RuntimeError("outer") from inner
        except RuntimeError as outer:
            exc = outer
        c = retry_if_exception_cause_type(ValueError)
        assert c(make_state(outcome=make_exc_outcome(exc))) is True

    def test_context_cause_true(self):
        try:
            try:
                raise ValueError("inner")
            except ValueError:
                raise RuntimeError("outer")
        except RuntimeError as outer:
            exc = outer
        assert exc.__cause__ is None
        assert isinstance(exc.__context__, ValueError)
        c = retry_if_exception_cause_type(ValueError)
        assert c(make_state(outcome=make_exc_outcome(exc))) is True

    def test_nested_cause_true(self):
        try:
            try:
                try:
                    raise KeyError("deep")
                except KeyError:
                    raise ValueError("mid")
            except ValueError as mid:
                raise RuntimeError("outer") from mid
        except RuntimeError as outer:
            exc = outer
        c = retry_if_exception_cause_type(KeyError)
        assert c(make_state(outcome=make_exc_outcome(exc))) is True

    def test_wrong_cause_type_false(self):
        try:
            try:
                raise KeyError("inner")
            except KeyError as inner:
                raise RuntimeError("outer") from inner
        except RuntimeError as outer:
            exc = outer
        c = retry_if_exception_cause_type(ValueError)
        assert c(make_state(outcome=make_exc_outcome(exc))) is False


# ---------------------------------------------------------------------------
# Operator composition integration
# ---------------------------------------------------------------------------

class TestOperatorComposition:
    def test_stop_or_short_circuit(self):
        calls = []
        a = stop_after_attempt(0)
        b = stop_never()
        combined = a | b
        assert combined(make_state(tries=0)) is True

    def test_stop_and_only_true_when_both(self):
        combined = stop_after_attempt(1) & stop_after_delay(0.0)
        assert combined(make_state(tries=1, elapsed=0.0)) is True
        assert combined(make_state(tries=0, elapsed=0.0)) is False
        assert combined(make_state(tries=1, elapsed=-1.0)) is False

    def test_retry_or_short_circuit(self):
        combined = retry_always() | retry_never()
        assert combined(make_state()) is True

    def test_retry_and(self):
        combined = retry_always() & retry_never()
        assert combined(make_state()) is False

    def test_deep_composition(self):
        combined = (retry_always() & retry_never()) | retry_always()
        assert combined(make_state()) is True
