from __future__ import annotations

import re
import threading

import pytest

from backon._conditions import (
    Stop,
    stop_after_attempt,
    stop_after_delay,
    stop_before_delay,
    stop_all,
    stop_any,
    stop_never,
    stop_when_event_set,
    RetryCondition,
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
from backon._state import RetryState


class Outcome:
    """A minimal stand-in for the Attempt object stored in RetryState.outcome."""

    def __init__(self, value=None, exception=None, wait=0.0):
        self.value = value
        self.exception = exception
        self.wait = wait


def make_state(tries=0, elapsed=0.0, outcome=None, idle_for=0.0):
    return RetryState(tries=tries, elapsed=elapsed, outcome=outcome, idle_for=idle_for)


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


def test_stop_base_not_implemented():
    with pytest.raises(NotImplementedError):
        Stop()(make_state())


def test_retry_condition_base_not_implemented():
    with pytest.raises(NotImplementedError):
        RetryCondition()(make_state())


# ---------------------------------------------------------------------------
# stop_after_attempt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tries, max_attempts, expected",
    [
        (0, 0, True),
        (0, 1, False),
        (1, 1, True),
        (2, 1, True),
        (4, 5, False),
        (5, 5, True),
        (6, 5, True),
    ],
)
def test_stop_after_attempt(tries, max_attempts, expected):
    cond = stop_after_attempt(max_attempts)
    assert cond.max_attempts == max_attempts
    assert cond(make_state(tries=tries)) is expected


# ---------------------------------------------------------------------------
# stop_after_delay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "elapsed, max_delay, expected",
    [
        (0.0, 0.0, True),
        (0.0, 1.0, False),
        (1.0, 1.0, True),
        (0.5, 1.0, False),
        (2.0, 1.0, True),
    ],
)
def test_stop_after_delay(elapsed, max_delay, expected):
    cond = stop_after_delay(max_delay)
    assert cond.max_delay == max_delay
    assert cond(make_state(elapsed=elapsed)) is expected


# ---------------------------------------------------------------------------
# stop_before_delay
# ---------------------------------------------------------------------------


def test_stop_before_delay_no_outcome():
    cond = stop_before_delay(1.0)
    # wait defaults to 0.0 when there is no outcome
    assert cond(make_state(elapsed=0.5)) is False
    assert cond(make_state(elapsed=1.0)) is True
    assert cond.make_delay if False else True  # no-op to keep flake quiet


def test_stop_before_delay_with_wait():
    cond = stop_before_delay(2.0)
    state = make_state(elapsed=1.0, outcome=Outcome(wait=1.0))
    assert cond(state) is True  # 1.0 + 1.0 >= 2.0
    state2 = make_state(elapsed=1.0, outcome=Outcome(wait=0.5))
    assert cond(state2) is False  # 1.5 < 2.0


def test_stop_before_delay_outcome_without_wait_attr():
    class NoWait:
        pass

    cond = stop_before_delay(1.0)
    state = make_state(elapsed=0.5, outcome=NoWait())
    assert cond(state) is False
    state2 = make_state(elapsed=1.0, outcome=NoWait())
    assert cond(state2) is True


def test_stop_before_delay_stores_max_delay():
    cond = stop_before_delay(3.5)
    assert cond.max_delay == 3.5


# ---------------------------------------------------------------------------
# stop_all / stop_any
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tries, elapsed, expected",
    [
        (0, 0.0, False),
        (5, 0.0, False),
        (0, 1.0, False),
        (5, 1.0, True),
    ],
)
def test_stop_all(tries, elapsed, expected):
    cond = stop_all(stop_after_attempt(5), stop_after_delay(1.0))
    assert len(cond.stops) == 2
    assert cond(make_state(tries=tries, elapsed=elapsed)) is expected


def test_stop_all_empty():
    # all() on empty is True
    assert stop_all()(make_state()) is True


def test_stop_all_short_circuits_false():
    calls = []

    class Recorder(Stop):
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def __call__(self, state):
            calls.append(self.name)
            return self.value

    cond = stop_all(Recorder("a", False), Recorder("b", True))
    assert cond(make_state()) is False
    assert calls == ["a"]


@pytest.mark.parametrize(
    "tries, elapsed, expected",
    [
        (0, 0.0, False),
        (5, 0.0, True),
        (0, 1.0, True),
        (5, 1.0, True),
    ],
)
def test_stop_any(tries, elapsed, expected):
    cond = stop_any(stop_after_attempt(5), stop_after_delay(1.0))
    assert len(cond.stops) == 2
    assert cond(make_state(tries=tries, elapsed=elapsed)) is expected


def test_stop_any_empty():
    # any() on empty is False
    assert stop_any()(make_state()) is False


def test_stop_any_short_circuits_true():
    calls = []

    class Recorder(Stop):
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def __call__(self, state):
            calls.append(self.name)
            return self.value

    cond = stop_any(Recorder("a", True), Recorder("b", False))
    assert cond(make_state()) is True
    assert calls == ["a"]


def test_stop_or_operator():
    cond = stop_after_attempt(3) | stop_after_delay(1.0)
    assert isinstance(cond, stop_any)
    assert cond(make_state(tries=3)) is True
    assert cond(make_state(tries=0, elapsed=1.0)) is True
    assert cond(make_state(tries=0, elapsed=0.0)) is False


def test_stop_and_operator():
    cond = stop_after_attempt(3) & stop_after_delay(1.0)
    assert isinstance(cond, stop_all)
    assert cond(make_state(tries=3, elapsed=1.0)) is True
    assert cond(make_state(tries=3, elapsed=0.0)) is False
    assert cond(make_state(tries=0, elapsed=1.0)) is False


# ---------------------------------------------------------------------------
# stop_never
# ---------------------------------------------------------------------------


def test_stop_never():
    cond = stop_never()
    assert cond(make_state()) is False
    assert cond(make_state(tries=1000, elapsed=999.0)) is False


# ---------------------------------------------------------------------------
# stop_when_event_set
# ---------------------------------------------------------------------------


def test_stop_when_event_set():
    event = threading.Event()
    cond = stop_when_event_set(event)
    assert cond(make_state()) is False
    event.set()
    assert cond(make_state()) is True
    event.clear()
    assert cond(make_state()) is False


# ---------------------------------------------------------------------------
# retry_if_exception_type
# ---------------------------------------------------------------------------


def test_retry_if_exception_type_single():
    cond = retry_if_exception_type(ValueError)
    assert cond.exc_types == (ValueError,)
    assert cond(make_state(outcome=Outcome(exception=ValueError("x")))) is True
    assert cond(make_state(outcome=Outcome(exception=TypeError("x")))) is False


def test_retry_if_exception_type_sequence():
    cond = retry_if_exception_type((ValueError, TypeError))
    assert cond.exc_types == (ValueError, TypeError)
    assert cond(make_state(outcome=Outcome(exception=ValueError()))) is True
    assert cond(make_state(outcome=Outcome(exception=TypeError()))) is True
    assert cond(make_state(outcome=Outcome(exception=KeyError()))) is False


def test_retry_if_exception_type_subclass():
    cond = retry_if_exception_type(Exception)
    assert cond(make_state(outcome=Outcome(exception=ValueError()))) is True


def test_retry_if_exception_type_no_outcome():
    cond = retry_if_exception_type(ValueError)
    assert cond(make_state(outcome=None)) is False


def test_retry_if_exception_type_no_exception():
    cond = retry_if_exception_type(ValueError)
    assert cond(make_state(outcome=Outcome(value=1))) is False


# ---------------------------------------------------------------------------
# retry_if_exception
# ---------------------------------------------------------------------------


def test_retry_if_exception_bool():
    cond = retry_if_exception(lambda exc: isinstance(exc, ValueError))
    assert cond(make_state(outcome=Outcome(exception=ValueError()))) is True
    assert cond(make_state(outcome=Outcome(exception=TypeError()))) is False


def test_retry_if_exception_numeric_returns_float():
    cond = retry_if_exception(lambda exc: 2)
    result = cond(make_state(outcome=Outcome(exception=ValueError())))
    assert result == 2.0
    assert isinstance(result, float)


def test_retry_if_exception_float_returned():
    cond = retry_if_exception(lambda exc: 1.5)
    result = cond(make_state(outcome=Outcome(exception=ValueError())))
    assert result == 1.5
    assert isinstance(result, float)


def test_retry_if_exception_bool_not_converted_to_float():
    cond = retry_if_exception(lambda exc: True)
    result = cond(make_state(outcome=Outcome(exception=ValueError())))
    assert result is True


def test_retry_if_exception_passes_exception_object():
    seen = []

    def predicate(exc):
        seen.append(exc)
        return False

    exc_instance = ValueError("boom")
    cond = retry_if_exception(predicate)
    cond(make_state(outcome=Outcome(exception=exc_instance)))
    assert seen == [exc_instance]


def test_retry_if_exception_no_outcome():
    cond = retry_if_exception(lambda exc: True)
    assert cond(make_state(outcome=None)) is False


def test_retry_if_exception_no_exception():
    cond = retry_if_exception(lambda exc: True)
    assert cond(make_state(outcome=Outcome(value=1))) is False


# ---------------------------------------------------------------------------
# retry_if_exception_message
# ---------------------------------------------------------------------------


def test_retry_if_exception_message_substring():
    cond = retry_if_exception_message("boom")
    assert cond.message == "boom"
    assert cond.match is None
    assert cond(make_state(outcome=Outcome(exception=ValueError("a boom b")))) is True
    assert cond(make_state(outcome=Outcome(exception=ValueError("nope")))) is False


def test_retry_if_exception_message_regex():
    cond = retry_if_exception_message(r"bo+m", match="re")
    assert cond(make_state(outcome=Outcome(exception=ValueError("a booom b")))) is True
    assert cond(make_state(outcome=Outcome(exception=ValueError("bam")))) is False


def test_retry_if_exception_message_no_outcome():
    cond = retry_if_exception_message("boom")
    assert cond(make_state(outcome=None)) is False


def test_retry_if_exception_message_no_exception():
    cond = retry_if_exception_message("boom")
    assert cond(make_state(outcome=Outcome(value=1))) is False


# ---------------------------------------------------------------------------
# retry_if_result / retry_if_not_result
# ---------------------------------------------------------------------------


def test_retry_if_result():
    cond = retry_if_result(lambda v: v < 0)
    assert cond(make_state(outcome=Outcome(value=-1))) is True
    assert cond(make_state(outcome=Outcome(value=1))) is False


def test_retry_if_result_no_outcome():
    cond = retry_if_result(lambda v: True)
    assert cond(make_state(outcome=None)) is False


def test_retry_if_result_exception_present():
    cond = retry_if_result(lambda v: True)
    assert cond(make_state(outcome=Outcome(value=1, exception=ValueError()))) is False


def test_retry_if_result_predicate_receives_value():
    seen = []

    def predicate(v):
        seen.append(v)
        return True

    cond = retry_if_result(predicate)
    cond(make_state(outcome=Outcome(value="abc")))
    assert seen == ["abc"]


def test_retry_if_not_result():
    cond = retry_if_not_result(lambda v: v < 0)
    assert cond(make_state(outcome=Outcome(value=-1))) is False
    assert cond(make_state(outcome=Outcome(value=1))) is True


def test_retry_if_not_result_no_outcome():
    cond = retry_if_not_result(lambda v: True)
    assert cond(make_state(outcome=None)) is False


def test_retry_if_not_result_exception_present():
    cond = retry_if_not_result(lambda v: False)
    assert cond(make_state(outcome=Outcome(value=1, exception=ValueError()))) is False


# ---------------------------------------------------------------------------
# retry_all / retry_any
# ---------------------------------------------------------------------------


def test_retry_all_true():
    cond = retry_all(retry_always(), retry_always())
    assert len(cond.conditions) == 2
    assert cond(make_state()) is True


def test_retry_all_false():
    cond = retry_all(retry_always(), retry_never())
    assert cond(make_state()) is False


def test_retry_all_empty():
    assert retry_all()(make_state()) is True


def test_retry_all_short_circuits():
    calls = []

    class Rec(RetryCondition):
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def __call__(self, state):
            calls.append(self.name)
            return self.value

    cond = retry_all(Rec("a", False), Rec("b", True))
    assert cond(make_state()) is False
    assert calls == ["a"]


def test_retry_all_ignores_numeric_results():
    # Numeric (non-bool) results do not cause a False short-circuit.
    cond = retry_all(retry_if_exception(lambda exc: 0.0), retry_always())
    result = cond(make_state(outcome=Outcome(exception=ValueError())))
    assert result is True


def test_retry_any_true():
    cond = retry_any(retry_never(), retry_always())
    assert len(cond.conditions) == 2
    assert cond(make_state()) is True


def test_retry_any_false():
    cond = retry_any(retry_never(), retry_never())
    assert cond(make_state()) is False


def test_retry_any_empty():
    assert retry_any()(make_state()) is False


def test_retry_any_returns_float():
    cond = retry_any(retry_if_exception(lambda exc: 3))
    result = cond(make_state(outcome=Outcome(exception=ValueError())))
    assert result == 3.0
    assert isinstance(result, float)


def test_retry_any_short_circuits_true():
    calls = []

    class Rec(RetryCondition):
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def __call__(self, state):
            calls.append(self.name)
            return self.value

    cond = retry_any(Rec("a", True), Rec("b", False))
    assert cond(make_state()) is True
    assert calls == ["a"]


def test_retry_any_bool_true_short_circuits_before_numeric():
    cond = retry_any(retry_always(), retry_if_exception(lambda exc: 1))
    assert cond(make_state(outcome=Outcome(exception=ValueError()))) is True


def test_retry_or_operator():
    cond = retry_never() | retry_always()
    assert isinstance(cond, retry_any)
    assert cond(make_state()) is True


def test_retry_and_operator():
    cond = retry_always() & retry_never()
    assert isinstance(cond, retry_all)
    assert cond(make_state()) is False


# ---------------------------------------------------------------------------
# retry_always / retry_never
# ---------------------------------------------------------------------------


def test_retry_always():
    assert retry_always()(make_state()) is True


def test_retry_never():
    assert retry_never()(make_state()) is False


# ---------------------------------------------------------------------------
# retry_if_not_exception_type / retry_unless_exception_type
# ---------------------------------------------------------------------------


def test_retry_if_not_exception_type_single():
    cond = retry_if_not_exception_type(ValueError)
    assert cond.exc_types == (ValueError,)
    assert cond(make_state(outcome=Outcome(exception=ValueError()))) is False
    assert cond(make_state(outcome=Outcome(exception=TypeError()))) is True


def test_retry_if_not_exception_type_sequence():
    cond = retry_if_not_exception_type((ValueError, TypeError))
    assert cond.exc_types == (ValueError, TypeError)
    assert cond(make_state(outcome=Outcome(exception=ValueError()))) is False
    assert cond(make_state(outcome=Outcome(exception=TypeError()))) is False
    assert cond(make_state(outcome=Outcome(exception=KeyError()))) is True


def test_retry_if_not_exception_type_no_outcome():
    cond = retry_if_not_exception_type(ValueError)
    assert cond(make_state(outcome=None)) is False


def test_retry_if_not_exception_type_no_exception():
    cond = retry_if_not_exception_type(ValueError)
    assert cond(make_state(outcome=Outcome(value=1))) is False


def test_retry_unless_exception_type_is_subclass_and_behavior():
    assert issubclass(retry_unless_exception_type, retry_if_not_exception_type)
    cond = retry_unless_exception_type(ValueError)
    assert cond(make_state(outcome=Outcome(exception=ValueError()))) is False
    assert cond(make_state(outcome=Outcome(exception=TypeError()))) is True


# ---------------------------------------------------------------------------
# retry_if_not_exception_message
# ---------------------------------------------------------------------------


def test_retry_if_not_exception_message_substring():
    cond = retry_if_not_exception_message("boom")
    assert cond.match == "boom"
    assert cond.regex is False
    assert cond(make_state(outcome=Outcome(exception=ValueError("a boom b")))) is False
    assert cond(make_state(outcome=Outcome(exception=ValueError("nope")))) is True


def test_retry_if_not_exception_message_regex():
    cond = retry_if_not_exception_message(r"bo+m", regex=True)
    assert cond(make_state(outcome=Outcome(exception=ValueError("a booom b")))) is False
    assert cond(make_state(outcome=Outcome(exception=ValueError("bam")))) is True


def test_retry_if_not_exception_message_no_outcome():
    cond = retry_if_not_exception_message("boom")
    assert cond(make_state(outcome=None)) is False


def test_retry_if_not_exception_message_no_exception():
    cond = retry_if_not_exception_message("boom")
    assert cond(make_state(outcome=Outcome(value=1))) is False


# ---------------------------------------------------------------------------
# retry_if_exception_cause_type
# ---------------------------------------------------------------------------


def test_retry_if_exception_cause_type_direct_cause():
    cause = ValueError("cause")
    exc = RuntimeError("outer")
    exc.__cause__ = cause
    cond = retry_if_exception_cause_type(ValueError)
    assert cond.exc_types == (ValueError,)
    assert cond(make_state(outcome=Outcome(exception=exc))) is True


def test_retry_if_exception_cause_type_nested_cause():
    root = ValueError("root")
    mid = TypeError("mid")
    mid.__cause__ = root
    exc = RuntimeError("outer")
    exc.__cause__ = mid
    cond = retry_if_exception_cause_type(ValueError)
    assert cond(make_state(outcome=Outcome(exception=exc))) is True


def test_retry_if_exception_cause_type_no_match():
    cause = KeyError("cause")
    exc = RuntimeError("outer")
    exc.__cause__ = cause
    cond = retry_if_exception_cause_type(ValueError)
    assert cond(make_state(outcome=Outcome(exception=exc))) is False


def test_retry_if_exception_cause_type_uses_context():
    try:
        try:
            raise ValueError("inner")
        except ValueError as inner:
            raise RuntimeError("outer") from None
    except RuntimeError as outer:
        # __cause__ is suppressed (None) but __context__ retains the ValueError
        cond = retry_if_exception_cause_type(ValueError)
        state = make_state(outcome=Outcome(exception=outer))
        assert outer.__cause__ is None
        assert isinstance(outer.__context__, ValueError)
        assert cond(state) is True


def test_retry_if_exception_cause_type_sequence():
    root = KeyError("root")
    exc = RuntimeError("outer")
    exc.__cause__ = root
    cond = retry_if_exception_cause_type((ValueError, KeyError))
    assert cond.exc_types == (ValueError, KeyError)
    assert cond(make_state(outcome=Outcome(exception=exc))) is True


def test_retry_if_exception_cause_type_no_cause():
    exc = RuntimeError("outer")
    cond = retry_if_exception_cause_type(ValueError)
    assert cond(make_state(outcome=Outcome(exception=exc))) is False


def test_retry_if_exception_cause_type_no_outcome():
    cond = retry_if_exception_cause_type(ValueError)
    assert cond(make_state(outcome=None)) is False


def test_retry_if_exception_cause_type_no_exception():
    cond = retry_if_exception_cause_type(ValueError)
    assert cond(make_state(outcome=Outcome(value=1))) is False
