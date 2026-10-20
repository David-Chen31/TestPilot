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


class FakeOutcome:
    """Minimal stand-in for backon._state.Attempt used by conditions."""

    def __init__(self, exception=None, value=None, wait=0.0, has_exception=True):
        self.exception = exception
        self.value = value
        self.wait = wait
        if not has_exception:
            self.exception = None


def make_state(outcome=None, tries=0, elapsed=0.0, idle_for=0.0):
    state = RetryState(tries=tries, elapsed=elapsed, idle_for=idle_for, outcome=outcome)
    return state


# ---------------------------------------------------------------------------
# Stop base class
# ---------------------------------------------------------------------------


def test_stop_base_call_raises():
    with pytest.raises(NotImplementedError):
        Stop()(make_state())


def test_stop_or_returns_stop_any():
    a = stop_never()
    b = stop_never()
    combined = a | b
    assert isinstance(combined, stop_any)
    assert combined.stops == (a, b)


def test_stop_and_returns_stop_all():
    a = stop_never()
    b = stop_never()
    combined = a & b
    assert isinstance(combined, stop_all)
    assert combined.stops == (a, b)


# ---------------------------------------------------------------------------
# stop_after_attempt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tries, max_attempts, expected",
    [(0, 1, False), (1, 1, True), (2, 1, True), (0, 0, True), (0, 3, False)],
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
    [(0.0, 1.0, False), (0.99, 1.0, False), (1.0, 1.0, True), (2.0, 1.0, True)],
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
    assert cond(make_state(elapsed=0.5)) is False
    assert cond(make_state(elapsed=1.0)) is True
    assert cond(make_state(elapsed=1.5)) is True


def test_stop_before_delay_with_wait():
    cond = stop_before_delay(1.0)
    outcome = FakeOutcome(wait=0.6)
    # 0.5 + 0.6 = 1.1 >= 1.0
    assert cond(make_state(outcome=outcome, elapsed=0.5)) is True
    # 0.2 + 0.6 = 0.8 < 1.0
    assert cond(make_state(outcome=outcome, elapsed=0.2)) is False


def test_stop_before_delay_outcome_without_wait_attr():
    cond = stop_before_delay(1.0)

    class NoWait:
        exception = None
        value = None

    assert cond(make_state(outcome=NoWait(), elapsed=0.4)) is False
    assert cond(make_state(outcome=NoWait(), elapsed=1.0)) is True


# ---------------------------------------------------------------------------
# stop_all / stop_any / stop_never
# ---------------------------------------------------------------------------


def test_stop_all_true_when_all_true():
    s = stop_all(stop_after_attempt(1), stop_after_delay(1.0))
    assert s(make_state(tries=1, elapsed=1.0)) is True


def test_stop_all_false_when_any_false():
    s = stop_all(stop_after_attempt(1), stop_after_delay(1.0))
    assert s(make_state(tries=1, elapsed=0.0)) is False
    assert s(make_state(tries=0, elapsed=1.0)) is False


def test_stop_all_empty_is_true():
    assert stop_all()(make_state()) is True


def test_stop_any_true_when_any_true():
    s = stop_any(stop_after_attempt(5), stop_after_delay(1.0))
    assert s(make_state(tries=0, elapsed=2.0)) is True
    assert s(make_state(tries=9, elapsed=0.0)) is True


def test_stop_any_false_when_none_true():
    s = stop_any(stop_after_attempt(5), stop_after_delay(1.0))
    assert s(make_state(tries=0, elapsed=0.0)) is False


def test_stop_any_empty_is_false():
    assert stop_any()(make_state()) is False


def test_stop_never_always_false():
    assert stop_never()(make_state()) is False


# ---------------------------------------------------------------------------
# stop_when_event_set
# ---------------------------------------------------------------------------


def test_stop_when_event_set():
    event = threading.Event()
    cond = stop_when_event_set(event)
    assert cond.event is event
    assert cond(make_state()) is False
    event.set()
    assert cond(make_state()) is True


# ---------------------------------------------------------------------------
# RetryCondition base and operator overloads
# ---------------------------------------------------------------------------


def test_retry_condition_base_call_raises():
    with pytest.raises(NotImplementedError):
        RetryCondition()(make_state())


def test_retry_condition_or_returns_retry_any():
    a = retry_always()
    b = retry_never()
    combined = a | b
    assert isinstance(combined, retry_any)
    assert combined.conditions == (a, b)


def test_retry_condition_and_returns_retry_all():
    a = retry_always()
    b = retry_never()
    combined = a & b
    assert isinstance(combined, retry_all)
    assert combined.conditions == (a, b)


# ---------------------------------------------------------------------------
# retry_if_exception_type
# ---------------------------------------------------------------------------


def test_retry_if_exception_type_single():
    cond = retry_if_exception_type(ValueError)
    assert cond.exc_types == (ValueError,)
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("x")))) is True
    assert cond(make_state(outcome=FakeOutcome(exception=TypeError("x")))) is False


def test_retry_if_exception_type_sequence():
    cond = retry_if_exception_type((ValueError, TypeError))
    assert cond.exc_types == (ValueError, TypeError)
    assert cond(make_state(outcome=FakeOutcome(exception=TypeError("x")))) is True


def test_retry_if_exception_type_no_outcome():
    cond = retry_if_exception_type(ValueError)
    assert cond(make_state()) is False


def test_retry_if_exception_type_no_exception():
    cond = retry_if_exception_type(ValueError)
    outcome = FakeOutcome(value=42)
    assert cond(make_state(outcome=outcome)) is False


# ---------------------------------------------------------------------------
# retry_if_exception
# ---------------------------------------------------------------------------


def test_retry_if_exception_bool_predicate():
    cond = retry_if_exception(lambda e: isinstance(e, ValueError))
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError()))) is True
    assert cond(make_state(outcome=FakeOutcome(exception=TypeError()))) is False


def test_retry_if_exception_numeric_returns_float():
    cond = retry_if_exception(lambda e: 3)
    result = cond(make_state(outcome=FakeOutcome(exception=ValueError())))
    assert result == 3.0
    assert isinstance(result, float)


def test_retry_if_exception_float_result():
    cond = retry_if_exception(lambda e: 1.5)
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError()))) == 1.5


def test_retry_if_exception_bool_is_not_numeric():
    cond = retry_if_exception(lambda e: True)
    result = cond(make_state(outcome=FakeOutcome(exception=ValueError())))
    # booleans are not coerced to float
    assert result is True


def test_retry_if_exception_no_outcome():
    cond = retry_if_exception(lambda e: True)
    assert cond(make_state()) is False


def test_retry_if_exception_no_exception():
    cond = retry_if_exception(lambda e: True)
    assert cond(make_state(outcome=FakeOutcome(value=1))) is False


# ---------------------------------------------------------------------------
# retry_if_exception_message
# ---------------------------------------------------------------------------


def test_retry_if_exception_message_substring():
    cond = retry_if_exception_message("boom")
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("big boom here")))) is True
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("nope")))) is False


def test_retry_if_exception_message_regex():
    cond = retry_if_exception_message(r"bo+m", match="re")
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("boooms")))) is True
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("bam")))) is False


def test_retry_if_exception_message_no_outcome():
    cond = retry_if_exception_message("x")
    assert cond(make_state()) is False


def test_retry_if_exception_message_no_exception():
    cond = retry_if_exception_message("x")
    assert cond(make_state(outcome=FakeOutcome(value=1))) is False


# ---------------------------------------------------------------------------
# retry_if_result / retry_if_not_result
# ---------------------------------------------------------------------------


def test_retry_if_result():
    cond = retry_if_result(lambda v: v == 5)
    assert cond(make_state(outcome=FakeOutcome(value=5))) is True
    assert cond(make_state(outcome=FakeOutcome(value=6))) is False


def test_retry_if_result_no_outcome():
    cond = retry_if_result(lambda v: True)
    assert cond(make_state()) is False


def test_retry_if_result_with_exception():
    cond = retry_if_result(lambda v: True)
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError()))) is False


def test_retry_if_not_result():
    cond = retry_if_not_result(lambda v: v == 5)
    assert cond(make_state(outcome=FakeOutcome(value=5))) is False
    assert cond(make_state(outcome=FakeOutcome(value=6))) is True


def test_retry_if_not_result_no_outcome():
    cond = retry_if_not_result(lambda v: True)
    assert cond(make_state()) is False


def test_retry_if_not_result_with_exception():
    cond = retry_if_not_result(lambda v: True)
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError()))) is False


# ---------------------------------------------------------------------------
# retry_all / retry_any
# ---------------------------------------------------------------------------


def test_retry_all_all_true():
    cond = retry_all(retry_always(), retry_always())
    assert cond(make_state()) is True


def test_retry_all_one_false():
    cond = retry_all(retry_always(), retry_never())
    assert cond(make_state()) is False


def test_retry_all_short_circuits():
    calls = []

    def make(flag):
        def inner(state):
            calls.append(flag)
            return flag

        return inner

    cond = retry_all(make(False), make(True))
    assert cond(make_state()) is False
    assert calls == [False]


def test_retry_all_numeric_not_blocking():
    # numeric results are skipped (only False bool returns trigger early exit)
    cond = retry_all(retry_if_exception(lambda e: 2), retry_always())
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError()))) is True


def test_retry_all_empty_true():
    assert retry_all()(make_state()) is True


def test_retry_any_one_true():
    cond = retry_any(retry_never(), retry_always())
    assert cond(make_state()) is True


def test_retry_any_all_false():
    cond = retry_any(retry_never(), retry_never())
    assert cond(make_state()) is False


def test_retry_any_numeric_returns_float():
    cond = retry_any(retry_never(), retry_if_exception(lambda e: 4))
    result = cond(make_state(outcome=FakeOutcome(exception=ValueError())))
    assert result == 4.0
    assert isinstance(result, float)


def test_retry_any_short_circuits_on_true():
    calls = []

    def make(flag):
        def inner(state):
            calls.append(flag)
            return flag

        return inner

    cond = retry_any(make(True), make(False))
    assert cond(make_state()) is True
    assert calls == [True]


def test_retry_any_empty_false():
    assert retry_any()(make_state()) is False


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
    assert cond(make_state(outcome=FakeOutcome(exception=TypeError()))) is True
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError()))) is False


def test_retry_if_not_exception_type_sequence():
    cond = retry_if_not_exception_type((ValueError, TypeError))
    assert cond.exc_types == (ValueError, TypeError)
    assert cond(make_state(outcome=FakeOutcome(exception=KeyError()))) is True
    assert cond(make_state(outcome=FakeOutcome(exception=TypeError()))) is False


def test_retry_if_not_exception_type_no_outcome():
    cond = retry_if_not_exception_type(ValueError)
    assert cond(make_state()) is False


def test_retry_if_not_exception_type_no_exception():
    cond = retry_if_not_exception_type(ValueError)
    assert cond(make_state(outcome=FakeOutcome(value=1))) is False


def test_retry_unless_exception_type_is_subclass():
    assert issubclass(retry_unless_exception_type, retry_if_not_exception_type)
    cond = retry_unless_exception_type(ValueError)
    assert cond(make_state(outcome=FakeOutcome(exception=TypeError()))) is True
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError()))) is False


# ---------------------------------------------------------------------------
# retry_if_not_exception_message
# ---------------------------------------------------------------------------


def test_retry_if_not_exception_message_substring():
    cond = retry_if_not_exception_message("boom")
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("nope")))) is True
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("a boom")))) is False


def test_retry_if_not_exception_message_regex():
    cond = retry_if_not_exception_message(r"bo+m", regex=True)
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("bam")))) is True
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("booom")))) is False


def test_retry_if_not_exception_message_no_outcome():
    cond = retry_if_not_exception_message("x")
    assert cond(make_state()) is False


def test_retry_if_not_exception_message_no_exception():
    cond = retry_if_not_exception_message("x")
    assert cond(make_state(outcome=FakeOutcome(value=1))) is False


# ---------------------------------------------------------------------------
# retry_if_exception_cause_type
# ---------------------------------------------------------------------------


def test_retry_if_exception_cause_type_direct_cause():
    try:
        try:
            raise KeyError("root")
        except KeyError as e:
            raise ValueError("outer") from e
    except ValueError as outer:
        exc = outer
    cond = retry_if_exception_cause_type(KeyError)
    assert cond.exc_types == (KeyError,)
    assert cond(make_state(outcome=FakeOutcome(exception=exc))) is True


def test_retry_if_exception_cause_type_no_match():
    try:
        try:
            raise KeyError("root")
        except KeyError as e:
            raise ValueError("outer") from e
    except ValueError as outer:
        exc = outer
    cond = retry_if_exception_cause_type(TypeError)
    assert cond(make_state(outcome=FakeOutcome(exception=exc))) is False


def test_retry_if_exception_cause_type_deep_chain():
    try:
        try:
            try:
                raise KeyError("deep")
            except KeyError as e3:
                raise TypeError("mid") from e3
        except TypeError as e2:
            raise ValueError("outer") from e2
    except ValueError as outer:
        exc = outer
    cond = retry_if_exception_cause_type(KeyError)
    assert cond(make_state(outcome=FakeOutcome(exception=exc))) is True


def test_retry_if_exception_cause_type_context():
    try:
        try:
            raise KeyError("ctx")
        except KeyError:
            raise ValueError("outer")
    except ValueError as outer:
        exc = outer
    cond = retry_if_exception_cause_type(KeyError)
    assert cond(make_state(outcome=FakeOutcome(exception=exc))) is True


def test_retry_if_exception_cause_type_no_outcome():
    cond = retry_if_exception_cause_type(ValueError)
    assert cond(make_state()) is False


def test_retry_if_exception_cause_type_no_exception():
    cond = retry_if_exception_cause_type(ValueError)
    assert cond(make_state(outcome=FakeOutcome(value=1))) is False


def test_retry_if_exception_cause_type_no_cause():
    cond = retry_if_exception_cause_type(ValueError)
    exc = ValueError("lonely")
    assert cond(make_state(outcome=FakeOutcome(exception=exc))) is False


def test_retry_if_exception_cause_type_sequence():
    try:
        try:
            raise KeyError("root")
        except KeyError as e:
            raise ValueError("outer") from e
    except ValueError as outer:
        exc = outer
    cond = retry_if_exception_cause_type((TypeError, KeyError))
    assert cond.exc_types == (TypeError, KeyError)
    assert cond(make_state(outcome=FakeOutcome(exception=exc))) is True
