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
from backon._state import Attempt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state(tries=0, elapsed=0.0, outcome=None) -> RetryState:
    return RetryState(tries=tries, elapsed=elapsed, outcome=outcome)


class _WaitOutcome:
    """Stand-in outcome object exposing a `wait` attribute."""

    def __init__(self, wait):
        self.wait = wait


# ---------------------------------------------------------------------------
# Attempt construction sanity (used for building outcomes)
# ---------------------------------------------------------------------------


def test_attempt_available_for_building_outcomes():
    # Make sure we can construct Attempt objects; otherwise tests below that
    # rely on real outcomes would be unreliable.
    att = Attempt(value=1)
    assert att.value == 1
    assert att.exception is None


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


def test_stop_base_call_raises():
    with pytest.raises(NotImplementedError):
        Stop()(make_state())


def test_retry_condition_base_call_raises():
    with pytest.raises(NotImplementedError):
        RetryCondition()(make_state())


# ---------------------------------------------------------------------------
# stop_after_attempt
# ---------------------------------------------------------------------------


def test_stop_after_attempt_basic():
    cond = stop_after_attempt(3)
    assert cond.max_attempts == 3
    assert cond(make_state(tries=0)) is False
    assert cond(make_state(tries=2)) is False
    assert cond(make_state(tries=3)) is True
    assert cond(make_state(tries=10)) is True


def test_stop_after_attempt_or_and_operator():
    a = stop_after_attempt(1)
    b = stop_never()
    combined_or = a | b
    assert isinstance(combined_or, stop_any)
    assert combined_or(make_state(tries=1)) is True

    combined_and = a & b
    assert isinstance(combined_and, stop_all)
    assert combined_and(make_state(tries=1)) is False


# ---------------------------------------------------------------------------
# stop_after_delay
# ---------------------------------------------------------------------------


def test_stop_after_delay_basic():
    cond = stop_after_delay(5.0)
    assert cond.max_delay == 5.0
    assert cond(make_state(elapsed=0.0)) is False
    assert cond(make_state(elapsed=4.9)) is False
    assert cond(make_state(elapsed=5.0)) is True
    assert cond(make_state(elapsed=10.0)) is True


# ---------------------------------------------------------------------------
# stop_before_delay
# ---------------------------------------------------------------------------


def test_stop_before_delay_no_outcome():
    cond = stop_before_delay(5.0)
    assert cond.max_delay == 5.0
    assert cond(make_state(elapsed=4.0)) is False
    assert cond(make_state(elapsed=5.0)) is True


def test_stop_before_delay_with_wait_outcome():
    cond = stop_before_delay(5.0)
    assert cond(make_state(elapsed=4.0, outcome=_WaitOutcome(1.0))) is True
    assert cond(make_state(elapsed=4.0, outcome=_WaitOutcome(0.5))) is False
    assert cond(make_state(elapsed=0.0, outcome=_WaitOutcome(5.0))) is True


def test_stop_before_delay_outcome_without_wait_attr():
    class NoWait:
        pass

    cond = stop_before_delay(5.0)
    assert cond(make_state(elapsed=4.0, outcome=NoWait())) is False
    assert cond(make_state(elapsed=5.0, outcome=NoWait())) is True


# ---------------------------------------------------------------------------
# stop_all / stop_any / stop_never
# ---------------------------------------------------------------------------


def test_stop_all():
    cond = stop_all(stop_after_attempt(2), stop_after_delay(10.0))
    assert cond.stops == (cond.stops[0], cond.stops[1])
    assert cond(make_state(tries=2, elapsed=10.0)) is True
    assert cond(make_state(tries=2, elapsed=9.0)) is False
    assert cond(make_state(tries=1, elapsed=10.0)) is False
    assert cond(make_state(tries=1, elapsed=1.0)) is False


def test_stop_all_empty():
    cond = stop_all()
    assert cond(make_state()) is True


def test_stop_any():
    cond = stop_any(stop_after_attempt(2), stop_after_delay(10.0))
    assert cond(make_state(tries=2, elapsed=0.0)) is True
    assert cond(make_state(tries=0, elapsed=10.0)) is True
    assert cond(make_state(tries=2, elapsed=10.0)) is True
    assert cond(make_state(tries=1, elapsed=9.0)) is False


def test_stop_any_empty():
    cond = stop_any()
    assert cond(make_state()) is False


def test_stop_never():
    cond = stop_never()
    assert cond(make_state()) is False
    assert cond(make_state(tries=100, elapsed=100.0)) is False


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
# retry_if_exception_type
# ---------------------------------------------------------------------------


def test_retry_if_exception_type_single():
    cond = retry_if_exception_type(ValueError)
    assert cond.exc_types == (ValueError,)
    assert cond(make_state()) is False
    assert cond(make_state(outcome=Attempt(value=None))) is False
    assert cond(make_state(outcome=Attempt(exception=ValueError()))) is True
    assert cond(make_state(outcome=Attempt(exception=TypeError()))) is False


def test_retry_if_exception_type_sequence():
    cond = retry_if_exception_type((ValueError, KeyError))
    assert cond.exc_types == (ValueError, KeyError)
    assert cond(make_state(outcome=Attempt(exception=ValueError()))) is True
    assert cond(make_state(outcome=Attempt(exception=KeyError()))) is True
    assert cond(make_state(outcome=Attempt(exception=RuntimeError()))) is False


def test_retry_if_exception_type_subclass():
    cond = retry_if_exception_type(Exception)
    assert cond(make_state(outcome=Attempt(exception=ValueError()))) is True


def test_retry_if_exception_type_or_and_operator():
    a = retry_if_exception_type(ValueError)
    b = retry_always()
    combined_or = a | b
    assert isinstance(combined_or, retry_any)
    combined_and = a & b
    assert isinstance(combined_and, retry_all)


# ---------------------------------------------------------------------------
# retry_if_exception
# ---------------------------------------------------------------------------


def test_retry_if_exception_basic():
    cond = retry_if_exception(lambda e: isinstance(e, ValueError))
    assert cond(make_state()) is False
    assert cond(make_state(outcome=Attempt(value=1))) is False
    assert cond(make_state(outcome=Attempt(exception=ValueError()))) is True
    assert cond(make_state(outcome=Attempt(exception=TypeError()))) is False


def test_retry_if_exception_numeric_result_converted_to_float():
    cond = retry_if_exception(lambda e: 1.5)
    assert cond(make_state(outcome=Attempt(exception=ValueError()))) == 1.5
    assert isinstance(cond(make_state(outcome=Attempt(exception=ValueError()))), float)

    cond_int = retry_if_exception(lambda e: 2)
    assert cond_int(make_state(outcome=Attempt(exception=ValueError()))) == 2.0


def test_retry_if_exception_bool_result():
    # bool is subclass of int but should be returned as bool
    cond = retry_if_exception(lambda e: True)
    result = cond(make_state(outcome=Attempt(exception=ValueError())))
    assert result is True


def test_retry_if_exception_passes_exception():
    seen = []

    def pred(exc):
        seen.append(exc)
        return False

    cond = retry_if_exception(pred)
    exc = ValueError("boom")
    cond(make_state(outcome=Attempt(exception=exc)))
    assert seen == [exc]


# ---------------------------------------------------------------------------
# retry_if_exception_message
# ---------------------------------------------------------------------------


def test_retry_if_exception_message_substring():
    cond = retry_if_exception_message("boom")
    assert cond.match is None
    assert cond(make_state()) is False
    assert cond(make_state(outcome=Attempt(value=1))) is False
    assert cond(make_state(outcome=Attempt(exception=ValueError("boom!")))) is True
    assert cond(make_state(outcome=Attempt(exception=ValueError("nope")))) is False


def test_retry_if_exception_message_regex():
    cond = retry_if_exception_message(r"bo+m", match="re")
    assert cond(make_state(outcome=Attempt(exception=ValueError("booom")))) is True
    assert cond(make_state(outcome=Attempt(exception=ValueError("bom")))) is True
    assert cond(make_state(outcome=Attempt(exception=ValueError("nothing")))) is False


def test_retry_if_exception_message_stringifies():
    cond = retry_if_exception_message("42")
    assert cond(make_state(outcome=Attempt(exception=ValueError(42)))) is True


# ---------------------------------------------------------------------------
# retry_if_result / retry_if_not_result
# ---------------------------------------------------------------------------


def test_retry_if_result():
    cond = retry_if_result(lambda v: v == 5)
    assert cond(make_state()) is False
    assert cond(make_state(outcome=Attempt(value=5))) is True
    assert cond(make_state(outcome=Attempt(value=4))) is False
    assert cond(make_state(outcome=Attempt(exception=ValueError(), value=5))) is False


def test_retry_if_not_result():
    cond = retry_if_not_result(lambda v: v == 5)
    assert cond(make_state()) is False
    assert cond(make_state(outcome=Attempt(value=5))) is False
    assert cond(make_state(outcome=Attempt(value=4))) is True
    assert cond(make_state(outcome=Attempt(exception=ValueError(), value=4))) is False


# ---------------------------------------------------------------------------
# retry_all / retry_any
# ---------------------------------------------------------------------------


def test_retry_all():
    cond = retry_all(retry_always(), retry_always())
    assert cond(make_state()) is True

    cond2 = retry_all(retry_always(), retry_never())
    assert cond2(make_state()) is False


def test_retry_all_empty():
    cond = retry_all()
    assert cond(make_state()) is True


def test_retry_all_ignores_numeric():
    # numeric (float) results are truthy-ish but retry_all only rejects on
    # False (bool); a numeric result should not stop the chain.
    cond = retry_all(retry_if_exception(lambda e: 0.0), retry_always())
    # 0.0 is returned as float, not a bool False, so retry_all does not fail
    assert cond(make_state(outcome=Attempt(exception=ValueError()))) is True


def test_retry_any():
    cond = retry_any(retry_never(), retry_always())
    assert cond(make_state()) is True

    cond2 = retry_any(retry_never(), retry_never())
    assert cond2(make_state()) is False


def test_retry_any_empty():
    cond = retry_any()
    assert cond(make_state()) is False


def test_retry_any_numeric_returns_float():
    cond = retry_any(retry_never(), retry_if_exception(lambda e: 3))
    result = cond(make_state(outcome=Attempt(exception=ValueError())))
    assert result == 3.0
    assert isinstance(result, float)


def test_retry_any_numeric_zero_truthy_float_returned():
    # 0.0 is not a bool, so retry_any returns it as float 0.0
    cond = retry_any(retry_if_exception(lambda e: 0.0))
    result = cond(make_state(outcome=Attempt(exception=ValueError())))
    assert result == 0.0
    assert isinstance(result, float)


def test_retry_any_short_circuits_on_true():
    calls = []

    def make(value):
        def cond(state):
            calls.append(value)
            return value

        return cond

    cond = retry_any(make(True), make(False))
    assert cond(make_state()) is True
    assert calls == [True]


# ---------------------------------------------------------------------------
# retry_always / retry_never
# ---------------------------------------------------------------------------


def test_retry_always():
    cond = retry_always()
    assert cond(make_state()) is True


def test_retry_never():
    cond = retry_never()
    assert cond(make_state()) is False


# ---------------------------------------------------------------------------
# retry_if_not_exception_type / retry_unless_exception_type
# ---------------------------------------------------------------------------


def test_retry_if_not_exception_type_single():
    cond = retry_if_not_exception_type(ValueError)
    assert cond.exc_types == (ValueError,)
    assert cond(make_state()) is False
    assert cond(make_state(outcome=Attempt(value=1))) is False
    assert cond(make_state(outcome=Attempt(exception=ValueError()))) is False
    assert cond(make_state(outcome=Attempt(exception=TypeError()))) is True


def test_retry_if_not_exception_type_sequence():
    cond = retry_if_not_exception_type((ValueError, KeyError))
    assert cond.exc_types == (ValueError, KeyError)
    assert cond(make_state(outcome=Attempt(exception=ValueError()))) is False
    assert cond(make_state(outcome=Attempt(exception=KeyError()))) is False
    assert cond(make_state(outcome=Attempt(exception=RuntimeError()))) is True


def test_retry_unless_exception_type_is_subclass():
    assert issubclass(retry_unless_exception_type, retry_if_not_exception_type)
    cond = retry_unless_exception_type(ValueError)
    assert cond(make_state(outcome=Attempt(exception=ValueError()))) is False
    assert cond(make_state(outcome=Attempt(exception=TypeError()))) is True


# ---------------------------------------------------------------------------
# retry_if_not_exception_message
# ---------------------------------------------------------------------------


def test_retry_if_not_exception_message_substring():
    cond = retry_if_not_exception_message("boom")
    assert cond.regex is False
    assert cond(make_state()) is False
    assert cond(make_state(outcome=Attempt(value=1))) is False
    assert cond(make_state(outcome=Attempt(exception=ValueError("boom!")))) is False
    assert cond(make_state(outcome=Attempt(exception=ValueError("nope")))) is True


def test_retry_if_not_exception_message_regex():
    cond = retry_if_not_exception_message(r"bo+m", regex=True)
    assert cond.regex is True
    assert cond(make_state(outcome=Attempt(exception=ValueError("booom")))) is False
    assert cond(make_state(outcome=Attempt(exception=ValueError("nothing")))) is True


# ---------------------------------------------------------------------------
# retry_if_exception_cause_type
# ---------------------------------------------------------------------------


def test_retry_if_exception_cause_type_no_cause():
    cond = retry_if_exception_cause_type(ValueError)
    assert cond.exc_types == (ValueError,)
    assert cond(make_state()) is False
    assert cond(make_state(outcome=Attempt(value=1))) is False
    assert cond(make_state(outcome=Attempt(exception=RuntimeError()))) is False


def test_retry_if_exception_cause_type_direct_cause():
    cond = retry_if_exception_cause_type(ValueError)
    try:
        try:
            raise ValueError("root")
        except ValueError as e:
            raise RuntimeError("outer") from e
    except RuntimeError as exc:
        assert cond(make_state(outcome=Attempt(exception=exc))) is True


def test_retry_if_exception_cause_type_context():
    cond = retry_if_exception_cause_type(ValueError)
    try:
        try:
            raise ValueError("root")
        except ValueError:
            raise RuntimeError("outer")
    except RuntimeError as exc:
        assert exc.__cause__ is None
        assert isinstance(exc.__context__, ValueError)
        assert cond(make_state(outcome=Attempt(exception=exc))) is True


def test_retry_if_exception_cause_type_nested_chain():
    cond = retry_if_exception_cause_type(KeyError)
    try:
        try:
            try:
                raise KeyError("deep")
            except KeyError as e:
                raise ValueError("mid") from e
        except ValueError as e:
            raise RuntimeError("outer") from e
    except RuntimeError as exc:
        assert cond(make_state(outcome=Attempt(exception=exc))) is True


def test_retry_if_exception_cause_type_mismatch():
    cond = retry_if_exception_cause_type(KeyError)
    try:
        try:
            raise ValueError("root")
        except ValueError as e:
            raise RuntimeError("outer") from e
    except RuntimeError as exc:
        assert cond(make_state(outcome=Attempt(exception=exc))) is False


def test_retry_if_exception_cause_type_sequence():
    cond = retry_if_exception_cause_type((ValueError, KeyError))
    assert cond.exc_types == (ValueError, KeyError)
    try:
        try:
            raise KeyError("root")
        except KeyError as e:
            raise RuntimeError("outer") from e
    except RuntimeError as exc:
        assert cond(make_state(outcome=Attempt(exception=exc))) is True
