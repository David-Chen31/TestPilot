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


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------
@dataclass
class FakeOutcome:
    """Mimics the bits of ``Attempt`` that conditions inspect."""

    value: Any = None
    exception: BaseException | None = None
    wait: float = 0.0


def make_state(
    *,
    tries: int = 0,
    elapsed: float = 0.0,
    idle_for: float = 0.0,
    outcome: FakeOutcome | None = None,
) -> RetryState:
    state = RetryState(tries=tries, elapsed=elapsed, idle_for=idle_for)
    state.outcome = outcome  # type: ignore[assignment]
    return state


def state_with_exception(exc: BaseException, **kwargs: Any) -> RetryState:
    return make_state(outcome=FakeOutcome(exception=exc), **kwargs)


def state_with_value(value: Any, **kwargs: Any) -> RetryState:
    return make_state(outcome=FakeOutcome(value=value), **kwargs)


# ---------------------------------------------------------------------------
# Base classes: abstract __call__ raise NotImplementedError
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
@pytest.mark.parametrize(
    ("tries", "max_attempts", "expected"),
    [
        (0, 1, False),
        (1, 1, True),
        (2, 1, True),
        (0, 3, False),
        (2, 3, False),
        (3, 3, True),
        (99, 3, True),
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
    ("elapsed", "max_delay", "expected"),
    [
        (0.0, 1.0, False),
        (0.999, 1.0, False),
        (1.0, 1.0, True),
        (1.5, 1.0, True),
        (0.0, 0.0, True),
    ],
)
def test_stop_after_delay(elapsed, max_delay, expected):
    cond = stop_after_delay(max_delay)
    assert cond.max_delay == max_delay
    assert cond(make_state(elapsed=elapsed)) is expected


# ---------------------------------------------------------------------------
# stop_before_delay
# ---------------------------------------------------------------------------
def test_stop_before_delay_no_outcome_uses_zero_wait():
    cond = stop_before_delay(1.0)
    assert cond.max_delay == 1.0
    assert cond(make_state(elapsed=0.5)) is False
    assert cond(make_state(elapsed=1.0)) is True


def test_stop_before_delay_with_outcome_wait():
    cond = stop_before_delay(1.0)
    # elapsed + wait == 1.0 -> True
    assert cond(make_state(elapsed=0.5, outcome=FakeOutcome(wait=0.5))) is True
    # elapsed + wait < 1.0 -> False
    assert cond(make_state(elapsed=0.5, outcome=FakeOutcome(wait=0.4))) is False
    # elapsed + wait > 1.0 -> True
    assert cond(make_state(elapsed=0.5, outcome=FakeOutcome(wait=0.6))) is True


def test_stop_before_delay_outcome_without_wait_attr():
    @dataclass
    class NoWaitOutcome:
        value: Any = None
        exception: BaseException | None = None

    state = make_state(elapsed=0.9)
    state.outcome = NoWaitOutcome()  # type: ignore[assignment]
    assert stop_before_delay(1.0)(state) is False
    state.elapsed = 1.0
    assert stop_before_delay(1.0)(state) is True


# ---------------------------------------------------------------------------
# stop_all / stop_any / stop_never
# ---------------------------------------------------------------------------
def test_stop_all_requires_all():
    cond = stop_all(stop_after_attempt(2), stop_after_delay(1.0))
    assert cond.stops == (cond.stops[0], cond.stops[1])
    assert cond(make_state(tries=2, elapsed=1.0)) is True
    assert cond(make_state(tries=1, elapsed=1.0)) is False
    assert cond(make_state(tries=2, elapsed=0.5)) is False


def test_stop_all_empty_is_true():
    assert stop_all()(make_state()) is True


def test_stop_any_requires_any():
    cond = stop_any(stop_after_attempt(2), stop_after_delay(1.0))
    assert cond(make_state(tries=2, elapsed=0.0)) is True
    assert cond(make_state(tries=0, elapsed=1.0)) is True
    assert cond(make_state(tries=1, elapsed=0.5)) is False


def test_stop_any_empty_is_false():
    assert stop_any()(make_state()) is False


def test_stop_never():
    assert stop_never()(make_state()) is False
    assert stop_never()(make_state(tries=100, elapsed=1e9)) is False


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
# Operator overloads: | and &
# ---------------------------------------------------------------------------
def test_stop_or_returns_stop_any():
    a = stop_after_attempt(1)
    b = stop_after_delay(1.0)
    combined = a | b
    assert isinstance(combined, stop_any)
    assert combined.stops == (a, b)


def test_stop_and_returns_stop_all():
    a = stop_after_attempt(1)
    b = stop_after_delay(1.0)
    combined = a & b
    assert isinstance(combined, stop_all)
    assert combined.stops == (a, b)


def test_retry_or_returns_retry_any():
    a = retry_always()
    b = retry_never()
    combined = a | b
    assert isinstance(combined, retry_any)
    assert combined.conditions == (a, b)


def test_retry_and_returns_retry_all():
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
    assert cond(state_with_exception(ValueError())) is True
    assert cond(state_with_exception(KeyError())) is False


def test_retry_if_exception_type_sequence():
    cond = retry_if_exception_type([ValueError, KeyError])
    assert cond.exc_types == (ValueError, KeyError)
    assert cond(state_with_exception(ValueError())) is True
    assert cond(state_with_exception(KeyError())) is True
    assert cond(state_with_exception(TypeError())) is False


def test_retry_if_exception_type_no_outcome():
    assert retry_if_exception_type(ValueError)(make_state()) is False


def test_retry_if_exception_type_outcome_without_exception():
    assert retry_if_exception_type(ValueError)(state_with_value(5)) is False


# ---------------------------------------------------------------------------
# retry_if_not_exception_type / retry_unless_exception_type
# ---------------------------------------------------------------------------
def test_retry_if_not_exception_type():
    cond = retry_if_not_exception_type(ValueError)
    assert cond.exc_types == (ValueError,)
    assert cond(state_with_exception(KeyError())) is True
    assert cond(state_with_exception(ValueError())) is False


def test_retry_if_not_exception_type_sequence():
    cond = retry_if_not_exception_type([ValueError, KeyError])
    assert cond(state_with_exception(TypeError())) is True
    assert cond(state_with_exception(ValueError())) is False
    assert cond(state_with_exception(KeyError())) is False


def test_retry_if_not_exception_type_no_outcome():
    assert retry_if_not_exception_type(ValueError)(make_state()) is False


def test_retry_if_not_exception_type_outcome_without_exception():
    assert retry_if_not_exception_type(ValueError)(state_with_value(1)) is False


def test_retry_unless_exception_type_is_subclass():
    assert issubclass(retry_unless_exception_type, retry_if_not_exception_type)
    cond = retry_unless_exception_type(ValueError)
    assert cond(state_with_exception(KeyError())) is True
    assert cond(state_with_exception(ValueError())) is False


# ---------------------------------------------------------------------------
# retry_if_exception
# ---------------------------------------------------------------------------
def test_retry_if_exception_bool_result():
    cond = retry_if_exception(lambda exc: isinstance(exc, ValueError))
    assert cond(state_with_exception(ValueError())) is True
    assert cond(state_with_exception(KeyError())) is False


def test_retry_if_exception_numeric_result_becomes_float():
    cond = retry_if_exception(lambda exc: 1.5)
    assert cond(state_with_exception(ValueError())) == 1.5
    assert isinstance(cond(state_with_exception(ValueError())), float)


def test_retry_if_exception_int_result_becomes_float():
    cond = retry_if_exception(lambda exc: 2)
    result = cond(state_with_exception(ValueError()))
    assert result == 2.0
    assert isinstance(result, float)


def test_retry_if_exception_no_outcome():
    called = []

    def predicate(exc):
        called.append(exc)
        return True

    cond = retry_if_exception(predicate)
    assert cond(make_state()) is False
    assert called == []


def test_retry_if_exception_outcome_without_exception():
    called = []
    cond = retry_if_exception(lambda e: called.append(e) or True)
    assert cond(state_with_value(1)) is False
    assert called == []


# ---------------------------------------------------------------------------
# retry_if_exception_message
# ---------------------------------------------------------------------------
def test_retry_if_exception_message_substring():
    cond = retry_if_exception_message("boom")
    assert cond.message == "boom"
    assert cond.match is None
    assert cond(state_with_exception(ValueError("big boom here"))) is True
    assert cond(state_with_exception(ValueError("quiet"))) is False


def test_retry_if_exception_message_regex():
    cond = retry_if_exception_message(r"bo+m", match="re")
    assert cond(state_with_exception(ValueError("a booom"))) is True
    assert cond(state_with_exception(ValueError("nothing"))) is False


def test_retry_if_exception_message_no_outcome():
    cond = retry_if_exception_message("x")
    assert cond(make_state()) is False


def test_retry_if_exception_message_outcome_without_exception():
    cond = retry_if_exception_message("x")
    assert cond(state_with_value(1)) is False


# ---------------------------------------------------------------------------
# retry_if_not_exception_message
# ---------------------------------------------------------------------------
def test_retry_if_not_exception_message_substring():
    cond = retry_if_not_exception_message("boom")
    assert cond(state_with_exception(ValueError("big boom here"))) is False
    assert cond(state_with_exception(ValueError("quiet"))) is True


def test_retry_if_not_exception_message_regex():
    cond = retry_if_not_exception_message(r"bo+m", regex=True)
    assert cond(state_with_exception(ValueError("a booom"))) is False
    assert cond(state_with_exception(ValueError("nothing"))) is True


def test_retry_if_not_exception_message_no_outcome():
    cond = retry_if_not_exception_message("x")
    assert cond(make_state()) is False


def test_retry_if_not_exception_message_outcome_without_exception():
    cond = retry_if_not_exception_message("x")
    assert cond(state_with_value(1)) is False


# ---------------------------------------------------------------------------
# retry_if_result / retry_if_not_result
# ---------------------------------------------------------------------------
def test_retry_if_result():
    cond = retry_if_result(lambda v: v is None)
    assert cond(state_with_value(None)) is True
    assert cond(state_with_value(5)) is False


def test_retry_if_result_no_outcome():
    called = []
    cond = retry_if_result(lambda v: called.append(v) or True)
    assert cond(make_state()) is False
    assert called == []


def test_retry_if_result_exception_outcome():
    called = []
    cond = retry_if_result(lambda v: called.append(v) or True)
    assert cond(state_with_exception(ValueError())) is False
    assert called == []


def test_retry_if_not_result():
    cond = retry_if_not_result(lambda v: v is None)
    assert cond(state_with_value(None)) is False
    assert cond(state_with_value(5)) is True


def test_retry_if_not_result_no_outcome():
    cond = retry_if_not_result(lambda v: False)
    assert cond(make_state()) is False


def test_retry_if_not_result_exception_outcome():
    cond = retry_if_not_result(lambda v: False)
    assert cond(state_with_exception(ValueError())) is False


# ---------------------------------------------------------------------------
# retry_all / retry_any
# ---------------------------------------------------------------------------
def test_retry_all_all_true():
    cond = retry_all(retry_always(), retry_always())
    assert cond(make_state()) is True


def test_retry_all_short_circuits_on_false():
    calls = []

    def make(result, label):
        def f(state):
            calls.append(label)
            return result

        return retry_if_exception(f)

    cond = retry_all(make(False, "a"), make(True, "b"))
    assert cond(state_with_exception(ValueError())) is False
    assert calls == ["a"]


def test_retry_all_ignores_numeric_results():
    # numeric results are not bool, so don't trip the "False" short circuit
    numeric = retry_if_exception(lambda exc: 2.5)
    cond = retry_all(numeric, retry_always())
    assert cond(state_with_exception(ValueError())) is True


def test_retry_all_empty_is_true():
    assert retry_all()(make_state()) is True


def test_retry_any_short_circuits_on_true():
    calls = []

    def make(result, label):
        def f(state):
            calls.append(label)
            return result

        return retry_if_exception(f)

    cond = retry_any(make(True, "a"), make(True, "b"))
    assert cond(state_with_exception(ValueError())) is True
    assert calls == ["a"]


def test_retry_any_returns_numeric_as_float():
    cond = retry_any(
        retry_never(),
        retry_if_exception(lambda exc: 3),
        retry_always(),
    )
    result = cond(state_with_exception(ValueError()))
    assert result == 3.0
    assert isinstance(result, float)


def test_retry_any_none_true_returns_false():
    cond = retry_any(retry_never(), retry_never())
    assert cond(make_state()) is False


def test_retry_any_empty_is_false():
    assert retry_any()(make_state()) is False


# ---------------------------------------------------------------------------
# retry_always / retry_never
# ---------------------------------------------------------------------------
def test_retry_always_never():
    assert retry_always()(make_state()) is True
    assert retry_never()(make_state()) is False


# ---------------------------------------------------------------------------
# retry_if_exception_cause_type
# ---------------------------------------------------------------------------
def test_retry_if_exception_cause_type_direct_cause():
    try:
        try:
            raise ValueError("cause")
        except ValueError as cause:
            raise TypeError("outer") from cause
    except TypeError as exc:
        state = state_with_exception(exc)
    assert retry_if_exception_cause_type(ValueError)(state) is True
    assert retry_if_exception_cause_type(KeyError)(state) is False


def test_retry_if_exception_cause_type_context():
    try:
        try:
            raise ValueError("context")
        except ValueError:
            raise TypeError("outer")
    except TypeError as exc:
        state = state_with_exception(exc)
    assert retry_if_exception_cause_type(ValueError)(state) is True


def test_retry_if_exception_cause_type_nested():
    try:
        try:
            try:
                raise KeyError("deep")
            except KeyError as deep:
                raise ValueError("mid") from deep
        except ValueError as mid:
            raise TypeError("outer") from mid
    except TypeError as exc:
        state = state_with_exception(exc)
    assert retry_if_exception_cause_type(KeyError)(state) is True
    assert retry_if_exception_cause_type(ValueError)(state) is True
    assert retry_if_exception_cause_type(RuntimeError)(state) is False


def test_retry_if_exception_cause_type_no_cause():
    state = state_with_exception(ValueError("plain"))
    assert retry_if_exception_cause_type(ValueError)(state) is False


def test_retry_if_exception_cause_type_sequence():
    try:
        try:
            raise ValueError("cause")
        except ValueError as cause:
            raise TypeError("outer") from cause
    except TypeError as exc:
        state = state_with_exception(exc)
    cond = retry_if_exception_cause_type([KeyError, ValueError])
    assert cond.exc_types == (KeyError, ValueError)
    assert cond(state) is True


def test_retry_if_exception_cause_type_no_outcome():
    assert retry_if_exception_cause_type(ValueError)(make_state()) is False


def test_retry_if_exception_cause_type_outcome_without_exception():
    assert retry_if_exception_cause_type(ValueError)(state_with_value(1)) is False
