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


# ---------------------------------------------------------------------------
# Helpers to build fake outcomes / states
# ---------------------------------------------------------------------------


class FakeOutcome:
    def __init__(self, value=None, exception=None, wait=None):
        self.value = value
        self.exception = exception
        self._has_wait = wait is not None
        self._wait = wait

    @property
    def wait(self):
        if not self._has_wait:
            raise AttributeError("wait")
        return self._wait


def make_state(tries=0, elapsed=0.0, outcome=None):
    return RetryState(tries=tries, elapsed=elapsed, outcome=outcome)


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


def test_stop_base_call_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        Stop()(make_state())


def test_retry_condition_base_call_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        RetryCondition()(make_state())


# ---------------------------------------------------------------------------
# stop_after_attempt
# ---------------------------------------------------------------------------


def test_stop_after_attempt_below_and_at_threshold():
    cond = stop_after_attempt(3)
    assert cond(make_state(tries=0)) is False
    assert cond(make_state(tries=2)) is False
    assert cond(make_state(tries=3)) is True
    assert cond(make_state(tries=4)) is True
    assert cond.max_attempts == 3


# ---------------------------------------------------------------------------
# stop_after_delay
# ---------------------------------------------------------------------------


def test_stop_after_delay():
    cond = stop_after_delay(1.5)
    assert cond(make_state(elapsed=0.0)) is False
    assert cond(make_state(elapsed=1.0)) is False
    assert cond(make_state(elapsed=1.5)) is True
    assert cond(make_state(elapsed=2.0)) is True
    assert cond.max_delay == 1.5


# ---------------------------------------------------------------------------
# stop_before_delay
# ---------------------------------------------------------------------------


def test_stop_before_delay_no_outcome():
    cond = stop_before_delay(5.0)
    assert cond(make_state(elapsed=1.0, outcome=None)) is False
    assert cond(make_state(elapsed=5.0, outcome=None)) is True


def test_stop_before_delay_outcome_without_wait_attribute():
    cond = stop_before_delay(5.0)
    outcome = FakeOutcome()  # no wait attribute -> getattr default 0.0
    assert cond(make_state(elapsed=1.0, outcome=outcome)) is False
    assert cond(make_state(elapsed=5.0, outcome=outcome)) is True


def test_stop_before_delay_outcome_with_wait():
    cond = stop_before_delay(5.0)
    outcome = FakeOutcome(wait=2.0)
    assert cond(make_state(elapsed=2.0, outcome=outcome)) is False
    assert cond(make_state(elapsed=3.0, outcome=outcome)) is True
    assert cond(make_state(elapsed=4.0, outcome=outcome)) is True


# ---------------------------------------------------------------------------
# stop_all / stop_any
# ---------------------------------------------------------------------------


def test_stop_all_semantics():
    cond = stop_all(stop_after_attempt(1), stop_after_delay(1.0))
    state = make_state(tries=1, elapsed=1.0)
    assert cond(state) is True
    assert cond(make_state(tries=0, elapsed=5.0)) is False
    assert cond(make_state(tries=5, elapsed=0.0)) is False


def test_stop_all_empty_is_true():
    assert stop_all()(make_state()) is True


def test_stop_any_semantics():
    cond = stop_any(stop_after_attempt(1), stop_after_delay(1.0))
    assert cond(make_state(tries=1, elapsed=0.0)) is True
    assert cond(make_state(tries=0, elapsed=1.0)) is True
    assert cond(make_state(tries=0, elapsed=0.0)) is False


def test_stop_any_empty_is_false():
    assert stop_any()(make_state()) is False


def test_stop_or_and_operators():
    a = stop_after_attempt(1)
    b = stop_after_delay(1.0)

    any_cond = a | b
    assert isinstance(any_cond, stop_any)
    assert any_cond(make_state(tries=1)) is True
    assert any_cond(make_state(elapsed=1.0)) is True
    assert any_cond(make_state()) is False

    all_cond = a & b
    assert isinstance(all_cond, stop_all)
    assert all_cond(make_state(tries=1, elapsed=1.0)) is True
    assert all_cond(make_state(tries=1, elapsed=0.0)) is False


# ---------------------------------------------------------------------------
# stop_never / stop_when_event_set
# ---------------------------------------------------------------------------


def test_stop_never():
    cond = stop_never()
    assert cond(make_state(tries=100, elapsed=1e9)) is False


def test_stop_when_event_set():
    event = threading.Event()
    cond = stop_when_event_set(event)
    assert cond(make_state()) is False
    event.set()
    assert cond(make_state()) is True


# ---------------------------------------------------------------------------
# retry_if_exception_type
# ---------------------------------------------------------------------------


def test_retry_if_exception_type_single():
    cond = retry_if_exception_type(ValueError)
    assert cond(make_state()) is False  # no outcome
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("x")))) is True
    assert cond(make_state(outcome=FakeOutcome(exception=TypeError("x")))) is False
    assert cond(make_state(outcome=FakeOutcome(value=1))) is False  # no exception


def test_retry_if_exception_type_sequence():
    cond = retry_if_exception_type([ValueError, KeyError])
    assert cond(make_state(outcome=FakeOutcome(exception=KeyError("k")))) is True
    assert cond(make_state(outcome=FakeOutcome(exception=RuntimeError("r")))) is False
    assert cond.exc_types == (ValueError, KeyError)


def test_retry_if_exception_type_subclass():
    cond = retry_if_exception_type(Exception)
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("v")))) is True


# ---------------------------------------------------------------------------
# retry_if_exception
# ---------------------------------------------------------------------------


def test_retry_if_exception_no_outcome():
    cond = retry_if_exception(lambda e: True)
    assert cond(make_state()) is False


def test_retry_if_exception_no_exception():
    cond = retry_if_exception(lambda e: True)
    assert cond(make_state(outcome=FakeOutcome(value=1))) is False


def test_retry_if_exception_bool_true():
    cond = retry_if_exception(lambda e: isinstance(e, ValueError))
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError()))) is True
    assert cond(make_state(outcome=FakeOutcome(exception=TypeError()))) is False


def test_retry_if_exception_numeric_result_to_float():
    cond = retry_if_exception(lambda e: 2)
    result = cond(make_state(outcome=FakeOutcome(exception=ValueError())))
    assert result == 2.0
    assert isinstance(result, float)


def test_retry_if_exception_bool_is_not_coerced_to_float():
    cond = retry_if_exception(lambda e: True)
    result = cond(make_state(outcome=FakeOutcome(exception=ValueError())))
    assert result is True


def test_retry_if_exception_passes_exception_instance():
    seen = []

    def predicate(exc):
        seen.append(exc)
        return True

    exc = ValueError("hello")
    retry_if_exception(predicate)(make_state(outcome=FakeOutcome(exception=exc)))
    assert seen == [exc]


# ---------------------------------------------------------------------------
# retry_if_exception_message
# ---------------------------------------------------------------------------


def test_retry_if_exception_message_no_outcome_or_exception():
    cond = retry_if_exception_message("boom")
    assert cond(make_state()) is False
    assert cond(make_state(outcome=FakeOutcome(value=1))) is False


def test_retry_if_exception_message_substring():
    cond = retry_if_exception_message("boom")
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("kaboom!")))) is True
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("nope")))) is False


def test_retry_if_exception_message_regex():
    cond = retry_if_exception_message(r"err\d+", match="re")
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("err42")))) is True
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("err")))) is False
    assert cond.match == "re"


def test_retry_if_exception_message_regex_special_used_as_substring():
    cond = retry_if_exception_message(r"err\d+")
    # Without match="re", treated as plain substring, so fails.
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("err42")))) is False
    # And literal regex text matches when present.
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError(r"err\d+")))) is True


# ---------------------------------------------------------------------------
# retry_if_result / retry_if_not_result
# ---------------------------------------------------------------------------


def test_retry_if_result_no_outcome():
    cond = retry_if_result(lambda v: True)
    assert cond(make_state()) is False


def test_retry_if_result_exception_present():
    cond = retry_if_result(lambda v: True)
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError()))) is False


def test_retry_if_result_predicate():
    cond = retry_if_result(lambda v: v is None)
    assert cond(make_state(outcome=FakeOutcome(value=None))) is True
    assert cond(make_state(outcome=FakeOutcome(value=5))) is False


def test_retry_if_not_result_no_outcome():
    cond = retry_if_not_result(lambda v: True)
    assert cond(make_state()) is False


def test_retry_if_not_result_exception_present():
    cond = retry_if_not_result(lambda v: True)
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError()))) is False


def test_retry_if_not_result_predicate():
    cond = retry_if_not_result(lambda v: v is None)
    assert cond(make_state(outcome=FakeOutcome(value=None))) is False
    assert cond(make_state(outcome=FakeOutcome(value=5))) is True


# ---------------------------------------------------------------------------
# retry_all / retry_any
# ---------------------------------------------------------------------------


def test_retry_all_true_when_all_pass():
    cond = retry_all(retry_always(), retry_never() | retry_always())
    assert cond(make_state()) is True


def test_retry_all_false_when_any_bool_false():
    cond = retry_all(retry_always(), retry_never(), retry_always())
    assert cond(make_state()) is False


def test_retry_all_ignores_numeric_results():
    retry_num = retry_if_exception(lambda e: 3)
    state = make_state(outcome=FakeOutcome(exception=ValueError()))
    cond = retry_all(retry_num, retry_always())
    assert cond(state) is True


def test_retry_all_empty_true():
    assert retry_all()(make_state()) is True


def test_retry_all_short_circuits():
    calls = []

    def make(name, value):
        def cond(state):
            calls.append(name)
            return value
        return cond

    cond = retry_all(make("first", False), make("second", True))
    assert cond(make_state()) is False
    assert calls == ["first"]


def test_retry_any_true_when_any_true():
    cond = retry_any(retry_never(), retry_always())
    assert cond(make_state()) is True


def test_retry_any_false_when_all_false():
    cond = retry_any(retry_never(), retry_never())
    assert cond(make_state()) is False


def test_retry_any_numeric_result():
    retry_num = retry_if_exception(lambda e: 4)
    state = make_state(outcome=FakeOutcome(exception=ValueError()))
    result = retry_any(retry_num)
    assert result(state) == 4.0
    assert isinstance(result(state), float)


def test_retry_any_empty_false():
    assert retry_any()(make_state()) is False


def test_retry_any_short_circuits_on_true():
    calls = []

    def make(name, value):
        def cond(state):
            calls.append(name)
            return value
        return cond

    cond = retry_any(make("first", True), make("second", True))
    assert cond(make_state()) is True
    assert calls == ["first"]


# ---------------------------------------------------------------------------
# retry operators
# ---------------------------------------------------------------------------


def test_retry_condition_or_and():
    a = retry_always()
    b = retry_never()

    or_cond = a | b
    assert isinstance(or_cond, retry_any)
    assert or_cond(make_state()) is True

    and_cond = a & b
    assert isinstance(and_cond, retry_all)
    assert and_cond(make_state()) is False


# ---------------------------------------------------------------------------
# retry_always / retry_never
# ---------------------------------------------------------------------------


def test_retry_always_never():
    assert retry_always()(make_state()) is True
    assert retry_never()(make_state()) is False


# ---------------------------------------------------------------------------
# retry_if_not_exception_type / retry_unless_exception_type
# ---------------------------------------------------------------------------


def test_retry_if_not_exception_type_no_outcome():
    cond = retry_if_not_exception_type(ValueError)
    assert cond(make_state()) is False


def test_retry_if_not_exception_type_no_exception():
    cond = retry_if_not_exception_type(ValueError)
    assert cond(make_state(outcome=FakeOutcome(value=1))) is False


def test_retry_if_not_exception_type_single():
    cond = retry_if_not_exception_type(ValueError)
    assert cond(make_state(outcome=FakeOutcome(exception=TypeError()))) is True
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError()))) is False


def test_retry_if_not_exception_type_sequence():
    cond = retry_if_not_exception_type([ValueError, KeyError])
    assert cond(make_state(outcome=FakeOutcome(exception=TypeError()))) is True
    assert cond(make_state(outcome=FakeOutcome(exception=KeyError()))) is False
    assert cond.exc_types == (ValueError, KeyError)


def test_retry_unless_exception_type_is_subclass():
    assert issubclass(retry_unless_exception_type, retry_if_not_exception_type)
    cond = retry_unless_exception_type(ValueError)
    assert cond(make_state(outcome=FakeOutcome(exception=TypeError()))) is True
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError()))) is False


# ---------------------------------------------------------------------------
# retry_if_not_exception_message
# ---------------------------------------------------------------------------


def test_retry_if_not_exception_message_no_outcome_or_exception():
    cond = retry_if_not_exception_message("boom")
    assert cond(make_state()) is False
    assert cond(make_state(outcome=FakeOutcome(value=1))) is False


def test_retry_if_not_exception_message_substring():
    cond = retry_if_not_exception_message("boom")
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("kaboom")))) is False
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("fine")))) is True


def test_retry_if_not_exception_message_regex():
    cond = retry_if_not_exception_message(r"err\d+", regex=True)
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("err42")))) is False
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("ok")))) is True
    assert cond.regex is True


def test_retry_if_not_exception_message_regex_default_false():
    cond = retry_if_not_exception_message(r"err\d+")
    assert cond.regex is False
    # Without regex, literal substring check.
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError(r"err\d+")))) is False
    assert cond(make_state(outcome=FakeOutcome(exception=ValueError("err42")))) is True


# ---------------------------------------------------------------------------
# retry_if_exception_cause_type
# ---------------------------------------------------------------------------


def test_retry_if_exception_cause_type_no_outcome_or_exception():
    cond = retry_if_exception_cause_type(ValueError)
    assert cond(make_state()) is False
    assert cond(make_state(outcome=FakeOutcome(value=1))) is False


def test_retry_if_exception_cause_type_direct_cause():
    try:
        try:
            raise ValueError("inner")
        except ValueError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as exc:
        state = make_state(outcome=FakeOutcome(exception=exc))

    cond = retry_if_exception_cause_type(ValueError)
    assert cond(state) is True
    assert retry_if_exception_cause_type(KeyError)(state) is False


def test_retry_if_exception_cause_type_uses_context_when_no_cause():
    try:
        try:
            raise ValueError("inner")
        except ValueError:
            raise RuntimeError("outer")
    except RuntimeError as exc:
        state = make_state(outcome=FakeOutcome(exception=exc))

    cond = retry_if_exception_cause_type(ValueError)
    assert cond(state) is True


def test_retry_if_exception_cause_type_nested_chain():
    try:
        try:
            try:
                raise KeyError("deep")
            except KeyError as deep:
                raise ValueError("middle") from deep
        except ValueError as mid:
            raise RuntimeError("top") from mid
    except RuntimeError as exc:
        state = make_state(outcome=FakeOutcome(exception=exc))

    cond = retry_if_exception_cause_type(KeyError)
    assert cond(state) is True
    assert retry_if_exception_cause_type(TypeError)(state) is False


def test_retry_if_exception_cause_type_no_cause():
    state = make_state(outcome=FakeOutcome(exception=ValueError("solo")))
    assert retry_if_exception_cause_type(ValueError)(state) is False


def test_retry_if_exception_cause_type_sequence():
    try:
        try:
            raise KeyError("inner")
        except KeyError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as exc:
        state = make_state(outcome=FakeOutcome(exception=exc))

    cond = retry_if_exception_cause_type([ValueError, KeyError])
    assert cond(state) is True
    assert cond.exc_types == (ValueError, KeyError)
