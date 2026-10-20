from __future__ import annotations

import re
import threading

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
)
from backon._state import RetryState


class _Outcome:
    """Minimal stand-in for backon._state.Attempt."""

    def __init__(self, value=None, exception=None, wait=0.0):
        self.value = value
        self.exception = exception
        self.wait = wait


def make_state(tries=0, elapsed=0.0, outcome=None):
    return RetryState(tries=tries, elapsed=elapsed, outcome=outcome)


class _Base(self_less=True):
    pass


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


def test_stop_base_call_raises():
    with pytest.raises(NotImplementedError):
        Stop()(make_state())


def test_retry_condition_base_call_raises():
    with pytest.raises(NotImplementedError):
        RetryCondition()(make_state())


def test_stop_or_returns_stop_any():
    s = Stop.__or__(Stop(), Stop())
    from backon._conditions import stop_any

    assert isinstance(s, stop_any)


def test_stop_and_returns_stop_all():
    s = Stop.__and__(Stop(), Stop())
    from backon._conditions import stop_all

    assert isinstance(s, stop_all)


def test_retry_condition_or_returns_retry_any():
    r = RetryCondition.__or__(RetryCondition(), RetryCondition())
    assert isinstance(r, retry_any)


def test_retry_condition_and_returns_retry_all():
    r = RetryCondition.__and__(RetryCondition(), RetryCondition())
    assert isinstance(r, retry_all)


# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------


def test_stop_after_attempt():
    from backon._conditions import stop_after_attempt

    cond = stop_after_attempt(3)
    assert cond.max_attempts == 3
    assert cond(make_state(tries=0)) is False
    assert cond(make_state(tries=2)) is False
    assert cond(make_state(tries=3)) is True
    assert cond(make_state(tries=5)) is True


def test_stop_after_delay():
    from backon._conditions import stop_after_delay

    cond = stop_after_delay(10.0)
    assert cond.max_delay == 10.0
    assert cond(make_state(elapsed=0.0)) is False
    assert cond(make_state(elapsed=9.99)) is False
    assert cond(make_state(elapsed=10.0)) is True
    assert cond(make_state(elapsed=100.0)) is True


def test_stop_before_delay_includes_wait():
    from backon._conditions import stop_before_delay

    cond = stop_before_delay(10.0)
    assert cond.max_delay == 10.0
    # no outcome -> wait is 0
    assert cond(make_state(elapsed=5.0)) is False
    assert cond(make_state(elapsed=10.0)) is True
    # with outcome wait
    st = make_state(elapsed=6.0, outcome=_Outcome(wait=3.0))
    assert cond(st) is False
    st = make_state(elapsed=6.0, outcome=_Outcome(wait=4.0))
    assert cond(st) is True


def test_stop_before_delay_no_wait_attr():
    from backon._conditions import stop_before_delay

    cond = stop_before_delay(5.0)

    class NoWait:
        pass

    st = make_state(elapsed=4.0, outcome=NoWait())
    assert cond(st) is False
    st = make_state(elapsed=5.0, outcome=NoWait())
    assert cond(st) is True


def test_stop_all():
    from backon._conditions import stop_after_attempt, stop_all

    cond = stop_all(stop_after_attempt(2), stop_after_attempt(3))
    assert cond.stops is not None and len(cond.stops) == 2
    assert cond(make_state(tries=1)) is False
    assert cond(make_state(tries=2)) is False
    assert cond(make_state(tries=3)) is True


def test_stop_all_empty_is_true():
    from backon._conditions import stop_all

    assert stop_all()(make_state()) is True


def test_stop_any():
    from backon._conditions import stop_after_attempt, stop_any

    cond = stop_any(stop_after_attempt(2), stop_after_attempt(3))
    assert len(cond.stops) == 2
    assert cond(make_state(tries=1)) is False
    assert cond(make_state(tries=2)) is True
    assert cond(make_state(tries=3)) is True


def test_stop_any_empty_is_false():
    from backon._conditions import stop_any

    assert stop_any()(make_state()) is False


def test_stop_never():
    from backon._conditions import stop_never

    cond = stop_never()
    assert cond(make_state(tries=1000, elapsed=99999.0)) is False


def test_stop_when_event_set():
    from backon._conditions import stop_when_event_set

    ev = threading.Event()
    cond = stop_when_event_set(ev)
    assert cond.event is ev
    assert cond(make_state()) is False
    ev.set()
    assert cond(make_state()) is True


def test_stop_operator_composition():
    from backon._conditions import stop_after_attempt

    a = stop_after_attempt(5)
    b = stop_after_attempt(2)
    any_cond = a | b
    all_cond = a & b
    assert any_cond(make_state(tries=2)) is True
    assert any_cond(make_state(tries=1)) is False
    assert all_cond(make_state(tries=2)) is False
    assert all_cond(make_state(tries=5)) is True


# ---------------------------------------------------------------------------
# retry_if_exception_type / retry_if_not_exception_type
# ---------------------------------------------------------------------------


def test_retry_if_exception_type_single():
    cond = retry_if_exception_type(ValueError)
    assert cond.exc_types == (ValueError,)
    assert cond(make_state()) is False
    assert cond(make_state(outcome=_Outcome(exception=ValueError()))) is True
    assert cond(make_state(outcome=_Outcome(exception=TypeError()))) is False
    assert cond(make_state(outcome=_Outcome(value=1))) is False


def test_retry_if_exception_type_sequence():
    cond = retry_if_exception_type([ValueError, KeyError])
    assert cond.exc_types == (ValueError, KeyError)
    assert cond(make_state(outcome=_Outcome(exception=KeyError()))) is True
    assert cond(make_state(outcome=_Outcome(exception=RuntimeError()))) is False


def test_retry_if_exception_type_subclass():
    cond = retry_if_exception_type(Exception)
    assert cond(make_state(outcome=_Outcome(exception=ValueError()))) is True


def test_retry_if_not_exception_type():
    cond = retry_if_not_exception_type(ValueError)
    assert cond.exc_types == (ValueError,)
    assert cond(make_state()) is False
    assert cond(make_state(outcome=_Outcome(exception=ValueError()))) is False
    assert cond(make_state(outcome=_Outcome(exception=TypeError()))) is True
    assert cond(make_state(outcome=_Outcome(value=1))) is False


def test_retry_unless_exception_type_alias():
    assert issubclass(retry_unless_exception_type, retry_if_not_exception_type)
    cond = retry_unless_exception_type(ValueError)
    assert cond(make_state(outcome=_Outcome(exception=KeyError()))) is True
    assert cond(make_state(outcome=_Outcome(exception=ValueError()))) is False


# ---------------------------------------------------------------------------
# retry_if_exception
# ---------------------------------------------------------------------------


def test_retry_if_exception_bool():
    cond = retry_if_exception(lambda e: isinstance(e, ValueError))
    assert cond(make_state()) is False
    assert cond(make_state(outcome=_Outcome(value=1))) is False
    assert cond(make_state(outcome=_Outcome(exception=ValueError()))) is True
    assert cond(make_state(outcome=_Outcome(exception=TypeError()))) is False


def test_retry_if_exception_float_returns_float():
    cond = retry_if_exception(lambda e: 0.5)
    result = cond(make_state(outcome=_Outcome(exception=ValueError())))
    assert result == 0.5
    assert isinstance(result, float)


def test_retry_if_exception_int_returns_float():
    cond = retry_if_exception(lambda e: 3)
    result = cond(make_state(outcome=_Outcome(exception=ValueError())))
    assert result == 3.0
    assert isinstance(result, float)


def test_retry_if_exception_bool_true_not_converted_to_float():
    cond = retry_if_exception(lambda e: True)
    result = cond(make_state(outcome=_Outcome(exception=ValueError())))
    assert result is True
    assert isinstance(result, bool)


def test_retry_if_exception_non_numeric_coerced_to_bool():
    cond = retry_if_exception(lambda e: "truthy")
    result = cond(make_state(outcome=_Outcome(exception=ValueError())))
    assert result is True


# ---------------------------------------------------------------------------
# retry_if_exception_message
# ---------------------------------------------------------------------------


def test_retry_if_exception_message_contains():
    cond = retry_if_exception_message("boom")
    assert cond.message == "boom"
    assert cond.match is None
    assert cond(make_state()) is False
    assert cond(make_state(outcome=_Outcome(value=1))) is False
    assert cond(make_state(outcome=_Outcome(exception=ValueError("boom!")))) is True
    assert cond(make_state(outcome=_Outcome(exception=ValueError("nope")))) is False


def test_retry_if_exception_message_regex():
    cond = retry_if_exception_message(r"bo+m", match="re")
    assert cond(make_state(outcome=_Outcome(exception=ValueError("boom")))) is True
    assert cond(make_state(outcome=_Outcome(exception=ValueError("bm")))) is False


# ---------------------------------------------------------------------------
# retry_if_not_exception_message
# ---------------------------------------------------------------------------


def test_retry_if_not_exception_message_plain():
    cond = retry_if_not_exception_message("boom")
    assert cond.match == "boom"
    assert cond.regex is False
    assert cond(make_state()) is False
    assert cond(make_state(outcome=_Outcome(value=1))) is False
    assert cond(make_state(outcome=_Outcome(exception=ValueError("boom!")))) is False
    assert cond(make_state(outcome=_Outcome(exception=ValueError("nope")))) is True


def test_retry_if_not_exception_message_regex():
    cond = retry_if_not_exception_message(r"bo+m", regex=True)
    assert cond.regex is True
    assert cond(make_state(outcome=_Outcome(exception=ValueError("boom")))) is False
    assert cond(make_state(outcome=_Outcome(exception=ValueError("other")))) is True


# ---------------------------------------------------------------------------
# retry_if_result / retry_if_not_result
# ---------------------------------------------------------------------------


def test_retry_if_result():
    cond = retry_if_result(lambda v: v < 0)
    assert cond(make_state()) is False
    assert cond(make_state(outcome=_Outcome(exception=ValueError()))) is False
    assert cond(make_state(outcome=_Outcome(value=-1))) is True
    assert cond(make_state(outcome=_Outcome(value=1))) is False


def test_retry_if_not_result():
    cond = retry_if_not_result(lambda v: v < 0)
    assert cond(make_state()) is False
    assert cond(make_state(outcome=_Outcome(exception=ValueError()))) is False
    assert cond(make_state(outcome=_Outcome(value=-1))) is False
    assert cond(make_state(outcome=_Outcome(value=1))) is True


# ---------------------------------------------------------------------------
# retry_all / retry_any
# ---------------------------------------------------------------------------


def test_retry_all():
    always = retry_always()
    never = retry_never()
    assert retry_all(always, always)(make_state()) is True
    assert retry_all(always, never)(make_state()) is False
    assert retry_all()(make_state()) is True


def test_retry_all_length_and_order():
    cond = retry_all(retry_always(), retry_always())
    assert len(cond.conditions) == 2


def test_retry_all_short_circuits_on_false():
    calls = []

    class Tracker(RetryCondition):
        def __init__(self, ret):
            self.ret = ret

        def __call__(self, state):
            calls.append(self.ret)
            return self.ret

    cond = retry_all(Tracker(False), Tracker(True))
    assert cond(make_state()) is False
    assert calls == [False]


def test_retry_all_ignores_non_bool_results():
    class FloatCond(RetryCondition):
        def __call__(self, state):
            return 1.5

    class TrueCond(RetryCondition):
        def __call__(self, state):
            return True

    assert retry_all(FloatCond(), TrueCond())(make_state()) is True


def test_retry_any():
    always = retry_always()
    never = retry_never()
    assert retry_any(never, never)(make_state()) is False
    assert retry_any(never, always)(make_state()) is True
    assert retry_any()(make_state()) is False


def test_retry_any_length_and_order():
    cond = retry_any(retry_never(), retry_never())
    assert len(cond.conditions) == 2


def test_retry_any_short_circuits_on_true():
    calls = []

    class Tracker(RetryCondition):
        def __init__(self, ret):
            self.ret = ret

        def __call__(self, state):
            calls.append(self.ret)
            return self.ret

    cond = retry_any(Tracker(True), Tracker(False))
    assert cond(make_state()) is True
    assert calls == [True]


def test_retry_any_returns_float_result():
    class FloatCond(RetryCondition):
        def __call__(self, state):
            return 2

    class FalseCond(RetryCondition):
        def __call__(self, state):
            return False

    result = retry_any(FalseCond(), FloatCond())(make_state())
    assert result == 2.0
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# retry_always / retry_never
# ---------------------------------------------------------------------------


def test_retry_always():
    assert retry_always()(make_state()) is True


def test_retry_never():
    assert retry_never()(make_state()) is False


# ---------------------------------------------------------------------------
# retry_if_exception_cause_type
# ---------------------------------------------------------------------------


def test_retry_if_exception_cause_type_no_outcome():
    cond = retry_if_exception_cause_type(ValueError)
    assert cond(make_state()) is False


def test_retry_if_exception_cause_type_no_exception():
    cond = retry_if_exception_cause_type(ValueError)
    assert cond(make_state(outcome=_Outcome(value=1))) is False


def test_retry_if_exception_cause_type_direct_cause():
    cond = retry_if_exception_cause_type(ValueError)
    exc = RuntimeError("outer")
    exc.__cause__ = ValueError("inner")
    assert cond(make_state(outcome=_Outcome(exception=exc))) is True


def test_retry_if_exception_cause_type_context_fallback():
    cond = retry_if_exception_cause_type(ValueError)
    exc = RuntimeError("outer")
    exc.__context__ = ValueError("inner")
    assert cond(make_state(outcome=_Outcome(exception=exc))) is True


def test_retry_if_exception_cause_type_chain():
    cond = retry_if_exception_cause_type(KeyError)
    exc = RuntimeError("outer")
    mid = ValueError("mid")
    inner = KeyError("inner")
    exc.__cause__ = mid
    mid.__cause__ = inner
    assert cond(make_state(outcome=_Outcome(exception=exc))) is True


def test_retry_if_exception_cause_type_no_match():
    cond = retry_if_exception_cause_type(KeyError)
    exc = RuntimeError("outer")
    exc.__cause__ = ValueError("inner")
    assert cond(make_state(outcome=_Outcome(exception=exc))) is False


def test_retry_if_exception_cause_type_no_cause():
    cond = retry_if_exception_cause_type(ValueError)
    assert cond(make_state(outcome=_Outcome(exception=RuntimeError()))) is False


def test_retry_if_exception_cause_type_sequence():
    cond = retry_if_exception_cause_type((ValueError, KeyError))
    assert cond.exc_types == (ValueError, KeyError)
    exc = RuntimeError()
    exc.__cause__ = KeyError()
    assert cond(make_state(outcome=_Outcome(exception=exc))) is True
