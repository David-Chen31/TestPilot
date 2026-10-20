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


class Attempt:
    """Minimal stand-in for backon._state.Attempt used as state.outcome."""

    def __init__(self, value=None, exception=None, wait=0.0):
        self.value = value
        self.exception = exception
        self.wait = wait


def make_state(tries=0, elapsed=0.0, outcome=None):
    return RetryState(tries=tries, elapsed=elapsed, outcome=outcome)


# --------------------------------------------------------------------------
# Stop base class
# --------------------------------------------------------------------------

def test_stop_base_call_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        Stop()(make_state())


def test_stop_or_returns_stop_any():
    s1 = stop_after_attempt(1)
    s2 = stop_after_attempt(2)
    combined = s1 | s2
    assert isinstance(combined, stop_any)
    assert combined.stops == (s1, s2)


def test_stop_and_returns_stop_all():
    s1 = stop_after_attempt(1)
    s2 = stop_after_attempt(2)
    combined = s1 & s2
    assert isinstance(combined, stop_all)
    assert combined.stops == (s1, s2)


# --------------------------------------------------------------------------
# stop_after_attempt
# --------------------------------------------------------------------------

def test_stop_after_attempt_stores_max_attempts():
    s = stop_after_attempt(3)
    assert s.max_attempts == 3


@pytest.mark.parametrize(
    "tries,max_attempts,expected",
    [
        (0, 1, False),
        (1, 1, True),
        (2, 1, True),
        (0, 3, False),
        (2, 3, False),
        (3, 3, True),
        (4, 3, True),
    ],
)
def test_stop_after_attempt_call(tries, max_attempts, expected):
    s = stop_after_attempt(max_attempts)
    assert s(make_state(tries=tries)) is expected


# --------------------------------------------------------------------------
# stop_after_delay
# --------------------------------------------------------------------------

def test_stop_after_delay_stores_max_delay():
    s = stop_after_delay(5.0)
    assert s.max_delay == 5.0


@pytest.mark.parametrize(
    "elapsed,max_delay,expected",
    [
        (0.0, 1.0, False),
        (0.5, 1.0, False),
        (1.0, 1.0, True),
        (1.5, 1.0, True),
        (0.0, 0.0, True),
    ],
)
def test_stop_after_delay_call(elapsed, max_delay, expected):
    s = stop_after_delay(max_delay)
    assert s(make_state(elapsed=elapsed)) is expected


# --------------------------------------------------------------------------
# stop_before_delay
# --------------------------------------------------------------------------

def test_stop_before_delay_stores_max_delay():
    s = stop_before_delay(5.0)
    assert s.max_delay == 5.0


def test_stop_before_delay_no_outcome_uses_zero_wait():
    s = stop_before_delay(1.0)
    assert s(make_state(elapsed=0.5, outcome=None)) is False
    assert s(make_state(elapsed=1.0, outcome=None)) is True


def test_stop_before_delay_with_outcome_wait():
    s = stop_before_delay(1.0)
    outcome = Attempt(wait=0.6)
    # 0.5 + 0.6 = 1.1 >= 1.0 -> True
    assert s(make_state(elapsed=0.5, outcome=outcome)) is True


def test_stop_before_delay_with_outcome_wait_below():
    s = stop_before_delay(1.0)
    outcome = Attempt(wait=0.3)
    # 0.5 + 0.3 = 0.8 < 1.0 -> False
    assert s(make_state(elapsed=0.5, outcome=outcome)) is False


def test_stop_before_delay_outcome_without_wait_attr():
    s = stop_before_delay(1.0)

    class NoWait:
        pass

    outcome = NoWait()
    # getattr falls back to 0.0
    assert s(make_state(elapsed=0.9, outcome=outcome)) is False
    assert s(make_state(elapsed=1.0, outcome=outcome)) is True


def test_stop_before_delay_boundary_equality():
    s = stop_before_delay(2.0)
    outcome = Attempt(wait=0.5)
    assert s(make_state(elapsed=1.5, outcome=outcome)) is True


# --------------------------------------------------------------------------
# stop_all
# --------------------------------------------------------------------------

def test_stop_all_empty_returns_true():
    # all() of empty is True
    assert stop_all()(make_state()) is True


def test_stop_all_stores_stops():
    s1 = stop_after_attempt(1)
    s2 = stop_after_attempt(2)
    combined = stop_all(s1, s2)
    assert combined.stops == (s1, s2)


def test_stop_all_all_true():
    combined = stop_all(stop_after_attempt(1), stop_after_delay(0.5))
    assert combined(make_state(tries=1, elapsed=0.6)) is True


def test_stop_all_one_false():
    combined = stop_all(stop_after_attempt(1), stop_after_delay(1.0))
    assert combined(make_state(tries=1, elapsed=0.2)) is False


def test_stop_all_short_circuits():
    calls = []

    class Tracking(Stop):
        def __init__(self, result):
            self.result = result

        def __call__(self, state):
            calls.append(self.result)
            return self.result

    combined = stop_all(Tracking(False), Tracking(True))
    assert combined(make_state()) is False
    assert calls == [False]


# --------------------------------------------------------------------------
# stop_any
# --------------------------------------------------------------------------

def test_stop_any_empty_returns_false():
    assert stop_any()(make_state()) is False


def test_stop_any_stores_stops():
    s1 = stop_after_attempt(1)
    s2 = stop_after_attempt(2)
    combined = stop_any(s1, s2)
    assert combined.stops == (s1, s2)


def test_stop_any_one_true():
    combined = stop_any(stop_after_attempt(1), stop_after_delay(10.0))
    assert combined(make_state(tries=1, elapsed=0.0)) is True


def test_stop_any_all_false():
    combined = stop_any(stop_after_attempt(5), stop_after_delay(10.0))
    assert combined(make_state(tries=1, elapsed=0.0)) is False


def test_stop_any_short_circuits():
    calls = []

    class Tracking(Stop):
        def __init__(self, result):
            self.result = result

        def __call__(self, state):
            calls.append(self.result)
            return self.result

    combined = stop_any(Tracking(True), Tracking(False))
    assert combined(make_state()) is True
    assert calls == [True]


# --------------------------------------------------------------------------
# stop_never
# --------------------------------------------------------------------------

def test_stop_never_always_false():
    s = stop_never()
    assert s(make_state()) is False
    assert s(make_state(tries=100, elapsed=1000.0)) is False


# --------------------------------------------------------------------------
# stop_when_event_set
# --------------------------------------------------------------------------

def test_stop_when_event_set_stores_event():
    event = threading.Event()
    s = stop_when_event_set(event)
    assert s.event is event


def test_stop_when_event_set_false_when_not_set():
    event = threading.Event()
    s = stop_when_event_set(event)
    assert s(make_state()) is False


def test_stop_when_event_set_true_when_set():
    event = threading.Event()
    s = stop_when_event_set(event)
    event.set()
    assert s(make_state()) is True


# --------------------------------------------------------------------------
# RetryCondition base class
# --------------------------------------------------------------------------

def test_retry_condition_base_call_raises():
    with pytest.raises(NotImplementedError):
        RetryCondition()(make_state())


def test_retry_condition_or_returns_retry_any():
    c1 = retry_always()
    c2 = retry_never()
    combined = c1 | c2
    assert isinstance(combined, retry_any)
    assert combined.conditions == (c1, c2)


def test_retry_condition_and_returns_retry_all():
    c1 = retry_always()
    c2 = retry_never()
    combined = c1 & c2
    assert isinstance(combined, retry_all)
    assert combined.conditions == (c1, c2)


# --------------------------------------------------------------------------
# retry_if_exception_type
# --------------------------------------------------------------------------

def test_retry_if_exception_type_single_type_stored_as_tuple():
    c = retry_if_exception_type(ValueError)
    assert c.exc_types == (ValueError,)


def test_retry_if_exception_type_sequence_stored_as_tuple():
    c = retry_if_exception_type([ValueError, TypeError])
    assert c.exc_types == (ValueError, TypeError)


def test_retry_if_exception_type_none_outcome_false():
    c = retry_if_exception_type(ValueError)
    assert c(make_state(outcome=None)) is False


def test_retry_if_exception_type_no_exception_false():
    c = retry_if_exception_type(ValueError)
    assert c(make_state(outcome=Attempt(value=1))) is False


def test_retry_if_exception_type_matching():
    c = retry_if_exception_type(ValueError)
    assert c(make_state(outcome=Attempt(exception=ValueError()))) is True


def test_retry_if_exception_type_non_matching():
    c = retry_if_exception_type(ValueError)
    assert c(make_state(outcome=Attempt(exception=TypeError()))) is False


def test_retry_if_exception_type_multiple():
    c = retry_if_exception_type((ValueError, TypeError))
    assert c(make_state(outcome=Attempt(exception=TypeError()))) is True
    assert c(make_state(outcome=Attempt(exception=KeyError()))) is False


def test_retry_if_exception_type_subclass():
    class MyError(ValueError):
        pass

    c = retry_if_exception_type(ValueError)
    assert c(make_state(outcome=Attempt(exception=MyError()))) is True


# --------------------------------------------------------------------------
# retry_if_exception
# --------------------------------------------------------------------------

def test_retry_if_exception_stores_predicate():
    pred = lambda exc: True
    c = retry_if_exception(pred)
    assert c.predicate is pred


def test_retry_if_exception_none_outcome_false():
    c = retry_if_exception(lambda exc: True)
    assert c(make_state(outcome=None)) is False


def test_retry_if_exception_no_exception_false():
    c = retry_if_exception(lambda exc: True)
    assert c(make_state(outcome=Attempt(value=1))) is False


def test_retry_if_exception_predicate_called_with_exception():
    seen = []

    def pred(exc):
        seen.append(exc)
        return True

    exc = ValueError("bad")
    c = retry_if_exception(pred)
    result = c(make_state(outcome=Attempt(exception=exc)))
    assert result is True
    assert seen == [exc]


def test_retry_if_exception_bool_result_passthrough():
    c = retry_if_exception(lambda exc: False)
    assert c(make_state(outcome=Attempt(exception=ValueError()))) is False


def test_retry_if_exception_numeric_result_converted_to_float():
    c = retry_if_exception(lambda exc: 5)
    result = c(make_state(outcome=Attempt(exception=ValueError())))
    assert result == 5.0
    assert isinstance(result, float)


def test_retry_if_exception_float_result_converted_to_float():
    c = retry_if_exception(lambda exc: 2.5)
    result = c(make_state(outcome=Attempt(exception=ValueError())))
    assert result == 2.5
    assert isinstance(result, float)


def test_retry_if_exception_bool_not_converted_to_float():
    c = retry_if_exception(lambda exc: True)
    result = c(make_state(outcome=Attempt(exception=ValueError())))
    assert result is True


def test_retry_if_exception_non_numeric_truthy_coerced_to_bool():
    c = retry_if_exception(lambda exc: "yes")
    result = c(make_state(outcome=Attempt(exception=ValueError())))
    assert result is True


# --------------------------------------------------------------------------
# retry_if_exception_message
# --------------------------------------------------------------------------

def test_retry_if_exception_message_stores_attrs():
    c = retry_if_exception_message("boom")
    assert c.message == "boom"
    assert c.match is None


def test_retry_if_exception_message_stores_match():
    c = retry_if_exception_message("boom", match="re")
    assert c.match == "re"


def test_retry_if_exception_message_none_outcome_false():
    c = retry_if_exception_message("boom")
    assert c(make_state(outcome=None)) is False


def test_retry_if_exception_message_no_exception_false():
    c = retry_if_exception_message("boom")
    assert c(make_state(outcome=Attempt(value=1))) is False


def test_retry_if_exception_message_substring_match():
    c = retry_if_exception_message("boom")
    assert c(make_state(outcome=Attempt(exception=ValueError("big boom here")))) is True


def test_retry_if_exception_message_substring_no_match():
    c = retry_if_exception_message("boom")
    assert c(make_state(outcome=Attempt(exception=ValueError("quiet")))) is False


def test_retry_if_exception_message_regex_match():
    c = retry_if_exception_message(r"b\w+m", match="re")
    assert c(make_state(outcome=Attempt(exception=ValueError("boom")))) is True


def test_retry_if_exception_message_regex_no_match():
    c = retry_if_exception_message(r"^boom$", match="re")
    assert c(make_state(outcome=Attempt(exception=ValueError("boom!")))) is False


def test_retry_if_exception_message_regex_uses_search():
    c = retry_if_exception_message(r"boom", match="re")
    assert c(make_state(outcome=Attempt(exception=ValueError("xx boom yy")))) is True


# --------------------------------------------------------------------------
# retry_if_result
# --------------------------------------------------------------------------

def test_retry_if_result_none_outcome_false():
    c = retry_if_result(lambda v: True)
    assert c(make_state(outcome=None)) is False


def test_retry_if_result_exception_outcome_false():
    c = retry_if_result(lambda v: True)
    assert c(make_state(outcome=Attempt(exception=ValueError()))) is False


def test_retry_if_result_predicate_true():
    c = retry_if_result(lambda v: v == 5)
    assert c(make_state(outcome=Attempt(value=5))) is True


def test_retry_if_result_predicate_false():
    c = retry_if_result(lambda v: v == 5)
    assert c(make_state(outcome=Attempt(value=6))) is False


# --------------------------------------------------------------------------
# retry_if_not_result
# --------------------------------------------------------------------------

def test_retry_if_not_result_none_outcome_false():
    c = retry_if_not_result(lambda v: True)
    assert c(make_state(outcome=None)) is False


def test_retry_if_not_result_exception_outcome_false():
    c = retry_if_not_result(lambda v: True)
    assert c(make_state(outcome=Attempt(exception=ValueError()))) is False


def test_retry_if_not_result_inverts_predicate_true():
    c = retry_if_not_result(lambda v: v == 5)
    assert c(make_state(outcome=Attempt(value=6))) is True


def test_retry_if_not_result_inverts_predicate_false():
    c = retry_if_not_result(lambda v: v == 5)
    assert c(make_state(outcome=Attempt(value=5))) is False


# --------------------------------------------------------------------------
# retry_all
# --------------------------------------------------------------------------

def test_retry_all_stores_conditions():
    c1 = retry_always()
    c2 = retry_never()
    combined = retry_all(c1, c2)
    assert combined.conditions == (c1, c2)


def test_retry_all_empty_returns_true():
    assert retry_all()(make_state()) is True


def test_retry_all_all_true():
    combined = retry_all(retry_always(), retry_always())
    assert combined(make_state()) is True


def test_retry_all_one_bool_false():
    combined = retry_all(retry_always(), retry_never())
    assert combined(make_state()) is False


def test_retry_all_numeric_result_is_not_false():
    # numeric results are not bool, so they do not trigger early False
    c = retry_if_exception(lambda exc: 5.0)
    combined = retry_all(c, retry_always())
    assert combined(make_state(outcome=Attempt(exception=ValueError()))) is True


def test_retry_all_zero_numeric_does_not_short_circuit():
    # 0.0 is an int/float but not bool, so loop continues and final returns True
    c = retry_if_exception(lambda exc: 0.0)
    combined = retry_all(c)
    assert combined(make_state(outcome=Attempt(exception=ValueError()))) is True


def test_retry_all_short_circuits_on_bool_false():
    calls = []

    class Tracking(RetryCondition):
        def __init__(self, result):
            self.result = result

        def __call__(self, state):
            calls.append(self.result)
            return self.result

    combined = retry_all(Tracking(False), Tracking(True))
    assert combined(make_state()) is False
    assert calls == [False]


# --------------------------------------------------------------------------
# retry_any
# --------------------------------------------------------------------------

def test_retry_any_stores_conditions():
    c1 = retry_always()
    c2 = retry_never()
    combined = retry_any(c1, c2)
    assert combined.conditions == (c1, c2)


def test_retry_any_empty_returns_false():
    assert retry_any()(make_state()) is False


def test_retry_any_one_true():
    combined = retry_any(retry_never(), retry_always())
    assert combined(make_state()) is True


def test_retry_any_all_false():
    combined = retry_any(retry_never(), retry_never())
    assert combined(make_state()) is False


def test_retry_any_numeric_result_returned_as_float():
    c = retry_if_exception(lambda exc: 5)
    combined = retry_any(retry_never(), c)
    result = combined(make_state(outcome=Attempt(exception=ValueError())))
    assert result == 5.0
    assert isinstance(result, float)


def test_retry_any_bool_true_short_circuits_before_numeric():
    calls = []

    class Tracking(RetryCondition):
        def __init__(self, result):
            self.result = result

        def __call__(self, state):
            calls.append(self.result)
            return self.result

    combined = retry_any(Tracking(True), Tracking(5.0))
    assert combined(make_state()) is True
    assert calls == [True]


def test_retry_any_numeric_short_circuits():
    calls = []

    class Tracking(RetryCondition):
        def __init__(self, result):
            self.result = result

        def __call__(self, state):
            calls.append(self.result)
            return self.result

    combined = retry_any(Tracking(3.0), Tracking(True))
    result = combined(make_state())
    assert result == 3.0
    assert calls == [3.0]


# --------------------------------------------------------------------------
# retry_always / retry_never
# --------------------------------------------------------------------------

def test_retry_always_true():
    assert retry_always()(make_state()) is True


def test_retry_never_false():
    assert retry_never()(make_state()) is False


# --------------------------------------------------------------------------
# retry_if_not_exception_type
# --------------------------------------------------------------------------

def test_retry_if_not_exception_type_single_stored_as_tuple():
    c = retry_if_not_exception_type(ValueError)
    assert c.exc_types == (ValueError,)


def test_retry_if_not_exception_type_none_outcome_false():
    c = retry_if_not_exception_type(ValueError)
    assert c(make_state(outcome=None)) is False


def test_retry_if_not_exception_type_no_exception_false():
    c = retry_if_not_exception_type(ValueError)
    assert c(make_state(outcome=Attempt(value=1))) is False


def test_retry_if_not_exception_type_non_matching_true():
    c = retry_if_not_exception_type(ValueError)
    assert c(make_state(outcome=Attempt(exception=TypeError()))) is True


def test_retry_if_not_exception_type_matching_false():
    c = retry_if_not_exception_type(ValueError)
    assert c(make_state(outcome=Attempt(exception=ValueError()))) is False


def test_retry_if_not_exception_type_subclass_matching_false():
    class MyError(ValueError):
        pass

    c = retry_if_not_exception_type(ValueError)
    assert c(make_state(outcome=Attempt(exception=MyError()))) is False


# --------------------------------------------------------------------------
# retry_unless_exception_type
# --------------------------------------------------------------------------

def test_retry_unless_exception_type_is_subclass():
    c = retry_unless_exception_type(ValueError)
    assert isinstance(c, retry_if_not_exception_type)


def test_retry_unless_exception_type_behavior_non_matching_true():
    c = retry_unless_exception_type(ValueError)
    assert c(make_state(outcome=Attempt(exception=TypeError()))) is True


def test_retry_unless_exception_type_behavior_matching_false():
    c = retry_unless_exception_type(ValueError)
    assert c(make_state(outcome=Attempt(exception=ValueError()))) is False


def test_retry_unless_exception_type_none_outcome_false():
    c = retry_unless_exception_type(ValueError)
    assert c(make_state(outcome=None)) is False


# --------------------------------------------------------------------------
# retry_if_not_exception_message
# --------------------------------------------------------------------------

def test_retry_if_not_exception_message_stores_attrs():
    c = retry_if_not_exception_message("boom")
    assert c.match == "boom"
    assert c.regex is False


def test_retry_if_not_exception_message_stores_regex():
    c = retry_if_not_exception_message("boom", regex=True)
    assert c.regex is True


def test_retry_if_not_exception_message_none_outcome_false():
    c = retry_if_not_exception_message("boom")
    assert c(make_state(outcome=None)) is False


def test_retry_if_not_exception_message_no_exception_false():
    c = retry_if_not_exception_message("boom")
    assert c(make_state(outcome=Attempt(value=1))) is False


def test_retry_if_not_exception_message_substring_not_present_true():
    c = retry_if_not_exception_message("boom")
    assert c(make_state(outcome=Attempt(exception=ValueError("quiet")))) is True


def test_retry_if_not_exception_message_substring_present_false():
    c = retry_if_not_exception_message("boom")
    assert c(make_state(outcome=Attempt(exception=ValueError("big boom")))) is False


def test_retry_if_not_exception_message_regex_not_matching_true():
    c = retry_if_not_exception_message(r"^boom$", regex=True)
    assert c(make_state(outcome=Attempt(exception=ValueError("boom!")))) is True


def test_retry_if_not_exception_message_regex_matching_false():
    c = retry_if_not_exception_message(r"boom", regex=True)
    assert c(make_state(outcome=Attempt(exception=ValueError("big boom")))) is False


# --------------------------------------------------------------------------
# retry_if_exception_cause_type
# --------------------------------------------------------------------------

def test_retry_if_exception_cause_type_single_stored_as_tuple():
    c = retry_if_exception_cause_type(ValueError)
    assert c.exc_types == (ValueError,)


def test_retry_if_exception_cause_type_sequence_stored_as_tuple():
    c = retry_if_exception_cause_type([ValueError, TypeError])
    assert c.exc_types == (ValueError, TypeError)


def test_retry_if_exception_cause_type_none_outcome_false():
    c = retry_if_exception_cause_type(ValueError)
    assert c(make_state(outcome=None)) is False


def test_retry_if_exception_cause_type_no_exception_false():
    c = retry_if_exception_cause_type(ValueError)
    assert c(make_state(outcome=Attempt(value=1))) is False


def test_retry_if_exception_cause_type_no_cause_false():
    c = retry_if_exception_cause_type(ValueError)
    assert c(make_state(outcome=Attempt(exception=RuntimeError("x")))) is False


def test_retry_if_exception_cause_type_direct_cause_match():
    inner = ValueError("inner")
    outer = RuntimeError("outer")
    outer.__cause__ = inner
    c = retry_if_exception_cause_type(ValueError)
    assert c(make_state(outcome=Attempt(exception=outer))) is True


def test_retry_if_exception_cause_type_direct_cause_non_match():
    inner = TypeError("inner")
    outer = RuntimeError("outer")
    outer.__cause__ = inner
    c = retry_if_exception_cause_type(ValueError)
    assert c(make_state(outcome=Attempt(exception=outer))) is False


def test_retry_if_exception_cause_type_uses_context_when_no_cause():
    inner = ValueError("inner")
    outer = RuntimeError("outer")
    outer.__context__ = inner
    c = retry_if_exception_cause_type(ValueError)
    assert c(make_state(outcome=Attempt(exception=outer))) is True


def test_retry_if_exception_cause_type_walks_chain():
    innermost = ValueError("deep")
    middle = RuntimeError("middle")
    middle.__cause__ = innermost
    outer = RuntimeError("outer")
    outer.__cause__ = middle
    c = retry_if_exception_cause_type(ValueError)
    assert c(make_state(outcome=Attempt(exception=outer))) is True


def test_retry_if_exception_cause_type_chain_mixed_cause_context():
    innermost = KeyError("deep")
    middle = RuntimeError("middle")
    middle.__context__ = innermost
    outer = RuntimeError("outer")
    outer.__cause__ = middle
    c = retry_if_exception_cause_type(KeyError)
    assert c(make_state(outcome=Attempt(exception=outer))) is True


def test_retry_if_exception_cause_type_multiple_types():
    inner = TypeError("inner")
    outer = RuntimeError("outer")
    outer.__cause__ = inner
    c = retry_if_exception_cause_type((ValueError, TypeError))
    assert c(make_state(outcome=Attempt(exception=outer))) is True


# --------------------------------------------------------------------------
# Composition integration
# --------------------------------------------------------------------------

def test_stop_composition_or_then_call():
    combined = stop_after_attempt(2) | stop_after_delay(5.0)
    assert combined(make_state(tries=1, elapsed=0.0)) is False
    assert combined(make_state(tries=2, elapsed=0.0)) is True
    assert combined(make_state(tries=0, elapsed=5.0)) is True


def test_stop_composition_and_then_call():
    combined = stop_after_attempt(2) & stop_after_delay(5.0)
    assert combined(make_state(tries=2, elapsed=5.0)) is True
    assert combined(make_state(tries=2, elapsed=0.0)) is False
    assert combined(make_state(tries=0, elapsed=5.0)) is False


def test_retry_composition_or_then_call():
    combined = retry_if_result(lambda v: v == 1) | retry_if_result(lambda v: v == 2)
    assert combined(make_state(outcome=Attempt(value=1))) is True
    assert combined(make_state(outcome=Attempt(value=2))) is True
    assert combined(make_state(outcome=Attempt(value=3))) is False


def test_retry_composition_and_then_call():
    combined = retry_if_result(lambda v: v > 0) & retry_if_result(lambda v: v < 10)
    assert combined(make_state(outcome=Attempt(value=5))) is True
    assert combined(make_state(outcome=Attempt(value=0))) is False
    assert combined(make_state(outcome=Attempt(value=20))) is False
