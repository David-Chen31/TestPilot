from __future__ import annotations

import threading
import types

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
# Helpers
# ---------------------------------------------------------------------------


class _Outcome:
    """A lightweight stand-in for Attempt carrying the fields used."""

    def __init__(self, exception=None, value=None, wait=0.0):
        self.exception = exception
        self.value = value
        self.wait = wait


def make_state(exception=None, value=None,
               wait=0.0, tries=0, elapsed=0.0, outcome=None):
    """Build a RetryState."""
    state = RetryState(tries=tries, elapsed=elapsed)
    if outcome is not None:
        state.outcome = outcome
    elif exception is not None or value is not None or wait != 0.0:
        state.outcome = _Outcome(exception=exception, value=value, wait=wait)
    state.outcome = outcome if outcome is not None else state.outcome
    return state


def state_no_outcome(tries=0, elapsed=0.0):
    return RetryState(tries=tries, elapsed=elapsed, outcome=None)


def state_exc(exc, tries=0, elapsed=0.0):
    st = RetryState(tries=tries, elapsed=elapsed)
    st.outcome = _Outcome(exception=exc)
    return st


def state_result(value, tries=0, elapsed=0.0):
    st = RetryState(tries=tries, elapsed=elapsed)
    st.outcome = _Outcome(exception=None, value=value)
    return st


def state_outcome(outcome, tries=0, elapsed=0.0):
    st = RetryState(tries=tries, elapsed=elapsed)
    st.outcome = outcome
    return st


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class TestBaseClasses:
    def test_stop_base_call_raises(self):
        with pytest.raises(NotImplementedError):
            Stop()(state_no_outcome())

    def test_retry_condition_base_call_raises(self):
        with pytest.raises(NotImplementedError):
            RetryCondition()(state_no_outcome())

    def test_stop_or_returns_stop_any(self):
        a = stop_never()
        b = stop_never()
        combined = a | b
        assert isinstance(combined, stop_any)
        assert combined.stops == (a, b)

    def test_stop_and_returns_stop_all(self):
        a = stop_never()
        b = stop_never()
        combined = a & b
        assert isinstance(combined, stop_all)
        assert combined.stops == (a, b)

    def test_retry_condition_or_returns_retry_any(self):
        a = retry_never()
        b = retry_never()
        combined = a | b
        assert isinstance(combined, retry_any)
        assert combined.conditions == (a, b)

    def test_retry_condition_and_returns_retry_all(self):
        a = retry_never()
        b = retry_never()
        combined = a & b
        assert isinstance(combined, retry_all)
        assert combined.conditions == (a, b)


# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------


class TestStopAfterAttempt:
    def test_below_threshold(self):
        assert stop_after_attempt(3)(state_no_outcome(tries=2)) is False

    def test_equal_threshold(self):
        assert stop_after_attempt(3)(state_no_outcome(tries=3)) is True

    def test_above_threshold(self):
        assert stop_after_attempt(3)(state_no_outcome(tries=10)) is True

    def test_stores_max_attempts(self):
        s = stop_after_attempt(5)
        assert s.max_attempts == 5

    def test_zero_attempts(self):
        assert stop_after_attempt(0)(state_no_outcome(tries=0)) is True


class TestStopAfterDelay:
    def test_below(self):
        assert stop_after_delay(10.0)(state_no_outcome(elapsed=9.99)) is False

    def test_equal(self):
        assert stop_after_delay(10.0)(state_no_outcome(elapsed=10.0)) is True

    def test_above(self):
        assert stop_after_delay(10.0)(state_no_outcome(elapsed=100.0)) is True

    def test_stores_max_delay(self):
        assert stop_after_delay(2.5).max_delay == 2.5


class TestStopBeforeDelay:
    def test_without_outcome_uses_zero_wait(self):
        assert stop_before_delay(10.0)(state_no_outcome(elapsed=9.0)) is False
        assert stop_before_delay(10.0)(state_no_outcome(elapsed=10.0)) is True

    def test_with_wait_below(self):
        st = state_outcome(_Outcome(wait=2.0), elapsed=7.0)
        assert stop_before_delay(10.0)(st) is False

    def test_with_wait_equal(self):
        st = state_outcome(_Outcome(wait=2.0), elapsed=8.0)
        assert stop_before_delay(10.0)(st) is True

    def test_with_wait_above(self):
        st = state_outcome(_Outcome(wait=5.0), elapsed=8.0)
        assert stop_before_delay(10.0)(st) is True

    def test_outcome_without_wait_attr(self):
        # outcome exists but has no `wait` attribute -> getattr default 0.0
        outcome = types.SimpleNamespace(exception=None, value=None)
        st = state_outcome(outcome, elapsed=10.0)
        assert stop_before_delay(10.0)(st) is True

    def test_outcome_without_wait_attr_below(self):
        # getattr default must be 0.0, not 1.0: elapsed+0.0 < max_delay
        outcome = types.SimpleNamespace(exception=None, value=None)
        st = state_outcome(outcome, elapsed=9.5)
        assert stop_before_delay(10.0)(st) is False

    def test_outcome_without_wait_attr_boundary(self):
        # with default wait 0.0, elapsed exactly max_delay -> True
        outcome = types.SimpleNamespace(exception=None, value=None)
        assert stop_before_delay(10.0)(state_outcome(outcome, elapsed=10.0)) is True
        # if default were 1.0, elapsed 9.0 would become 10.0 -> True (but real is 9.0 -> False)
        assert stop_before_delay(10.0)(state_outcome(outcome, elapsed=9.0)) is False

    def test_stores_max_delay(self):
        assert stop_before_delay(3.0).max_delay == 3.0

    def test_no_outcome_below_boundary_strict(self):
        # No outcome -> wait must be 0.0. If the conditional were broken to
        # always access state.outcome, this would raise AttributeError.
        st = state_no_outcome(elapsed=1.0)
        assert st.outcome is None
        assert stop_before_delay(5.0)(st) is False

    def test_no_outcome_does_not_access_attr(self):
        # A state whose outcome is None must not trigger any attribute access
        # on outcome. If the ternary check were removed/always-true, accessing
        # state.outcome.wait would raise AttributeError here.
        st = RetryState(elapsed=0.0, outcome=None)
        assert st.outcome is None
        assert stop_before_delay(1.0)(st) is False

    def test_no_outcome_exact_boundary(self):
        # With no outcome, wait is 0.0, so elapsed == max_delay -> True.
        st = RetryState(elapsed=4.0, outcome=None)
        assert st.outcome is None
        assert stop_before_delay(4.0)(st) is True

    def test_no_outcome_just_below(self):
        st = RetryState(elapsed=3.999, outcome=None)
        assert st.outcome is None
        assert stop_before_delay(4.0)(st) is False

    def test_outcome_present_no_outcome_is_used_for_wait(self):
        # When an outcome is present but wait == 0.0, the result depends only
        # on elapsed. This distinguishes the two branches of the ternary:
        # the "outcome present" branch reads `wait` (0.0 here), the fallback
        # also yields 0.0, so results coincide. Use an outcome whose `wait`
        # attribute equals elapsed to force the "outcome present" branch.
        # If `state.outcome` were None, the fallback would be used instead.
        st_no = state_no_outcome(elapsed=5.0)
        st_yes = state_outcome(_Outcome(wait=0.0), elapsed=5.0)
        # Both give wait 0.0 -> same result.
        assert stop_before_delay(5.0)(st_no) is True
        assert stop_before_delay(5.0)(st_yes) is True
        assert stop_before_delay(5.5)(st_no) is False
        assert stop_before_delay(5.5)(st_yes) is False

    def test_outcome_none_vs_present_with_nonzero_wait_differs(self):
        # A state with outcome=None must use wait 0.0; a state with outcome
        # whose wait is large must include that wait. This pins down the
        # `if state.outcome else 0.0` short-circuit: with outcome None the
        # max(...) must NOT include any wait, so elapsed < max_delay -> False.
        st_no = RetryState(elapsed=1.0, outcome=None)
        st_yes = state_outcome(_Outcome(wait=100.0), elapsed=1.0)
        assert stop_before_delay(2.0)(st_no) is False
        assert stop_before_delay(2.0)(st_yes) is True

    def test_outcome_none_boundary_does_not_add_phantom_wait(self):
        # If the ternary wrongly always read a wait (or default were nonzero),
        # elapsed=1.0 with max_delay=1.5 could stop. Real code: wait 0.0,
        # elapsed 1.0 < 1.5 -> False.
        st = RetryState(elapsed=1.0, outcome=None)
        assert st.outcome is None
        assert stop_before_delay(1.5)(st) is False
        # And exactly at boundary it stops.
        assert stop_before_delay(1.0)(st) is True

    def test_outcome_falsy_falls_back_to_zero_wait(self):
        # The ternary tests truthiness of state.outcome. A falsy outcome
        # object with a nonzero wait must be ignored (fallback 0.0).
        class _Falsy:
            wait = 100.0

            def __bool__(self):
                return False

        st = state_outcome(_Falsy(), elapsed=1.0)
        # outcome is falsy -> wait = 0.0 -> 1.0 < 2.0 -> False
        assert stop_before_delay(2.0)(st) is False

    def test_outcome_truthy_uses_wait(self):
        st = state_outcome(_Outcome(wait=100.0), elapsed=1.0)
        assert stop_before_delay(2.0)(st) is True

    def test_outcome_none_uses_zero_wait_not_attribute_error(self):
        # An outcome=None state with a small elapsed must not raise even if
        # the code would try to inspect outcome.wait.
        st = RetryState(elapsed=0.5, outcome=None)
        assert stop_before_delay(1.0)(st) is False


class TestStopAll:
    def test_all_true(self):
        stop = stop_all(stop_after_attempt(1), stop_after_delay(1.0))
        st = state_no_outcome(tries=1, elapsed=1.0)
        assert stop(st) is True

    def test_one_false(self):
        stop = stop_all(stop_after_attempt(1), stop_after_delay(5.0))
        st = state_no_outcome(tries=1, elapsed=1.0)
        assert stop(st) is False

    def test_none_true(self):
        assert stop_all()(state_no_outcome()) is True

    def test_short_circuits(self):
        calls = []

        def make(name, result):
            class _S(Stop):
                def __call__(self, state):
                    calls.append(name)
                    return result
            return _S()

        stop = stop_all(make("a", False), make("b", True))
        assert stop(state_no_outcome()) is False
        assert calls == ["a"]

    def test_stores_stops(self):
        a = stop_never()
        b = stop_never()
        s = stop_all(a, b)
        assert s.stops == (a, b)


class TestStopAny:
    def test_any_true(self):
        stop = stop_any(stop_after_attempt(1), stop_after_delay(5.0))
        st = state_no_outcome(tries=1, elapsed=1.0)
        assert stop(st) is True

    def test_all_false(self):
        stop = stop_any(stop_after_attempt(1), stop_after_delay(5.0))
        st = state_no_outcome(tries=0, elapsed=1.0)
        assert stop(st) is False

    def test_none_false(self):
        assert stop_any()(state_no_outcome()) is False

    def test_short_circuits(self):
        calls = []

        def make(name, result):
            class _S(Stop):
                def __call__(self, state):
                    calls.append(name)
                    return result
            return _S()

        stop = stop_any(make("a", True), make("b", False))
        assert stop(state_no_outcome()) is True
        assert calls == ["a"]


class TestStopNever:
    def test_always_false(self):
        assert stop_never()(state_no_outcome()) is False
        assert stop_never()(state_exc(ValueError())) is False


class TestStopWhenEventSet:
    def test_not_set(self):
        event = threading.Event()
        assert stop_when_event_set(event)(state_no_outcome()) is False

    def test_set(self):
        event = threading.Event()
        event.set()
        assert stop_when_event_set(event)(state_no_outcome()) is True

    def test_stores_event(self):
        event = threading.Event()
        s = stop_when_event_set(event)
        assert s.event is event


# ---------------------------------------------------------------------------
# retry_if_exception_type
# ---------------------------------------------------------------------------


class TestRetryIfExceptionType:
    def test_single_type_match(self):
        c = retry_if_exception_type(ValueError)
        assert c(state_exc(ValueError("x"))) is True

    def test_single_type_no_match(self):
        c = retry_if_exception_type(ValueError)
        assert c(state_exc(TypeError("x"))) is False

    def test_sequence_match(self):
        c = retry_if_exception_type((ValueError, TypeError))
        assert c(state_exc(KeyError("x"))) is False
        assert c(state_exc(TypeError("x"))) is True

    def test_subclass_match(self):
        class MyError(ValueError):
            pass

        c = retry_if_exception_type(ValueError)
        assert c(state_exc(MyError())) is True

    def test_no_outcome(self):
        c = retry_if_exception_type(ValueError)
        assert c(state_no_outcome()) is False

    def test_outcome_no_exception(self):
        c = retry_if_exception_type(ValueError)
        assert c(state_result(42)) is False

    def test_normalizes_to_tuple(self):
        c = retry_if_exception_type(ValueError)
        assert c.exc_types == (ValueError,)

    def test_sequence_normalizes(self):
        c = retry_if_exception_type([ValueError, TypeError])
        assert c.exc_types == (ValueError, TypeError)


# ---------------------------------------------------------------------------
# retry_if_exception
# ---------------------------------------------------------------------------


class TestRetryIfException:
    def test_predicate_true(self):
        c = retry_if_exception(lambda e: isinstance(e, ValueError))
        assert c(state_exc(ValueError())) is True

    def test_predicate_false(self):
        c = retry_if_exception(lambda e: isinstance(e, ValueError))
        assert c(state_exc(TypeError())) is False

    def test_no_outcome(self):
        c = retry_if_exception(lambda e: True)
        assert c(state_no_outcome()) is False

    def test_outcome_no_exception(self):
        c = retry_if_exception(lambda e: True)
        assert c(state_result(1)) is False

    def test_float_result_passthrough(self):
        c = retry_if_exception(lambda e: 2)
        result = c(state_exc(ValueError()))
        assert result == 2.0
        assert isinstance(result, float)

    def test_float_input_passthrough(self):
        c = retry_if_exception(lambda e: 1.5)
        assert c(state_exc(ValueError())) == 1.5

    def test_bool_true_not_converted(self):
        c = retry_if_exception(lambda e: True)
        result = c(state_exc(ValueError()))
        assert result is True

    def test_bool_false_not_converted(self):
        c = retry_if_exception(lambda e: False)
        assert c(state_exc(ValueError())) is False

    def test_truthy_non_number_boolified(self):
        c = retry_if_exception(lambda e: "yes")
        assert c(state_exc(ValueError())) is True

    def test_predicate_receives_exception(self):
        received = []

        def predicate(exc):
            received.append(exc)
            return True

        err = ValueError("boom")
        retry_if_exception(predicate)(state_exc(err))
        assert received == [err]


# ---------------------------------------------------------------------------
# retry_if_exception_message
# ---------------------------------------------------------------------------


class TestRetryIfExceptionMessage:
    def test_substring_match(self):
        c = retry_if_exception_message("boom")
        assert c(state_exc(ValueError("big boom here"))) is True

    def test_substring_no_match(self):
        c = retry_if_exception_message("boom")
        assert c(state_exc(ValueError("nothing"))) is False

    def test_exact_match_substring(self):
        c = retry_if_exception_message("boom")
        assert c(state_exc(ValueError("boom"))) is True

    def test_regex_match(self):
        c = retry_if_exception_message(r"bo+m", match="re")
        assert c(state_exc(ValueError("boooom"))) is True

    def test_regex_no_match(self):
        c = retry_if_exception_message(r"^boom$", match="re")
        assert c(state_exc(ValueError("a boom b"))) is False

    def test_no_outcome(self):
        c = retry_if_exception_message("boom")
        assert c(state_no_outcome()) is False

    def test_outcome_no_exception(self):
        c = retry_if_exception_message("boom")
        assert c(state_result("boom")) is False

    def test_stores_attrs(self):
        c = retry_if_exception_message("boom", match="re")
        assert c.message == "boom"
        assert c.match == "re"


# ---------------------------------------------------------------------------
# retry_if_result / retry_if_not_result
# ---------------------------------------------------------------------------


class TestRetryIfResult:
    def test_predicate_true(self):
        c = retry_if_result(lambda v: v == 1)
        assert c(state_result(1)) is True

    def test_predicate_false(self):
        c = retry_if_result(lambda v: v == 1)
        assert c(state_result(2)) is False

    def test_no_outcome(self):
        c = retry_if_result(lambda v: True)
        assert c(state_no_outcome()) is False

    def test_outcome_with_exception(self):
        c = retry_if_result(lambda v: True)
        assert c(state_exc(ValueError())) is False

    def test_predicate_receives_value(self):
        received = []
        retry_if_result(lambda v: received.append(v) or True)(state_result("abc"))
        assert received == ["abc"]


class TestRetryIfNotResult:
    def test_predicate_true_returns_false(self):
        c = retry_if_not_result(lambda v: v == 1)
        assert c(state_result(1)) is False

    def test_predicate_false_returns_true(self):
        c = retry_if_not_result(lambda v: v == 1)
        assert c(state_result(2)) is True

    def test_no_outcome(self):
        c = retry_if_not_result(lambda v: True)
        assert c(state_no_outcome()) is False

    def test_outcome_with_exception(self):
        c = retry_if_not_result(lambda v: False)
        assert c(state_exc(ValueError())) is False


# ---------------------------------------------------------------------------
# retry_all / retry_any
# ---------------------------------------------------------------------------


def _cond(result):
    class _C(RetryCondition):
        def __call__(self, state):
            return result
    return _C()


class TestRetryAll:
    def test_all_true(self):
        c = retry_all(_cond(True), _cond(True))
        assert c(state_no_outcome()) is True

    def test_one_false(self):
        c = retry_all(_cond(True), _cond(False))
        assert c(state_no_outcome()) is False

    def test_none_true(self):
        assert retry_all()(state_no_outcome()) is True

    def test_short_circuits_on_false(self):
        calls = []

        def make(name, result):
            class _C(RetryCondition):
                def __call__(self, state):
                    calls.append(name)
                    return result
            return _C()

        c = retry_all(make("a", False), make("b", True))
        assert c(state_no_outcome()) is False
        assert calls == ["a"]

    def test_float_values_are_not_false(self):
        # numeric (non-bool) results are not treated as False by retry_all
        c = retry_all(_cond(0.0), _cond(True))
        assert c(state_no_outcome()) is True

    def test_passes_state_to_conditions(self):
        received = []

        class _C(RetryCondition):
            def __call__(self, state):
                received.append(state)
                return True

        st = state_no_outcome()
        c = retry_all(_C())
        assert c(st) is True
        assert received == [st]
        assert received[0] is st

    def test_stores_conditions(self):
        a, b = retry_never(), retry_never()
        c = retry_all(a, b)
        assert c.conditions == (a, b)


class TestRetryAny:
    def test_any_true(self):
        c = retry_any(_cond(False), _cond(True))
        assert c(state_no_outcome()) is True

    def test_all_false(self):
        c = retry_any(_cond(False), _cond(False))
        assert c(state_no_outcome()) is False

    def test_none_false(self):
        assert retry_any()(state_no_outcome()) is False

    def test_returns_float_when_numeric(self):
        c = retry_any(_cond(2), _cond(True))
        result = c(state_no_outcome())
        assert result == 2.0
        assert isinstance(result, float)

    def test_short_circuits_on_true(self):
        calls = []

        def make(name, result):
            class _C(RetryCondition):
                def __call__(self, state):
                    calls.append(name)
                    return result
            return _C()

        c = retry_any(make("a", True), make("b", False))
        assert c(state_no_outcome()) is True
        assert calls == ["a"]

    def test_passes_state_to_conditions(self):
        received = []

        class _C(RetryCondition):
            def __call__(self, state):
                received.append(state)
                return True

        st = state_no_outcome()
        c = retry_any(_C())
        assert c(st) is True
        assert received == [st]
        assert received[0] is st

    def test_stores_conditions(self):
        a, b = retry_never(), retry_never()
        c = retry_any(a, b)
        assert c.conditions == (a, b)


# ---------------------------------------------------------------------------
# retry_always / retry_never
# ---------------------------------------------------------------------------


class TestRetryAlwaysNever:
    def test_always_true(self):
        assert retry_always()(state_no_outcome()) is True

    def test_never_false(self):
        assert retry_never()(state_no_outcome()) is False
        assert retry_never()(state_exc(ValueError())) is False


# ---------------------------------------------------------------------------
# retry_if_not_exception_type / retry_unless_exception_type
# ---------------------------------------------------------------------------


class TestRetryIfNotExceptionType:
    def test_not_matching_type(self):
        c = retry_if_not_exception_type(ValueError)
        assert c(state_exc(TypeError())) is True

    def test_matching_type(self):
        c = retry_if_not_exception_type(ValueError)
        assert c(state_exc(ValueError())) is False

    def test_subclass_match(self):
        class MyError(ValueError):
            pass

        c = retry_if_not_exception_type(ValueError)
        assert c(state_exc(MyError())) is False

    def test_sequence(self):
        c = retry_if_not_exception_type((ValueError, TypeError))
        assert c(state_exc(KeyError())) is True
        assert c(state_exc(TypeError())) is False

    def test_no_outcome(self):
        c = retry_if_not_exception_type(ValueError)
        assert c(state_no_outcome()) is False

    def test_outcome_no_exception(self):
        c = retry_if_not_exception_type(ValueError)
        assert c(state_result(1)) is False

    def test_normalizes_to_tuple(self):
        assert retry_if_not_exception_type(ValueError).exc_types == (ValueError,)


class TestRetryUnlessExceptionType:
    def test_is_subclass(self):
        assert issubclass(retry_unless_exception_type, retry_if_not_exception_type)

    def test_behavior(self):
        c = retry_unless_exception_type(ValueError)
        assert c(state_exc(TypeError())) is True
        assert c(state_exc(ValueError())) is False


# ---------------------------------------------------------------------------
# retry_if_not_exception_message
# ---------------------------------------------------------------------------


class TestRetryIfNotExceptionMessage:
    def test_substring_not_present(self):
        c = retry_if_not_exception_message("boom")
        assert c(state_exc(ValueError("nothing"))) is True

    def test_substring_present(self):
        c = retry_if_not_exception_message("boom")
        assert c(state_exc(ValueError("big boom"))) is False

    def test_regex_not_present(self):
        c = retry_if_not_exception_message(r"bo+m", regex=True)
        assert c(state_exc(ValueError("nothing"))) is True

    def test_regex_present(self):
        c = retry_if_not_exception_message(r"bo+m", regex=True)
        assert c(state_exc(ValueError("boooom"))) is False

    def test_default_regex_is_false(self):
        # default regex=False -> literal substring, not regex
        c = retry_if_not_exception_message(r"bo+m")
        assert c.regex is False
        # "bo+m" literally not present in "boooom" -> True
        assert c(state_exc(ValueError("boooom"))) is True
        # literal present -> False
        assert c(state_exc(ValueError("x bo+m y"))) is False

    def test_default_matches_literal_dot(self):
        # If regex defaulted True, "." would match any char. With literal,
        # "a.c" only matches the literal substring.
        c = retry_if_not_exception_message("a.c")
        assert c(state_exc(ValueError("axc"))) is True
        assert c(state_exc(ValueError("a.c"))) is False

    def test_no_outcome(self):
        c = retry_if_not_exception_message("boom")
        assert c(state_no_outcome()) is False

    def test_outcome_no_exception(self):
        c = retry_if_not_exception_message("boom")
        assert c(state_result("boom")) is False

    def test_stores_attrs(self):
        c = retry_if_not_exception_message("x", regex=True)
        assert c.match == "x"
        assert c.regex is True


# ---------------------------------------------------------------------------
# retry_if_exception_cause_type
# ---------------------------------------------------------------------------


class TestRetryIfExceptionCauseType:
    def test_direct_cause_via_raise_from(self):
        try:
            try:
                raise ValueError("inner")
            except ValueError as inner:
                raise RuntimeError("outer") from inner
        except RuntimeError as exc:
            outer = exc
        c = retry_if_exception_cause_type(ValueError)
        assert c(state_exc(outer)) is True

    def test_direct_cause_no_match(self):
        try:
            try:
                raise ValueError("inner")
            except ValueError as inner:
                raise RuntimeError("outer") from inner
        except RuntimeError as exc:
            outer = exc
        c = retry_if_exception_cause_type(TypeError)
        assert c(state_exc(outer)) is False

    def test_context_chain(self):
        # __context__ set implicitly when raising inside except block
        try:
            try:
                raise ValueError("inner")
            except ValueError:
                raise RuntimeError("outer")
        except RuntimeError as exc:
            outer = exc
        c = retry_if_exception_cause_type(ValueError)
        assert c(state_exc(outer)) is True

    def test_deep_chain(self):
        try:
            try:
                try:
                    raise KeyError("level0")
                except KeyError as e0:
                    raise ValueError("level1") from e0
            except ValueError as e1:
                raise TypeError("level2") from e1
        except TypeError as e2:
            top = e2
        c = retry_if_exception_cause_type(KeyError)
        assert c(state_exc(top)) is True

    def test_deep_chain_via_context(self):
        # Build a chain purely through __context__ (raise inside except).
        try:
            try:
                try:
                    raise KeyError("level0")
                except KeyError:
                    raise ValueError("level1")
            except ValueError:
                raise TypeError("level2")
        except TypeError as e2:
            top = e2
        c = retry_if_exception_cause_type(KeyError)
        assert c(state_exc(top)) is True

    def test_cause_walk_uses_context_when_no_cause(self):
        # Each link only has __context__ (no __cause__). The loop must
        # fall back to __context__ at each step of the chain.
        outer = RuntimeError("outer")
        middle = ValueError("middle")
        inner = KeyError("inner")
        outer.__context__ = middle
        middle.__context__ = inner
        # Ensure no __cause__ anywhere
        assert outer.__cause__ is None
        assert middle.__cause__ is None
        assert inner.__cause__ is None
        c = retry_if_exception_cause_type(KeyError)
        assert c(state_exc(outer)) is True

    def test_no_cause(self):
        c = retry_if_exception_cause_type(ValueError)
        assert c(state_exc(ValueError("standalone"))) is False

    def test_no_outcome(self):
        c = retry_if_exception_cause_type(ValueError)
        assert c(state_no_outcome()) is False

    def test_outcome_no_exception(self):
        c = retry_if_exception_cause_type(ValueError)
        assert c(state_result(1)) is False

    def test_sequence_types(self):
        try:
            try:
                raise KeyError("inner")
            except KeyError as inner:
                raise RuntimeError("outer") from inner
        except RuntimeError as exc:
            outer = exc
        c = retry_if_exception_cause_type((ValueError, KeyError))
        assert c(state_exc(outer)) is True

    def test_normalizes_to_tuple(self):
        assert retry_if_exception_cause_type(ValueError).exc_types == (ValueError,)
