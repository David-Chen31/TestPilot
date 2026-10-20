from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeAttempt:
    value: Any = None
    exception: BaseException | None = None
    wait: float = 0.0


@dataclass
class FakeState:
    tries: int = 0
    elapsed: float = 0.0
    outcome: FakeAttempt | None = None

    def set_outcome(self, **kwargs: Any) -> "FakeState":
        self.outcome = FakeAttempt(**kwargs)
        return self


def make_state(**kwargs: Any) -> FakeState:
    return FakeState(**kwargs)


class CustomStop(Stop):
    def __init__(self, result: bool) -> None:
        self.result = result
        self.calls = 0

    def __call__(self, state: Any) -> bool:
        self.calls += 1
        return self.result


class CustomRetry(RetryCondition):
    def __init__(self, result: bool | float) -> None:
        self.result = result
        self.calls = 0

    def __call__(self, state: Any) -> bool | float:
        self.calls += 1
        return self.result


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class TestStopBase:
    def test_call_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            Stop()(make_state())

    def test_or_returns_stop_any(self) -> None:
        a = CustomStop(True)
        b = CustomStop(False)
        combined = a | b
        assert isinstance(combined, stop_any)
        assert combined.stops == (a, b)

    def test_and_returns_stop_all(self) -> None:
        a = CustomStop(True)
        b = CustomStop(False)
        combined = a & b
        assert isinstance(combined, stop_all)
        assert combined.stops == (a, b)


class TestRetryConditionBase:
    def test_call_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            RetryCondition()(make_state())

    def test_or_returns_retry_any(self) -> None:
        a = CustomRetry(True)
        b = CustomRetry(False)
        combined = a | b
        assert isinstance(combined, retry_any)
        assert combined.conditions == (a, b)

    def test_and_returns_retry_all(self) -> None:
        a = CustomRetry(True)
        b = CustomRetry(False)
        combined = a & b
        assert isinstance(combined, retry_all)
        assert combined.conditions == (a, b)


# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------


class TestStopAfterAttempt:
    def test_init_stores_max_attempts(self) -> None:
        s = stop_after_attempt(5)
        assert s.max_attempts == 5

    @pytest.mark.parametrize(
        "tries,expected",
        [
            (0, False),
            (4, False),
            (5, True),
            (6, True),
        ],
    )
    def test_call(self, tries: int, expected: bool) -> None:
        s = stop_after_attempt(5)
        assert s(make_state(tries=tries)) is expected

    def test_max_attempts_zero(self) -> None:
        s = stop_after_attempt(0)
        assert s(make_state(tries=0)) is True


class TestStopAfterDelay:
    def test_init_stores_max_delay(self) -> None:
        s = stop_after_delay(1.5)
        assert s.max_delay == 1.5

    @pytest.mark.parametrize(
        "elapsed,expected",
        [
            (0.0, False),
            (0.99, False),
            (1.0, True),
            (2.0, True),
        ],
    )
    def test_call(self, elapsed: float, expected: bool) -> None:
        s = stop_after_delay(1.0)
        assert s(make_state(elapsed=elapsed)) is expected


class TestStopBeforeDelay:
    def test_init_stores_max_delay(self) -> None:
        s = stop_before_delay(2.0)
        assert s.max_delay == 2.0

    def test_no_outcome_uses_zero_wait(self) -> None:
        s = stop_before_delay(1.0)
        assert s(make_state(elapsed=0.5, outcome=None)) is False
        assert s(make_state(elapsed=1.0, outcome=None)) is True

    def test_with_wait(self) -> None:
        s = stop_before_delay(1.0)
        state = make_state(elapsed=0.5).set_outcome(wait=0.4)
        assert s(state) is False
        state = make_state(elapsed=0.5).set_outcome(wait=0.5)
        assert s(state) is True

    def test_outcome_without_wait_attr(self) -> None:
        s = stop_before_delay(1.0)

        class NoWait:
            pass

        state = make_state(elapsed=0.9)
        state.outcome = NoWait()  # type: ignore[assignment]
        assert s(state) is False
        state = make_state(elapsed=1.0)
        state.outcome = NoWait()  # type: ignore[assignment]
        assert s(state) is True


class TestStopAll:
    def test_all_true(self) -> None:
        s = stop_all(stop_after_attempt(1), stop_after_delay(1.0))
        assert s(make_state(tries=1, elapsed=2.0)) is True

    def test_one_false(self) -> None:
        s = stop_all(stop_after_attempt(1), stop_after_delay(1.0))
        assert s(make_state(tries=1, elapsed=0.0)) is False

    def test_empty_is_true(self) -> None:
        s = stop_all()
        assert s(make_state()) is True

    def test_short_circuits(self) -> None:
        a = CustomStop(False)
        b = CustomStop(True)
        s = stop_all(a, b)
        assert s(make_state()) is False
        assert a.calls == 1
        assert b.calls == 0


class TestStopAny:
    def test_any_true(self) -> None:
        s = stop_any(stop_after_attempt(1), stop_after_delay(10.0))
        assert s(make_state(tries=1, elapsed=0.0)) is True

    def test_all_false(self) -> None:
        s = stop_any(stop_after_attempt(1), stop_after_delay(10.0))
        assert s(make_state(tries=0, elapsed=0.0)) is False

    def test_empty_is_false(self) -> None:
        s = stop_any()
        assert s(make_state()) is False

    def test_short_circuits(self) -> None:
        a = CustomStop(True)
        b = CustomStop(False)
        s = stop_any(a, b)
        assert s(make_state()) is True
        assert a.calls == 1
        assert b.calls == 0


class TestStopNever:
    def test_always_false(self) -> None:
        s = stop_never()
        assert s(make_state()) is False
        assert s(make_state(tries=100, elapsed=1000.0)) is False


class TestStopWhenEventSet:
    def test_event_not_set(self) -> None:
        event = threading.Event()
        s = stop_when_event_set(event)
        assert s.event is event
        assert s(make_state()) is False

    def test_event_set(self) -> None:
        event = threading.Event()
        event.set()
        s = stop_when_event_set(event)
        assert s(make_state()) is True


# ---------------------------------------------------------------------------
# retry_if_exception_type
# ---------------------------------------------------------------------------


class TestRetryIfExceptionType:
    def test_no_outcome(self) -> None:
        c = retry_if_exception_type(ValueError)
        assert c(make_state(outcome=None)) is False

    def test_single_type_matches(self) -> None:
        c = retry_if_exception_type(ValueError)
        state = make_state().set_outcome(exception=ValueError("x"))
        assert c(state) is True

    def test_single_type_does_not_match(self) -> None:
        c = retry_if_exception_type(ValueError)
        state = make_state().set_outcome(exception=TypeError("x"))
        assert c(state) is False

    def test_sequence_matches(self) -> None:
        c = retry_if_exception_type([ValueError, TypeError])
        assert c.exc_types == (ValueError, TypeError)
        assert c(make_state().set_outcome(exception=TypeError("x"))) is True
        assert c(make_state().set_outcome(exception=KeyError("x"))) is False

    def test_single_type_regex_excludes_regex_type(self) -> None:
        # str is a type but not an Exception subclass; isinstance still works.
        c = retry_if_exception_type(str)  # type: ignore[arg-type]
        assert c.exc_types == (str,)

    def test_outcome_no_exception(self) -> None:
        c = retry_if_exception_type(ValueError)
        state = make_state().set_outcome(value=1)
        assert c(state) is False

    def test_subclass_matches(self) -> None:
        class MyError(ValueError):
            pass

        c = retry_if_exception_type(ValueError)
        assert c(make_state().set_outcome(exception=MyError("x"))) is True


# ---------------------------------------------------------------------------
# retry_if_exception
# ---------------------------------------------------------------------------


class TestRetryIfException:
    def test_no_outcome(self) -> None:
        c = retry_if_exception(lambda exc: True)
        assert c(make_state(outcome=None)) is False

    def test_no_exception(self) -> None:
        c = retry_if_exception(lambda exc: True)
        assert c(make_state().set_outcome(value=1)) is False

    def test_predicate_true(self) -> None:
        c = retry_if_exception(lambda exc: isinstance(exc, ValueError))
        assert c(make_state().set_outcome(exception=ValueError("x"))) is True

    def test_predicate_false(self) -> None:
        c = retry_if_exception(lambda exc: isinstance(exc, TypeError))
        assert c(make_state().set_outcome(exception=ValueError("x"))) is False

    def test_predicate_returns_float(self) -> None:
        c = retry_if_exception(lambda exc: 2.5)
        result = c(make_state().set_outcome(exception=ValueError("x")))
        assert result == 2.5
        assert isinstance(result, float)

    def test_predicate_returns_int_becomes_float(self) -> None:
        c = retry_if_exception(lambda exc: 3)
        result = c(make_state().set_outcome(exception=ValueError("x")))
        assert result == 3.0
        assert isinstance(result, float)

    def test_predicate_returns_bool_not_float(self) -> None:
        c = retry_if_exception(lambda exc: True)
        result = c(make_state().set_outcome(exception=ValueError("x")))
        assert result is True
        assert isinstance(result, bool)

    def test_predicate_returns_zero(self) -> None:
        c = retry_if_exception(lambda exc: 0)
        result = c(make_state().set_outcome(exception=ValueError("x")))
        assert result == 0.0
        assert isinstance(result, float)

    def test_passes_exception(self) -> None:
        seen = []
        exc = ValueError("boom")
        c = retry_if_exception(lambda e: seen.append(e) or True)
        c(make_state().set_outcome(exception=exc))
        assert seen == [exc]


# ---------------------------------------------------------------------------
# retry_if_exception_message
# ---------------------------------------------------------------------------


class TestRetryIfExceptionMessage:
    def test_no_outcome(self) -> None:
        c = retry_if_exception_message("x")
        assert c(make_state(outcome=None)) is False

    def test_no_exception(self) -> None:
        c = retry_if_exception_message("x")
        assert c(make_state().set_outcome(value=1)) is False

    def test_substring_match(self) -> None:
        c = retry_if_exception_message("boom")
        assert c(make_state().set_outcome(exception=ValueError("a boom b"))) is True

    def test_substring_no_match(self) -> None:
        c = retry_if_exception_message("boom")
        assert c(make_state().set_outcome(exception=ValueError("nope"))) is False

    def test_init_stores_values(self) -> None:
        c = retry_if_exception_message("msg", match="re")
        assert c.message == "msg"
        assert c.match == "re"

    def test_regex_match(self) -> None:
        c = retry_if_exception_message(r"bo+m", match="re")
        assert c(make_state().set_outcome(exception=ValueError("a boom b"))) is True

    def test_regex_no_match(self) -> None:
        c = retry_if_exception_message(r"bo+m", match="re")
        assert c(make_state().set_outcome(exception=ValueError("bam"))) is False

    def test_regex_is_search_not_fullmatch(self) -> None:
        c = retry_if_exception_message(r"\d+", match="re")
        assert c(make_state().set_outcome(exception=ValueError("abc123xyz"))) is True

    def test_invalid_regex_raises(self) -> None:
        c = retry_if_exception_message("[", match="re")
        with pytest.raises(re.error):
            c(make_state().set_outcome(exception=ValueError("x")))


# ---------------------------------------------------------------------------
# retry_if_result / retry_if_not_result
# ---------------------------------------------------------------------------


class TestRetryIfResult:
    def test_no_outcome(self) -> None:
        c = retry_if_result(lambda v: True)
        assert c(make_state(outcome=None)) is False

    def test_has_exception(self) -> None:
        c = retry_if_result(lambda v: True)
        assert c(make_state().set_outcome(exception=ValueError("x"))) is False

    def test_predicate_true(self) -> None:
        c = retry_if_result(lambda v: v == 42)
        assert c(make_state().set_outcome(value=42)) is True

    def test_predicate_false(self) -> None:
        c = retry_if_result(lambda v: v == 42)
        assert c(make_state().set_outcome(value=0)) is False

    def test_passes_value(self) -> None:
        seen = []
        c = retry_if_result(lambda v: seen.append(v) or False)
        c(make_state().set_outcome(value="hello"))
        assert seen == ["hello"]


class TestRetryIfNotResult:
    def test_no_outcome(self) -> None:
        c = retry_if_not_result(lambda v: True)
        assert c(make_state(outcome=None)) is False

    def test_has_exception(self) -> None:
        c = retry_if_not_result(lambda v: True)
        assert c(make_state().set_outcome(exception=ValueError("x"))) is False

    def test_predicate_false_gives_true(self) -> None:
        c = retry_if_not_result(lambda v: v == 42)
        assert c(make_state().set_outcome(value=0)) is True

    def test_predicate_true_gives_false(self) -> None:
        c = retry_if_not_result(lambda v: v == 42)
        assert c(make_state().set_outcome(value=42)) is False


# ---------------------------------------------------------------------------
# retry_all / retry_any
# ---------------------------------------------------------------------------


class TestRetryAll:
    def test_all_true(self) -> None:
        c = retry_all(retry_always(), retry_always())
        assert c(make_state()) is True

    def test_one_false(self) -> None:
        c = retry_all(retry_always(), retry_never())
        assert c(make_state()) is False

    def test_empty_is_true(self) -> None:
        c = retry_all()
        assert c(make_state()) is True

    def test_float_treated_as_truthy(self) -> None:
        c = retry_all(CustomRetry(1.5), retry_always())
        assert c(make_state()) is True

    def test_short_circuits(self) -> None:
        a = CustomRetry(False)
        b = CustomRetry(True)
        c = retry_all(a, b)
        assert c(make_state()) is False
        assert a.calls == 1
        assert b.calls == 0


class TestRetryAny:
    def test_any_true(self) -> None:
        c = retry_any(retry_never(), retry_always())
        assert c(make_state()) is True

    def test_all_false(self) -> None:
        c = retry_any(retry_never(), retry_never())
        assert c(make_state()) is False

    def test_empty_is_false(self) -> None:
        c = retry_any()
        assert c(make_state()) is False

    def test_float_result_returned(self) -> None:
        c = retry_any(retry_never(), CustomRetry(2.5))
        result = c(make_state())
        assert result == 2.5
        assert isinstance(result, float)

    def test_int_result_converted_to_float(self) -> None:
        c = retry_any(CustomRetry(3))
        result = c(make_state())
        assert result == 3.0
        assert isinstance(result, float)

    def test_bool_short_circuits_before_float(self) -> None:
        c = retry_any(CustomRetry(True), CustomRetry(2.5))
        assert c(make_state()) is True

    def test_short_circuits(self) -> None:
        a = CustomRetry(True)
        b = CustomRetry(False)
        c = retry_any(a, b)
        assert c(make_state()) is True
        assert a.calls == 1
        assert b.calls == 0


# ---------------------------------------------------------------------------
# retry_always / retry_never
# ---------------------------------------------------------------------------


class TestRetryAlwaysNever:
    def test_always_true(self) -> None:
        assert retry_always()(make_state()) is True

    def test_never_false(self) -> None:
        assert retry_never()(make_state()) is False


# ---------------------------------------------------------------------------
# retry_if_not_exception_type / retry_unless_exception_type
# ---------------------------------------------------------------------------


class TestRetryIfNotExceptionType:
    def test_no_outcome(self) -> None:
        c = retry_if_not_exception_type(ValueError)
        assert c(make_state(outcome=None)) is False

    def test_no_exception(self) -> None:
        c = retry_if_not_exception_type(ValueError)
        assert c(make_state().set_outcome(value=1)) is False

    def test_single_type_not_matching_returns_true(self) -> None:
        c = retry_if_not_exception_type(ValueError)
        assert c(make_state().set_outcome(exception=TypeError("x"))) is True

    def test_single_type_matching_returns_false(self) -> None:
        c = retry_if_not_exception_type(ValueError)
        assert c(make_state().set_outcome(exception=ValueError("x"))) is False

    def test_sequence(self) -> None:
        c = retry_if_not_exception_type([ValueError, TypeError])
        assert c.exc_types == (ValueError, TypeError)
        assert c(make_state().set_outcome(exception=KeyError("x"))) is True
        assert c(make_state().set_outcome(exception=TypeError("x"))) is False

    def test_subclass(self) -> None:
        class MyError(ValueError):
            pass

        c = retry_if_not_exception_type(ValueError)
        assert c(make_state().set_outcome(exception=MyError("x"))) is False


class TestRetryUnlessExceptionType:
    def test_is_subclass(self) -> None:
        c = retry_unless_exception_type(ValueError)
        assert isinstance(c, retry_if_not_exception_type)

    def test_behavior(self) -> None:
        c = retry_unless_exception_type(ValueError)
        assert c(make_state().set_outcome(exception=TypeError("x"))) is True
        assert c(make_state().set_outcome(exception=ValueError("x"))) is False


# ---------------------------------------------------------------------------
# retry_if_not_exception_message
# ---------------------------------------------------------------------------


class TestRetryIfNotExceptionMessage:
    def test_no_outcome(self) -> None:
        c = retry_if_not_exception_message("x")
        assert c(make_state(outcome=None)) is False

    def test_no_exception(self) -> None:
        c = retry_if_not_exception_message("x")
        assert c(make_state().set_outcome(value=1)) is False

    def test_substring_absent_returns_true(self) -> None:
        c = retry_if_not_exception_message("boom")
        assert c(make_state().set_outcome(exception=ValueError("nope"))) is True

    def test_substring_present_returns_false(self) -> None:
        c = retry_if_not_exception_message("boom")
        assert c(make_state().set_outcome(exception=ValueError("a boom"))) is False

    def test_init_stores_values(self) -> None:
        c = retry_if_not_exception_message("msg", regex=True)
        assert c.match == "msg"
        assert c.regex is True

    def test_regex_absent_returns_true(self) -> None:
        c = retry_if_not_exception_message(r"bo+m", regex=True)
        assert c(make_state().set_outcome(exception=ValueError("bam"))) is True

    def test_regex_present_returns_false(self) -> None:
        c = retry_if_not_exception_message(r"bo+m", regex=True)
        assert c(make_state().set_outcome(exception=ValueError("a boom"))) is False


# ---------------------------------------------------------------------------
# retry_if_exception_cause_type
# ---------------------------------------------------------------------------


class TestRetryIfExceptionCauseType:
    def test_no_outcome(self) -> None:
        c = retry_if_exception_cause_type(ValueError)
        assert c(make_state(outcome=None)) is False

    def test_no_exception(self) -> None:
        c = retry_if_exception_cause_type(ValueError)
        assert c(make_state().set_outcome(value=1)) is False

    def test_single_type_stored(self) -> None:
        c = retry_if_exception_cause_type(ValueError)
        assert c.exc_types == (ValueError,)

    def test_sequence_stored(self) -> None:
        c = retry_if_exception_cause_type([ValueError, TypeError])
        assert c.exc_types == (ValueError, TypeError)

    def test_no_cause(self) -> None:
        c = retry_if_exception_cause_type(ValueError)
        exc = RuntimeError("top")
        assert c(make_state().set_outcome(exception=exc)) is False

    def test_direct_cause_matches(self) -> None:
        c = retry_if_exception_cause_type(ValueError)
        try:
            try:
                raise ValueError("cause")
            except ValueError as e:
                raise RuntimeError("top") from e
        except RuntimeError as outer:
            exc = outer
        assert c(make_state().set_outcome(exception=exc)) is True

    def test_direct_cause_no_match(self) -> None:
        c = retry_if_exception_cause_type(KeyError)
        try:
            try:
                raise ValueError("cause")
            except ValueError as e:
                raise RuntimeError("top") from e
        except RuntimeError as outer:
            exc = outer
        assert c(make_state().set_outcome(exception=exc)) is False

    def test_nested_cause_matches(self) -> None:
        c = retry_if_exception_cause_type(ValueError)
        try:
            try:
                try:
                    raise ValueError("root")
                except ValueError as e1:
                    raise TypeError("mid") from e1
            except TypeError as e2:
                raise RuntimeError("top") from e2
        except RuntimeError as outer:
            exc = outer
        assert c(make_state().set_outcome(exception=exc)) is True

    def test_context_matches(self) -> None:
        c = retry_if_exception_cause_type(ValueError)

        class Exc:
            __cause__ = None
            __context__ = None

        # Build a chain using __context__ only.
        try:
            try:
                raise ValueError("ctx")
            except ValueError:
                raise RuntimeError("top")
        except RuntimeError as outer:
            exc = outer
        # __context__ is set when raised during handling.
        assert c(make_state().set_outcome(exception=exc)) is True

    def test_sequence_any_match(self) -> None:
        c = retry_if_exception_cause_type([ValueError, TypeError])
        try:
            try:
                raise TypeError("cause")
            except TypeError as e:
                raise RuntimeError("top") from e
        except RuntimeError as outer:
            exc = outer
        assert c(make_state().set_outcome(exception=exc)) is True


# ---------------------------------------------------------------------------
# Operator composition integration tests
# ---------------------------------------------------------------------------


class TestOperatorComposition:
    def test_stop_or_behavior(self) -> None:
        combined = stop_after_attempt(2) | stop_after_delay(5.0)
        assert combined(make_state(tries=2, elapsed=0.0)) is True
        assert combined(make_state(tries=0, elapsed=0.0)) is False

    def test_stop_and_behavior(self) -> None:
        combined = stop_after_attempt(2) & stop_after_delay(5.0)
        assert combined(make_state(tries=2, elapsed=5.0)) is True
        assert combined(make_state(tries=2, elapsed=0.0)) is False

    def test_retry_or_behavior(self) -> None:
        combined = retry_never() | retry_always()
        assert combined(make_state()) is True

    def test_retry_and_behavior(self) -> None:
        combined = retry_always() & retry_never()
        assert combined(make_state()) is False

    def test_chained_operator(self) -> None:
        combined = retry_never() | retry_never() | retry_always()
        assert isinstance(combined, retry_any)
        # Left-to-right: (retry_never | retry_never) | retry_always
        assert combined(make_state()) is True
