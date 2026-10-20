from __future__ import annotations

import threading
import time
from unittest import mock

import pytest

from backon._state import (
    Attempt,
    AttemptResult,
    AttemptTimeoutError,
    RetryCallState,
    RetryError,
    RetryState,
    TryAgain,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TestExceptionClasses:
    def test_try_again_is_exception(self):
        assert issubclass(TryAgain, Exception)
        with pytest.raises(TryAgain):
            raise TryAgain("retry please")

    def test_try_again_message(self):
        exc = TryAgain("hello")
        assert exc.args == ("hello",)

    def test_attempt_timeout_error_is_exception(self):
        assert issubclass(AttemptTimeoutError, Exception)
        with pytest.raises(AttemptTimeoutError):
            raise AttemptTimeoutError()


# ---------------------------------------------------------------------------
# Attempt
# ---------------------------------------------------------------------------

class TestAttempt:
    def test_defaults(self):
        attempt = Attempt()
        assert attempt.tries == 0
        assert attempt.exception is None
        assert attempt.value is None
        assert attempt.wait == 0.0
        assert attempt.elapsed == 0.0

    def test_custom_fields(self):
        err = ValueError("bad")
        attempt = Attempt(tries=3, exception=err, value=42, wait=1.5, elapsed=2.5)
        assert attempt.tries == 3
        assert attempt.exception is err
        assert attempt.value == 42
        assert attempt.wait == 1.5
        assert attempt.elapsed == 2.5

    def test_equality(self):
        assert Attempt(tries=1, value="x") == Attempt(tries=1, value="x")
        assert Attempt(tries=1) != Attempt(tries=2)


# ---------------------------------------------------------------------------
# AttemptResult
# ---------------------------------------------------------------------------

class TestAttemptResult:
    def test_defaults(self):
        result = AttemptResult()
        assert result.value is None
        assert result.exception is None

    def test_value(self):
        result = AttemptResult(value=10)
        assert result.value == 10
        assert result.exception is None

    def test_exception(self):
        err = RuntimeError("boom")
        result = AttemptResult(exception=err)
        assert result.value is None
        assert result.exception is err

    def test_equality(self):
        assert AttemptResult(value=1) == AttemptResult(value=1)
        assert AttemptResult(value=1) != AttemptResult(value=2)


# ---------------------------------------------------------------------------
# RetryError
# ---------------------------------------------------------------------------

class TestRetryError:
    def test_message_and_last_attempt(self):
        cause = ValueError("cause")
        attempt = Attempt(tries=4, exception=cause)
        err = RetryError(attempt)
        assert err.last_attempt is attempt
        assert str(err) == "Retry failed after 4 tries"
        assert err.__cause__ is cause

    def test_no_exception_no_cause(self):
        attempt = Attempt(tries=2, exception=None)
        err = RetryError(attempt)
        assert err.last_attempt is attempt
        assert str(err) == "Retry failed after 2 tries"
        assert err.__cause__ is None

    def test_reraise_raises_cause(self):
        cause = KeyError("key")
        err = RetryError(Attempt(tries=1, exception=cause))
        with pytest.raises(KeyError) as excinfo:
            err.reraise()
        assert excinfo.value is cause
        # raised "from None" so no context/cause
        assert excinfo.value.__cause__ is None
        assert excinfo.value.__suppress_context__ is True

    def test_reraise_without_cause_does_nothing(self):
        err = RetryError(Attempt(tries=1, exception=None))
        # Should simply return without raising anything
        assert err.reraise() is None

    def test_is_runtime_error_subclass(self):
        assert issubclass(RetryError, Exception)


# ---------------------------------------------------------------------------
# RetryState
# ---------------------------------------------------------------------------

class TestRetryState:
    def test_defaults(self):
        state = RetryState()
        assert callable(state.target)
        assert state.target() is None
        assert state.args == ()
        assert state.kwargs == {}
        assert state.tries == 0
        assert state.start_time == 0.0
        assert state.elapsed == 0.0
        assert state.idle_for == 0.0
        assert state.outcome is None

    def test_custom_init(self):
        def target(a, b):
            return a + b

        state = RetryState(target=target, args=(1, 2), kwargs={"c": 3}, tries=5)
        assert state.target is target
        assert state.args == (1, 2)
        assert state.kwargs == {"c": 3}
        assert state.tries == 5

    def test_kwargs_default_isolated_between_instances(self):
        s1 = RetryState()
        s2 = RetryState()
        s1.kwargs["x"] = 1
        assert s2.kwargs == {}

    def test_post_init_creates_lock(self):
        state = RetryState()
        lock = state._lock.lock
        assert isinstance(lock, type(threading.Lock()))
        # Lock is not held initially
        assert lock.acquire(blocking=False) is True
        lock.release()

    def test_post_init_lock_is_per_instance_thread_local(self):
        s1 = RetryState()
        s2 = RetryState()
        assert s1._lock.lock is not s2._lock.lock

    def test_statistics(self):
        state = RetryState(tries=3, start_time=10.0, idle_for=1.5, elapsed=2.5)
        assert state.statistics == {
            "start_time": 10.0,
            "attempt_number": 3,
            "idle_for": 1.5,
            "elapsed": 2.5,
        }

    def test_statistics_defaults(self):
        state = RetryState()
        assert state.statistics == {
            "start_time": 0.0,
            "attempt_number": 0,
            "idle_for": 0.0,
            "elapsed": 0.0,
        }

    def test_to_details(self):
        def target():
            pass

        args = (1,)
        kwargs = {"a": 2}
        state = RetryState(target=target, args=args, kwargs=kwargs, tries=2, elapsed=3.0)
        details = state.to_details()
        assert details == {
            "target": target,
            "args": args,
            "kwargs": kwargs,
            "tries": 2,
            "elapsed": 3.0,
        }
        assert details["target"] is target
        assert details["args"] is args
        assert details["kwargs"] is kwargs


# ---------------------------------------------------------------------------
# RetryCallState
# ---------------------------------------------------------------------------

class TestRetryCallState:
    def test_defaults(self):
        state = RetryCallState()
        assert state.fn is None
        assert state.args == ()
        assert state.kwargs == {}
        assert state.attempt_number == 1
        assert state.outcome is None
        assert state.outcome_timestamp is None
        assert state.start_time == 0.0
        assert state.idle_for == 0.0
        assert state.upcoming_sleep == 0.0

    def test_kwargs_default_isolated(self):
        s1 = RetryCallState()
        s2 = RetryCallState()
        s1.kwargs["a"] = 1
        assert s2.kwargs == {}

    def test_custom_init(self):
        def fn(x):
            return x

        state = RetryCallState(
            fn=fn,
            args=(1,),
            kwargs={"k": 2},
            attempt_number=4,
            outcome_timestamp=123.0,
            start_time=5.0,
            idle_for=0.5,
            upcoming_sleep=1.0,
        )
        assert state.fn is fn
        assert state.args == (1,)
        assert state.kwargs == {"k": 2}
        assert state.attempt_number == 4
        assert state.outcome_timestamp == 123.0
        assert state.start_time == 5.0
        assert state.idle_for == 0.5
        assert state.upcoming_sleep == 1.0

    def test_elapsed_zero_when_start_time_zero(self):
        state = RetryCallState(start_time=0.0)
        assert state.elapsed == 0.0

    def test_elapsed_uses_monotonic(self):
        state = RetryCallState(start_time=100.0)
        with mock.patch("backon._state.time_module.monotonic", return_value=105.5):
            assert state.elapsed == 5.5

    def test_seconds_since_start_equals_elapsed(self):
        state = RetryCallState(start_time=50.0)
        with mock.patch("backon._state.time_module.monotonic", return_value=53.25):
            assert state.seconds_since_start == state.elapsed == 3.25

    def test_seconds_since_start_zero(self):
        state = RetryCallState()
        assert state.seconds_since_start == 0.0

    def test_statistics(self):
        state = RetryCallState(
            attempt_number=3,
            start_time=10.0,
            idle_for=1.5,
        )
        with mock.patch("backon._state.time_module.monotonic", return_value=12.5):
            assert state.statistics == {
                "start_time": 10.0,
                "attempt_number": 3,
                "idle_for": 1.5,
                "elapsed": 2.5,
            }

    def test_statistics_zero_start(self):
        state = RetryCallState()
        assert state.statistics == {
            "start_time": 0.0,
            "attempt_number": 1,
            "idle_for": 0.0,
            "elapsed": 0.0,
        }

    def test_to_details(self):
        def fn():
            pass

        args = ("a",)
        kwargs = {"b": 1}
        state = RetryCallState(
            fn=fn,
            args=args,
            kwargs=kwargs,
            attempt_number=2,
            start_time=10.0,
        )
        with mock.patch("backon._state.time_module.monotonic", return_value=13.0):
            details = state.to_details()
        assert details == {
            "target": fn,
            "args": args,
            "kwargs": kwargs,
            "tries": 2,
            "elapsed": 3.0,
        }
        assert details["target"] is fn
        assert details["args"] is args
        assert details["kwargs"] is kwargs

    def test_to_details_zero_start(self):
        state = RetryCallState(attempt_number=1)
        details = state.to_details()
        assert details == {
            "target": None,
            "args": (),
            "kwargs": {},
            "tries": 1,
            "elapsed": 0.0,
        }

    def test_outcome_can_hold_attempt(self):
        attempt = Attempt(tries=1, value="ok")
        state = RetryCallState(outcome=attempt)
        assert state.outcome is attempt
