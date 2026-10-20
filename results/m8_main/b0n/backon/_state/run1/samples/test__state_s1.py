from __future__ import annotations

import threading
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
# Exception classes
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_try_again_is_exception(self):
        exc = TryAgain("x")
        assert isinstance(exc, Exception)
        assert str(exc) == "x"

    def test_attempt_timeout_error_is_exception(self):
        exc = AttemptTimeoutError("t")
        assert isinstance(exc, Exception)
        assert str(exc) == "t"


class TestRetryError:
    def test_message_and_cause(self):
        cause = ValueError("boom")
        attempt = Attempt(tries=3, exception=cause)
        err = RetryError(attempt)
        assert err.last_attempt is attempt
        assert str(err) == "Retry failed after 3 tries"
        assert err.__cause__ is cause

    def test_no_exception_means_no_cause(self):
        attempt = Attempt(tries=1, exception=None)
        err = RetryError(attempt)
        assert err.last_attempt is attempt
        assert str(err) == "Retry failed after 1 tries"
        assert err.__cause__ is None

    def test_reraise_raises_stored_cause(self):
        cause = KeyError("k")
        attempt = Attempt(tries=2, exception=cause)
        err = RetryError(attempt)
        with pytest.raises(KeyError) as info:
            err.reraise()
        assert info.value is cause
        # raised with "from None", so no context
        assert info.value.__cause__ is None

    def test_reraise_without_cause_returns_none(self):
        attempt = Attempt(tries=1, exception=None)
        err = RetryError(attempt)
        # No exception to raise, so reraise silently returns.
        assert err.reraise() is None

    def test_reraise_no_context_on_raise(self):
        cause = RuntimeError("inner")
        attempt = Attempt(tries=1, exception=cause)
        err = RetryError(attempt)
        try:
            err.reraise()
        except RuntimeError as caught:
            assert caught is cause
            assert caught.__cause__ is None
            assert caught.__suppress_context__ is True
        else:
            pytest.fail("expected RuntimeError")


# ---------------------------------------------------------------------------
# Attempt / AttemptResult dataclasses
# ---------------------------------------------------------------------------


class TestAttempt:
    def test_defaults(self):
        a = Attempt()
        assert a.tries == 0
        assert a.exception is None
        assert a.value is None
        assert a.wait == 0.0
        assert a.elapsed == 0.0

    def test_positional_construction(self):
        exc = ValueError("e")
        a = Attempt(2, exc, "val", 1.5, 2.5)
        assert a.tries == 2
        assert a.exception is exc
        assert a.value == "val"
        assert a.wait == 1.5
        assert a.elapsed == 2.5

    def test_equality(self):
        assert Attempt(tries=1, value=5) == Attempt(tries=1, value=5)
        assert Attempt(tries=1) != Attempt(tries=2)


class TestAttemptResult:
    def test_defaults(self):
        r = AttemptResult()
        assert r.value is None
        assert r.exception is None

    def test_value(self):
        r = AttemptResult(value=42)
        assert r.value == 42
        assert r.exception is None

    def test_exception(self):
        exc = ValueError("e")
        r = AttemptResult(exception=exc)
        assert r.exception is exc
        assert r.value is None

    def test_equality(self):
        assert AttemptResult(value=1) == AttemptResult(value=1)
        assert AttemptResult(value=1) != AttemptResult(value=2)


# ---------------------------------------------------------------------------
# RetryState
# ---------------------------------------------------------------------------


class TestRetryState:
    def test_defaults(self):
        s = RetryState()
        assert callable(s.target)
        assert s.target() is None
        assert s.args == ()
        assert s.kwargs == {}
        assert s.tries == 0
        assert s.start_time == 0.0
        assert s.elapsed == 0.0
        assert s.idle_for == 0.0
        assert s.outcome is None

    def test_post_init_creates_lock(self):
        s = RetryState()
        assert hasattr(s._lock, "lock")
        assert isinstance(s._lock.lock, type(threading.Lock()))

    def test_kwargs_independent_between_instances(self):
        s1 = RetryState()
        s2 = RetryState()
        s1.kwargs["a"] = 1
        assert s2.kwargs == {}

    def test_custom_fields(self):
        def target(a, b):
            return a + b

        s = RetryState(
            target=target,
            args=(1, 2),
            kwargs={"c": 3},
            tries=4,
            start_time=10.0,
            elapsed=1.2,
            idle_for=0.3,
        )
        assert s.target is target
        assert s.args == (1, 2)
        assert s.kwargs == {"c": 3}
        assert s.tries == 4
        assert s.start_time == 10.0
        assert s.elapsed == 1.2
        assert s.idle_for == 0.3

    def test_statistics(self):
        s = RetryState(tries=2, start_time=100.0, idle_for=0.5, elapsed=3.0)
        assert s.statistics == {
            "start_time": 100.0,
            "attempt_number": 2,
            "idle_for": 0.5,
            "elapsed": 3.0,
        }

    def test_statistics_is_fresh_dict(self):
        s = RetryState()
        d1 = s.statistics
        d2 = s.statistics
        assert d1 == d2
        assert d1 is not d2

    def test_to_details(self):
        def target():
            pass

        s = RetryState(
            target=target,
            args=(1,),
            kwargs={"k": "v"},
            tries=3,
            elapsed=1.5,
        )
        assert s.to_details() == {
            "target": target,
            "args": (1,),
            "kwargs": {"k": "v"},
            "tries": 3,
            "elapsed": 1.5,
        }

    def test_lock_is_per_instance(self):
        s1 = RetryState()
        s2 = RetryState()
        assert s1._lock is not s2._lock


# ---------------------------------------------------------------------------
# RetryCallState
# ---------------------------------------------------------------------------


class TestRetryCallState:
    def test_defaults(self):
        s = RetryCallState()
        assert s.fn is None
        assert s.args == ()
        assert s.kwargs == {}
        assert s.attempt_number == 1
        assert s.outcome is None
        assert s.outcome_timestamp is None
        assert s.start_time == 0.0
        assert s.idle_for == 0.0
        assert s.upcoming_sleep == 0.0

    def test_kwargs_independent_between_instances(self):
        s1 = RetryCallState()
        s2 = RetryCallState()
        s1.kwargs["x"] = 1
        assert s2.kwargs == {}

    def test_elapsed_zero_when_start_time_zero(self):
        s = RetryCallState()
        assert s.elapsed == 0.0
        assert s.seconds_since_start == 0.0

    def test_elapsed_uses_monotonic(self):
        s = RetryCallState(start_time=100.0)
        with mock.patch(
            "backon._state.time_module.monotonic", return_value=103.5
        ):
            assert s.elapsed == pytest.approx(3.5)

    def test_seconds_since_start_matches_elapsed(self):
        s = RetryCallState(start_time=50.0)
        with mock.patch(
            "backon._state.time_module.monotonic", return_value=52.0
        ):
            assert s.elapsed == pytest.approx(2.0)
            assert s.seconds_since_start == pytest.approx(2.0)

    def test_statistics(self):
        s = RetryCallState(
            attempt_number=5,
            start_time=10.0,
            idle_for=0.7,
        )
        with mock.patch(
            "backon._state.time_module.monotonic", return_value=12.0
        ):
            assert s.statistics == {
                "start_time": 10.0,
                "attempt_number": 5,
                "idle_for": 0.7,
                "elapsed": pytest.approx(2.0),
            }

    def test_statistics_without_start(self):
        s = RetryCallState(attempt_number=0, idle_for=0.0)
        assert s.statistics == {
            "start_time": 0.0,
            "attempt_number": 0,
            "idle_for": 0.0,
            "elapsed": 0.0,
        }

    def test_to_details(self):
        def fn():
            pass

        s = RetryCallState(
            fn=fn,
            args=(1, 2),
            kwargs={"a": 1},
            attempt_number=2,
            start_time=1.0,
        )
        with mock.patch(
            "backon._state.time_module.monotonic", return_value=3.0
        ):
            assert s.to_details() == {
                "target": fn,
                "args": (1, 2),
                "kwargs": {"a": 1},
                "tries": 2,
                "elapsed": pytest.approx(2.0),
            }

    def test_outcome_stored(self):
        a = Attempt(tries=1, value="ok")
        s = RetryCallState(outcome=a, outcome_timestamp=123.0)
        assert s.outcome is a
        assert s.outcome_timestamp == 123.0
