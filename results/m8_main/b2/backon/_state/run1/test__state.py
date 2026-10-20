from __future__ import annotations

import threading

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


class TestExceptions:
    def test_try_again_is_exception(self):
        assert issubclass(TryAgain, Exception)
        err = TryAgain("retry please")
        assert str(err) == "retry please"

    def test_attempt_timeout_error_is_exception(self):
        assert issubclass(AttemptTimeoutError, Exception)
        err = AttemptTimeoutError()
        assert isinstance(err, Exception)


class TestRetryError:
    def test_message_and_last_attempt(self):
        exc = ValueError("boom")
        attempt = Attempt(tries=3, exception=exc)
        err = RetryError(attempt)
        assert str(err) == "Retry failed after 3 tries"
        assert err.last_attempt is attempt
        assert err.__cause__ is exc

    def test_no_exception_means_no_cause(self):
        attempt = Attempt(tries=2, exception=None)
        err = RetryError(attempt)
        assert str(err) == "Retry failed after 2 tries"
        assert err.last_attempt is attempt
        assert err.__cause__ is None

    def test_default_attempt_zero_tries(self):
        attempt = Attempt()
        err = RetryError(attempt)
        assert str(err) == "Retry failed after 0 tries"
        assert err.__cause__ is None

    def test_reraise_raises_cause_without_context(self):
        exc = RuntimeError("original")
        attempt = Attempt(tries=1, exception=exc)
        err = RetryError(attempt)
        with pytest.raises(RuntimeError) as excinfo:
            err.reraise()
        assert excinfo.value is exc
        assert excinfo.value.__context__ is None
        assert excinfo.value.__suppress_context__ is True

    def test_reraise_no_exception_is_noop(self):
        attempt = Attempt(tries=1, exception=None)
        err = RetryError(attempt)
        # Should simply return without raising.
        assert err.reraise() is None

    def test_reraise_custom_base_exception(self):
        exc = KeyboardInterrupt()
        attempt = Attempt(tries=1, exception=exc)
        err = RetryError(attempt)
        with pytest.raises(KeyboardInterrupt):
            err.reraise()


# ---------------------------------------------------------------------------
# Attempt / AttemptResult dataclasses
# ---------------------------------------------------------------------------


class TestAttempt:
    def test_defaults(self):
        attempt = Attempt()
        assert attempt.tries == 0
        assert attempt.exception is None
        assert attempt.value is None
        assert attempt.wait == 0.0
        assert attempt.elapsed == 0.0

    def test_custom_values(self):
        exc = ValueError("x")
        attempt = Attempt(tries=5, exception=exc, value="v", wait=1.5, elapsed=2.5)
        assert attempt.tries == 5
        assert attempt.exception is exc
        assert attempt.value == "v"
        assert attempt.wait == 1.5
        assert attempt.elapsed == 2.5

    def test_equality(self):
        assert Attempt(tries=1, value=5) == Attempt(tries=1, value=5)
        assert Attempt(tries=1) != Attempt(tries=2)


class TestAttemptResult:
    def test_defaults(self):
        result = AttemptResult()
        assert result.value is None
        assert result.exception is None

    def test_custom(self):
        exc = ValueError("x")
        result = AttemptResult(value=42, exception=exc)
        assert result.value == 42
        assert result.exception is exc

    def test_equality(self):
        assert AttemptResult(value=1) == AttemptResult(value=1)


# ---------------------------------------------------------------------------
# RetryState
# ---------------------------------------------------------------------------


class TestRetryState:
    def test_defaults(self):
        state = RetryState()
        assert state.args == ()
        assert state.kwargs == {}
        assert state.tries == 0
        assert state.start_time == 0.0
        assert state.elapsed == 0.0
        assert state.idle_for == 0.0
        assert state.outcome is None

    def test_default_target_callable_returns_none(self):
        state = RetryState()
        assert state.target() is None

    def test_post_init_creates_lock(self):
        state = RetryState()
        assert isinstance(state._lock.lock, type(threading.Lock()))

    def test_each_instance_has_own_lock(self):
        s1 = RetryState()
        s2 = RetryState()
        # thread-local objects are distinct per instance (default_factory)
        assert s1._lock is not s2._lock
        # Since __post_init__ creates a fresh threading.Lock per instance,
        # the stored lock attributes in the same thread are distinct objects.
        assert s1._lock.lock is not s2._lock.lock

    def test_lock_is_thread_local(self):
        # The lock attribute is only set in the thread where __post_init__ ran.
        state = RetryState()
        assert hasattr(state._lock, "lock")
        captured = {}

        def worker():
            captured["has_lock"] = hasattr(state._lock, "lock")

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert captured["has_lock"] is False

    def test_kwargs_default_factory_not_shared(self):
        s1 = RetryState()
        s2 = RetryState()
        s1.kwargs["a"] = 1
        assert s2.kwargs == {}

    def test_custom_values(self):
        target = lambda x: x

        def fn(*args, **kwargs):
            return None

        state = RetryState(
            target=fn,
            args=(1, 2),
            kwargs={"k": "v"},
            tries=3,
            start_time=10.0,
            elapsed=1.5,
            idle_for=0.5,
        )
        assert state.target is fn
        assert state.args == (1, 2)
        assert state.kwargs == {"k": "v"}
        assert state.tries == 3
        assert state.start_time == 10.0
        assert state.elapsed == 1.5
        assert state.idle_for == 0.5

    def test_statistics(self):
        state = RetryState(tries=4, start_time=100.0, elapsed=2.0, idle_for=0.25)
        assert state.statistics == {
            "start_time": 100.0,
            "attempt_number": 4,
            "idle_for": 0.25,
            "elapsed": 2.0,
        }

    def test_statistics_keys(self):
        state = RetryState()
        stats = state.statistics
        assert set(stats.keys()) == {
            "start_time",
            "attempt_number",
            "idle_for",
            "elapsed",
        }

    def test_to_details(self):
        def target():
            return None

        kwargs = {"a": 1}
        state = RetryState(target=target, args=(1,), kwargs=kwargs, tries=2, elapsed=3.0)
        details = state.to_details()
        assert details == {
            "target": target,
            "args": (1,),
            "kwargs": kwargs,
            "tries": 2,
            "elapsed": 3.0,
        }

    def test_to_details_default(self):
        state = RetryState()
        details = state.to_details()
        assert details["args"] == ()
        assert details["kwargs"] == {}
        assert details["tries"] == 0
        assert details["elapsed"] == 0.0
        assert details["target"] is state.target


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

    def test_elapsed_zero_when_start_time_zero(self, monkeypatch):
        state = RetryCallState()
        # Even if monotonic advances, start_time == 0 yields 0.0
        monkeypatch.setattr(
            "backon._state.time_module.monotonic", lambda: 12345.0
        )
        assert state.elapsed == 0.0

    def test_elapsed_computed_from_start_time(self, monkeypatch):
        monkeypatch.setattr(
            "backon._state.time_module.monotonic", lambda: 100.0
        )
        state = RetryCallState(start_time=40.0)
        assert state.elapsed == 60.0

    def test_elapsed_negative_when_start_in_future(self, monkeypatch):
        monkeypatch.setattr(
            "backon._state.time_module.monotonic", lambda: 10.0
        )
        state = RetryCallState(start_time=20.0)
        assert state.elapsed == -10.0

    def test_seconds_since_start_equals_elapsed(self, monkeypatch):
        monkeypatch.setattr(
            "backon._state.time_module.monotonic", lambda: 50.5
        )
        state = RetryCallState(start_time=0.5)
        assert state.seconds_since_start == state.elapsed
        assert state.seconds_since_start == 50.0

    def test_seconds_since_start_zero(self):
        state = RetryCallState()
        assert state.seconds_since_start == 0.0

    def test_statistics(self, monkeypatch):
        monkeypatch.setattr(
            "backon._state.time_module.monotonic", lambda: 10.0
        )
        state = RetryCallState(
            attempt_number=3, idle_for=0.5, start_time=4.0
        )
        assert state.statistics == {
            "start_time": 4.0,
            "attempt_number": 3,
            "idle_for": 0.5,
            "elapsed": 6.0,
        }

    def test_statistics_with_zero_start(self):
        state = RetryCallState(attempt_number=2, idle_for=1.0)
        assert state.statistics == {
            "start_time": 0.0,
            "attempt_number": 2,
            "idle_for": 1.0,
            "elapsed": 0.0,
        }

    def test_to_details(self, monkeypatch):
        monkeypatch.setattr(
            "backon._state.time_module.monotonic", lambda: 20.0
        )

        def fn():
            return None

        kwargs = {"a": 1}
        state = RetryCallState(
            fn=fn, args=(1, 2), kwargs=kwargs, attempt_number=4, start_time=5.0
        )
        details = state.to_details()
        assert details == {
            "target": fn,
            "args": (1, 2),
            "kwargs": kwargs,
            "tries": 4,
            "elapsed": 15.0,
        }

    def test_to_details_default(self):
        state = RetryCallState()
        details = state.to_details()
        assert details["target"] is None
        assert details["args"] == ()
        assert details["kwargs"] == {}
        assert details["tries"] == 1
        assert details["elapsed"] == 0.0

    def test_kwargs_default_factory_not_shared(self):
        s1 = RetryCallState()
        s2 = RetryCallState()
        s1.kwargs["x"] = 1
        assert s2.kwargs == {}

    def test_outcome_and_timestamp_assignable(self):
        state = RetryCallState()
        attempt = Attempt(tries=1, value=7)
        state.outcome = attempt
        state.outcome_timestamp = 123.0
        assert state.outcome is attempt
        assert state.outcome_timestamp == 123.0
