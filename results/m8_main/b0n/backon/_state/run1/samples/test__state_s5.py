import threading
import time

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


class TestExceptions:
    def test_try_again_is_exception(self):
        exc = TryAgain("boom")
        assert isinstance(exc, Exception)
        assert str(exc) == "boom"

    def test_try_again_can_be_raised(self):
        with pytest.raises(TryAgain):
            raise TryAgain()

    def test_attempt_timeout_error(self):
        exc = AttemptTimeoutError("timeout")
        assert isinstance(exc, Exception)
        assert str(exc) == "timeout"

    def test_attempt_timeout_error_raised(self):
        with pytest.raises(AttemptTimeoutError):
            raise AttemptTimeoutError()

    def test_retry_error_message_and_cause(self):
        cause = ValueError("original")
        last = Attempt(tries=3, exception=cause)
        err = RetryError(last)
        assert err.last_attempt is last
        assert str(err) == "Retry failed after 3 tries"
        assert err.__cause__ is cause

    def test_retry_error_no_cause(self):
        last = Attempt(tries=5, exception=None)
        err = RetryError(last)
        assert err.last_attempt is last
        assert str(err) == "Retry failed after 5 tries"
        assert err.__cause__ is None

    def test_retry_error_reraise_raises_cause_from_none(self):
        cause = KeyError("k")
        err = RetryError(Attempt(tries=1, exception=cause))
        with pytest.raises(KeyError) as excinfo:
            err.reraise()
        assert excinfo.value is cause
        assert excinfo.value.__cause__ is None
        assert excinfo.value.__suppress_context__ is True

    def test_retry_error_reraise_no_exception_returns_none(self):
        err = RetryError(Attempt(tries=1, exception=None))
        assert err.reraise() is None

    def test_retry_error_reraise_with_base_exception(self):
        cause = KeyboardInterrupt()
        err = RetryError(Attempt(tries=1, exception=cause))
        with pytest.raises(KeyboardInterrupt) as excinfo:
            err.reraise()
        assert excinfo.value is cause


class TestAttempt:
    def test_defaults(self):
        a = Attempt()
        assert a.tries == 0
        assert a.exception is None
        assert a.value is None
        assert a.wait == 0.0
        assert a.elapsed == 0.0

    def test_custom_values(self):
        exc = RuntimeError("x")
        a = Attempt(tries=2, exception=exc, value=42, wait=1.5, elapsed=0.25)
        assert a.tries == 2
        assert a.exception is exc
        assert a.value == 42
        assert a.wait == 1.5
        assert a.elapsed == 0.25


class TestAttemptResult:
    def test_defaults(self):
        r = AttemptResult()
        assert r.value is None
        assert r.exception is None

    def test_with_values(self):
        exc = ValueError("v")
        r = AttemptResult(value="ok", exception=exc)
        assert r.value == "ok"
        assert r.exception is exc


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
        assert callable(state.target)
        assert state.target() is None

    def test_lock_initialized_per_instance(self):
        s1 = RetryState()
        s2 = RetryState()
        assert isinstance(s1._lock.lock, type(threading.Lock()))
        assert s1._lock.lock is not s2._lock.lock

    def test_lock_usable(self):
        s = RetryState()
        with s._lock.lock:
            assert True
        # lock must be released after use
        acquired = s._lock.lock.acquire(timeout=1.0)
        assert acquired is True
        s._lock.lock.release()

    def test_kwargs_default_factory_is_unique(self):
        s1 = RetryState()
        s2 = RetryState()
        s1.kwargs["a"] = 1
        assert s2.kwargs == {}

    def test_custom_values(self):
        def target():
            return 7

        state = RetryState(
            target=target,
            args=(1, 2),
            kwargs={"x": 3},
            tries=4,
            start_time=100.0,
            elapsed=2.5,
            idle_for=0.1,
            outcome=Attempt(tries=4),
        )
        assert state.target is target
        assert state.args == (1, 2)
        assert state.kwargs == {"x": 3}
        assert state.tries == 4
        assert state.start_time == 100.0
        assert state.elapsed == 2.5
        assert state.idle_for == 0.1
        assert state.outcome.tries == 4

    def test_statistics(self):
        state = RetryState(start_time=10.0, tries=3, idle_for=0.5, elapsed=1.25)
        assert state.statistics == {
            "start_time": 10.0,
            "attempt_number": 3,
            "idle_for": 0.5,
            "elapsed": 1.25,
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
            return None

        state = RetryState(
            target=target,
            args=(1,),
            kwargs={"b": 2},
            tries=5,
            elapsed=3.0,
        )
        assert state.to_details() == {
            "target": target,
            "args": (1,),
            "kwargs": {"b": 2},
            "tries": 5,
            "elapsed": 3.0,
        }


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

    def test_kwargs_default_factory_is_unique(self):
        s1 = RetryCallState()
        s2 = RetryCallState()
        s1.kwargs["z"] = 1
        assert s2.kwargs == {}

    def test_elapsed_zero_when_start_time_zero(self, monkeypatch):
        # Even if monotonic advances, elapsed is 0 when start_time == 0.
        monkeypatch.setattr(time, "monotonic", lambda: 12345.0)
        state = RetryCallState()
        assert state.elapsed == 0.0
        assert state.seconds_since_start == 0.0

    def test_elapsed_uses_monotonic(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 100.0)
        state = RetryCallState(start_time=40.0)
        assert state.elapsed == pytest.approx(60.0)
        assert state.seconds_since_start == pytest.approx(60.0)

    def test_elapsed_negative_time_difference(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 10.0)
        state = RetryCallState(start_time=15.0)
        assert state.elapsed == pytest.approx(-5.0)

    def test_elapsed_is_dynamic(self, monkeypatch):
        now = {"t": 100.0}
        monkeypatch.setattr(time, "monotonic", lambda: now["t"])
        state = RetryCallState(start_time=90.0)
        assert state.elapsed == pytest.approx(10.0)
        now["t"] = 95.0
        assert state.elapsed == pytest.approx(5.0)

    def test_statistics(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 110.0)
        state = RetryCallState(
            start_time=100.0,
            attempt_number=4,
            idle_for=0.75,
        )
        assert state.statistics == {
            "start_time": 100.0,
            "attempt_number": 4,
            "idle_for": 0.75,
            "elapsed": pytest.approx(10.0),
        }

    def test_statistics_zero_start(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 500.0)
        state = RetryCallState(attempt_number=2, idle_for=0.2)
        assert state.statistics == {
            "start_time": 0.0,
            "attempt_number": 2,
            "idle_for": 0.2,
            "elapsed": 0.0,
        }

    def test_to_details(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 200.0)

        def fn():
            return 1

        state = RetryCallState(
            fn=fn,
            args=("a",),
            kwargs={"k": "v"},
            attempt_number=3,
            start_time=150.0,
        )
        details = state.to_details()
        assert details["target"] is fn
        assert details["args"] == ("a",)
        assert details["kwargs"] == {"k": "v"}
        assert details["tries"] == 3
        assert details["elapsed"] == pytest.approx(50.0)

    def test_outcome_and_timestamp(self):
        outcome = Attempt(tries=2, value="x")
        state = RetryCallState(outcome=outcome, outcome_timestamp=123.5)
        assert state.outcome is outcome
        assert state.outcome_timestamp == 123.5

    def test_upcoming_sleep_field(self):
        state = RetryCallState(upcoming_sleep=1.25)
        assert state.upcoming_sleep == 1.25
