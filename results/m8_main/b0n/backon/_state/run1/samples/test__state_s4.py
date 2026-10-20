from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TestExceptionClasses:
    def test_try_again_is_exception(self):
        assert issubclass(TryAgain, Exception)
        with pytest.raises(TryAgain):
            raise TryAgain()

    def test_attempt_timeout_error_is_exception(self):
        assert issubclass(AttemptTimeoutError, Exception)
        with pytest.raises(AttemptTimeoutError):
            raise AttemptTimeoutError()

    def test_retry_error_stores_last_attempt(self):
        exc = ValueError("boom")
        attempt = Attempt(tries=3, exception=exc)
        err = RetryError(attempt)
        assert err.last_attempt is attempt
        assert str(err) == "Retry failed after 3 tries"
        assert err.__cause__ is exc

    def test_retry_error_without_cause(self):
        attempt = Attempt(tries=0, exception=None)
        err = RetryError(attempt)
        assert err.last_attempt is attempt
        assert str(err) == "Retry failed after 0 tries"
        assert err.__cause__ is None

    def test_reraise_raises_cause_from_none(self):
        exc = ValueError("boom")
        err = RetryError(Attempt(tries=1, exception=exc))
        with pytest.raises(ValueError) as info:
            err.reraise()
        assert info.value is exc
        # raised "from None", so no context chained in
        assert info.value.__suppress_context__ is True
        assert info.value.__cause__ is None

    def test_reraise_without_cause_does_nothing(self):
        err = RetryError(Attempt(tries=2, exception=None))
        # Should simply return without raising.
        assert err.reraise() is None

    def test_retry_error_explicit_cause_assignment(self):
        exc = RuntimeError("x")
        err = RetryError(Attempt(tries=5, exception=exc))
        assert isinstance(err, Exception)
        assert err.__cause__ is exc


# ---------------------------------------------------------------------------
# Attempt
# ---------------------------------------------------------------------------

class TestAttempt:
    def test_defaults(self):
        a = Attempt()
        assert a.tries == 0
        assert a.exception is None
        assert a.value is None
        assert a.wait == 0.0
        assert a.elapsed == 0.0

    def test_custom_values(self):
        exc = KeyError("k")
        a = Attempt(tries=4, exception=exc, value="v", wait=1.5, elapsed=2.5)
        assert a.tries == 4
        assert a.exception is exc
        assert a.value == "v"
        assert a.wait == 1.5
        assert a.elapsed == 2.5

    def test_equality(self):
        assert Attempt(tries=1, value=2) == Attempt(tries=1, value=2)
        assert Attempt(tries=1) != Attempt(tries=2)


# ---------------------------------------------------------------------------
# AttemptResult
# ---------------------------------------------------------------------------

class TestAttemptResult:
    def test_defaults(self):
        r = AttemptResult()
        assert r.value is None
        assert r.exception is None

    def test_with_value(self):
        r = AttemptResult(value=42)
        assert r.value == 42
        assert r.exception is None

    def test_with_exception(self):
        exc = ValueError("boom")
        r = AttemptResult(exception=exc)
        assert r.value is None
        assert r.exception is exc


# ---------------------------------------------------------------------------
# RetryState
# ---------------------------------------------------------------------------

class TestRetryState:
    def test_defaults(self):
        state = RetryState()
        assert callable(state.target)
        # default target lambda returns None
        assert state.target() is None
        assert state.args == ()
        assert state.kwargs == {}
        assert state.tries == 0
        assert state.start_time == 0.0
        assert state.elapsed == 0.0
        assert state.idle_for == 0.0
        assert state.outcome is None

    def test_post_init_creates_lock(self):
        state = RetryState()
        assert isinstance(state._lock, threading.local)
        assert isinstance(state._lock.lock, type(threading.Lock()))

    def test_kwargs_and_args_default_factories_independent(self):
        s1 = RetryState()
        s2 = RetryState()
        s1.kwargs["a"] = 1
        assert s2.kwargs == {}

    def test_custom_values(self):
        def target(x, y=0):
            return x + y

        outcome = Attempt(tries=1, value=5)
        state = RetryState(
            target=target,
            args=(1,),
            kwargs={"y": 2},
            tries=3,
            start_time=100.0,
            elapsed=2.0,
            idle_for=0.5,
            outcome=outcome,
        )
        assert state.target is target
        assert state.args == (1,)
        assert state.kwargs == {"y": 2}
        assert state.tries == 3
        assert state.start_time == 100.0
        assert state.elapsed == 2.0
        assert state.idle_for == 0.5
        assert state.outcome is outcome

    def test_statistics(self):
        state = RetryState(
            tries=7,
            start_time=123.5,
            elapsed=1.25,
            idle_for=0.75,
        )
        assert state.statistics == {
            "start_time": 123.5,
            "attempt_number": 7,
            "idle_for": 0.75,
            "elapsed": 1.25,
        }

    def test_to_details(self):
        def target():
            return 1

        state = RetryState(
            target=target,
            args=(1, 2),
            kwargs={"a": 3},
            tries=2,
            elapsed=0.5,
        )
        details = state.to_details()
        assert details == {
            "target": target,
            "args": (1, 2),
            "kwargs": {"a": 3},
            "tries": 2,
            "elapsed": 0.5,
        }

    def test_to_details_default_target_is_callable(self):
        state = RetryState()
        d = state.to_details()
        assert callable(d["target"])
        assert d["args"] == ()
        assert d["kwargs"] == {}
        assert d["tries"] == 0
        assert d["elapsed"] == 0.0


# ---------------------------------------------------------------------------
# RetryCallState
# ---------------------------------------------------------------------------

class TestRetryCallState:
    def test_defaults(self):
        cs = RetryCallState()
        assert cs.fn is None
        assert cs.args == ()
        assert cs.kwargs == {}
        assert cs.attempt_number == 1
        assert cs.outcome is None
        assert cs.outcome_timestamp is None
        assert cs.start_time == 0.0
        assert cs.idle_for == 0.0
        assert cs.upcoming_sleep == 0.0

    def test_kwargs_default_independent(self):
        c1 = RetryCallState()
        c2 = RetryCallState()
        c1.kwargs["x"] = 1
        assert c2.kwargs == {}

    def test_custom_values(self):
        def fn():
            return "ok"

        outcome = Attempt(tries=2, value="ok")
        cs = RetryCallState(
            fn=fn,
            args=(1,),
            kwargs={"k": "v"},
            attempt_number=2,
            outcome=outcome,
            outcome_timestamp=123.0,
            start_time=120.0,
            idle_for=0.25,
            upcoming_sleep=2.0,
        )
        assert cs.fn is fn
        assert cs.args == (1,)
        assert cs.kwargs == {"k": "v"}
        assert cs.attempt_number == 2
        assert cs.outcome is outcome
        assert cs.outcome_timestamp == 123.0
        assert cs.start_time == 120.0
        assert cs.idle_for == 0.25
        assert cs.upcoming_sleep == 2.0

    def test_elapsed_zero_when_start_time_zero(self):
        cs = RetryCallState()
        assert cs.elapsed == 0.0

    def test_elapsed_uses_monotonic(self, monkeypatch):
        calls = []

        def fake_monotonic():
            calls.append(True)
            return 200.0

        monkeypatch.setattr(time, "monotonic", fake_monotonic)
        cs = RetryCallState(start_time=150.0)
        assert cs.elapsed == 50.0
        assert len(calls) == 1

    def test_seconds_since_start_matches_elapsed(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 105.0)
        cs = RetryCallState(start_time=100.0)
        assert cs.seconds_since_start == cs.elapsed
        assert cs.seconds_since_start == 5.0

    def test_seconds_since_start_zero(self):
        cs = RetryCallState()
        assert cs.seconds_since_start == 0.0

    def test_statistics(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 110.0)
        cs = RetryCallState(
            attempt_number=4,
            start_time=100.0,
            idle_for=0.5,
        )
        assert cs.statistics == {
            "start_time": 100.0,
            "attempt_number": 4,
            "idle_for": 0.5,
            "elapsed": 10.0,
        }

    def test_statistics_with_zero_start(self):
        cs = RetryCallState(attempt_number=1, idle_for=0.0)
        assert cs.statistics == {
            "start_time": 0.0,
            "attempt_number": 1,
            "idle_for": 0.0,
            "elapsed": 0.0,
        }

    def test_to_details(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 102.0)

        def fn(a, b=0):
            return a + b

        cs = RetryCallState(
            fn=fn,
            args=(1,),
            kwargs={"b": 2},
            attempt_number=3,
            start_time=100.0,
        )
        details = cs.to_details()
        assert details == {
            "target": fn,
            "args": (1,),
            "kwargs": {"b": 2},
            "tries": 3,
            "elapsed": 2.0,
        }

    def test_to_details_default(self):
        cs = RetryCallState()
        assert cs.to_details() == {
            "target": None,
            "args": (),
            "kwargs": {},
            "tries": 1,
            "elapsed": 0.0,
        }
