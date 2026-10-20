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
# Exception classes
# ---------------------------------------------------------------------------

def test_try_again_is_exception():
    assert issubclass(TryAgain, Exception)
    with pytest.raises(TryAgain):
        raise TryAgain("retry me")


def test_attempt_timeout_error_is_exception():
    assert issubclass(AttemptTimeoutError, Exception)
    with pytest.raises(AttemptTimeoutError):
        raise AttemptTimeoutError()


def test_retry_error_with_cause():
    cause = ValueError("boom")
    attempt = Attempt(tries=3, exception=cause)
    err = RetryError(attempt)

    assert err.last_attempt is attempt
    assert isinstance(err, Exception)
    assert str(err) == "Retry failed after 3 tries"
    assert err.__cause__ is cause


def test_retry_error_without_cause():
    attempt = Attempt(tries=5, exception=None, value="ok")
    err = RetryError(attempt)

    assert err.last_attempt is attempt
    assert str(err) == "Retry failed after 5 tries"
    assert err.__cause__ is None


def test_retry_error_reraise_raises_cause():
    cause = RuntimeError("original")
    attempt = Attempt(tries=2, exception=cause)
    err = RetryError(attempt)

    with pytest.raises(RuntimeError) as exc_info:
        err.reraise()

    assert exc_info.value is cause
    # raised "from None" -> no implicit context chain set via __cause__
    assert exc_info.value.__cause__ is None


def test_retry_error_reraise_without_cause_does_nothing():
    attempt = Attempt(tries=1, exception=None)
    err = RetryError(attempt)

    # No exception should be raised.
    assert err.reraise() is None


# ---------------------------------------------------------------------------
# Attempt / AttemptResult dataclasses
# ---------------------------------------------------------------------------

def test_attempt_defaults():
    a = Attempt()
    assert a.tries == 0
    assert a.exception is None
    assert a.value is None
    assert a.wait == 0.0
    assert a.elapsed == 0.0


def test_attempt_custom_values():
    exc = KeyError("x")
    a = Attempt(tries=4, exception=exc, value="v", wait=1.5, elapsed=2.5)
    assert a.tries == 4
    assert a.exception is exc
    assert a.value == "v"
    assert a.wait == 1.5
    assert a.elapsed == 2.5


def test_attempt_is_dataclass_like():
    a = Attempt(tries=1, value=10)
    b = Attempt(tries=1, value=10)
    assert a == b
    c = Attempt(tries=2, value=10)
    assert a != c


def test_attempt_result_defaults():
    r = AttemptResult()
    assert r.value is None
    assert r.exception is None


def test_attempt_result_custom():
    exc = ValueError("e")
    r = AttemptResult(value=42, exception=exc)
    assert r.value == 42
    assert r.exception is exc


# ---------------------------------------------------------------------------
# RetryState
# ---------------------------------------------------------------------------

def test_retry_state_defaults():
    s = RetryState()
    assert s.args == ()
    assert s.kwargs == {}
    assert s.tries == 0
    assert s.start_time == 0.0
    assert s.elapsed == 0.0
    assert s.idle_for == 0.0
    assert s.outcome is None
    # default target is a callable returning None
    assert callable(s.target)
    assert s.target() is None
    # lock is created per-instance via threading.local
    assert isinstance(s._lock.lock, threading.Lock().__class__)


def test_retry_state_lock_is_per_instance():
    s1 = RetryState()
    s2 = RetryState()
    assert s1._lock is not s2._lock
    assert s1._lock.lock is not s2._lock.lock


def test_retry_state_custom_args_kwargs():
    def fn(a, b, c=0):
        return a + b + c

    s = RetryState(target=fn, args=(1, 2), kwargs={"c": 3})
    assert s.target is fn
    assert s.args == (1, 2)
    assert s.kwargs == {"c": 3}
    assert s.target(*s.args, **s.kwargs) == 6


def test_retry_state_statistics():
    s = RetryState()
    s.start_time = 100.0
    s.tries = 3
    s.idle_for = 0.5
    s.elapsed = 1.25

    stats = s.statistics
    assert stats == {
        "start_time": 100.0,
        "attempt_number": 3,
        "idle_for": 0.5,
        "elapsed": 1.25,
    }


def test_retry_state_to_details():
    def fn():
        pass

    s = RetryState(target=fn, args=(1,), kwargs={"k": "v"})
    s.tries = 2
    s.elapsed = 0.75

    details = s.to_details()
    assert details == {
        "target": fn,
        "args": (1,),
        "kwargs": {"k": "v"},
        "tries": 2,
        "elapsed": 0.75,
    }


def test_retry_state_outcome():
    s = RetryState()
    a = Attempt(tries=1, value=5)
    s.outcome = a
    assert s.outcome is a


# ---------------------------------------------------------------------------
# RetryCallState
# ---------------------------------------------------------------------------

def test_retry_call_state_defaults():
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


def test_retry_call_state_elapsed_zero_when_no_start():
    s = RetryCallState()
    assert s.start_time == 0.0
    assert s.elapsed == 0.0


def test_retry_call_state_elapsed_with_start(monkeypatch):
    fixed_now = 1000.0
    monkeypatch.setattr(time, "monotonic", lambda: fixed_now)

    s = RetryCallState(start_time=990.0)
    assert s.elapsed == 10.0
    assert s.seconds_since_start == 10.0


def test_retry_call_state_elapsed_uses_monotonic(monkeypatch):
    calls = []

    def fake_monotonic():
        calls.append(1)
        return 250.0

    monkeypatch.setattr(time, "monotonic", fake_monotonic)

    s = RetryCallState(start_time=200.0)
    assert s.elapsed == 50.0
    assert len(calls) == 1


def test_retry_call_state_statistics(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 500.0)
    s = RetryCallState(start_time=490.0)
    s.attempt_number = 4
    s.idle_for = 0.25

    stats = s.statistics
    assert stats == {
        "start_time": 490.0,
        "attempt_number": 4,
        "idle_for": 0.25,
        "elapsed": 10.0,
    }


def test_retry_call_state_to_details(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 300.0)

    def fn():
        pass

    s = RetryCallState(fn=fn, args=(1, 2), kwargs={"x": "y"})
    s.start_time = 275.0
    s.attempt_number = 2

    details = s.to_details()
    assert details == {
        "target": fn,
        "args": (1, 2),
        "kwargs": {"x": "y"},
        "tries": 2,
        "elapsed": 25.0,
    }


def test_retry_call_state_custom():
    def fn(a, b):
        return a + b

    s = RetryCallState(fn=fn, args=(2, 3), kwargs={}, attempt_number=7)
    assert s.fn is fn
    assert s.args == (2, 3)
    assert s.attempt_number == 7
    assert s.fn(*s.args) == 5
