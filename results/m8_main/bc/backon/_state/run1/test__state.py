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


def test_try_again_is_exception():
    exc = TryAgain("msg")
    assert isinstance(exc, Exception)
    assert str(exc) == "msg"


def test_attempt_timeout_error_is_exception():
    exc = AttemptTimeoutError()
    assert isinstance(exc, Exception)
    assert str(exc) == ""


def test_attempt_timeout_error_with_message():
    exc = AttemptTimeoutError("boom")
    assert isinstance(exc, Exception)
    assert str(exc) == "boom"


def test_retry_error_attributes_and_message():
    cause = ValueError("bad")
    attempt = Attempt(tries=3, exception=cause)
    err = RetryError(attempt)

    assert isinstance(err, Exception)
    assert err.last_attempt is attempt
    assert err.__cause__ is cause
    assert str(err) == "Retry failed after 3 tries"


def test_retry_error_without_cause():
    attempt = Attempt(tries=5, exception=None)
    err = RetryError(attempt)

    assert err.last_attempt is attempt
    assert err.__cause__ is None
    assert str(err) == "Retry failed after 5 tries"


def test_retry_error_reraise_with_cause():
    cause = KeyError("k")
    attempt = Attempt(tries=1, exception=cause)
    err = RetryError(attempt)

    with pytest.raises(KeyError) as excinfo:
        err.reraise()

    assert excinfo.value is cause
    # raised ``from None`` -> suppress context
    assert excinfo.value.__suppress_context__ is True


def test_retry_error_reraise_without_cause_returns_none():
    attempt = Attempt(tries=1, exception=None)
    err = RetryError(attempt)

    # No exception should be raised; returns None.
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


def test_attempt_custom_values_and_equality():
    exc = RuntimeError("x")
    a = Attempt(tries=2, exception=exc, value=42, wait=0.5, elapsed=1.25)
    assert a.tries == 2
    assert a.exception is exc
    assert a.value == 42
    assert a.wait == 0.5
    assert a.elapsed == 1.25

    b = Attempt(tries=2, exception=exc, value=42, wait=0.5, elapsed=1.25)
    assert a == b


def test_attempt_result_defaults():
    r = AttemptResult()
    assert r.value is None
    assert r.exception is None


def test_attempt_result_custom_values():
    exc = ValueError("v")
    r = AttemptResult(value="hi", exception=exc)
    assert r.value == "hi"
    assert r.exception is exc
    assert r == AttemptResult(value="hi", exception=exc)


# ---------------------------------------------------------------------------
# RetryState
# ---------------------------------------------------------------------------


def test_retry_state_defaults():
    state = RetryState()
    assert callable(state.target)
    assert state.args == ()
    assert state.kwargs == {}
    assert state.tries == 0
    assert state.start_time == 0.0
    assert state.elapsed == 0.0
    assert state.idle_for == 0.0
    assert state.outcome is None


def test_retry_state_default_target_returns_none():
    state = RetryState()
    assert state.target() is None


def test_retry_state_custom_values():
    def fn(*a, **k):
        return "x"

    outcome = Attempt(tries=1, value="x")
    state = RetryState(
        target=fn,
        args=(1, 2),
        kwargs={"a": 3},
        tries=1,
        start_time=100.0,
        elapsed=2.0,
        idle_for=0.5,
        outcome=outcome,
    )
    assert state.target is fn
    assert state.args == (1, 2)
    assert state.kwargs == {"a": 3}
    assert state.tries == 1
    assert state.start_time == 100.0
    assert state.elapsed == 2.0
    assert state.idle_for == 0.5
    assert state.outcome is outcome


def test_retry_state_post_init_creates_lock_in_main_thread():
    # ``__post_init__`` attaches the lock only to the constructing thread's
    # ``threading.local`` storage; other threads have no ``lock`` attribute.
    state = RetryState()
    assert isinstance(state._lock, threading.local)

    main_lock = state._lock.lock
    assert isinstance(main_lock, type(threading.Lock()))
    assert state._lock.lock is main_lock


def test_retry_state_lock_not_shared_with_other_threads():
    state = RetryState()
    main_lock = state._lock.lock

    captured = {}

    def worker():
        # The lock attribute only exists in the thread that ran __post_init__.
        captured["has_lock"] = hasattr(state._lock, "lock")

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert captured["has_lock"] is False
    # Main-thread lock unchanged.
    assert state._lock.lock is main_lock


def test_retry_state_kwargs_default_is_per_instance():
    s1 = RetryState()
    s2 = RetryState()
    s1.kwargs["a"] = 1
    assert s2.kwargs == {}


def test_retry_state_statistics():
    state = RetryState(
        tries=4,
        start_time=10.0,
        idle_for=0.25,
        elapsed=3.5,
    )
    stats = state.statistics
    assert stats == {
        "start_time": 10.0,
        "attempt_number": 4,
        "idle_for": 0.25,
        "elapsed": 3.5,
    }


def test_retry_state_to_details():
    def fn():
        return 1

    state = RetryState(
        target=fn,
        args=(1,),
        kwargs={"b": 2},
        tries=2,
        elapsed=1.5,
    )
    details = state.to_details()
    assert details == {
        "target": fn,
        "args": (1,),
        "kwargs": {"b": 2},
        "tries": 2,
        "elapsed": 1.5,
    }


# ---------------------------------------------------------------------------
# RetryCallState
# ---------------------------------------------------------------------------


def test_retry_call_state_defaults():
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


def test_retry_call_state_custom_values():
    def fn():
        return 1

    outcome = Attempt(tries=1, value=1)
    state = RetryCallState(
        fn=fn,
        args=(1, 2),
        kwargs={"x": 3},
        attempt_number=5,
        outcome=outcome,
        outcome_timestamp=123.0,
        start_time=50.0,
        idle_for=0.25,
        upcoming_sleep=1.0,
    )
    assert state.fn is fn
    assert state.args == (1, 2)
    assert state.kwargs == {"x": 3}
    assert state.attempt_number == 5
    assert state.outcome is outcome
    assert state.outcome_timestamp == 123.0
    assert state.start_time == 50.0
    assert state.idle_for == 0.25
    assert state.upcoming_sleep == 1.0


def test_retry_call_state_kwargs_default_is_per_instance():
    s1 = RetryCallState()
    s2 = RetryCallState()
    s1.kwargs["a"] = 1
    assert s2.kwargs == {}


def test_retry_call_state_elapsed_zero_start_time():
    state = RetryCallState(start_time=0.0)
    assert state.elapsed == 0.0


def test_retry_call_state_elapsed_uses_monotonic(monkeypatch):
    from backon import _state

    monkeypatch.setattr(_state.time_module, "monotonic", lambda: 100.0)
    state = RetryCallState(start_time=40.0)
    assert state.elapsed == 60.0
    assert state.seconds_since_start == 60.0


def test_retry_call_state_seconds_since_start_zero(monkeypatch):
    from backon import _state

    monkeypatch.setattr(_state.time_module, "monotonic", lambda: 999.0)
    state = RetryCallState(start_time=0.0)
    assert state.seconds_since_start == 0.0


def test_retry_call_state_statistics(monkeypatch):
    from backon import _state

    monkeypatch.setattr(_state.time_module, "monotonic", lambda: 100.0)
    state = RetryCallState(
        attempt_number=3,
        start_time=90.0,
        idle_for=0.5,
    )
    stats = state.statistics
    assert stats == {
        "start_time": 90.0,
        "attempt_number": 3,
        "idle_for": 0.5,
        "elapsed": 10.0,
    }


def test_retry_call_state_to_details(monkeypatch):
    from backon import _state

    monkeypatch.setattr(_state.time_module, "monotonic", lambda: 20.0)

    def fn():
        return 1

    state = RetryCallState(
        fn=fn,
        args=(1,),
        kwargs={"k": 2},
        attempt_number=4,
        start_time=15.0,
    )
    details = state.to_details()
    assert details == {
        "target": fn,
        "args": (1,),
        "kwargs": {"k": 2},
        "tries": 4,
        "elapsed": 5.0,
    }
