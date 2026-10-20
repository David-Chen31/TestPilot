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

class TestTryAgain:
    def test_is_exception(self):
        assert issubclass(TryAgain, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(TryAgain):
            raise TryAgain("retry now")

    def test_message(self):
        exc = TryAgain("boom")
        assert str(exc) == "boom"


class TestAttemptTimeoutError:
    def test_is_exception(self):
        assert issubclass(AttemptTimeoutError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(AttemptTimeoutError):
            raise AttemptTimeoutError()

    def test_raise_from_none(self):
        with pytest.raises(AttemptTimeoutError) as exc_info:
            raise AttemptTimeoutError() from None
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__ is True


class TestRetryError:
    def test_attributes_and_message(self):
        cause = ValueError("bad")
        last = Attempt(tries=5, exception=cause, elapsed=1.5)
        err = RetryError(last)
        assert err.last_attempt is last
        assert str(err) == "Retry failed after 5 tries"
        assert err.__cause__ is cause

    def test_no_exception_sets_no_cause(self):
        last = Attempt(tries=3, exception=None)
        err = RetryError(last)
        assert err.last_attempt is last
        assert str(err) == "Retry failed after 3 tries"
        # __cause__ should not be set explicitly; it stays None by default.
        assert err.__cause__ is None

    def test_reraise_raises_cause(self):
        cause = KeyError("k")
        last = Attempt(tries=2, exception=cause)
        err = RetryError(last)
        with pytest.raises(KeyError) as exc_info:
            err.reraise()
        assert exc_info.value is cause
        # raise ... from None -> suppress context
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__ is True

    def test_reraise_without_cause_returns_none(self):
        last = Attempt(tries=1, exception=None)
        err = RetryError(last)
        # No exception to reraise: returns silently.
        assert err.reraise() is None

    def test_is_exception(self):
        assert issubclass(RetryError, Exception)


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
        exc = RuntimeError("x")
        a = Attempt(tries=4, exception=exc, value="v", wait=0.25, elapsed=2.0)
        assert a.tries == 4
        assert a.exception is exc
        assert a.value == "v"
        assert a.wait == 0.25
        assert a.elapsed == 2.0

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

    def test_values(self):
        exc = ValueError("v")
        r = AttemptResult(value=42, exception=exc)
        assert r.value == 42
        assert r.exception is exc

    def test_equality(self):
        assert AttemptResult(value=1) == AttemptResult(value=1)
        assert AttemptResult(value=1) != AttemptResult(value=2)


# ---------------------------------------------------------------------------
# RetryState
# ---------------------------------------------------------------------------

class TestRetryState:
    def test_defaults(self):
        st = RetryState()
        assert st.args == ()
        assert st.kwargs == {}
        assert st.tries == 0
        assert st.start_time == 0.0
        assert st.elapsed == 0.0
        assert st.idle_for == 0.0
        assert st.outcome is None
        # default target is a callable returning None
        assert st.target() is None

    def test_post_init_creates_lock(self):
        st = RetryState()
        assert isinstance(st._lock.lock, type(threading.Lock()))
        # The lock is functional.
        assert st._lock.lock.acquire(blocking=False) is True
        st._lock.lock.release()

    def test_locks_are_per_instance(self):
        a = RetryState()
        b = RetryState()
        assert a._lock is not b._lock
        assert a._lock.lock is not b._lock.lock

    def test_custom_fields(self):
        def target():
            return 1

        outcome = Attempt(tries=2, value=1)
        st = RetryState(
            target=target,
            args=(1, 2),
            kwargs={"a": 3},
            tries=2,
            start_time=10.0,
            elapsed=1.0,
            idle_for=0.5,
            outcome=outcome,
        )
        assert st.target is target
        assert st.args == (1, 2)
        assert st.kwargs == {"a": 3}
        assert st.tries == 2
        assert st.start_time == 10.0
        assert st.elapsed == 1.0
        assert st.idle_for == 0.5
        assert st.outcome is outcome

    def test_kwargs_default_factory_independent(self):
        a = RetryState()
        b = RetryState()
        a.kwargs["x"] = 1
        assert b.kwargs == {}

    def test_statistics(self):
        st = RetryState(tries=3, start_time=5.0, idle_for=0.75, elapsed=2.5)
        assert st.statistics == {
            "start_time": 5.0,
            "attempt_number": 3,
            "idle_for": 0.75,
            "elapsed": 2.5,
        }

    def test_to_details(self):
        def target():
            return None

        st = RetryState(
            target=target,
            args=(1,),
            kwargs={"b": 2},
            tries=3,
            elapsed=2.5,
        )
        assert st.to_details() == {
            "target": target,
            "args": (1,),
            "kwargs": {"b": 2},
            "tries": 3,
            "elapsed": 2.5,
        }


# ---------------------------------------------------------------------------
# RetryCallState
# ---------------------------------------------------------------------------

class TestRetryCallState:
    def test_defaults(self):
        st = RetryCallState()
        assert st.fn is None
        assert st.args == ()
        assert st.kwargs == {}
        assert st.attempt_number == 1
        assert st.outcome is None
        assert st.outcome_timestamp is None
        assert st.start_time == 0.0
        assert st.idle_for == 0.0
        assert st.upcoming_sleep == 0.0

    def test_kwargs_default_factory_independent(self):
        a = RetryCallState()
        b = RetryCallState()
        a.kwargs["x"] = 1
        assert b.kwargs == {}

    def test_custom_fields(self):
        def fn():
            return None

        outcome = Attempt(tries=2)
        st = RetryCallState(
            fn=fn,
            args=(1,),
            kwargs={"k": "v"},
            attempt_number=2,
            outcome=outcome,
            outcome_timestamp=123.0,
            start_time=50.0,
            idle_for=0.3,
            upcoming_sleep=1.0,
        )
        assert st.fn is fn
        assert st.args == (1,)
        assert st.kwargs == {"k": "v"}
        assert st.attempt_number == 2
        assert st.outcome is outcome
        assert st.outcome_timestamp == 123.0
        assert st.start_time == 50.0
        assert st.idle_for == 0.3
        assert st.upcoming_sleep == 1.0

    def test_elapsed_zero_when_start_time_zero(self, monkeypatch):
        # Even if monotonic returns something big, elapsed must be 0.
        monkeypatch.setattr(time, "monotonic", lambda: 99999.0)
        st = RetryCallState(start_time=0.0)
        assert st.elapsed == 0.0

    def test_elapsed_nonzero(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 105.0)
        st = RetryCallState(start_time=100.0)
        assert st.elapsed == pytest.approx(5.0)

    def test_elapsed_negative_start(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 10.0)
        st = RetryCallState(start_time=-5.0)
        assert st.elapsed == pytest.approx(15.0)

    def test_seconds_since_start_equals_elapsed(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 20.0)
        st = RetryCallState(start_time=15.0)
        assert st.seconds_since_start == st.elapsed == pytest.approx(5.0)

    def test_seconds_since_start_zero(self):
        st = RetryCallState(start_time=0.0)
        assert st.seconds_since_start == 0.0

    def test_statistics(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 110.0)
        st = RetryCallState(
            attempt_number=4, start_time=100.0, idle_for=0.5
        )
        assert st.statistics == {
            "start_time": 100.0,
            "attempt_number": 4,
            "idle_for": 0.5,
            "elapsed": pytest.approx(10.0),
        }

    def test_statistics_zero_start(self):
        st = RetryCallState(attempt_number=1, start_time=0.0)
        stats = st.statistics
        assert stats["start_time"] == 0.0
        assert stats["attempt_number"] == 1
        assert stats["idle_for"] == 0.0
        assert stats["elapsed"] == 0.0

    def test_to_details(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 103.0)

        def fn():
            return None

        st = RetryCallState(
            fn=fn,
            args=(1, 2),
            kwargs={"c": 3},
            attempt_number=3,
            start_time=100.0,
        )
        details = st.to_details()
        assert details["target"] is fn
        assert details["args"] == (1, 2)
        assert details["kwargs"] == {"c": 3}
        assert details["tries"] == 3
        assert details["elapsed"] == pytest.approx(3.0)

    def test_to_details_zero_start(self):
        st = RetryCallState()
        details = st.to_details()
        assert details["target"] is None
        assert details["args"] == ()
        assert details["kwargs"] == {}
        assert details["tries"] == 1
        assert details["elapsed"] == 0.0
