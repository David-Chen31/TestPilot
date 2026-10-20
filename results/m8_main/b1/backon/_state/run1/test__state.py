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
        exc = AttemptTimeoutError("too slow")
        assert isinstance(exc, Exception)
        assert str(exc) == "too slow"

    def test_attempt_timeout_error_raised(self):
        with pytest.raises(AttemptTimeoutError):
            raise AttemptTimeoutError()


class TestRetryError:
    def test_message_and_last_attempt(self):
        cause = ValueError("original")
        attempt = Attempt(tries=3, exception=cause)
        err = RetryError(attempt)
        assert isinstance(err, Exception)
        assert err.last_attempt is attempt
        assert str(err) == "Retry failed after 3 tries"
        assert err.__cause__ is cause

    def test_no_cause(self):
        attempt = Attempt(tries=0)
        err = RetryError(attempt)
        assert err.last_attempt is attempt
        assert str(err) == "Retry failed after 0 tries"
        assert err.__cause__ is None

    def test_reraise_with_cause(self):
        cause = RuntimeError("oh no")
        attempt = Attempt(tries=2, exception=cause)
        err = RetryError(attempt)
        with pytest.raises(RuntimeError) as info:
            err.reraise()
        assert info.value is cause

    def test_reraise_without_cause_does_nothing(self):
        attempt = Attempt(tries=1)
        err = RetryError(attempt)
        # No exception raised
        assert err.reraise() is None

    def test_reraise_does_not_chain(self):
        cause = KeyError("k")
        err = RetryError(Attempt(tries=1, exception=cause))
        try:
            err.reraise()
        except KeyError as e:
            assert e.__context__ is None or e.__suppress_context__
        else:
            pytest.fail("reraise did not raise")


class TestAttempt:
    def test_defaults(self):
        a = Attempt()
        assert a.tries == 0
        assert a.exception is None
        assert a.value is None
        assert a.wait == 0.0
        assert a.elapsed == 0.0

    def test_custom_values(self):
        exc = ValueError("x")
        a = Attempt(tries=5, exception=exc, value=10, wait=1.5, elapsed=2.5)
        assert a.tries == 5
        assert a.exception is exc
        assert a.value == 10
        assert a.wait == 1.5
        assert a.elapsed == 2.5


class TestAttemptResult:
    def test_defaults(self):
        r = AttemptResult()
        assert r.value is None
        assert r.exception is None

    def test_custom(self):
        exc = ValueError("bad")
        r = AttemptResult(value=42, exception=exc)
        assert r.value == 42
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
        assert state._lock.lock is not None

    def test_default_target_is_callable(self):
        state = RetryState()
        assert callable(state.target)
        assert state.target() is None

    def test_default_kwargs_are_independent(self):
        s1 = RetryState()
        s2 = RetryState()
        s1.kwargs["a"] = 1
        assert s2.kwargs == {}

    def test_custom_values(self):
        outcome = Attempt(tries=1)
        state = RetryState(
            target="foo",
            args=(1, 2),
            kwargs={"x": 1},
            tries=3,
            start_time=100.0,
            elapsed=4.0,
            idle_for=2.0,
            outcome=outcome,
        )
        assert state.target == "foo"
        assert state.args == (1, 2)
        assert state.kwargs == {"x": 1}
        assert state.tries == 3
        assert state.start_time == 100.0
        assert state.elapsed == 4.0
        assert state.idle_for == 2.0
        assert state.outcome is outcome

    def test_lock_is_created(self):
        state = RetryState()
        assert isinstance(state._lock.lock, type(threading.Lock()))

    def test_lock_is_per_instance(self):
        s1 = RetryState()
        s2 = RetryState()
        assert s1._lock.lock is not s2._lock.lock

    def test_statistics(self):
        state = RetryState(tries=4, start_time=10.0, elapsed=5.0, idle_for=1.5)
        assert state.statistics == {
            "start_time": 10.0,
            "attempt_number": 4,
            "idle_for": 1.5,
            "elapsed": 5.0,
        }

    def test_to_details(self):
        def target():
            pass

        state = RetryState(
            target=target,
            args=(1,),
            kwargs={"k": "v"},
            tries=2,
            elapsed=3.0,
        )
        details = state.to_details()
        assert details["target"] is target
        assert details["args"] == (1,)
        assert details["kwargs"] == {"k": "v"}
        assert details["tries"] == 2
        assert details["elapsed"] == 3.0


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

    def test_default_kwargs_independent(self):
        s1 = RetryCallState()
        s2 = RetryCallState()
        s1.kwargs["z"] = 1
        assert s2.kwargs == {}

    def test_custom_values(self):
        def fn():
            pass

        outcome = Attempt(tries=1)
        state = RetryCallState(
            fn=fn,
            args=(1, 2),
            kwargs={"a": 1},
            attempt_number=3,
            outcome=outcome,
            outcome_timestamp=50.0,
            start_time=100.0,
            idle_for=1.0,
            upcoming_sleep=2.0,
        )
        assert state.fn is fn
        assert state.args == (1, 2)
        assert state.kwargs == {"a": 1}
        assert state.attempt_number == 3
        assert state.outcome is outcome
        assert state.outcome_timestamp == 50.0
        assert state.start_time == 100.0
        assert state.idle_for == 1.0
        assert state.upcoming_sleep == 2.0

    def test_elapsed_zero_when_start_time_zero(self):
        state = RetryCallState()
        assert state.elapsed == 0.0
        assert state.seconds_since_start == 0.0

    def test_elapsed_uses_monotonic(self, monkeypatch):
        fake_now = 1000.0
        monkeypatch.setattr(time, "monotonic", lambda: fake_now)
        state = RetryCallState(start_time=950.0)
        assert state.elapsed == 50.0
        assert state.seconds_since_start == 50.0

    def test_elapsed_advances(self, monkeypatch):
        current = {"now": 100.0}
        monkeypatch.setattr(time, "monotonic", lambda: current["now"])
        state = RetryCallState(start_time=90.0)
        assert state.elapsed == 10.0
        current["now"] = 95.0
        assert state.elapsed == 5.0

    def test_statistics(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 110.0)
        state = RetryCallState(
            attempt_number=2,
            start_time=100.0,
            idle_for=0.5,
        )
        assert state.statistics == {
            "start_time": 100.0,
            "attempt_number": 2,
            "idle_for": 0.5,
            "elapsed": 10.0,
        }

    def test_statistics_zero_start_time(self):
        state = RetryCallState(attempt_number=1)
        stats = state.statistics
        assert stats["start_time"] == 0.0
        assert stats["attempt_number"] == 1
        assert stats["idle_for"] == 0.0
        assert stats["elapsed"] == 0.0

    def test_to_details(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 105.0)

        def fn():
            pass

        state = RetryCallState(
            fn=fn,
            args=(1,),
            kwargs={"b": 2},
            attempt_number=4,
            start_time=100.0,
        )
        details = state.to_details()
        assert details["target"] is fn
        assert details["args"] == (1,)
        assert details["kwargs"] == {"b": 2}
        assert details["tries"] == 4
        assert details["elapsed"] == 5.0
