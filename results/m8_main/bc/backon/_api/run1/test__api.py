import asyncio
import logging
from unittest import mock

import pytest

from backon._retry import _api
from backon._retry._api import retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fast_sleep(sleeps):
    def _sleep(seconds):
        sleeps.append(seconds)

    return _sleep


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# retry: dispatch
# ---------------------------------------------------------------------------


def test_retry_dispatches_sync(monkeypatch):
    sync_mock = mock.Mock(return_value="sync-result")
    async_mock = mock.Mock()
    monkeypatch.setattr(_api, "_retry_sync", sync_mock)
    monkeypatch.setattr(_api, "_retry_async", async_mock)

    def target():
        return 1

    result = retry(target)

    assert result == "sync-result"
    sync_mock.assert_called_once()
    async_mock.assert_not_called()
    assert sync_mock.call_args.args[0] is target


def test_retry_dispatches_async(monkeypatch):
    async def target():
        return 1

    async_result = object()
    async_mock = mock.Mock(return_value=async_result)
    sync_mock = mock.Mock()
    monkeypatch.setattr(_api, "_retry_async", async_mock)
    monkeypatch.setattr(_api, "_retry_sync", sync_mock)

    result = retry(target)

    assert result is async_result
    async_mock.assert_called_once()
    sync_mock.assert_not_called()
    assert async_mock.call_args.args[0] is target


def test_retry_passes_through_kwargs_to_sync(monkeypatch):
    sync_mock = mock.Mock(return_value=None)
    monkeypatch.setattr(_api, "_retry_sync", sync_mock)

    def target():
        return 1

    handler = lambda details: None
    cond = object()
    stop = object()
    limiter = object()

    retry(
        target,
        max_tries=3,
        max_time=10.0,
        raise_on_giveup=False,
        logger="mylogger",
        backoff_log_level=logging.WARNING,
        giveup_log_level=logging.CRITICAL,
        on_success=handler,
        on_backoff=handler,
        on_giveup=handler,
        on_attempt=handler,
        before_sleep=handler,
        before=handler,
        after=handler,
        retry_error_callback=handler,
        condition=cond,
        stop=stop,
        rate_limit=limiter,
        attempt_timeout=2.0,
        key1=1,
        key2="two",
    )

    kwargs = sync_mock.call_args.kwargs
    assert kwargs["max_tries"] == 3
    assert kwargs["max_time"] == 10.0
    assert kwargs["raise_on_giveup"] is False
    assert kwargs["logger"] == "mylogger"
    assert kwargs["backoff_log_level"] == logging.WARNING
    assert kwargs["giveup_log_level"] == logging.CRITICAL
    assert kwargs["condition"] is cond
    assert kwargs["stop"] is stop
    assert kwargs["rate_limit"] is limiter
    assert kwargs["attempt_timeout"] == 2.0
    assert kwargs["wait_gen_kwargs"] == {"key1": 1, "key2": "two"}


def test_retry_passes_through_kwargs_to_async(monkeypatch):
    async_mock = mock.Mock(return_value=None)
    monkeypatch.setattr(_api, "_retry_async", async_mock)

    async def target():
        return 1

    retry(target, max_tries=5, foo="bar")

    kwargs = async_mock.call_args.kwargs
    assert kwargs["max_tries"] == 5
    assert kwargs["wait_gen_kwargs"] == {"foo": "bar"}


def test_retry_default_wait_gen_is_expo():
    sleeps = []

    def failing():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        retry(
            failing,
            exception=ValueError,
            sleep=_fast_sleep(sleeps),
            max_tries=2,
            jitter=None,
        )
    assert len(sleeps) == 1
    # expo default base=1 -> first retry wait is 1.0 (no jitter)
    assert sleeps[0] == 1.0


# ---------------------------------------------------------------------------
# retry: sync behaviour
# ---------------------------------------------------------------------------


def test_sync_success_first_try():
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        return "ok"

    assert retry(target) == "ok"
    assert calls["n"] == 1


def test_sync_retries_until_success():
    calls = {"n": 0}
    sleeps = []

    def target():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return "done"

    result = retry(
        target,
        exception=ValueError,
        max_tries=5,
        sleep=_fast_sleep(sleeps),
        jitter=None,
    )
    assert result == "done"
    assert calls["n"] == 3
    assert len(sleeps) == 2


def test_sync_gives_up_and_raises():
    calls = {"n": 0}
    sleeps = []

    def target():
        calls["n"] += 1
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        retry(
            target,
            exception=RuntimeError,
            max_tries=3,
            sleep=_fast_sleep(sleeps),
            jitter=None,
        )
    assert calls["n"] == 3
    assert len(sleeps) == 2


def test_sync_raise_on_giveup_false_returns_none():
    def target():
        raise RuntimeError("last")

    result = retry(
        target,
        exception=RuntimeError,
        max_tries=1,
        raise_on_giveup=False,
        sleep=lambda s: None,
    )
    # Fast path with raise_on_giveup=False and no callback yields None.
    assert result is None


def test_sync_retry_error_callback_used():
    def target():
        raise RuntimeError("cb")

    result = retry(
        target,
        exception=RuntimeError,
        max_tries=2,
        raise_on_giveup=False,
        retry_error_callback=lambda details: "fallback",
        sleep=lambda s: None,
    )
    assert result == "fallback"


def test_sync_predicate_based_condition():
    calls = {"n": 0}
    sleeps = []

    def target():
        calls["n"] += 1
        return calls["n"] < 3

    # predicate=bool: retry while predicate(value) is truthy.
    # Attempt 1 -> True -> retry; attempt 2 -> True -> retry; attempt 3 -> False -> stop.
    result = retry(
        target,
        predicate=bool,
        max_tries=5,
        sleep=_fast_sleep(sleeps),
        jitter=None,
    )
    assert result is False
    assert calls["n"] == 3
    assert len(sleeps) == 2


def test_sync_handlers_called():
    events = []

    calls = {"n": 0}

    def target():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("x")
        return "ok"

    retry(
        target,
        exception=ValueError,
        max_tries=3,
        sleep=lambda s: None,
        on_attempt=lambda d: events.append(("attempt", d["tries"])),
        on_success=lambda d: events.append(("success", d["tries"])),
        on_backoff=lambda d: events.append(("backoff", d["tries"])),
    )
    assert ("success", 2) in events
    assert any(e[0] == "attempt" for e in events)
    assert any(e[0] == "backoff" for e in events)


def test_sync_before_after_handlers():
    order = []

    def before(details):
        order.append("before")

    def after(details):
        order.append("after")

    def target():
        return 1

    retry(target, before=before, after=after)
    assert order == ["before", "after"]


def test_sync_giveup_predicate_honoured():
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        raise ValueError("stop-me")

    with pytest.raises(ValueError):
        retry(
            target,
            exception=ValueError,
            giveup=lambda e: True,
            max_tries=10,
            sleep=lambda s: None,
        )
    assert calls["n"] == 1


def test_sync_custom_logger_used(caplog):
    caplog.set_level(logging.INFO)

    def target():
        raise ValueError("logged")

    with pytest.raises(ValueError):
        retry(
            target,
            exception=ValueError,
            max_tries=2,
            sleep=lambda s: None,
            logger="backon.test.logger",
        )
    assert any("Backing off" in r.message for r in caplog.records)


def test_sync_custom_sleep_used():
    sleeps = []

    def target():
        raise ValueError("x")

    with pytest.raises(ValueError):
        retry(
            target,
            exception=ValueError,
            max_tries=3,
            sleep=_fast_sleep(sleeps),
            jitter=None,
        )
    assert len(sleeps) == 2


def test_sync_condition_and_stop_override_defaults():
    sleeps = []
    cond_calls = []
    stop_calls = []

    def target():
        raise ValueError("v")

    def condition(state):
        cond_calls.append(state)
        return True

    def stop(state):
        stop_calls.append(state)
        return len(stop_calls) >= 2

    result = retry(
        target,
        condition=condition,
        stop=stop,
        raise_on_giveup=False,
        sleep=_fast_sleep(sleeps),
    )
    # Fast path returns None when giving up without a callback and raise_on_giveup=False.
    assert result is None
    assert cond_calls
    assert len(stop_calls) >= 1


def test_sync_wait_gen_kwargs_used():
    sleeps = []

    def target():
        raise ValueError("x")

    with pytest.raises(ValueError):
        retry(
            target,
            exception=ValueError,
            max_tries=3,
            sleep=_fast_sleep(sleeps),
            jitter=None,
            base=2,
            factor=2,
        )
    # expo with base=2, factor=2 -> first wait 2.0
    assert sleeps[0] == 2.0


# ---------------------------------------------------------------------------
# retry: disabled
# ---------------------------------------------------------------------------


def test_retry_disabled_calls_target_directly(monkeypatch):
    # `_retry_sync` checks is_enabled() itself and calls the target directly.
    # It is still invoked by `retry`, so rather than prohibiting the call we
    # verify the target gets called directly with no retry logic engaged.
    calls = {"n": 0}

    def target(*args, **kwargs):
        calls["n"] += 1
        return ("direct", args, kwargs)

    monkeypatch.setattr(_api, "is_enabled", lambda: False)

    result = retry(target, max_tries=5)
    assert calls["n"] == 1
    assert result[0] == "direct"
    assert result[1] == ()
    assert result[2] == {}


def test_retry_disabled_async_calls_target_directly(monkeypatch):
    monkeypatch.setattr(_api, "is_enabled", lambda: False)

    async def target():
        return "direct-async"

    coro = retry(target, max_tries=5)
    result = _run(coro)
    assert result == "direct-async"


# ---------------------------------------------------------------------------
# retry: async behaviour
# ---------------------------------------------------------------------------


def test_async_success_first_try():
    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        return "ok"

    assert _run(retry(target)) == "ok"
    assert calls["n"] == 1


def test_async_retries_until_success():
    calls = {"n": 0}
    sleeps = []

    async def target():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return "done"

    async def main():
        async def sleepy(seconds):
            sleeps.append(seconds)

        return await retry(
            target,
            exception=ValueError,
            max_tries=5,
            sleep=sleepy,
            jitter=None,
        )

    assert _run(main()) == "done"
    assert calls["n"] == 3
    assert len(sleeps) == 2


def test_async_gives_up_and_raises():
    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        raise RuntimeError("nope")

    async def main():
        async def sleepy(seconds):
            return None

        with pytest.raises(RuntimeError, match="nope"):
            await retry(
                target,
                exception=RuntimeError,
                max_tries=3,
                sleep=sleepy,
                jitter=None,
            )

    _run(main())
    assert calls["n"] == 3


def test_async_raise_on_giveup_false_returns_none():
    async def target():
        raise RuntimeError("last")

    async def main():
        async def sleepy(seconds):
            return None

        return await retry(
            target,
            exception=RuntimeError,
            max_tries=1,
            raise_on_giveup=False,
            sleep=sleepy,
        )

    result = _run(main())
    # Fast path with raise_on_giveup=False and no callback yields None.
    assert result is None


def test_async_retry_error_callback():
    async def target():
        raise RuntimeError("cb")

    async def main():
        async def sleepy(seconds):
            return None

        return await retry(
            target,
            exception=RuntimeError,
            max_tries=2,
            raise_on_giveup=False,
            retry_error_callback=lambda details: "fallback",
            sleep=sleepy,
        )

    assert _run(main()) == "fallback"


def test_async_handlers_called():
    events = []
    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("x")
        return "ok"

    async def main():
        async def sleepy(seconds):
            return None

        return await retry(
            target,
            exception=ValueError,
            max_tries=3,
            sleep=sleepy,
            jitter=None,
            on_attempt=lambda d: events.append(("attempt", d["tries"])),
            on_success=lambda d: events.append(("success", d["tries"])),
            on_backoff=lambda d: events.append(("backoff", d["tries"])),
        )

    assert _run(main()) == "ok"
    assert ("success", 2) in events
    assert any(e[0] == "attempt" for e in events)
    assert any(e[0] == "backoff" for e in events)


def test_async_condition_and_stop_override():
    stop_calls = []

    async def target():
        raise ValueError("v")

    async def main():
        async def sleepy(seconds):
            return None

        def condition(state):
            return True

        def stop(state):
            stop_calls.append(state)
            return len(stop_calls) >= 2

        return await retry(
            target,
            condition=condition,
            stop=stop,
            raise_on_giveup=False,
            sleep=sleepy,
        )

    result = _run(main())
    # Fast path returns None when giving up without a callback and raise_on_giveup=False.
    assert result is None
    assert len(stop_calls) >= 1


def test_async_wait_gen_kwargs_used():
    sleeps = []

    async def target():
        raise ValueError("x")

    async def main():
        async def sleepy(seconds):
            sleeps.append(seconds)

        with pytest.raises(ValueError):
            await retry(
                target,
                exception=ValueError,
                max_tries=3,
                sleep=sleepy,
                jitter=None,
                base=1,
                factor=1,
            )

    _run(main())
    assert len(sleeps) >= 1
    # With base=1, factor=1 the expo generator yields a constant 1.0.
    assert sleeps[0] == 1.0


# ---------------------------------------------------------------------------
# retry: rate limit / attempt timeout plumbing (smoke)
# ---------------------------------------------------------------------------


def test_sync_rate_limit_passed():
    from backon._rate_limiter import RateLimiter

    limiter = RateLimiter(max_calls=10, period=1.0)
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        return 1

    assert retry(target, rate_limit=limiter) == 1
    assert calls["n"] == 1


def test_sync_attempt_timeout_passed():
    def target():
        return 42

    assert retry(target, attempt_timeout=1.0) == 42


# ---------------------------------------------------------------------------
# retry: logger variations
# ---------------------------------------------------------------------------


def test_retry_accepts_logger_object():
    logger = logging.getLogger("backon.test.object")
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("x")
        return "ok"

    result = retry(
        target,
        exception=ValueError,
        max_tries=2,
        sleep=lambda s: None,
        jitter=None,
        logger=logger,
    )
    assert result == "ok"


def test_retry_accepts_no_logger_override():
    # Passing an explicit on_backoff should still work with default logger.
    events = []

    def target():
        raise ValueError("x")

    with pytest.raises(ValueError):
        retry(
            target,
            exception=ValueError,
            max_tries=2,
            on_backoff=lambda d: events.append(d),
            sleep=lambda s: None,
        )
    assert len(events) == 1


# ---------------------------------------------------------------------------
# _retry_sync / _retry_async: wait_gen_kwargs default handling
# ---------------------------------------------------------------------------


def test_retry_sync_wait_gen_kwargs_none_defaults_to_empty_dict(monkeypatch):
    """Calling _retry_sync directly with wait_gen_kwargs=None converts it to {}."""
    captured = {}

    def fake_inner(target, wait_gen, **kwargs):
        captured.update(kwargs)
        return "done"

    monkeypatch.setattr(_api, "_retry_fast_sync_inner", fake_inner)
    # Ensure retries enabled so we reach the inner dispatch.
    monkeypatch.setattr(_api, "is_enabled", lambda: True)

    def target():
        return 1

    result = _api._retry_sync(target, _api.expo, wait_gen_kwargs=None)

    assert result == "done"
    assert captured["wait_gen_kwargs"] == {}
    assert isinstance(captured["wait_gen_kwargs"], dict)


def test_retry_async_wait_gen_kwargs_none_defaults_to_empty_dict(monkeypatch):
    """Calling _retry_async directly with wait_gen_kwargs=None converts it to {}."""
    captured = {}

    async def fake_inner(target, wait_gen, **kwargs):
        captured.update(kwargs)
        return "done"

    monkeypatch.setattr(_api, "_retry_fast_async_inner", fake_inner)
    monkeypatch.setattr(_api, "is_enabled", lambda: True)

    async def target():
        return 1

    result = _run(_api._retry_async(target, _api.expo, wait_gen_kwargs=None))

    assert result == "done"
    assert captured["wait_gen_kwargs"] == {}
    assert isinstance(captured["wait_gen_kwargs"], dict)


def test_retry_sync_explicit_wait_gen_kwargs_preserved(monkeypatch):
    """An explicitly provided wait_gen_kwargs dict is passed through unchanged."""
    captured = {}

    def fake_inner(target, wait_gen, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(_api, "_retry_fast_sync_inner", fake_inner)
    monkeypatch.setattr(_api, "is_enabled", lambda: True)

    def target():
        return 1

    supplied = {"base": 3}
    result = _api._retry_sync(target, _api.expo, wait_gen_kwargs=supplied)

    assert result == "ok"
    assert captured["wait_gen_kwargs"] is supplied


def test_retry_async_explicit_wait_gen_kwargs_preserved(monkeypatch):
    captured = {}

    async def fake_inner(target, wait_gen, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(_api, "_retry_fast_async_inner", fake_inner)
    monkeypatch.setattr(_api, "is_enabled", lambda: True)

    async def target():
        return 1

    supplied = {"base": 4}
    result = _run(_api._retry_async(target, _api.expo, wait_gen_kwargs=supplied))

    assert result == "ok"
    assert captured["wait_gen_kwargs"] is supplied
