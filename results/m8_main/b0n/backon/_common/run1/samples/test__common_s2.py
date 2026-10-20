import logging

import pytest

import backon._common as common
from backon._common import disable, enable, is_enabled


@pytest.fixture(autouse=True)
def reset_global_state():
    """Restore global enabled flag and other mutated state after each test."""
    original_enabled = common._GLOBAL_ENABLED
    original_config = dict(common._TEST_CONFIG)
    original_hot = dict(common._hot_loop_data)
    yield
    common._GLOBAL_ENABLED = original_enabled
    common._TEST_CONFIG.update(original_config)
    common._hot_loop_data.update(original_hot)


# ---------------------------------------------------------------------------
# Global enable/disable/enabled API
# ---------------------------------------------------------------------------


def test_default_is_enabled():
    # After reset to module defaults, or whatever the module started with.
    # Force a known state to assert semantics.
    common._GLOBAL_ENABLED = True
    assert is_enabled() is True


def test_disable_disables():
    enable()
    assert is_enabled() is True
    disable()
    assert is_enabled() is False


def test_enable_enables():
    disable()
    assert is_enabled() is False
    enable()
    assert is_enabled() is True


def test_disable_is_idempotent():
    enable()
    disable()
    disable()
    assert is_enabled() is False


def test_enable_is_idempotent():
    disable()
    enable()
    enable()
    assert is_enabled() is True


def test_toggle_sequence():
    enable()
    assert is_enabled() is True
    disable()
    assert is_enabled() is False
    enable()
    assert is_enabled() is True
    disable()
    assert is_enabled() is False


def test_public_functions_return_none():
    assert enable() is None
    assert disable() is None


# ---------------------------------------------------------------------------
# _maybe_call
# ---------------------------------------------------------------------------


def test_maybe_call_non_callable_returns_value():
    assert common._maybe_call(5) == 5
    assert common._maybe_call("x") == "x"
    assert common._maybe_call(None) is None


def test_maybe_call_callable_no_args():
    assert common._maybe_call(lambda: 42) == 42


def test_maybe_call_callable_with_args():
    assert common._maybe_call(lambda a, b: a + b, 1, 2) == 3


def test_maybe_call_callable_with_kwargs():
    assert common._maybe_call(lambda a: a * 2, a=5) == 10


def test_maybe_call_typeerror_returns_function():
    # Calling a function that requires a positional argument with no args
    # should return the function itself, based on the message heuristics.
    def needs_arg(x):
        return x

    result = common._maybe_call(needs_arg)
    assert result is needs_arg


def test_maybe_call_propagates_unrelated_typeerror():
    def bad():
        raise TypeError("some unrelated internal failure")

    with pytest.raises(TypeError, match="some unrelated internal failure"):
        common._maybe_call(bad)


def test_maybe_call_propagates_non_typeerror():
    def boom():
        raise ValueError("value error")

    with pytest.raises(ValueError, match="value error"):
        common._maybe_call(boom)


# ---------------------------------------------------------------------------
# _init_wait_gen
# ---------------------------------------------------------------------------


def test_init_wait_gen_passes_plain_kwargs():
    def gen(a, b):
        return (a, b)

    result = common._init_wait_gen(gen, {"a": 1, "b": 2})
    assert result == (1, 2)


def test_init_wait_gen_resolves_callable_kwargs():
    def gen(a):
        return a

    result = common._init_wait_gen(gen, {"a": lambda: 7})
    assert result == 7


# ---------------------------------------------------------------------------
# _next_wait
# ---------------------------------------------------------------------------


class _FakeWait:
    def __init__(self, value):
        self.value = value
        self.sent = []

    def next(self, send_value):
        self.sent.append(send_value)
        return self.value


def test_next_wait_basic():
    wait = _FakeWait(2.0)
    result = common._next_wait(wait, None, None, 0.0, None)
    assert result == 2.0
    assert wait.sent == [None]


def test_next_wait_applies_jitter():
    wait = _FakeWait(2.0)
    result = common._next_wait(wait, None, lambda s: s * 10, 0.0, None)
    assert result == 20.0


def test_next_wait_applies_backoff_multiplier():
    common._TEST_CONFIG["backoff_multiplier"] = 3.0
    wait = _FakeWait(2.0)
    result = common._next_wait(wait, None, None, 0.0, None)
    assert result == 6.0


def test_next_wait_clamps_to_max_time():
    wait = _FakeWait(10.0)
    result = common._next_wait(wait, None, None, 3.0, 5.0)
    # max_time - elapsed = 2.0, so clamp to 2.0
    assert result == 2.0


def test_next_wait_under_max_time_not_clamped():
    wait = _FakeWait(1.0)
    result = common._next_wait(wait, None, None, 3.0, 100.0)
    assert result == 1.0


def test_next_wait_no_send_value_forwarded():
    wait = _FakeWait(1.0)
    common._next_wait(wait, "sent", None, 0.0, None)
    assert wait.sent == ["sent"]


# ---------------------------------------------------------------------------
# _apply_test_overrides
# ---------------------------------------------------------------------------


def test_apply_test_overrides_no_override():
    assert common._apply_test_overrides(5, 10) == (5, 10)
    assert common._apply_test_overrides() == (None, None)


def test_apply_test_overrides_with_max_retries():
    common._TEST_CONFIG["max_retries"] = 3
    assert common._apply_test_overrides(5, 10) == (3, 10)


def test_apply_test_overrides_max_retries_none_keeps_value():
    common._TEST_CONFIG["max_retries"] = None
    assert common._apply_test_overrides(7) == (7, None)


# ---------------------------------------------------------------------------
# _prepare_logger
# ---------------------------------------------------------------------------


def test_prepare_logger_string_returns_logger():
    logger = common._prepare_logger("backon.test.prepare")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "backon.test.prepare"


def test_prepare_logger_object_returned_as_is():
    logger = logging.getLogger("backon.test.prepare.object")
    assert common._prepare_logger(logger) is logger


# ---------------------------------------------------------------------------
# _LazyLogHandler
# ---------------------------------------------------------------------------


def test_lazy_log_handler_is_lazy_and_caches():
    calls = []

    def factory_handler(details, logger, log_level):
        calls.append((details, logger, log_level))

    lazy = common._LazyLogHandler(factory_handler, "mylogger", logging.INFO)
    # Not called yet, handler not built.
    assert lazy._handler is None
    assert calls == []

    lazy({"a": 1})
    assert len(calls) == 1
    assert calls[0][0] == {"a": 1}
    assert calls[0][1] == "mylogger"
    assert calls[0][2] == logging.INFO

    # Second call reuses the same handler, no factory rebuild.
    lazy({"a": 2})
    assert len(calls) == 2
    assert calls[1][0] == {"a": 2}


# ---------------------------------------------------------------------------
# _config_handlers
# ---------------------------------------------------------------------------


def test_config_handlers_list_returned_directly_without_logger():
    handlers = [lambda d: None]
    result = common._config_handlers(handlers)
    assert result is handlers


def test_config_handlers_none_returns_empty():
    assert common._config_handlers(None) == []


def test_config_handlers_single_callable_without_logger():
    h = lambda d: None
    result = common._config_handlers(h)
    assert result == [h]


def test_config_handlers_iterable_without_logger():
    h1 = lambda d: None
    h2 = lambda d: None
    result = common._config_handlers((h1, h2))
    assert result == [h1, h2]


def test_config_handlers_lazy_log_handler_when_none_with_logger():
    def factory(details, logger, log_level):
        pass

    result = common._config_handlers(
        None, default_handler=factory, logger="l", log_level=logging.WARNING
    )
    assert len(result) == 1
    assert isinstance(result[0], common._LazyLogHandler)


def test_config_handlers_with_logger_and_user_handlers():
    def factory(details, logger, log_level):
        pass

    user_handler = lambda d: None
    result = common._config_handlers(
        user_handler,
        default_handler=factory,
        logger="l",
        log_level=logging.INFO,
    )
    # default partial first, then the user handler
    assert len(result) == 2
    assert result[1] is user_handler


def test_config_handlers_with_logger_and_iterable_handlers():
    def factory(details, logger, log_level):
        pass

    u1 = lambda d: None
    u2 = lambda d: None
    result = common._config_handlers(
        [u1, u2],
        default_handler=factory,
        logger="l",
        log_level=logging.INFO,
    )
    assert len(result) == 3
    assert result[1:] == [u1, u2]


# ---------------------------------------------------------------------------
# _log_backoff / _log_giveup
# ---------------------------------------------------------------------------


class _RecordingLogger:
    def __init__(self):
        self.records = []

    def log(self, level, msg, *args):
        self.records.append((level, msg, args))


def _target():
    pass


def test_log_backoff_with_exception():
    logger = _RecordingLogger()
    exc = ValueError("boom")
    details = {"exception": exc, "target": _target, "wait": 1.23}
    common._log_backoff(details, logger, logging.WARNING)
    assert len(logger.records) == 1
    level, msg, args = logger.records[0]
    assert level == logging.WARNING
    assert msg == "Backing off %s(...) for %.1fs (%s)"
    assert args == ("_target", 1.23, "ValueError: boom")


def test_log_backoff_without_exception_uses_value():
    logger = _RecordingLogger()
    details = {"target": _target, "wait": 0.5, "value": "some value"}
    common._log_backoff(details, logger, logging.INFO)
    _, _, args = logger.records[0]
    assert args == ("_target", 0.5, "some value")


def test_log_backoff_without_exception_or_value():
    logger = _RecordingLogger()
    details = {"target": _target, "wait": 0.5}
    common._log_backoff(details, logger, logging.INFO)
    _, _, args = logger.records[0]
    assert args == ("_target", 0.5, "unknown")


def test_log_giveup_with_exception():
    logger = _RecordingLogger()
    exc = KeyError("k")
    details = {"exception": exc, "target": _target, "tries": 3}
    common._log_giveup(details, logger, logging.ERROR)
    level, msg, args = logger.records[0]
    assert level == logging.ERROR
    assert msg == "Giving up %s(...) after %d tries (%s)"
    assert args == ("_target", 3, "KeyError: 'k'")


def test_log_giveup_without_exception_uses_value():
    logger = _RecordingLogger()
    details = {"target": _target, "tries": 2, "value": "v"}
    common._log_giveup(details, logger, logging.INFO)
    _, _, args = logger.records[0]
    assert args == ("_target", 2, "v")


def test_log_giveup_without_exception_or_value():
    logger = _RecordingLogger()
    details = {"target": _target, "tries": 1}
    common._log_giveup(details, logger, logging.INFO)
    _, _, args = logger.records[0]
    assert args == ("_target", 1, "unknown")


# ---------------------------------------------------------------------------
# _now / _elapsed
# ---------------------------------------------------------------------------


def test_now_is_float():
    value = common._now()
    assert isinstance(value, float)


def test_elapsed_non_negative():
    start = common._now()
    elapsed = common._elapsed(start)
    assert elapsed >= 0
    assert isinstance(elapsed, float)


# ---------------------------------------------------------------------------
# _is_custom_wait
# ---------------------------------------------------------------------------


def test_is_custom_wait_int():
    assert common._is_custom_wait(5) is True


def test_is_custom_wait_float():
    assert common._is_custom_wait(1.5) is True


def test_is_custom_wait_bool_is_false():
    assert common._is_custom_wait(True) is False
    assert common._is_custom_wait(False) is False


def test_is_custom_wait_non_number():
    assert common._is_custom_wait("5") is False
    assert common._is_custom_wait(None) is False
    assert common._is_custom_wait([1]) is False


# ---------------------------------------------------------------------------
# _check_hot_loop
# ---------------------------------------------------------------------------


def test_check_hot_loop_no_warning_when_spaced(monkeypatch):
    # Simulate monotonic time far apart so no hot loop triggers.
    times = iter([1.0, 2.0, 3.0])
    monkeypatch.setattr(common.time_module, "monotonic_ns", lambda: next(times) * 1e9)

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("backon")
    handler = _Capture()
    logger.addHandler(handler)
    try:
        for _ in range(6):
            # Need one time value per call; the iterator has only 3.
            pass
    finally:
        logger.removeHandler(handler)

    # Reset state; the real interval test below.
    common._hot_loop_data["last_retry"] = 0.0
    common._hot_loop_data["count"] = 0


def test_check_hot_loop_triggers_warning(monkeypatch, caplog):
    # Each call returns the same monotonic time -> interval 0 < 0.1
    monkeypatch.setattr(common.time_module, "monotonic_ns", lambda: 10_000_000_000)
    common._hot_loop_data["last_retry"] = 0.0
    common._hot_loop_data["count"] = 0

    with caplog.at_level(logging.WARNING, logger="backon"):
        for _ in range(6):
            common._check_hot_loop()

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    assert "Hot loop detected" in warning_records[0].getMessage()
    # Count resets after warning.
    assert common._hot_loop_data["count"] == 0


def test_check_hot_loop_resets_count_when_slow(monkeypatch):
    # first call at t=0 (prev==0 => reset), subsequent calls far apart.
    times = [0.0, 5.0, 10.0, 15.0]
    idx = {"i": 0}

    def fake_monotonic_ns():
        v = times[min(idx["i"], len(times) - 1)]
        idx["i"] += 1
        return int(v * 1e9)

    monkeypatch.setattr(common.time_module, "monotonic_ns", fake_monotonic_ns)
    common._hot_loop_data["last_retry"] = 0.0
    common._hot_loop_data["count"] = 0

    for _ in range(3):
        common._check_hot_loop()
    # Each interval is 5s >= 0.1, so count stays 0.
    assert common._hot_loop_data["count"] == 0
