from __future__ import annotations

import logging

import pytest

import backon._common as common
from backon._common import (
    _GLOBAL_ENABLED,
    _LazyLogHandler,
    _apply_test_overrides,
    _check_hot_loop,
    _config_handlers,
    _elapsed,
    _hot_loop_data,
    _init_wait_gen,
    _is_custom_wait,
    _log_backoff,
    _log_giveup,
    _maybe_call,
    _next_wait,
    _now,
    _prepare_logger,
    disable,
    enable,
    is_enabled,
)


@pytest.fixture(autouse=True)
def _restore_global_enabled():
    """Ensure the global enabled flag is restored after each test."""
    original = common._GLOBAL_ENABLED
    yield
    common._GLOBAL_ENABLED = original


# ---------------------------------------------------------------------------
# disable / enable / is_enabled
# ---------------------------------------------------------------------------

def test_is_enabled_default_is_true():
    common._GLOBAL_ENABLED = True
    assert is_enabled() is True


def test_disable_sets_false_and_is_a_side_effect():
    common._GLOBAL_ENABLED = True
    assert disable() is None
    assert common._GLOBAL_ENABLED is False
    assert is_enabled() is False


def test_enable_sets_true_and_is_a_side_effect():
    common._GLOBAL_ENABLED = False
    assert enable() is None
    assert common._GLOBAL_ENABLED is True
    assert is_enabled() is True


def test_enable_disable_round_trip():
    disable()
    assert is_enabled() is False
    enable()
    assert is_enabled() is True
    disable()
    assert is_enabled() is False


# ---------------------------------------------------------------------------
# _maybe_call
# ---------------------------------------------------------------------------

def test_maybe_call_calls_callable_with_args():
    assert _maybe_call(lambda a, b: a + b, 1, 2) == 3


def test_maybe_call_calls_callable_with_kwargs():
    assert _maybe_call(lambda *, x: x * 2, x=5) == 10


def test_maybe_call_returns_non_callable_unchanged():
    sentinel = object()
    assert _maybe_call(sentinel) is sentinel
    assert _maybe_call(42) == 42


def test_maybe_call_returns_callable_if_typeerror_about_arguments():
    def needs_arg(x):
        return x

    # Calling needs_arg() raises a TypeError mentioning "required positional argument".
    assert _maybe_call(needs_arg) is needs_arg


def test_maybe_call_repropagates_unrelated_typeerror():
    def raises_typeerror():
        raise TypeError("some other error")

    with pytest.raises(TypeError, match="some other error"):
        _maybe_call(raises_typeerror)


def test_maybe_call_propagates_valueerror():
    def raises_valueerror():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        _maybe_call(raises_valueerror)


# ---------------------------------------------------------------------------
# _init_wait_gen
# ---------------------------------------------------------------------------

def test_init_wait_gen_passes_kwargs_and_maybe_calls():
    def wait_gen(a, b):
        return (a, b)

    result = _init_wait_gen(wait_gen, {"a": 1, "b": lambda: 2})
    assert result == (1, 2)


# ---------------------------------------------------------------------------
# _next_wait
# ---------------------------------------------------------------------------

class _FakeWait:
    def __init__(self, values):
        self.values = list(values)
        self.sent = []

    def next(self, value):
        self.sent.append(value)
        return self.values.pop(0)


@pytest.fixture()
def _reset_test_config():
    original = dict(common._TEST_CONFIG)
    yield
    common._TEST_CONFIG.clear()
    common._TEST_CONFIG.update(original)


def test_next_wait_returns_value_without_jitter(_reset_test_config):
    common._TEST_CONFIG["backoff_multiplier"] = 1.0
    wait = _FakeWait([3.0])
    assert _next_wait(wait, None, None, 0.0, None) == 3.0
    assert wait.sent == [None]


def test_next_wait_applies_jitter(_reset_test_config):
    common._TEST_CONFIG["backoff_multiplier"] = 1.0
    wait = _FakeWait([2.0])
    result = _next_wait(wait, "s", lambda v: v * 10, 0.0, None)
    assert result == 20.0
    assert wait.sent == ["s"]


def test_next_wait_applies_multiplier(_reset_test_config):
    common._TEST_CONFIG["backoff_multiplier"] = 0.5
    wait = _FakeWait([4.0])
    assert _next_wait(wait, None, None, 0.0, None) == 2.0


def test_next_wait_caps_at_max_time_minus_elapsed(_reset_test_config):
    common._TEST_CONFIG["backoff_multiplier"] = 1.0
    wait = _FakeWait([10.0])
    assert _next_wait(wait, None, None, 3.0, 5.0) == 2.0


def test_next_wait_does_not_cap_below_current_value(_reset_test_config):
    common._TEST_CONFIG["backoff_multiplier"] = 1.0
    wait = _FakeWait([1.0])
    # max_time - elapsed = 100, so min(1.0, 100) == 1.0
    assert _next_wait(wait, None, None, 5.0, 105.0) == 1.0


# ---------------------------------------------------------------------------
# _apply_test_overrides
# ---------------------------------------------------------------------------

def test_apply_test_overrides_no_override(_reset_test_config):
    common._TEST_CONFIG["max_retries"] = None
    assert _apply_test_overrides(max_tries=3, max_time=10) == (3, 10)


def test_apply_test_overrides_with_override(_reset_test_config):
    common._TEST_CONFIG["max_retries"] = 7
    assert _apply_test_overrides(max_tries=3, max_time=10) == (7, 10)
    assert _apply_test_overrides() == (7, None)


# ---------------------------------------------------------------------------
# _prepare_logger
# ---------------------------------------------------------------------------

def test_prepare_logger_from_string():
    lg = _prepare_logger("some.logger.name")
    assert isinstance(lg, logging.Logger)
    assert lg.name == "some.logger.name"


def test_prepare_logger_passthrough():
    lg = logging.getLogger("unchanged")
    assert _prepare_logger(lg) is lg


# ---------------------------------------------------------------------------
# _LazyLogHandler
# ---------------------------------------------------------------------------

def test_lazy_log_handler_lazy_and_caches():
    calls = []

    def handler(details, logger, log_level):
        calls.append((details, logger, log_level))

    lazy = _LazyLogHandler(handler, "mylogger", logging.INFO)
    assert lazy._handler is None
    lazy({"a": 1})
    first = lazy._handler
    assert first is not None
    lazy({"a": 2})
    assert lazy._handler is first
    assert len(calls) == 2
    assert calls[0] == ({"a": 1}, "mylogger", logging.INFO)
    assert calls[1][0] == {"a": 2}


# ---------------------------------------------------------------------------
# _config_handlers
# ---------------------------------------------------------------------------

def test_config_handlers_returns_user_list_when_no_logger():
    user = [object(), object()]
    assert _config_handlers(user, logger=None) is user


def test_config_handlers_none_returns_empty():
    assert _config_handlers(None) == []


def test_config_handlers_single_handler():
    h = object()
    result = _config_handlers(h)
    assert result == [h]


def test_config_handlers_iterable_of_handlers():
    h1, h2 = object(), object()
    result = _config_handlers((h1, h2))
    assert result == [h1, h2]


def test_config_handlers_with_logger_and_no_user_handlers(monkeypatch):
    log_level = logging.WARNING
    sentinel_handler = object()
    result = _config_handlers(
        None,
        default_handler=sentinel_handler,
        logger="mylog",
        log_level=log_level,
    )
    assert len(result) == 1
    lazy = result[0]
    assert isinstance(lazy, _LazyLogHandler)
    assert lazy._handler is None


def test_config_handlers_with_logger_and_user_handlers(monkeypatch):
    h1 = object()

    def default_handler(details, logger, log_level):
        pass

    result = _config_handlers(
        [h1],
        default_handler=default_handler,
        logger="mylog",
        log_level=logging.INFO,
    )
    # First element is a functools.partial, followed by user handlers.
    assert len(result) == 2
    assert result[1] is h1


def test_config_handlers_with_logger_and_single_user_handler():
    h1 = object()

    def default_handler(details, logger, log_level):
        pass

    result = _config_handlers(
        h1,
        default_handler=default_handler,
        logger="mylog",
        log_level=logging.INFO,
    )
    assert len(result) == 2
    assert result[1] is h1


def test_config_handlers_with_logger_requires_log_level():
    with pytest.raises(AssertionError):
        _config_handlers(None, default_handler=object(), logger="mylog")


# ---------------------------------------------------------------------------
# _log_backoff / _log_giveup
# ---------------------------------------------------------------------------

class _RecordingLogger:
    def __init__(self):
        self.records = []

    def log(self, level, msg, *args):
        self.records.append((level, msg, args))


def test_log_backoff_with_exception():
    logger = _RecordingLogger()

    def target():
        pass

    details = {
        "target": target,
        "wait": 1.5,
        "exception": ValueError("bad"),
    }
    _log_backoff(details, logger, logging.ERROR)
    assert len(logger.records) == 1
    level, msg, args = logger.records[0]
    assert level == logging.ERROR
    assert args == ("target", 1.5, "ValueError: bad")


def test_log_backoff_with_value_fallback():
    logger = _RecordingLogger()

    def target():
        pass

    details = {"target": target, "wait": 2.0, "value": "somevalue"}
    _log_backoff(details, logger, logging.INFO)
    level, msg, args = logger.records[0]
    assert args == ("target", 2.0, "somevalue")


def test_log_backoff_unknown_value():
    logger = _RecordingLogger()

    def target():
        pass

    details = {"target": target, "wait": 0.0}
    _log_backoff(details, logger, logging.INFO)
    _, _, args = logger.records[0]
    assert args == ("target", 0.0, "unknown")


def test_log_giveup_with_exception():
    logger = _RecordingLogger()

    def target():
        pass

    details = {"target": target, "tries": 4, "exception": KeyError("k")}
    _log_giveup(details, logger, logging.CRITICAL)
    level, msg, args = logger.records[0]
    assert level == logging.CRITICAL
    assert args == ("target", 4, "KeyError: 'k'")


def test_log_giveup_with_value_fallback():
    logger = _RecordingLogger()

    def target():
        pass

    details = {"target": target, "tries": 2, "value": "v"}
    _log_giveup(details, logger, logging.WARNING)
    _, _, args = logger.records[0]
    assert args == ("target", 2, "v")


# ---------------------------------------------------------------------------
# _now / _elapsed / _is_custom_wait
# ---------------------------------------------------------------------------

def test_now_returns_float_based_on_monotonic(monkeypatch):
    monkeypatch.setattr(common.time_module, "monotonic_ns", lambda: 2_000_000_000)
    assert _now() == 2.0


def test_elapsed_uses_now(monkeypatch):
    monkeypatch.setattr(common.time_module, "monotonic_ns", lambda: 3_000_000_000)
    assert _elapsed(1.0) == 2.0


def test_is_custom_wait_true_for_numbers():
    assert _is_custom_wait(1) is True
    assert _is_custom_wait(1.5) is True


def test_is_custom_wait_false_for_bool_and_others():
    assert _is_custom_wait(True) is False
    assert _is_custom_wait(False) is False
    assert _is_custom_wait("3") is False
    assert _is_custom_wait(None) is False


# ---------------------------------------------------------------------------
# _check_hot_loop
# ---------------------------------------------------------------------------

@pytest.fixture()
def _reset_hot_loop():
    original = dict(_hot_loop_data)
    yield
    _hot_loop_data.clear()
    _hot_loop_data.update(original)


def test_check_hot_loop_resets_when_gap_large(monkeypatch, _reset_hot_loop):
    times = iter([1.0, 2.0])

    def fake_ns():
        return int(next(times) * 1_000_000_000)

    monkeypatch.setattr(common.time_module, "monotonic_ns", fake_ns)

    _hot_loop_data["last_retry"] = 0.0
    _hot_loop_data["count"] = 0

    _check_hot_loop()
    assert _hot_loop_data["count"] == 0
    assert _hot_loop_data["last_retry"] == 1.0

    # gap is 1.0s > 0.1 => still resets
    _check_hot_loop()
    assert _hot_loop_data["count"] == 0
    assert _hot_loop_data["last_retry"] == 2.0


def test_check_hot_loop_increments_within_window(monkeypatch, _reset_hot_loop):
    # Five retries each 0.01s apart should trigger warning on 5th.
    base = 10.0
    times = iter([base + i * 0.01 for i in range(5)])

    def fake_ns():
        return int(next(times) * 1_000_000_000)

    monkeypatch.setattr(common.time_module, "monotonic_ns", fake_ns)

    _hot_loop_data["last_retry"] = 0.0
    _hot_loop_data["count"] = 0

    # First call sets last_retry, count resets to 0 (prev==0).
    _check_hot_loop()
    assert _hot_loop_data["count"] == 0

    for _ in range(4):
        _check_hot_loop()

    # After 5 total calls with small gaps, count reached 5 and reset to 0.
    assert _hot_loop_data["count"] == 0
    assert _hot_loop_data["last_retry"] == pytest.approx(base + 0.04)
