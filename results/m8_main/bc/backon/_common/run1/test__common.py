import logging

import pytest

import backon._common as common
from backon._common import disable, enable, is_enabled


@pytest.fixture(autouse=True)
def restore_global_state():
    """Ensure the module-global enabled flag is restored after each test."""
    original = common._GLOBAL_ENABLED
    yield
    common._GLOBAL_ENABLED = original


@pytest.fixture(autouse=True)
def restore_hot_loop_state():
    """Restore hot loop tracking state so tests don't bleed into each other."""
    saved = dict(common._hot_loop_data)
    yield
    with common._hot_loop_lock:
        common._hot_loop_data.clear()
        common._hot_loop_data.update(saved)


# ---------------------------------------------------------------------------
# enable / disable / is_enabled
# ---------------------------------------------------------------------------


def test_is_enabled_default_is_true():
    common._GLOBAL_ENABLED = True
    assert is_enabled() is True


def test_disable_sets_flag_false():
    assert is_enabled() is True
    result = disable()
    assert result is None
    assert is_enabled() is False


def test_enable_sets_flag_true():
    common._GLOBAL_ENABLED = False
    assert is_enabled() is False
    result = enable()
    assert result is None
    assert is_enabled() is True


def test_disable_is_idempotent():
    disable()
    disable()
    assert is_enabled() is False


def test_enable_is_idempotent():
    enable()
    enable()
    assert is_enabled() is True


def test_disable_enable_roundtrip():
    disable()
    assert is_enabled() is False
    enable()
    assert is_enabled() is True
    disable()
    assert is_enabled() is False
    enable()
    assert is_enabled() is True


def test_toggle_affects_module_global():
    disable()
    assert common._GLOBAL_ENABLED is False
    enable()
    assert common._GLOBAL_ENABLED is True


# ---------------------------------------------------------------------------
# Helpers / internal branches relevant to the public behavior
# ---------------------------------------------------------------------------


def test_maybe_call_with_callable_and_args():
    assert common._maybe_call(lambda a, b: a + b, 1, 2) == 3


def test_maybe_call_with_non_callable():
    value = 42
    assert common._maybe_call(value) is value


def test_maybe_call_with_callable_no_required_args():
    def fn():
        return "called"

    assert common._maybe_call(fn) == "called"


def test_maybe_call_with_callable_requiring_args_returns_itself():
    def fn(a):
        return a

    # Callable requires a positional argument; _maybe_call returns f itself.
    result = common._maybe_call(fn)
    assert result is fn


def test_maybe_call_propagates_unrelated_type_error():
    def fn():
        raise TypeError("boom happened inside")

    with pytest.raises(TypeError, match="boom happened inside"):
        common._maybe_call(fn)


def test_prepare_logger_with_string():
    logger = common._prepare_logger("backon._test_logger")
    assert logger is logging.getLogger("backon._test_logger")


def test_prepare_logger_with_object():
    logger = logging.getLogger("backon._test_obj")
    assert common._prepare_logger(logger) is logger


def test_test_config_defaults():
    assert common._TEST_CONFIG["max_retries"] is None
    assert common._TEST_CONFIG["backoff_multiplier"] == 1.0
    assert common._TEST_CONFIG["max_backoff"] is None


def test_apply_test_overrides_without_max_retries():
    saved = common._TEST_CONFIG["max_retries"]
    common._TEST_CONFIG["max_retries"] = None
    try:
        assert common._apply_test_overrides(max_tries=5, max_time=10) == (5, 10)
    finally:
        common._TEST_CONFIG["max_retries"] = saved


def test_apply_test_overrides_with_max_retries():
    saved = common._TEST_CONFIG["max_retries"]
    common._TEST_CONFIG["max_retries"] = 3
    try:
        assert common._apply_test_overrides(max_tries=5, max_time=10) == (3, 10)
    finally:
        common._TEST_CONFIG["max_retries"] = saved


def test_apply_test_overrides_defaults():
    saved = common._TEST_CONFIG["max_retries"]
    common._TEST_CONFIG["max_retries"] = None
    try:
        assert common._apply_test_overrides() == (None, None)
    finally:
        common._TEST_CONFIG["max_retries"] = saved


def test_is_custom_wait():
    assert common._is_custom_wait(1) is True
    assert common._is_custom_wait(1.5) is True
    assert common._is_custom_wait(True) is False
    assert common._is_custom_wait(False) is False
    assert common._is_custom_wait("x") is False
    assert common._is_custom_wait(None) is False


def test_init_wait_gen_calls_wait_gen():
    def gen(factor):
        yield factor

    result = common._init_wait_gen(gen, {"factor": 2})
    assert next(result) == 2


def test_next_wait_applies_jitter_and_multiplier():
    class FakeWait:
        def next(self, value):
            return 10

    saved = common._TEST_CONFIG["backoff_multiplier"]
    common._TEST_CONFIG["backoff_multiplier"] = 2.0
    try:
        seconds = common._next_wait(
            FakeWait(), None, lambda v: v + 5, elapsed=0, max_time=None
        )
        assert seconds == 30.0
    finally:
        common._TEST_CONFIG["backoff_multiplier"] = saved


def test_next_wait_caps_at_max_time_minus_elapsed():
    class FakeWait:
        def next(self, value):
            return 10

    saved = common._TEST_CONFIG["backoff_multiplier"]
    common._TEST_CONFIG["backoff_multiplier"] = 1.0
    try:
        seconds = common._next_wait(
            FakeWait(), None, None, elapsed=3, max_time=5
        )
        assert seconds == 2
    finally:
        common._TEST_CONFIG["backoff_multiplier"] = saved


def test_next_wait_without_jitter():
    class FakeWait:
        def next(self, value):
            return 7

    saved = common._TEST_CONFIG["backoff_multiplier"]
    common._TEST_CONFIG["backoff_multiplier"] = 1.0
    try:
        assert common._next_wait(FakeWait(), None, None, 0, None) == 7
    finally:
        common._TEST_CONFIG["backoff_multiplier"] = saved


def test_next_wait_passes_send_value():
    received = []

    class FakeWait:
        def next(self, value):
            received.append(value)
            return 1

    saved = common._TEST_CONFIG["backoff_multiplier"]
    common._TEST_CONFIG["backoff_multiplier"] = 1.0
    try:
        common._next_wait(FakeWait(), "sentinel", None, 0, None)
    finally:
        common._TEST_CONFIG["backoff_multiplier"] = saved
    assert received == ["sentinel"]


# ---------------------------------------------------------------------------
# _config_handlers branches
# ---------------------------------------------------------------------------


def test_config_handlers_list_with_no_logger_returns_same_list():
    handlers = [object(), object()]
    assert common._config_handlers(handlers, logger=None) is handlers


def test_config_handlers_none_returns_empty_list():
    assert common._config_handlers(None) == []


def test_config_handlers_single_callable_wrapped_in_list():
    def handler(details):
        return details

    result = common._config_handlers(handler)
    assert result == [handler]


def test_config_handlers_iterable_expanded():
    handlers = (object(), object())
    result = common._config_handlers(handlers)
    assert list(result) == list(handlers)


def test_config_handlers_with_logger_and_user_handlers_none_creates_lazy():
    def default_handler(details, logger, log_level):
        return None

    result = common._config_handlers(
        None, default_handler=default_handler, logger="x", log_level=logging.INFO
    )
    assert len(result) == 1
    assert isinstance(result[0], common._LazyLogHandler)


def test_config_handlers_with_logger_and_single_user_handler():
    def default_handler(details, logger, log_level):
        return None

    def user_handler(details):
        return details

    result = common._config_handlers(
        user_handler,
        default_handler=default_handler,
        logger="x",
        log_level=logging.INFO,
    )
    assert len(result) == 2
    assert result[1] is user_handler


def test_config_handlers_with_logger_and_iterable_user_handlers():
    def default_handler(details, logger, log_level):
        return None

    def user_handler_a(details):
        return details

    def user_handler_b(details):
        return details

    result = common._config_handlers(
        [user_handler_a, user_handler_b],
        default_handler=default_handler,
        logger="x",
        log_level=logging.INFO,
    )
    assert len(result) == 3
    assert result[0].func is default_handler
    assert result[1] is user_handler_a
    assert result[2] is user_handler_b


def test_config_handlers_default_handler_partial_has_kwargs():
    def default_handler(details, logger, log_level):
        return None

    log = logging.getLogger("backon._test_partial")
    result = common._config_handlers(
        None,
        default_handler=default_handler,
        logger=log,
        log_level=logging.INFO,
    )
    lazy = result[0]
    # Trigger lazy construction, then check the partial's bound kwargs.
    lazy({"x": 1})
    inner = lazy._handler
    assert inner.func is default_handler
    assert inner.keywords == {"logger": log, "log_level": logging.INFO}


def test_lazy_log_handler_calls_handler_once():
    calls = []

    def handler(details, logger, log_level):
        calls.append((details, logger, log_level))

    lazy = common._LazyLogHandler(handler, "logname", logging.WARNING)
    lazy({"a": 1})
    lazy({"b": 2})
    assert len(calls) == 2
    assert calls[0][0] == {"a": 1}
    assert calls[0][1] == "logname"
    assert calls[0][2] == logging.WARNING
    assert calls[1][0] == {"b": 2}


def test_lazy_log_handler_caches_factory_result():
    created = []

    def handler(details, logger, log_level):
        created.append(1)

    lazy = common._LazyLogHandler(handler, "logname", logging.WARNING)
    assert lazy._handler is None
    lazy({"a": 1})
    first = lazy._handler
    assert first is not None
    lazy({"b": 2})
    assert lazy._handler is first
    assert created == [1, 1]


def test_log_backoff_with_exception(caplog):
    logger = logging.getLogger("backon._test_backoff")

    def target():
        return None

    details = {
        "target": target,
        "wait": 1.5,
        "exception": ValueError("bad"),
    }
    with caplog.at_level(logging.WARNING):
        common._log_backoff(details, logger, logging.WARNING)
    assert "Backing off target(...) for 1.5s" in caplog.text
    assert "ValueError: bad" in caplog.text


def test_log_backoff_without_exception(caplog):
    logger = logging.getLogger("backon._test_backoff2")

    def target():
        return None

    details = {"target": target, "wait": 0.5, "value": "somevalue"}
    with caplog.at_level(logging.WARNING):
        common._log_backoff(details, logger, logging.WARNING)
    assert "somevalue" in caplog.text


def test_log_backoff_defaults_to_unknown(caplog):
    logger = logging.getLogger("backon._test_backoff3")

    def target():
        return None

    details = {"target": target, "wait": 0.5}
    with caplog.at_level(logging.WARNING):
        common._log_backoff(details, logger, logging.WARNING)
    assert "unknown" in caplog.text
    assert "target(...) for 0.5s" in caplog.text


def test_log_giveup_with_exception(caplog):
    logger = logging.getLogger("backon._test_giveup")

    def target():
        return None

    details = {"target": target, "tries": 3, "exception": RuntimeError("stop")}
    with caplog.at_level(logging.ERROR):
        common._log_giveup(details, logger, logging.ERROR)
    assert "Giving up target(...) after 3 tries" in caplog.text
    assert "RuntimeError: stop" in caplog.text


def test_log_giveup_without_exception(caplog):
    logger = logging.getLogger("backon._test_giveup2")

    def target():
        return None

    details = {"target": target, "tries": 2}
    with caplog.at_level(logging.ERROR):
        common._log_giveup(details, logger, logging.ERROR)
    assert "after 2 tries" in caplog.text
    assert "unknown" in caplog.text


# ---------------------------------------------------------------------------
# _now / _elapsed
# ---------------------------------------------------------------------------


def test_now_returns_positive_float():
    value = common._now()
    assert isinstance(value, float)
    assert value > 0


def test_elapsed_is_non_negative():
    start = common._now()
    elapsed = common._elapsed(start)
    assert isinstance(elapsed, float)
    assert elapsed >= 0


def test_elapsed_subtracts_start_from_now(monkeypatch):
    monkeypatch.setattr(common, "_now", lambda: 100.0)
    assert common._elapsed(40.0) == 60.0


# ---------------------------------------------------------------------------
# _check_hot_loop
# ---------------------------------------------------------------------------


def _reset_hot_loop():
    with common._hot_loop_lock:
        common._hot_loop_data["last_retry"] = 0.0
        common._hot_loop_data["count"] = 0


def test_check_hot_loop_first_call_sets_state(monkeypatch):
    _reset_hot_loop()
    monkeypatch.setattr(common.time_module, "monotonic_ns", lambda: 5_000_000_000)
    common._check_hot_loop()
    assert common._hot_loop_data["last_retry"] == 5.0
    assert common._hot_loop_data["count"] == 0


def test_check_hot_loop_fast_successive_calls_increment_count(monkeypatch):
    _reset_hot_loop()
    monkeypatch.setattr(common.time_module, "monotonic_ns", lambda: 1_000_000_000)
    common._check_hot_loop()
    assert common._hot_loop_data["count"] == 0
    # Second call within 0.1s increments the counter.
    monkeypatch.setattr(common.time_module, "monotonic_ns", lambda: 1_050_000_000)
    common._check_hot_loop()
    assert common._hot_loop_data["count"] == 1
    assert common._hot_loop_data["last_retry"] == 1.05


def test_check_hot_loop_slow_call_resets_count(monkeypatch):
    _reset_hot_loop()
    with common._hot_loop_lock:
        common._hot_loop_data["last_retry"] = 1.0
        common._hot_loop_data["count"] = 3
    # A gap >= 0.1s resets the count.
    monkeypatch.setattr(common.time_module, "monotonic_ns", lambda: 2_000_000_000)
    common._check_hot_loop()
    assert common._hot_loop_data["count"] == 0
    assert common._hot_loop_data["last_retry"] == 2.0


def test_check_hot_loop_warns_and_resets_after_threshold(monkeypatch, caplog):
    _reset_hot_loop()
    times = iter([1_000_000_000 + i * 10_000_000 for i in range(7)])
    monkeypatch.setattr(common.time_module, "monotonic_ns", lambda: next(times))

    # First call initializes; subsequent fast calls increment count.
    # After 5 increments (count reaches 5) a warning is emitted and count reset.
    with caplog.at_level(logging.WARNING, logger="backon"):
        for _ in range(6):
            common._check_hot_loop()

    assert "Hot loop detected" in caplog.text
    assert common._hot_loop_data["count"] == 0


def test_check_hot_loop_no_warning_below_threshold(monkeypatch, caplog):
    _reset_hot_loop()
    times = iter([1_000_000_000 + i * 10_000_000 for i in range(5)])
    monkeypatch.setattr(common.time_module, "monotonic_ns", lambda: next(times))

    with caplog.at_level(logging.WARNING, logger="backon"):
        for _ in range(5):
            common._check_hot_loop()

    # count reaches 4, just below threshold; no warning.
    assert "Hot loop detected" not in caplog.text
    assert common._hot_loop_data["count"] == 4
