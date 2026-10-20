import pytest

import backon._common as common
from backon._common import disable, enable, is_enabled


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Ensure the global enabled flag is restored after each test."""
    original = common._GLOBAL_ENABLED
    yield
    common._GLOBAL_ENABLED = original


class TestEnableDisable:
    def test_default_state_is_enabled(self):
        # The module-level default is True.
        assert common._GLOBAL_ENABLED is True
        assert is_enabled() is True

    def test_disable_returns_none_and_sets_flag(self):
        assert disable() is None
        assert common._GLOBAL_ENABLED is False
        assert is_enabled() is False

    def test_enable_returns_none_and_sets_flag(self):
        disable()
        assert enable() is None
        assert common._GLOBAL_ENABLED is True
        assert is_enabled() is True

    def test_disable_is_idempotent(self):
        disable()
        disable()
        assert is_enabled() is False

    def test_enable_is_idempotent(self):
        enable()
        enable()
        assert is_enabled() is True

    def test_toggle_sequence(self):
        assert is_enabled() is True
        disable()
        assert is_enabled() is False
        enable()
        assert is_enabled() is True
        disable()
        assert is_enabled() is False

    def test_is_enabled_returns_bool(self):
        # Precise check on the returned type/value.
        assert is_enabled() is True
        disable()
        assert is_enabled() is False


class TestTestingAliases:
    """The repository uses `_global_disable` / `_global_enable` aliases."""

    def test_global_disable_alias(self):
        common._global_disable() if hasattr(common, "_global_disable") else disable()
        assert is_enabled() is False

    def test_global_enable_alias(self):
        disable()
        common._global_enable() if hasattr(common, "_global_enable") else enable()
        assert is_enabled() is True


class TestMaybeCall:
    def test_calls_callable_with_args(self):
        result = common._maybe_call(lambda a, b=0: a + b, 2, b=3)
        assert result == 5

    def test_returns_non_callable_as_is(self):
        sentinel = object()
        assert common._maybe_call(sentinel) is sentinel
        assert common._maybe_call(42) == 42
        assert common._maybe_call("x") == "x"

    def test_returns_callable_when_signature_mismatch(self):
        # A callable that does not accept the given args returns itself.
        def needs_arg(x):
            return x

        result = common._maybe_call(needs_arg)
        assert result is needs_arg

    def test_propagates_other_type_errors(self):
        def raiser():
            raise TypeError("something unrelated broke")

        with pytest.raises(TypeError, match="something unrelated broke"):
            common._maybe_call(raiser, "unexpected")

    def test_propagates_non_type_errors(self):
        def raiser():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            common._maybe_call(raiser)


class TestInitWaitGen:
    def test_calls_maybe_call_on_kwargs(self):
        received = {}

        def wait_gen(**kwargs):
            received.update(kwargs)
            return "gen"

        result = common._init_wait_gen(
            wait_gen, {"a": 1, "b": lambda: "computed"}
        )
        assert result == "gen"
        assert received == {"a": 1, "b": "computed"}

    def test_no_kwargs(self):
        def wait_gen(**kwargs):
            return kwargs

        assert common._init_wait_gen(wait_gen, {}) == {}


class TestNextWait:
    def test_basic_wait_without_jitter_or_max_time(self):
        class FakeWait:
            def next(self, send_value):
                return send_value

        seconds = common._next_wait(
            FakeWait(), 2.0, None, elapsed=0.0, max_time=None
        )
        # backoff_multiplier default is 1.0
        assert seconds == 2.0

    def test_with_jitter(self):
        class FakeWait:
            def next(self, send_value):
                return send_value

        seconds = common._next_wait(
            FakeWait(), 4.0, jitter=lambda v: v / 2, elapsed=0.0, max_time=None
        )
        assert seconds == 2.0

    def test_capped_by_max_time(self):
        class FakeWait:
            def next(self, send_value):
                return send_value

        seconds = common._next_wait(
            FakeWait(), 10.0, None, elapsed=8.0, max_time=10.0
        )
        assert seconds == 2.0

    def test_negative_remaining_time(self):
        class FakeWait:
            def next(self, send_value):
                return send_value

        seconds = common._next_wait(
            FakeWait(), 10.0, None, elapsed=12.0, max_time=10.0
        )
        assert seconds == -2.0


class TestApplyTestOverrides:
    def test_no_override(self):
        orig = common._TEST_CONFIG["max_retries"]
        common._TEST_CONFIG["max_retries"] = None
        try:
            assert common._apply_test_overrides(5, 60) == (5, 60)
        finally:
            common._TEST_CONFIG["max_retries"] = orig

    def test_with_override(self):
        orig = common._TEST_CONFIG["max_retries"]
        common._TEST_CONFIG["max_retries"] = 3
        try:
            assert common._apply_test_overrides(5, 60) == (3, 60)
        finally:
            common._TEST_CONFIG["max_retries"] = orig


class TestPrepareLogger:
    def test_string_returns_logger(self):
        import logging

        logger = common._prepare_logger("backon.test")
        assert logger is logging.getLogger("backon.test")

    def test_logger_object_passthrough(self):
        import logging

        logger = logging.getLogger("some.other")
        assert common._prepare_logger(logger) is logger


class TestLazyLogHandler:
    def test_lazy_creation_and_delegation(self):
        created = []

        def factory(*, logger, log_level):
            created.append((logger, log_level))

            def handler(details):
                return ("handled", details)

            return handler

        lazy = common._LazyLogHandler(factory, "mylogger", 20)
        assert lazy._handler is None

        result = lazy("details-1")
        assert result == ("handled", "details-1")
        assert lazy._handler is not None
        assert created == [("mylogger", 20)]

        # Second call should not recreate the handler.
        lazy("details-2")
        assert created == [("mylogger", 20)]


class TestConfigHandlers:
    def test_list_returned_directly_when_no_logger(self):
        handlers = [object(), object()]
        assert common._config_handlers(handlers, logger=None) is handlers

    def test_none_and_no_logger_returns_empty(self):
        assert common._config_handlers(None, logger=None) == []

    def test_non_iterable_user_handler(self):
        class Handler:
            pass

        h = Handler()
        result = common._config_handlers(h, logger=None)
        assert result == [h]

    def test_iterable_user_handler(self):
        a, b = object(), object()
        result = common._config_handlers([a, b], logger=None)
        assert result == [a, b]

    def test_logger_with_none_user_handlers_creates_lazy(self):
        def default_handler(*, logger, log_level):
            return None

        result = common._config_handlers(
            None, default_handler=default_handler, logger="l", log_level=10
        )
        assert len(result) == 1
        assert isinstance(result[0], common._LazyLogHandler)

    def test_logger_with_iterable_user_handlers(self):
        def default_handler(*, logger, log_level):
            return None

        extras = [object(), object()]
        result = common._config_handlers(
            extras, default_handler=default_handler, logger="l", log_level=10
        )
        assert len(result) == 3
        assert callable(result[0])
        assert result[1:] == extras

    def test_logger_with_single_user_handler(self):
        def default_handler(*, logger, log_level):
            return None

        extra = object()
        result = common._config_handlers(
            extra, default_handler=default_handler, logger="l", log_level=10
        )
        assert len(result) == 2
        assert result[1] is extra


class TestLogBackoff:
    def test_logs_with_exception(self):
        import logging

        logger = logging.getLogger("backon.test.logbackoff")
        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        cap = Capture()
        logger.addHandler(cap)
        logger.setLevel(logging.DEBUG)
        try:
            def target():
                pass

            exc = ValueError("bad value")
            details = {
                "target": target,
                "wait": 1.25,
                "exception": exc,
            }
            common._log_backoff(details, logger, logging.WARNING)
        finally:
            logger.removeHandler(cap)

        assert len(records) == 1
        assert "Backing off" in records[0].getMessage()
        assert records[0].levelno == logging.WARNING

    def test_logs_without_exception_uses_value(self):
        import logging

        logger = logging.getLogger("backon.test.logbackoff.value")
        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        cap = Capture()
        logger.addHandler(cap)
        logger.setLevel(logging.DEBUG)
        try:
            def target():
                pass

            details = {"target": target, "wait": 0.5, "value": "somevalue"}
            common._log_backoff(details, logger, logging.INFO)
        finally:
            logger.removeHandler(cap)

        assert len(records) == 1
        assert "somevalue" in records[0].getMessage()


class TestLogGiveup:
    def test_logs_with_exception(self):
        import logging

        logger = logging.getLogger("backon.test.loggiveup")
        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        cap = Capture()
        logger.addHandler(cap)
        logger.setLevel(logging.DEBUG)
        try:
            def target():
                pass

            details = {
                "target": target,
                "tries": 3,
                "exception": RuntimeError("nope"),
            }
            common._log_giveup(details, logger, logging.ERROR)
        finally:
            logger.removeHandler(cap)

        assert len(records) == 1
        assert "Giving up" in records[0].getMessage()
        assert records[0].levelno == logging.ERROR

    def test_logs_without_exception_uses_unknown(self):
        import logging

        logger = logging.getLogger("backon.test.loggiveup.value")
        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        cap = Capture()
        logger.addHandler(cap)
        logger.setLevel(logging.DEBUG)
        try:
            def target():
                pass

            details = {"target": target, "tries": 1}
            common._log_giveup(details, logger, logging.WARNING)
        finally:
            logger.removeHandler(cap)

        assert len(records) == 1
        assert "unknown" in records[0].getMessage()


class TestTimeHelpers:
    def test_now_returns_float(self):
        assert isinstance(common._now(), float)

    def test_elapsed_non_negative(self, monkeypatch):
        monkeypatch.setattr(common, "_now", lambda: 100.0)
        assert common._elapsed(90.0) == 10.0

    def test_elapsed_zero(self, monkeypatch):
        monkeypatch.setattr(common, "_now", lambda: 50.0)
        assert common._elapsed(50.0) == 0.0


class TestIsCustomWait:
    def test_numeric_values_are_custom(self):
        assert common._is_custom_wait(1) is True
        assert common._is_custom_wait(1.5) is True
        assert common._is_custom_wait(0) is True

    def test_bool_is_not_custom(self):
        assert common._is_custom_wait(True) is False
        assert common._is_custom_wait(False) is False

    def test_non_numeric_is_not_custom(self):
        assert common._is_custom_wait("1") is False
        assert common._is_custom_wait(None) is False
        assert common._is_custom_wait([1]) is False


class TestCheckHotLoop:
    def test_no_warning_first_call(self, monkeypatch):
        import logging as logging_mod

        warnings = []

        class FakeLogger:
            def warning(self, msg, *args):
                warnings.append((msg, args))

        def fake_get_logger(name):
            return FakeLogger()

        # Reset hot loop state.
        common._hot_loop_data["last_retry"] = 0.0
        common._hot_loop_data["count"] = 0
        monkeypatch.setattr(logging_mod, "getLogger", fake_get_logger)
        monkeypatch.setattr(common.time_module, "monotonic_ns", lambda: 1_000_000_000)

        common._check_hot_loop()
        assert warnings == []
        assert common._hot_loop_data["count"] == 0

    def test_warning_after_five_quick_retries(self, monkeypatch):
        import logging as logging_mod

        warnings = []

        class FakeLogger:
            def warning(self, msg, *args):
                warnings.append((msg, args))

        monkeypatch.setattr(logging_mod, "getLogger", lambda name: FakeLogger())

        common._hot_loop_data["last_retry"] = 0.0
        common._hot_loop_data["count"] = 0

        # Providing constant time so gaps are 0 < 0.1, triggering count increments.
        monkeypatch.setattr(common.time_module, "monotonic_ns", lambda: 1_000_000_000)

        # First call establishes baseline (count stays 0).
        common._check_hot_loop()
        assert common._hot_loop_data["count"] == 0

        # Each subsequent call increments count.
        for expected in range(1, 5):
            common._check_hot_loop()
            assert common._hot_loop_data["count"] == expected

        assert warnings == []
        # 6th quick call: count becomes 5, triggers warning and resets.
        common._check_hot_loop()
        assert common._hot_loop_data["count"] == 0
        assert len(warnings) == 1
        assert "Hot loop detected" in warnings[0][0]

    def test_slow_gap_resets_count(self, monkeypatch):
        import logging as logging_mod

        warnings = []

        class FakeLogger:
            def warning(self, msg, *args):
                warnings.append((msg, args))

        monkeypatch.setattr(logging_mod, "getLogger", lambda name: FakeLogger())

        # First establish a baseline and one increment.
        common._hot_loop_data["last_retry"] = 0.0
        common._hot_loop_data["count"] = 0

        times = iter([1_000_000_000, 1_000_000_000, 2_000_000_000])
        monkeypatch.setattr(
            common.time_module, "monotonic_ns", lambda: next(times)
        )

        common._check_hot_loop()  # baseline, count 0
        common._check_hot_loop()  # quick, count 1
        assert common._hot_loop_data["count"] == 1

        # Now the gap is large, count should reset to 0.
        common._check_hot_loop()
        assert common._hot_loop_data["count"] == 0
        assert warnings == []
