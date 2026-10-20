import logging

import pytest

import backon._common as _common
from backon._common import disable, enable, is_enabled


@pytest.fixture(autouse=True)
def _reset_global_enabled():
    """Ensure global enabled state is restored after each test."""
    original = _common._GLOBAL_ENABLED
    yield
    _common._GLOBAL_ENABLED = original


def test_is_enabled_default_true():
    # The module is initialized with _GLOBAL_ENABLED = True.
    _common._GLOBAL_ENABLED = True
    assert is_enabled() is True


def test_disable_sets_false():
    _common._GLOBAL_ENABLED = True
    assert disable() is None
    assert _common._GLOBAL_ENABLED is False
    assert is_enabled() is False


def test_enable_sets_true():
    _common._GLOBAL_ENABLED = False
    assert enable() is None
    assert _common._GLOBAL_ENABLED is True
    assert is_enabled() is True


def test_enable_after_disable():
    disable()
    assert is_enabled() is False
    enable()
    assert is_enabled() is True


def test_disable_idempotent():
    disable()
    disable()
    assert is_enabled() is False


def test_enable_idempotent():
    enable()
    enable()
    assert is_enabled() is True


def test_disable_enable_roundtrip():
    initial = is_enabled()
    disable()
    assert is_enabled() is not initial
    enable()
    assert is_enabled() is initial


def test_global_enabled_variable_name():
    # Public functions manipulate the module-level _GLOBAL_ENABLED flag.
    disable()
    assert _common._GLOBAL_ENABLED is False
    enable()
    assert _common._GLOBAL_ENABLED is True


def test_disable_returns_none():
    assert disable() is None


def test_enable_returns_none():
    assert enable() is None


def test_is_enabled_returns_bool():
    assert isinstance(is_enabled(), bool)
    disable()
    assert isinstance(is_enabled(), bool)
