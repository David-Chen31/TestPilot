from __future__ import annotations

import math
import random

import pytest

from backon._wait_gen import (
    _CombinedWait,
    _Wait,
    _WaitChain,
    _WaitFactory,
    constant,
    decay,
    expo,
    fibo,
    runtime,
    wait_chain,
    wait_combine,
    wait_exception,
    wait_exponential_jitter,
    wait_incrementing,
    wait_none,
    wait_random,
    wait_random_exponential,
)


# ---------------------------------------------------------------------------
# wait_combine / _CombinedWait
# ---------------------------------------------------------------------------


def test_wait_combine_returns_combined_wait():
    cw = wait_combine(constant(1), constant(2))
    assert isinstance(cw, _CombinedWait)


def test_wait_combine_sums_next_values():
    cw = wait_combine(constant(1), constant(2), constant(3))
    assert cw.next() == 6.0
    assert cw.next() == 6.0


def test_wait_combine_empty_returns_zero():
    cw = wait_combine()
    assert cw.next() == 0.0


def test_wait_combine_tracks_state_independently():
    a = expo(base=2, factor=1)  # 1, 2, 4, ...
    b = expo(base=3, factor=1)  # 1, 3, 9, ...
    cw = wait_combine(a, b)
    assert cw.next() == 1 + 1
    assert cw.next() == 2 + 3
    assert cw.next() == 4 + 9


def test_combined_wait_add_wait():
    cw = wait_combine(constant(1))
    result = cw + constant(2)
    assert isinstance(result, _CombinedWait)
    assert result.next() == 3.0


def test_combined_wait_add_combined_wait():
    cw1 = wait_combine(constant(1), constant(2))
    cw2 = wait_combine(constant(3), constant(4))
    total = cw1 + cw2
    assert isinstance(total, _CombinedWait)
    # Flattened: 1+2+3+4
    assert total.next() == 10.0


def test_combined_wait_call_rebuilds_with_kwargs():
    cw = wait_combine(constant(1), constant(2))
    new_cw = cw(interval=5)
    assert isinstance(new_cw, _CombinedWait)
    assert new_cw.next() == 10.0


def test_combined_wait_call_resets_state():
    cw = wait_combine(expo(factor=1, base=2), constant(1))
    cw.next()  # advance expo state
    cw.next()
    fresh = cw()
    assert fresh.next() == 1 + 1


def test_combined_wait_is_wait_subclass():
    assert isinstance(wait_combine(), _Wait)


# ---------------------------------------------------------------------------
# wait_chain / _WaitChain
# ---------------------------------------------------------------------------


def test_wait_chain_returns_wait_chain():
    chain = wait_chain(constant(1))
    assert isinstance(chain, _WaitChain)


def test_wait_chain_empty_returns_zero():
    chain = wait_chain()
    assert chain.next() == 0.0


def test_wait_chain_cycles_through_waits():
    chain = wait_chain(constant(1), constant(2), constant(3))
    assert chain.next() == 1.0
    assert chain.next() == 2.0
    assert chain.next() == 3.0
    assert chain.next() == 1.0
    assert chain.next() == 2.0


def test_wait_chain_single_wait():
    chain = wait_chain(constant(7))
    for _ in range(3):
        assert chain.next() == 7.0


def test_wait_chain_with_expo_advances_inner_state():
    chain = wait_chain(expo(base=2, factor=1), constant(10))
    assert chain.next() == 1.0
    assert chain.next() == 10.0
    assert chain.next() == 2.0
    assert chain.next() == 10.0
    assert chain.next() == 4.0


def test_wait_chain_call_rebuilds():
    chain = wait_chain(constant(1), constant(2))
    chain.next()
    fresh = chain()
    assert isinstance(fresh, _WaitChain)
    assert fresh.next() == 1.0


def test_wait_chain_call_forwards_kwargs():
    chain = wait_chain(expo(factor=1), constant(1))
    fresh = chain(base=3)
    assert fresh.next() == 1.0  # expo base=3 -> factor*base^0 = 1
    assert fresh.next() == 1.0  # constant


def test_wait_chain_next_passes_send_value():
    received = []

    def capture(v):
        received.append(v)
        return 5.0

    chain = wait_chain(runtime(value=capture))
    assert chain.next("hello") == 5.0
    assert received == ["hello"]


# ---------------------------------------------------------------------------
# Combining / chaining interplay
# ---------------------------------------------------------------------------


def test_wait_combine_of_chains():
    c1 = wait_chain(constant(1), constant(2))
    c2 = wait_chain(constant(10), constant(20))
    combined = wait_combine(c1, c2)
    assert combined.next() == 11.0
    assert combined.next() == 22.0


def test_wait_chain_of_combinators():
    cw = wait_combine(constant(1), constant(2))
    chain = wait_chain(cw, constant(100))
    assert chain.next() == 3.0
    assert chain.next() == 100.0
    assert chain.next() == 3.0


# ---------------------------------------------------------------------------
# Factories and definitions used by these functions
# ---------------------------------------------------------------------------


def test_wait_factory_is_callable():
    w = expo(base=2, factor=1)
    assert isinstance(w, _Wait)
    assert w.next() == 1.0
    assert w.next() == 2.0


def test_wait_factory_add_wait():
    result = expo() + constant(5)
    assert isinstance(result, _CombinedWait)


def test_wait_factory_radd_with_combined():
    cw = wait_combine(constant(1))
    result = cw + expo(factor=1)
    assert result.next() == 1.0 + 1.0


def test_wait_add_combined_flattening():
    a = constant(1)
    cw = wait_combine(constant(2), constant(3))
    result = a + cw
    assert isinstance(result, _CombinedWait)
    assert result.next() == 6.0


# ---------------------------------------------------------------------------
# Deterministic helpers used with the public API
# ---------------------------------------------------------------------------


def test_constant_with_sequence():
    w = constant([1, 2, 3])
    assert w.next() == 1
    assert w.next() == 2
    assert w.next() == 3
    with pytest.raises(StopIteration):
        w.next()


def test_constant_repeat():
    w = constant(4)
    assert w.next() == 4.0
    assert w.next() == 4.0


def test_expo_next_values():
    w = expo(base=2, factor=1)
    assert w.next() == 1
    assert w.next() == 2
    assert w.next() == 4
    assert w.next() == 8


def test_expo_max_value():
    w = expo(base=2, factor=1, max_value=5)
    assert w.next() == 1
    assert w.next() == 2
    assert w.next() == 4
    assert w.next() == 5
    assert w.next() == 5


def test_decay_next_values():
    w = decay(initial_value=10, decay_factor=1)
    assert w.next() == pytest.approx(10.0)
    assert w.next() == pytest.approx(10.0 * math.e ** -1)


def test_decay_min_value():
    w = decay(initial_value=1, decay_factor=100, min_value=0.1)
    for _ in range(100):
        val = w.next()
        assert val >= 0.1


def test_fibo_next_values():
    w = fibo()
    assert w.next() == 1.0
    assert w.next() == 1.0
    assert w.next() == 2.0
    assert w.next() == 3.0
    assert w.next() == 5.0


def test_fibo_max_value():
    w = fibo(max_value=3)
    assert w.next() == 1.0
    assert w.next() == 1.0
    assert w.next() == 2.0
    assert w.next() == 3.0
    assert w.next() == 3.0


def test_runtime_uses_send_value():
    w = runtime(value=lambda v: (v or 0) + 1.0)
    assert w.next() == 1.0
    assert w.next(10) == 11.0


def test_wait_none_returns_zero():
    w = wait_none()
    assert w.next() == 0.0


def test_wait_exception_returns_callback_value():
    w = wait_exception(value=lambda v: 42.0)
    assert w.next() == 42.0


def test_wait_incrementing():
    w = wait_incrementing(start=1, increment=2, max_value=5)
    assert w.next() == 1
    assert w.next() == 3
    assert w.next() == 5
    assert w.next() == 5


# ---------------------------------------------------------------------------
# Random-based generators, tested deterministically via seeded random
# ---------------------------------------------------------------------------


def test_wait_random_uses_uniform(monkeypatch):
    calls = {}

    def fake_uniform(a, b):
        calls["args"] = (a, b)
        return 0.5

    monkeypatch.setattr(random, "uniform", fake_uniform)
    w = wait_random(min=2, max=4)
    assert w.next() == 0.5
    assert calls["args"] == (2, 4)


def test_wait_random_exponential_bounds(monkeypatch):
    captured = []

    def fake_uniform(a, b):
        captured.append((a, b))
        return a  # return lower bound for determinism

    monkeypatch.setattr(random, "uniform", fake_uniform)
    w = wait_random_exponential(multiplier=1, max_value=10, exp_base=2, min_value=1)
    values = [w.next() for _ in range(4)]
    assert values == [1.0, 1.0, 1.0, 1.0]
    # upper bounds: multiplier*exp_base^n clamped to max_value, then clamped by min
    assert [b for _, b in captured] == [1.0, 2.0, 4.0, 8.0]

    captured.clear()
    w2 = wait_random_exponential(multiplier=1, max_value=3, exp_base=2, min_value=0)
    [w2.next() for _ in range(4)]
    assert [b for _, b in captured] == [1.0, 2.0, 3.0, 3.0]


def test_wait_exponential_jitter_deterministic(monkeypatch):
    captured = []

    def fake_uniform(a, b):
        captured.append((a, b))
        return b  # add full jitter for determinism

    monkeypatch.setattr(random, "uniform", fake_uniform)
    w = wait_exponential_jitter(initial=1, max=100, exp_base=2, jitter=1)
    # n starts at 1 -> delay = min(1*2^1, 100) + jitter = 2 + 1 = 3
    assert w.next() == 3.0
    assert w.next() == 5.0
    assert w.next() == 9.0
    assert captured == [(0, 1), (0, 1), (0, 1)]


# ---------------------------------------------------------------------------
# Not-implemented base class
# ---------------------------------------------------------------------------


def test_wait_base_next_raises():
    w = _Wait()
    with pytest.raises(NotImplementedError):
        w.next()


def test_wait_factory_slots():
    f = _WaitFactory(_Wait)
    assert f() .__class__ is _Wait
