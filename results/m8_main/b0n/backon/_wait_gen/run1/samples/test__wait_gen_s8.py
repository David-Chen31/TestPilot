import itertools
import math
import random

import pytest

from backon._wait_gen import (
    _CombinedWait,
    _Constant,
    _Decay,
    _Expo,
    _Fibo,
    _Runtime,
    _Wait,
    _WaitChain,
    _WaitException,
    _WaitFactory,
    _WaitIncrementing,
    _WaitNone,
    _WaitRandom,
    _WaitRandomExponential,
    constant,
    decay,
    expo,
    fibo,
    runtime,
    wait_chain,
    wait_combine,
    wait_exception,
    wait_incrementing,
    wait_none,
    wait_random,
    wait_random_exponential,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def take(gen, n):
    return [gen.next() for _ in range(n)]


class _SeqWait(_Wait):
    """Deterministic wait that yields from a fixed sequence (looping)."""

    def __init__(self, values):
        super().__init__()
        self._values = list(values)
        self._iter = itertools.cycle(self._values)

    def next(self, send_value=None):
        return next(self._iter)


# ---------------------------------------------------------------------------
# wait_chain
# ---------------------------------------------------------------------------


def test_wait_chain_returns_wait_chain_instance():
    chain = wait_chain(expo(), expo())
    assert isinstance(chain, _WaitChain)
    assert isinstance(chain, _Wait)


def test_wait_chain_empty_returns_zero():
    chain = wait_chain()
    assert isinstance(chain, _WaitChain)
    assert [chain.next() for _ in range(5)] == [0.0] * 5


def test_wait_chain_cycles_through_waits():
    chain = wait_chain(constant(1), constant(2), constant(3))
    assert take(chain, 7) == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]


def test_wait_chain_single_wait():
    chain = wait_chain(constant(5))
    assert take(chain, 3) == [5.0, 5.0, 5.0]


def test_wait_chain_passes_send_value():
    received = []

    def rec(send_value):
        received.append(send_value)
        return float(send_value or 0)

    chain = wait_chain(runtime(value=rec))
    assert chain.next(10) == 10.0
    assert chain.next(20) == 20.0
    assert received == [10, 20]


def test_wait_chain_advances_inner_waits():
    chain = wait_chain(expo(base=2, factor=1), constant(0))
    # expo yields 1,2,4,... constant yields 0
    assert take(chain, 4) == [1.0, 0.0, 2.0, 0.0]


def test_wait_chain_call_reconstructs_waits():
    chain = wait_chain(constant(1), constant(2))
    new_chain = chain(interval=9)
    # Constant accepts an interval argument; both become 9
    assert take(new_chain, 4) == [9.0, 9.0, 9.0, 9.0]


def test_wait_chain_call_returns_wait_chain():
    chain = wait_chain(constant(1))
    new_chain = chain()
    assert isinstance(new_chain, _WaitChain)
    assert new_chain is not chain


# ---------------------------------------------------------------------------
# wait_combine
# ---------------------------------------------------------------------------


def test_wait_combine_returns_combined_wait_instance():
    combined = wait_combine(constant(1), constant(2))
    assert isinstance(combined, _CombinedWait)
    assert isinstance(combined, _Wait)


def test_wait_combine_empty_returns_zero():
    combined = wait_combine()
    assert isinstance(combined, _CombinedWait)
    assert [combined.next() for _ in range(3)] == [0.0, 0.0, 0.0]


def test_wait_combine_sums_waits():
    combined = wait_combine(constant(1), constant(2), constant(3))
    assert take(combined, 3) == [6.0, 6.0, 6.0]


def test_wait_combine_single_wait():
    combined = wait_combine(constant(4))
    assert take(combined, 2) == [4.0, 4.0]


def test_wait_combine_advances_inner_waits():
    combined = wait_combine(expo(base=2, factor=1), constant(10))
    # expo: 1,2,4 ; constant: 10
    assert take(combined, 3) == [11.0, 12.0, 14.0]


def test_wait_combine_passes_send_value():
    combined = wait_combine(
        runtime(value=lambda sv: float(sv or 0)),
        runtime(value=lambda sv: float((sv or 0) * 2)),
    )
    assert combined.next(5) == 15.0
    assert combined.next(1) == 3.0


def test_wait_combine_call_reconstructs_waits():
    combined = wait_combine(constant(1), constant(2))
    new_combined = combined(interval=7)
    assert isinstance(new_combined, _CombinedWait)
    assert take(new_combined, 3) == [14.0, 14.0, 14.0]


def test_wait_combine_and_wait_cha_in_combination():
    # combine of two chains sums their results
    c1 = wait_chain(constant(1), constant(2))
    c2 = wait_chain(constant(10), constant(20))
    combined = wait_combine(c1, c2)
    assert take(combined, 4) == [11.0, 22.0, 11.0, 22.0]


# ---------------------------------------------------------------------------
# _Wait.__add__ / __radd__ behaviors used with the public factories
# ---------------------------------------------------------------------------


def test_wait_add_basic():
    result = constant(1) + constant(2)
    assert isinstance(result, _CombinedWait)
    assert take(result, 3) == [3.0, 3.0, 3.0]


def test_wait_radd_basic():
    result = constant(1).__radd__(constant(2))
    assert isinstance(result, _CombinedWait)
    assert take(result, 2) == [3.0, 3.0]


def test_wait_add_with_combined_merges():
    left = constant(1) + constant(2)
    result = left + constant(3)
    assert isinstance(result, _CombinedWait)
    assert take(result, 2) == [6.0, 6.0]


def test_wait_add_combined_other_merges_flat():
    left = constant(1)
    other = wait_combine(constant(2), constant(3))
    result = left + other
    assert isinstance(result, _CombinedWait)
    assert len(result._waits) == 3
    assert take(result, 2) == [6.0, 6.0]


def test_wait_factory_add_returns_combined():
    result = constant + constant
    assert isinstance(result, _CombinedWait)
    assert take(result, 2) == [2.0, 2.0]


def test_wait_factory_radd_with_wait():
    result = constant(1).__radd__(constant(2))
    assert take(result, 1) == [3.0]


def test_wait_factory_radd_with_combined():
    combined = wait_combine(constant(1), constant(2))
    result = combined.__radd__(constant(3))
    assert isinstance(result, _CombinedWait)
    assert take(result, 1) == [6.0]


def test_wait_factory_add_to_combined():
    combined = wait_combine(constant(1), constant(2))
    result = combined + constant(3)
    assert isinstance(result, _CombinedWait)
    assert take(result, 1) == [6.0]


# ---------------------------------------------------------------------------
# Combined wait call semantics
# ---------------------------------------------------------------------------


def test_combined_wait_call_passes_kwargs_to_each():
    combined = wait_combine(constant(1), constant(1))
    new_combined = combined(interval=5)
    assert take(new_combined, 2) == [10.0, 10.0]


# ---------------------------------------------------------------------------
# Individual wait generators (used via public factories)
# ---------------------------------------------------------------------------


def test_constant_numeric():
    c = constant(3)
    assert take(c, 3) == [3.0, 3.0, 3.0]


def test_constant_int_becomes_float():
    c = constant(2)
    assert c.next() == 2.0
    assert isinstance(c.next(), float)


def test_constant_sequence():
    c = constant([1, 2, 3])
    assert [c.next(), c.next(), c.next()] == [1.0, 2.0, 3.0]
    with pytest.raises(StopIteration):
        c.next()


def test_expo_basic():
    e = expo(base=2, factor=1)
    assert take(e, 4) == [1.0, 2.0, 4.0, 8.0]


def test_expo_with_max_value_caps():
    e = expo(base=2, factor=1, max_value=5)
    assert take(e, 5) == [1.0, 2.0, 4.0, 5, 5]


def test_expo_factor():
    e = expo(base=3, factor=2)
    assert take(e, 3) == [2.0, 6.0, 18.0]


def test_decay_basic():
    d = decay(initial_value=1, decay_factor=1)
    values = take(d, 3)
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(math.e ** -1)
    assert values[2] == pytest.approx(math.e ** -2)


def test_decay_with_min_value():
    d = decay(initial_value=1, decay_factor=10, min_value=0.1)
    values = take(d, 5)
    assert values[0] == pytest.approx(1.0)
    # eventually saturates at min
    assert values[-1] == 0.1


def test_fibo_basic():
    f = fibo()
    assert take(f, 6) == [1.0, 1.0, 2.0, 3.0, 5.0, 8.0]


def test_fibo_max_value():
    f = fibo(max_value=5)
    assert take(f, 6) == [1.0, 1.0, 2.0, 3.0, 5.0, 5.0]


def test_runtime_uses_send_value():
    r = runtime(value=lambda sv: float(sv) * 2)
    assert r.next(3) == 6.0
    assert r.next(5) == 10.0


def test_runtime_none_send_value():
    r = runtime(value=lambda sv: 0.0 if sv is None else float(sv))
    assert r.next() == 0.0


def test_wait_incrementing_basic():
    w = wait_incrementing(start=1, increment=1)
    assert take(w, 4) == [1.0, 2.0, 3.0, 4.0]


def test_wait_incrementing_max_value():
    w = wait_incrementing(start=1, increment=2, max_value=5)
    assert take(w, 4) == [1.0, 3.0, 5.0, 5.0]


def test_wait_random_exponential_bounds(monkeypatch):
    calls = []

    def fake_uniform(a, b):
        calls.append((a, b))
        return (a + b) / 2

    monkeypatch.setattr(random, "uniform", fake_uniform)
    w = wait_random_exponential(multiplier=1, exp_base=2, max_value=4, min_value=0)
    res = take(w, 5)
    # a values are 1, 2, 4, 4, 4 (capped), result = a/2
    assert res == [0.5, 1.0, 2.0, 2.0, 2.0]


def test_wait_random_exponential_min_value(monkeypatch):
    monkeypatch.setattr(random, "uniform", lambda a, b: b)
    w = wait_random_exponential(multiplier=0.1, exp_base=2, min_value=1)
    # a sequence: 0.1, 0.2, 0.4, ... all below min 1 -> result uniform(0,1)=1
    res = take(w, 3)
    assert res == [1.0, 1.0, 1.0]


def test_wait_random_bounds(monkeypatch):
    monkeypatch.setattr(random, "uniform", lambda a, b: (a + b) / 2)
    w = wait_random(min=2, max=4)
    assert w.next() == 3.0


def test_wait_exception_delegates():
    exc = ValueError("x")
    w = wait_exception(value=lambda sv: 42.0)
    assert w.next() == 42.0


def test_wait_none_returns_zero():
    n = wait_none()
    assert [n.next() for _ in range(3)] == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# _Wait base class behaviors
# ---------------------------------------------------------------------------


def test_base_wait_next_not_implemented():
    base = _Wait()
    with pytest.raises(NotImplementedError):
        base.next()


def test_base_wait_call_clones():
    w = _Expo(base=2, factor=1)
    clone = w(factor=3)
    assert isinstance(clone, _Expo)
    assert clone is not w
    assert clone.next() == 3.0


def test_base_wait_call_positional_overrides():
    w = _Expo(base=2, factor=1)
    clone = w(3, 1)  # base=3, factor=1
    assert clone.next() == 1.0
    assert clone.next() == 3.0


def test_wait_factory_slots():
    f = _WaitFactory(_Expo)
    assert f.__slots__ == ("_cls",)


# ---------------------------------------------------------------------------
# Chained combination via + operator with chains
# ---------------------------------------------------------------------------


def test_chain_in_combination_with_plus():
    chain = wait_chain(constant(1), constant(2))
    result = chain + constant(10)
    assert isinstance(result, _CombinedWait)
    assert take(result, 4) == [11.0, 12.0, 11.0, 12.0]
