import math
import random

import pytest

import backon._wait_gen as wg
from backon._wait_gen import wait_chain, wait_combine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FixedWait(wg._Wait):
    """Deterministic wait that yields a sequence of values forever."""

    def __init__(self, *values):
        super().__init__(*values)
        self._values = list(values)
        self._idx = 0

    def next(self, send_value=None):
        v = self._values[self._idx % len(self._values)]
        self._idx += 1
        return v


class _CounterWait(wg._Wait):
    """Deterministic wait that yields 0,1,2,... and tracks send_value."""

    def __init__(self):
        super().__init__()
        self._n = 0
        self.received = []

    def next(self, send_value=None):
        self.received.append(send_value)
        v = self._n
        self._n += 1
        return float(v)


# ---------------------------------------------------------------------------
# wait_chain
# ---------------------------------------------------------------------------

def test_wait_chain_returns_waitchain_instance():
    chain = wait_chain()
    assert isinstance(chain, wg._WaitChain)
    assert isinstance(chain, wg._Wait)


def test_wait_chain_empty_returns_zero():
    chain = wait_chain()
    assert chain.next() == 0.0
    assert chain.next() == 0.0
    assert chain.next(123) == 0.0


def test_wait_chain_cycles_through_waits():
    a = _FixedWait(1)
    b = _FixedWait(2)
    c = _FixedWait(3)
    chain = wait_chain(a, b, c)
    assert [chain.next() for _ in range(7)] == [1, 2, 3, 1, 2, 3, 1]


def test_wait_chain_single_wait_repeats():
    a = _CounterWait()
    chain = wait_chain(a)
    assert [chain.next() for _ in range(4)] == [0.0, 1.0, 2.0, 3.0]


def test_wait_chain_uses_underlying_state_after_wrap():
    a = _CounterWait()
    b = _CounterWait()
    chain = wait_chain(a, b)
    assert [chain.next() for _ in range(4)] == [0.0, 0.0, 1.0, 1.0]


def test_wait_chain_passes_send_value():
    a = _CounterWait()
    b = _CounterWait()
    chain = wait_chain(a, b)
    chain.next("first")
    chain.next("second")
    chain.next("third")
    assert a.received == ["first", "third"]
    assert b.received == ["second"]


def test_wait_chain_stores_waits_list():
    a = _FixedWait(1)
    b = _FixedWait(2)
    chain = wait_chain(a, b)
    assert chain._waits == [a, b]


def test_wait_chain_call_rebuilds_chain():
    a = wg.expo(base=2, factor=1)
    b = wg.expo(base=3, factor=1)
    chain = wait_chain(a, b)
    new_chain = chain(base=10)
    assert isinstance(new_chain, wg._WaitChain)
    assert new_chain is not chain
    # each wrapped wait rebuilt with base=10
    assert new_chain.next() == 1.0  # first of first expo (factor*base**0)
    seq = [new_chain.next() for _ in range(2)]
    assert seq == [10.0, 10.0]


# ---------------------------------------------------------------------------
# wait_combine
# ---------------------------------------------------------------------------

def test_wait_combine_returns_combined_wait_instance():
    combined = wait_combine()
    assert isinstance(combined, wg._CombinedWait)
    assert isinstance(combined, wg._Wait)


def test_wait_combine_empty_sums_to_zero():
    combined = wait_combine()
    assert combined.next() == 0
    assert combined.next(5) == 0


def test_wait_combine_sums_next_values():
    combined = wait_combine(_FixedWait(1), _FixedWait(2), _FixedWait(3))
    assert combined.next() == 6


def test_wait_combine_sums_repeatedly():
    combined = wait_combine(_CounterWait(), _CounterWait())
    assert [combined.next() for _ in range(4)] == [0.0, 2.0, 4.0, 6.0]


def test_wait_combine_single_wait():
    a = _CounterWait()
    combined = wait_combine(a)
    assert [combined.next() for _ in range(3)] == [0.0, 1.0, 2.0]


def test_wait_combine_passes_send_value_to_each():
    a = _CounterWait()
    b = _CounterWait()
    combined = wait_combine(a, b)
    combined.next("x")
    combined.next("y")
    assert a.received == ["x", "y"]
    assert b.received == ["x", "y"]


def test_wait_combine_stores_waits_tuple():
    a = _FixedWait(1)
    b = _FixedWait(2)
    combined = wait_combine(a, b)
    assert combined._waits == (a, b)


def test_wait_combine_call_rebuilds_children():
    a = wg.expo(base=2)
    b = wg.expo(base=3)
    combined = wait_combine(a, b)
    new_combined = combined(base=10)
    assert isinstance(new_combined, wg._CombinedWait)
    assert new_combined.next() == 2.0  # 1 + 1


# ---------------------------------------------------------------------------
# Combined waits via __add__/__radd__
# ---------------------------------------------------------------------------

def test_add_two_waits_flattens():
    a = _FixedWait(1)
    b = _FixedWait(2)
    combined = a + b
    assert isinstance(combined, wg._CombinedWait)
    assert combined._waits == (a, b)
    assert combined.next() == 3


def test_add_wait_and_combined_flattens():
    a = _FixedWait(1)
    b = _FixedWait(2)
    c = _FixedWait(4)
    combined = a + (b + c)
    assert combined._waits == (a, b, c)
    assert combined.next() == 7


def test_combined_add_wait_flattens():
    a = _FixedWait(1)
    b = _FixedWait(2)
    c = _FixedWait(4)
    combined = wg._CombinedWait(a, b) + c
    assert combined._waits == (a, b, c)
    assert combined.next() == 7


def test_combined_add_combined_flattens():
    a = _FixedWait(1)
    b = _FixedWait(2)
    c = _FixedWait(4)
    d = _FixedWait(8)
    combined = wg._CombinedWait(a, b) + wg._CombinedWait(c, d)
    assert combined._waits == (a, b, c, d)
    assert combined.next() == 15


def test_factory_add_wait():
    combined = wg.expo + wg.fibo
    assert isinstance(combined, wg._CombinedWait)
    assert len(combined._waits) == 2
    assert combined.next() == 2.0  # expo 1 + fibo 1


def test_factory_add_combined():
    combined = wg.expo + wg.fibo + wg.constant
    assert isinstance(combined, wg._CombinedWait)
    assert len(combined._waits) == 3
    assert combined.next() == 3.0


def test_factory_radd_wait():
    combined = _FixedWait(1) + wg.expo
    assert isinstance(combined, wg._CombinedWait)
    assert combined.next() == 2.0


def test_factory_radd_combined():
    combined = wg._CombinedWait(_FixedWait(1)) + wg.fibo
    assert isinstance(combined, wg._CombinedWait)
    assert combined.next() == 2.0


# ---------------------------------------------------------------------------
# _Wait.__call__ merging behavior
# ---------------------------------------------------------------------------

def test_wait_call_returns_new_instance_same_class():
    e = wg.expo(base=2)
    e2 = e(base=10)
    assert isinstance(e2, wg._Expo)
    assert e2 is not e
    assert e2.next() == 1.0
    assert e2.next() == 10.0


def test_wait_call_overrides_kwargs():
    e = wg.expo(base=2, factor=3)
    e2 = e(base=5)
    assert e2.next() == 3.0
    assert e2.next() == 15.0


def test_wait_call_with_positional_args():
    e = wg.expo(base=2)
    # positional base argument
    e2 = e(3)
    assert e2.next() == 1.0
    assert e2.next() == 3.0


# ---------------------------------------------------------------------------
# _Wait.next base raises NotImplementedError
# ---------------------------------------------------------------------------

def test_base_wait_next_raises():
    w = wg._Wait()
    with pytest.raises(NotImplementedError):
        w.next()


# ---------------------------------------------------------------------------
# Other wait generators (light coverage of public factories)
# ---------------------------------------------------------------------------

def test_expo_values():
    e = wg.expo()
    assert [e.next() for _ in range(4)] == [1.0, 2.0, 4.0, 8.0]


def test_expo_max_value():
    e = wg.expo(base=2, factor=1, max_value=3)
    assert [e.next() for _ in range(4)] == [1.0, 2.0, 3.0, 3.0]


def test_decay_values_approx():
    d = wg.decay(initial_value=1, decay_factor=1)
    v1 = d.next()
    v2 = d.next()
    assert v1 == pytest.approx(1.0)
    assert v2 == pytest.approx(math.e ** -1)


def test_decay_min_value():
    d = wg.decay(initial_value=1, decay_factor=1, min_value=0.5)
    vals = [d.next() for _ in range(3)]
    assert vals[0] == pytest.approx(1.0)
    assert vals[2] == 0.5


def test_fibo_values():
    f = wg.fibo()
    assert [f.next() for _ in range(6)] == [1.0, 1.0, 2.0, 3.0, 5.0, 8.0]


def test_fibo_max_value():
    f = wg.fibo(max_value=3)
    assert [f.next() for _ in range(5)] == [1.0, 1.0, 2.0, 3.0, 3.0]


def test_constant_scalar():
    c = wg.constant(2.5)
    assert [c.next() for _ in range(3)] == [2.5, 2.5, 2.5]


def test_constant_sequence():
    c = wg.constant([1, 2, 3])
    assert [c.next() for _ in range(3)] == [1.0, 2.0, 3.0]
    with pytest.raises(StopIteration):
        c.next()


def test_runtime():
    r = wg.runtime(value=lambda sv: (sv or 0) + 1)
    assert r.next() == 1
    assert r.next(10) == 11


def test_wait_incrementing():
    w = wg.wait_incrementing(start=1, increment=2)
    assert [w.next() for _ in range(4)] == [1, 3, 5, 7]


def test_wait_incrementing_max_value():
    w = wg.wait_incrementing(start=1, increment=2, max_value=4)
    assert [w.next() for _ in range(4)] == [1, 3, 4, 4]


def test_wait_none():
    n = wg.wait_none()
    assert n.next() == 0.0
    assert n.next("x") == 0.0


def test_wait_exception():
    w = wg.wait_exception(value=lambda sv: 42.0)
    assert w.next() == 42.0


def test_wait_random_range():
    w = wg.wait_random(min=2, max=2)  # degenerate => deterministic
    assert w.next() == 2.0


def test_wait_random_exponential_bounds():
    random.seed(0)
    w = wg.wait_random_exponential(multiplier=1, max_value=4, exp_base=2, min_value=0)
    for _ in range(5):
        v = w.next()
        assert 0 <= v <= 4


def test_wait_exponential_jitter_bounds():
    random.seed(0)
    w = wg.wait_exponential_jitter(initial=1, max=10, exp_base=2, jitter=0)
    # with jitter=0, delay = min(1 * 2**n, 10)
    # n starts at 1
    assert w.next() == 2.0
    assert w.next() == 4.0
    assert w.next() == 8.0
    assert w.next() == 10.0
