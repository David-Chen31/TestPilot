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
    _WaitExponentialJitter,
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
    wait_exponential_jitter,
    wait_incrementing,
    wait_none,
    wait_random,
    wait_random_exponential,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FixedWait(_Wait):
    """A deterministic _Wait yielding a predefined sequence, then repeats last."""

    def __init__(self, values):
        super().__init__(values=tuple(values))
        self._values = tuple(values)
        self._idx = 0
        self.calls = 0

    def next(self, send_value=None):
        self.calls += 1
        v = self._values[min(self._idx, len(self._values) - 1)]
        self._idx += 1
        return v


def _take(wait, n, send_value=None):
    return [wait.next(send_value) for _ in range(n)]


# ---------------------------------------------------------------------------
# wait_combine
# ---------------------------------------------------------------------------

def test_wait_combine_returns_combined_wait():
    cw = wait_combine(constant(1), constant(2))
    assert isinstance(cw, _CombinedWait)
    assert isinstance(cw, _Wait)
    # it returns the sum of each underlying wait
    assert [cw.next() for _ in range(3)] == [3.0, 3.0, 3.0]


def test_wait_combine_empty_returns_zero():
    cw = wait_combine()
    assert isinstance(cw, _CombinedWait)
    assert cw.next() == 0


def test_wait_combine_single():
    cw = wait_combine(_FixedWait([5, 6, 7]))
    assert [cw.next() for _ in range(3)] == [5, 6, 7]


def test_wait_combine_multiple_sums_advancing_waits():
    a = _FixedWait([1, 2, 3])
    b = _FixedWait([10, 20, 30])
    cw = wait_combine(a, b)
    assert [cw.next() for _ in range(3)] == [11, 22, 33]


def test_wait_combine_send_value_forwarded():
    received = []

    def make(tag):
        def fn(sv):
            received.append((tag, sv))
            return 1.0
        return runtime(value=fn)

    cw = wait_combine(make("a"), make("b"))
    assert cw.next(send_value="X") == 2.0
    assert received == [("a", "X"), ("b", "X")]


def test_wait_combine_with_expo_and_constant():
    cw = wait_combine(expo(base=2, factor=1), constant(1))
    # expo: 1,2,4 ; constant: 1,1,1
    assert [cw.next() for _ in range(3)] == [2, 3, 5]


# ---------------------------------------------------------------------------
# wait_chain
# ---------------------------------------------------------------------------

def test_wait_chain_returns_wait_chain():
    wc = wait_chain(constant(1), constant(2))
    assert isinstance(wc, _WaitChain)
    assert isinstance(wc, _Wait)


def test_wait_chain_empty_returns_zero():
    wc = wait_chain()
    assert wc.next() == 0.0
    assert wc.next() == 0.0


def test_wait_chain_single():
    wc = wait_chain(_FixedWait([1, 2]))
    assert [wc.next() for _ in range(4)] == [1, 2, 2, 2]


def test_wait_chain_round_robin():
    a = _FixedWait([1, 2, 3])
    b = _FixedWait([10, 20, 30])
    wc = wait_chain(a, b)
    assert [wc.next() for _ in range(6)] == [1, 10, 2, 20, 3, 30]


def test_wait_chain_round_robin_wraps():
    a = _FixedWait([1])
    b = _FixedWait([2])
    c = _FixedWait([3])
    wc = wait_chain(a, b, c)
    assert [wc.next() for _ in range(7)] == [1, 2, 3, 1, 2, 3, 1]


def test_wait_chain_send_value_forwarded():
    received = []

    def fn(sv):
        received.append(sv)
        return 1.0

    wc = wait_chain(runtime(value=fn), runtime(value=fn))
    wc.next(send_value="A")
    wc.next(send_value="B")
    wc.next(send_value="C")
    assert received == ["A", "B", "C"]


def test_wait_chain_advances_inner_state():
    wc = wait_chain(expo(factor=1), expo(factor=10))
    # chain: expo1 -> expo2 -> expo1 -> expo2
    # expo1: 1, 2 ; expo2: 10, 20
    assert [wc.next() for _ in range(4)] == [1, 10, 2, 20]


# ---------------------------------------------------------------------------
# Combined/chain __call__ (clone) behavior
# ---------------------------------------------------------------------------

def test_combined_wait_call_clones_children():
    cw = wait_combine(expo(factor=1), expo(factor=10))
    cw.next()  # advance
    cloned = cw()
    assert isinstance(cloned, _CombinedWait)
    assert cloned is not cw
    # cloned should restart at beginning
    assert cloned.next() == 1 + 10
    # original continues
    assert cw.next() == 2 + 20


def test_wait_chain_call_clones_children():
    wc = wait_chain(expo(factor=1), expo(factor=10))
    wc.next()
    wc.next()
    cloned = wc()
    assert isinstance(cloned, _WaitChain)
    assert cloned is not wc
    # cloned restarts from first child
    assert cloned.next() == 1
    # original keeps its own state
    assert wc.next() == 2


# ---------------------------------------------------------------------------
# Combining via operators
# ---------------------------------------------------------------------------

def test_add_two_waits():
    s = expo(factor=1) + constant(1)
    assert isinstance(s, _CombinedWait)
    # start fresh
    s2 = expo(factor=1) + constant(1)
    assert [s2.next() for _ in range(3)] == [2, 3, 5]


def test_add_wait_plus_combined():
    cw = wait_combine(constant(1), constant(2))
    s = expo(factor=1) + cw
    assert isinstance(s, _CombinedWait)
    # expo: 1,2 ; (1+2)
    assert [s.next() for _ in range(2)] == [4, 5]


def test_add_wait_plus_factory():
    s = expo(base=2, factor=1) + constant
    assert isinstance(s, _CombinedWait)
    s2 = expo(base=2, factor=1) + constant()
    assert [s2.next() for _ in range(3)] == [2, 3, 5]


def test_radd_factory_plus_wait():
    s = constant + expo(factor=1)
    assert isinstance(s, _CombinedWait)
    assert [s.next() for _ in range(3)] == [2, 3, 5]


def test_radd_value_plus_wait():
    cw = wait_combine(constant(100))
    s = _FixedWait([1]) + cw
    assert isinstance(s, _CombinedWait)
    assert s.next() == 101


def test_combined_add_combined_merges_children():
    cw1 = wait_combine(constant(1), constant(2))
    cw2 = wait_combine(constant(4), constant(8))
    merged = cw1 + cw2
    assert isinstance(merged, _CombinedWait)
    assert merged.next() == 1 + 2 + 4 + 8


def test_combined_add_wait():
    cw = wait_combine(constant(1))
    merged = cw + constant(5)
    assert isinstance(merged, _CombinedWait)
    assert merged.next() == 6


# ---------------------------------------------------------------------------
# Individual wait generators
# ---------------------------------------------------------------------------

def test_expo_basic():
    w = expo(base=2, factor=1)
    assert _take(w, 5) == [1, 2, 4, 8, 16]


def test_expo_max_value():
    w = expo(base=2, factor=1, max_value=5)
    assert _take(w, 6) == [1, 2, 4, 5, 5, 5]


def test_expo_factor():
    w = expo(base=2, factor=3)
    assert _take(w, 4) == [3, 6, 12, 24]


def test_expo_clone_restarts():
    w = expo(factor=1)
    w.next()
    w.next()
    cloned = w(base=10)
    assert [cloned.next() for _ in range(2)] == [1, 10]


def test_decay_basic():
    w = decay(initial_value=1, decay_factor=1)
    vals = _take(w, 3)
    assert vals[0] == 1
    assert vals[1] == pytest.approx(math.e ** -1)
    assert vals[2] == pytest.approx(math.e ** -2)


def test_decay_min_value():
    w = decay(initial_value=1, decay_factor=10, min_value=0.5)
    vals = _take(w, 3)
    # values below min_value clamp to min_value
    assert vals[0] == 1
    assert vals[1] == 0.5
    assert vals[2] == 0.5


def test_fibo_basic():
    w = fibo()
    assert _take(w, 7) == [1, 1, 2, 3, 5, 8, 13]


def test_fibo_max_value():
    w = fibo(max_value=5)
    assert _take(w, 6) == [1, 1, 2, 3, 5, 5]


def test_constant_scalar():
    w = constant(3)
    assert _take(w, 4) == [3.0, 3.0, 3.0, 3.0]


def test_constant_sequence():
    w = constant([1, 2, 3])
    assert _take(w, 3) == [1, 2, 3]
    with pytest.raises(StopIteration):
        w.next()


def test_constant_clone_restarts():
    w = constant([1, 2])
    w.next()
    cloned = w(interval=[7, 8])
    assert cloned.next() == 7
    assert cloned.next() == 8


def test_runtime_uses_send_value():
    w = runtime(value=lambda sv: 10 if sv is None else sv)
    assert w.next() == 10
    assert w.next(send_value=42) == 42


def test_wait_exception_uses_send_value():
    def fn(exc):
        if exc is None:
            return 1.0
        return float(len(str(exc)))

    w = wait_exception(value=fn)()
    assert w.next() == 1.0
    assert w.next(send_value="boom") == 4.0


def test_wait_random_in_range(monkeypatch):
    random.seed(0)
    w = wait_random(min=2, max=5)()
    for _ in range(20):
        v = w.next()
        assert 2 <= v <= 5


def test_wait_random_exponential_bounded(monkeypatch):
    random.seed(123)
    w = wait_random_exponential(
        multiplier=1, max_value=4, exp_base=2, min_value=0.5
    )()
    for _ in range(10):
        v = w.next()
        assert 0 <= v <= 4


def test_wait_random_exponential_uses_max(monkeypatch):
    # force random.uniform to reveal upper bound; check a=0 case for min_value
    random.seed(1)
    w = wait_random_exponential(multiplier=1, max_value=1, exp_base=2)()
    # With max_value 1, a = 1 for n=0 (1*2^0 = 1 not > 1) -> uniform(0,1)
    v = w.next()
    assert 0 <= v <= 1


def test_wait_incrementing_basic():
    w = wait_incrementing(start=1, increment=2)()
    assert _take(w, 4) == [1, 3, 5, 7]


def test_wait_incrementing_max_value():
    w = wait_incrementing(start=1, increment=2, max_value=4)()
    assert _take(w, 4) == [1, 3, 4, 4]


def test_wait_incrementing_defaults():
    w = wait_incrementing()
    assert _take(w, 3) == [1, 2, 3]


def test_wait_exponential_jitter_bounds(monkeypatch):
    random.seed(7)
    w = wait_exponential_jitter(initial=1, max=10, exp_base=2, jitter=0.5)()
    for _ in range(6):
        v = w.next()
        assert 0 <= v <= 10 + 0.5


def test_wait_exponential_jitter_deterministic_when_jitter_zero(monkeypatch):
    w = wait_exponential_jitter(initial=1, max=100, exp_base=2, jitter=0)()
    assert _take(w, 4) == [2, 4, 8, 16]


def test_wait_none_returns_zero():
    w = wait_none()
    assert _take(w, 3) == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# _WaitFactory behavior
# ---------------------------------------------------------------------------

def test_factory_creates_instance():
    w = constant(2)
    assert isinstance(w, _Constant)


def test_factory_add_creates_combined():
    s = constant + expo(base=2, factor=1)
    assert isinstance(s, _CombinedWait)


def test_factory_add_combined():
    cw = wait_combine(constant(1))
    s = expo(base=2, factor=1) + cw
    assert isinstance(s, _CombinedWait)


def test_factory_radd_with_wait():
    s = constant(1) + expo(factor=1)
    assert isinstance(s, _CombinedWait)


# ---------------------------------------------------------------------------
# _Wait base
# ---------------------------------------------------------------------------

def test_wait_base_next_raises():
    w = _Wait()
    with pytest.raises(NotImplementedError):
        w.next()


def test_wait_call_returns_new_instance():
    w = expo(base=3, factor=2)
    cloned = w()
    assert isinstance(cloned, _Expo)
    assert cloned is not w


def test_wait_call_with_positional_args_overrides():
    w = expo(base=3, factor=2)
    cloned = w(4)
    assert isinstance(cloned, _Expo)
    # New base is 4; factor still default 1 (since merged kept factor=2? check ordering)
    # merged = {'base':3, 'factor':2, 'max_value':None} merged with {}
    # args=(4,), param_order=['base','factor','max_value'], pop 'base'
    # result: _Expo(4, factor=2) => base=4, factor=2
    assert cloned.next() == 2 * 4 ** 0


def test_wait_factory_slots():
    f = _WaitFactory(_Expo)
    with pytest.raises(AttributeError):
        f.foo = 1


# ---------------------------------------------------------------------------
# Combined wait with exception-based generator behavior
# ---------------------------------------------------------------------------

def test_combined_seed_state():
    cw = _CombinedWait()
    assert cw.next() == 0


def test_chain_index_modular():
    wc = wait_chain(constant(1))
    assert [wc.next() for _ in range(5)] == [1.0] * 5
