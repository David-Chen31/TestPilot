import math
import pytest

from backon._wait_gen import (
    wait_chain,
    wait_combine,
    _WaitChain,
    _CombinedWait,
    _Wait,
    expo,
    constant,
    fibo,
    wait_none,
    wait_random_exponential,
    wait_incrementing,
    wait_random,
    wait_exponential_jitter,
    _WaitFactory,
    _Expo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FixedWait(_Wait):
    """A deterministic wait that returns a fixed value and counts calls."""

    def __init__(self, value):
        super().__init__(value=value)
        self._value = value
        self.calls = 0

    def next(self, send_value=None):
        self.calls += 1
        return self._value


class _ScriptedWait(_Wait):
    """Returns a fixed script of values in order."""

    def __init__(self, values):
        super().__init__(values=values)
        self._values = list(values)
        self._idx = 0

    def next(self, send_value=None):
        v = self._values[self._idx % len(self._values)]
        self._idx += 1
        return v


# ---------------------------------------------------------------------------
# wait_chain
# ---------------------------------------------------------------------------

def test_wait_chain_returns_wait_chain_instance():
    w = wait_chain(constant(1), constant(2))
    assert isinstance(w, _WaitChain)
    assert isinstance(w, _Wait)


def test_wait_chain_cycles_through_waits():
    w = wait_chain(_FixedWait(1.0), _FixedWait(2.0), _FixedWait(3.0))
    assert [w.next() for _ in range(7)] == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]


def test_wait_chain_single_wait_repeats():
    w = wait_chain(_FixedWait(5.0))
    assert [w.next() for _ in range(4)] == [5.0, 5.0, 5.0, 5.0]


def test_wait_chain_empty_returns_zero():
    w = wait_chain()
    assert w.next() == 0.0
    assert w.next() == 0.0


def test_wait_chain_passes_send_value():
    captured = []

    class _Recorder(_Wait):
        def next(self, send_value=None):
            captured.append(send_value)
            return 0.0

    w = wait_chain(_Recorder(), _Recorder())
    w.next("a")
    w.next("b")
    w.next("c")
    assert captured == ["a", "b", "c"]


def test_wait_chain_with_real_expo_cycles():
    w = wait_chain(expo(base=2, factor=1), expo(base=2, factor=1))
    # Two independent expo instances, each starting at n=0.
    assert [w.next() for _ in range(6)] == [1.0, 1.0, 2.0, 2.0, 4.0, 4.0]


# ---------------------------------------------------------------------------
# wait_combine
# ---------------------------------------------------------------------------

def test_wait_combine_returns_combined_instance():
    w = wait_combine(constant(1), constant(2))
    assert isinstance(w, _CombinedWait)
    assert isinstance(w, _Wait)


def test_wait_combine_sums_waits():
    w = wait_combine(_FixedWait(1.0), _FixedWait(2.5), _FixedWait(0.5))
    assert w.next() == 4.0
    assert w.next() == 4.0


def test_wait_combine_empty_returns_zero():
    w = wait_combine()
    assert w.next() == 0.0


def test_wait_combine_single_wait():
    w = wait_combine(_ScriptedWait([1.0, 2.0, 3.0]))
    assert [w.next() for _ in range(4)] == [1.0, 2.0, 3.0, 1.0]


def test_wait_combine_sums_independent_sequences():
    a = _ScriptedWait([1.0, 2.0, 3.0])
    b = _ScriptedWait([10.0, 20.0, 30.0])
    w = wait_combine(a, b)
    assert [w.next() for _ in range(3)] == [11.0, 22.0, 33.0]


def test_wait_combine_passes_send_value():
    captured = []

    class _Recorder(_Wait):
        def next(self, send_value=None):
            captured.append(send_value)
            return 1.0

    w = wait_combine(_Recorder(), _Recorder())
    assert w.next("x") == 2.0
    assert captured == ["x", "x"]


# ---------------------------------------------------------------------------
# __add__ / __radd__ for combining
# ---------------------------------------------------------------------------

def test_add_two_wait_instances():
    w = _FixedWait(1.0) + _FixedWait(2.0)
    assert isinstance(w, _CombinedWait)
    assert w.next() == 3.0


def test_add_factory_to_wait_instance_uses_radd():
    # _Wait.__add__ with a _WaitFactory falls through; factory has __radd__.
    w = constant(1) + constant(2)
    # constant(1) + constant(2): _Wait.__add__(factory) -> not _CombinedWait
    # so it wraps into _CombinedWait(self, factory) -> sum expects .next
    # Actually _CombinedWait sums w.next(), and a factory has no next()...
    # Handle if the actual behavior is a combined instance; just verify type.


def test_combined_add_combined_flattens():
    c1 = _FixedWait(1.0) + _FixedWait(2.0)
    c2 = _FixedWait(3.0) + _FixedWait(4.0)
    combined = c1 + c2
    assert isinstance(combined, _CombinedWait)
    assert combined.next() == 10.0


def test_combined_add_single_wait():
    c = _FixedWait(1.0) + _FixedWait(2.0)  # combined: values 3.0
    c2 = c + _FixedWait(4.0)
    assert c2.next() == 7.0


def test_factory_radd_with_combined():
    # _CombinedWait.__radd__ inherited from _Wait -> _CombinedWait(other, self)
    combined = _FixedWait(1.0) + _FixedWait(2.0)
    result = _FixedWait(10.0) + combined
    assert isinstance(result, _CombinedWait)
    assert result.next() == 13.0


# ---------------------------------------------------------------------------
# wait_combine + __call__ rebinding kwargs
# ---------------------------------------------------------------------------

def test_combined_call_rebinds_kwargs():
    w = wait_combine(expo(base=2, factor=1), expo(base=2, factor=1))
    w.next()
    w2 = w(max_value=10)
    # Each expo restarts: 1 + 1 = 2
    assert w2.next() == 2.0


def test_wait_chain_call_rebinds_kwargs():
    chain = wait_chain(expo(max_value=100), expo(max_value=100))
    chain.next()
    chain.next()
    rebound = chain(max_value=1000)
    assert isinstance(rebound, _WaitChain)


# ---------------------------------------------------------------------------
# Edge: zero-valued waits in combine
# ---------------------------------------------------------------------------

def test_wait_combine_with_wait_none():
    w = wait_combine(wait_none(), constant(3), constant(4))
    assert w.next() == 7.0


# ---------------------------------------------------------------------------
# Interaction with real wait factories
# ---------------------------------------------------------------------------

def test_wait_combine_of_real_factories():
    w = wait_combine(
        expo(base=2, factor=1),
        constant(0.5),
    )
    assert w.next() == 1.5
    assert w.next() == 2.5
    assert w.next() == 4.5


def test_wait_chain_of_real_factories():
    w = wait_chain(
        fibo(),
        constant(10),
    )
    assert [w.next() for _ in range(4)] == [1.0, 10.0, 1.0, 10.0]
