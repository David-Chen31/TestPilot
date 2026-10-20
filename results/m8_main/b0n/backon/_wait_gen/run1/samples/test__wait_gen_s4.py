import itertools
import math
from unittest import mock

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


# --------------------------------------------------------------------------- #
# Basic building blocks
# --------------------------------------------------------------------------- #
class _FixedWait(_Wait):
    """A tiny deterministic wait used to test composition helpers."""

    def __init__(self, value):
        super().__init__()
        self._value = value

    def next(self, send_value=None):
        return self._value


# --------------------------------------------------------------------------- #
# _Wait base behavior
# --------------------------------------------------------------------------- #
def test_wait_next_not_implemented():
    w = _Wait()
    with pytest.raises(NotImplementedError):
        w.next()


def test_wait_call_returns_new_instance_of_same_class():
    w = _Expo()
    w2 = w()
    assert isinstance(w2, _Expo)
    assert w2 is not w


def test_wait_add_wraps_in_combined():
    a = _FixedWait(1)
    b = _FixedWait(2)
    combined = a + b
    assert isinstance(combined, _CombinedWait)
    assert combined._waits == (a, b)


def test_wait_add_with_combined_flattens():
    a = _FixedWait(1)
    b = _FixedWait(2)
    c = _FixedWait(3)
    combined = a + b
    combined2 = combined + c
    assert isinstance(combined2, _CombinedWait)
    assert combined2._waits == (a, b, c)


def test_wait_radd():
    a = _FixedWait(1)
    b = _FixedWait(2)
    combined = b.__radd__(a)
    assert isinstance(combined, _CombinedWait)
    assert combined._waits == (a, b)


# --------------------------------------------------------------------------- #
# _CombinedWait
# --------------------------------------------------------------------------- #
def test_combined_next_sums():
    combined = _CombinedWait(_FixedWait(1.5), _FixedWait(2.5))
    assert combined.next() == pytest.approx(4.0)


def test_combined_next_empty_is_zero():
    combined = _CombinedWait()
    assert combined.next() == 0.0


def test_combined_call_applies_kwargs():
    combined = _CombinedWait(_Expo(), _Expo())
    new = combined(base=3)
    assert isinstance(new, _CombinedWait)
    assert all(isinstance(w, _Expo) for w in new._waits)
    assert all(w._base == 3 for w in new._waits)
    # originals unchanged
    assert all(w._base == 2 for w in combined._waits)


def test_combined_add_combined():
    a = _FixedWait(1)
    b = _FixedWait(2)
    c = _FixedWait(3)
    d = _FixedWait(4)
    result = (a + b) + (c + d)
    assert isinstance(result, _CombinedWait)
    assert result._waits == (a, b, c, d)


def test_combined_add_single():
    a = _FixedWait(1)
    b = _FixedWait(2)
    c = _FixedWait(3)
    result = (a + b) + c
    assert result._waits == (a, b, c)


# --------------------------------------------------------------------------- #
# _WaitFactory
# --------------------------------------------------------------------------- #
def test_factory_call_creates_instance():
    w = expo(base=3)
    assert isinstance(w, _Expo)
    assert w._base == 3


def test_factory_add_creates_combined():
    combined = expo() + constant()
    assert isinstance(combined, _CombinedWait)
    assert isinstance(combined._waits[0], _Expo)
    assert isinstance(combined._waits[1], _Constant)


def test_factory_add_combined_flattens():
    combined = (expo() + constant()) + fibo()
    assert isinstance(combined, _CombinedWait)
    assert len(combined._waits) == 3
    assert isinstance(combined._waits[0], _Expo)
    assert isinstance(combined._waits[1], _Constant)
    assert isinstance(combined._waits[2], _Fibo)


def test_factory_radd_single():
    a = _Constant(interval=5)
    combined = a.__radd__(expo())
    # radd called on factory: expo.__radd__(a) would be interpreted; simulate
    combined = expo.__radd__(a)
    assert isinstance(combined, _CombinedWait)
    assert combined._waits[0] is a
    assert isinstance(combined._waits[1], _Expo)


def test_factory_radd_combined():
    a = _Constant(interval=5)
    combined = expo() + constant()
    result = expo.__radd__(combined)
    # When other is a _CombinedWait the factory's __radd__ handles it
    # but `a` here is not combined - this test targets combined branch
    combined2 = expo() + constant()
    result2 = expo.__radd__(combined2)
    assert isinstance(result2, _CombinedWait)
    assert isinstance(result2._waits[-1], _Expo)


# --------------------------------------------------------------------------- #
# _Expo
# --------------------------------------------------------------------------- #
def test_expo_defaults():
    w = _Expo()
    values = [w.next() for _ in range(4)]
    assert values == [1.0, 2.0, 4.0, 8.0]


def test_expo_base_factor():
    w = _Expo(base=3, factor=2)
    values = [w.next() for _ in range(3)]
    assert values == [2.0, 6.0, 18.0]


def test_expo_max_value_caps():
    w = _Expo(base=2, factor=1, max_value=3)
    values = [w.next() for _ in range(5)]
    assert values == [1.0, 2.0, 3, 3, 3]


def test_expo_call_resets_and_overrides():
    w = _Expo(base=2, factor=1)
    w.next()
    w2 = w(base=5)
    assert isinstance(w2, _Expo)
    assert w2._base == 5
    assert w2._n == 0


def test_expo_call_positional_args_override():
    w = _Expo(base=2, factor=1)
    w2 = w(3)
    assert isinstance(w2, _Expo)
    assert w2._base == 3


# --------------------------------------------------------------------------- #
# _Decay
# --------------------------------------------------------------------------- #
def test_decay_defaults():
    w = _Decay()
    v0 = w.next()
    v1 = w.next()
    assert v0 == pytest.approx(1.0)
    assert v1 == pytest.approx(math.e ** -1)


def test_decay_with_factor():
    w = _Decay(initial_value=10, decay_factor=2)
    v0 = w.next()
    v1 = w.next()
    assert v0 == pytest.approx(10.0)
    assert v1 == pytest.approx(10 * math.e ** -2)


def test_decay_min_value_caps():
    w = _Decay(initial_value=1, decay_factor=1, min_value=0.5)
    values = [w.next() for _ in range(5)]
    assert values[0] == pytest.approx(1.0)
    assert all(v >= 0.5 for v in values)
    assert values[-1] == 0.5


# --------------------------------------------------------------------------- #
# _Fibo
# --------------------------------------------------------------------------- #
def test_fibo_sequence():
    w = _Fibo()
    values = [w.next() for _ in range(6)]
    assert values == [1.0, 1.0, 2.0, 3.0, 5.0, 8.0]


def test_fibo_max_value_caps():
    w = _Fibo(max_value=5)
    values = [w.next() for _ in range(6)]
    assert values == [1.0, 1.0, 2.0, 3.0, 5.0, 5.0]


# --------------------------------------------------------------------------- #
# _Constant
# --------------------------------------------------------------------------- #
def test_constant_numeric():
    w = _Constant(interval=2)
    assert [w.next() for _ in range(3)] == [2.0, 2.0, 2.0]


def test_constant_sequence():
    w = _Constant(interval=[1.0, 2.0, 3.0])
    assert [w.next() for _ in range(3)] == [1.0, 2.0, 3.0]


def test_constant_sequence_exhausted():
    w = _Constant(interval=[1.0])
    assert w.next() == 1.0
    with pytest.raises(StopIteration):
        w.next()


# --------------------------------------------------------------------------- #
# _Runtime / _WaitException
# --------------------------------------------------------------------------- #
def test_runtime_calls_callable():
    w = _Runtime(value=lambda x: 42 if x is None else x + 1)
    assert w.next() == 42
    assert w.next(5) == 6


def test_wait_exception_calls_callable():
    w = _WaitException(value=lambda x: 99.0)
    assert w.next() == 99.0


# --------------------------------------------------------------------------- #
# _WaitRandom (deterministic via patch)
# --------------------------------------------------------------------------- #
def test_wait_random_uses_uniform(monkeypatch):
    calls = []

    def fake_uniform(a, b):
        calls.append((a, b))
        return 0.5

    monkeypatch.setattr("backon._wait_gen.random.uniform", fake_uniform)
    w = _WaitRandom(min=2, max=4)
    assert w.next() == 0.5
    assert calls == [(2, 4)]


# --------------------------------------------------------------------------- #
# _WaitRandomExponential
# --------------------------------------------------------------------------- #
def test_wait_random_exponential_uniform_bounds(monkeypatch):
    calls = []

    def fake_uniform(a, b):
        calls.append((a, b))
        return a

    monkeypatch.setattr("backon._wait_gen.random.uniform", fake_uniform)
    w = _WaitRandomExponential(multiplier=1, max_value=10, exp_base=2)
    for _ in range(5):
        w.next()
    bounds = [b for _, b in calls]
    assert bounds == [1, 2, 4, 8, 10]


def test_wait_random_exponential_min_value(monkeypatch):
    def fake_uniform(a, b):
        return a

    monkeypatch.setattr("backon._wait_gen.random.uniform", fake_uniform)
    w = _WaitRandomExponential(multiplier=1, min_value=5, max_value=3)
    # a is capped to max_value 3 then to min 5 -> bounds become 0..5
    result = w.next()
    assert result == 0


# --------------------------------------------------------------------------- #
# _WaitIncrementing
# --------------------------------------------------------------------------- #
def test_wait_incrementing():
    w = _WaitIncrementing(start=1, increment=2)
    values = [w.next() for _ in range(4)]
    assert values == [1, 3, 5, 7]


def test_wait_incrementing_max_caps():
    w = _WaitIncrementing(start=1, increment=2, max_value=4)
    values = [w.next() for _ in range(4)]
    assert values == [1, 3, 4, 4]


# --------------------------------------------------------------------------- #
# _WaitExponentialJitter
# --------------------------------------------------------------------------- #
def test_wait_exponential_jitter(monkeypatch):
    monkeypatch.setattr("backon._wait_gen.random.uniform", lambda a, b: 0.0)
    w = _WaitExponentialJitter(initial=1, max=100, exp_base=2, jitter=1)
    values = [w.next() for _ in range(4)]
    # starts at n=1, so 2**1, 2**2, ...
    assert values == [2, 4, 8, 16]


def test_wait_exponential_jitter_max(monkeypatch):
    monkeypatch.setattr("backon._wait_gen.random.uniform", lambda a, b: 0.0)
    w = _WaitExponentialJitter(initial=1, max=5, exp_base=2, jitter=1)
    values = [w.next() for _ in range(4)]
    assert values == [2, 4, 5, 5]


def test_wait_exponential_jitter_adds_jitter(monkeypatch):
    monkeypatch.setattr("backon._wait_gen.random.uniform", lambda a, b: 0.5)
    w = _WaitExponentialJitter(initial=1, max=100, exp_base=2, jitter=2)
    assert w.next() == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# _WaitNone
# --------------------------------------------------------------------------- #
def test_wait_none():
    w = _WaitNone()
    assert w.next() == 0.0
    assert w.next() == 0.0


# --------------------------------------------------------------------------- #
# _WaitChain
# --------------------------------------------------------------------------- #
def test_wait_chain_cycles():
    chain = wait_chain(_FixedWait(1), _FixedWait(2), _FixedWait(3))
    assert isinstance(chain, _WaitChain)
    values = [chain.next() for _ in range(7)]
    assert values == [1, 2, 3, 1, 2, 3, 1]


def test_wait_chain_empty():
    chain = _WaitChain()
    assert chain.next() == 0.0
    assert chain.next() == 0.0


def test_wait_chain_call_rebuilds():
    chain = wait_chain(_Expo(base=2), _Expo(base=3))
    new = chain(base=10)
    assert isinstance(new, _WaitChain)
    assert [w._base for w in new._waits] == [10, 10]
    # original unchanged
    assert [w._base for w in chain._waits] == [2, 3]


# --------------------------------------------------------------------------- #
# wait_combine
# --------------------------------------------------------------------------- #
def test_wait_combine():
    combined = wait_combine(_FixedWait(1), _FixedWait(2))
    assert isinstance(combined, _CombinedWait)
    assert combined.next() == 3


def test_wait_combine_empty():
    combined = wait_combine()
    assert combined.next() == 0.0


def test_wait_combine_public_with_real_waits():
    combined = wait_combine(expo(), constant(interval=1))
    v_expo = combined.next()
    # first expo is 1, first constant is 1
    assert v_expo == 2.0


# --------------------------------------------------------------------------- #
# Factories returning expected types
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "factory,cls",
    [
        (expo, _Expo),
        (decay, _Decay),
        (fibo, _Fibo),
        (constant, _Constant),
        (runtime, _Runtime),
        (wait_random_exponential, _WaitRandomExponential),
        (wait_incrementing, _WaitIncrementing),
        (wait_exception, _WaitException),
        (wait_random, _WaitRandom),
        (wait_exponential_jitter, _WaitExponentialJitter),
        (wait_none, _WaitNone),
    ],
)
def test_factories_return_correct_type(factory, cls):
    if cls in (_Runtime, _WaitException):
        w = factory(value=lambda x: 1.0)
    else:
        w = factory()
    assert isinstance(w, cls)


# --------------------------------------------------------------------------- #
# Chained / combined integration with real primitives
# --------------------------------------------------------------------------- #
def test_combined_with_chain():
    chain = wait_chain(_FixedWait(1), _FixedWait(2))
    combined = wait_combine(chain, _FixedWait(10))
    assert combined.next() == 11
    assert combined.next() == 12
    assert combined.next() == 11


def test_chain_with_no_args_returns_zero_forever():
    chain = wait_chain()
    for _ in range(3):
        assert chain.next() == 0.0
