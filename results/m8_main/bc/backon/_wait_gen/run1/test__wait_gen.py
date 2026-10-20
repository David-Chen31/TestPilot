import math
import random

import pytest

from backon._wait_gen import (
    wait_chain,
    wait_combine,
    wait_none,
    wait_random_exponential,
    wait_incrementing,
    wait_random,
    wait_exponential_jitter,
    constant,
    expo,
    decay,
    fibo,
    runtime,
    wait_exception,
    _Wait,
    _CombinedWait,
    _WaitChain,
    _WaitFactory,
    _Expo,
    _Decay,
    _Fibo,
    _Constant,
    _Runtime,
    _WaitRandomExponential,
    _WaitIncrementing,
    _WaitRandom,
    _WaitExponentialJitter,
    _WaitNone,
    _WaitException,
)


# --------------------------------------------------------------------------- #
# wait_combine
# --------------------------------------------------------------------------- #
def test_wait_combine_returns_combined_wait():
    w = wait_combine(wait_none())
    assert isinstance(w, _CombinedWait)
    assert w.next() == 0.0


def test_wait_combine_sums_waits():
    w = wait_combine(constant(1), constant(2), constant(3))
    assert w.next() == 6.0
    assert w.next() == 6.0


def test_wait_combine_empty_returns_zero():
    w = wait_combine()
    assert isinstance(w, _CombinedWait)
    assert w.next() == 0.0


def test_wait_combine_with_incrementing():
    w = wait_combine(wait_incrementing(start=1, increment=1), constant(10))
    assert w.next() == 11.0
    assert w.next() == 12.0
    assert w.next() == 13.0


def test_wait_combine_passes_send_value():
    def fn(v):
        return float(v)

    w = wait_combine(runtime(value=fn), runtime(value=fn))
    assert w.next(5) == 10.0


def test_wait_combine_call_kwargs_propagates():
    w = wait_combine(wait_incrementing(start=1), wait_incrementing(start=1))
    w2 = w(increment=5)
    assert isinstance(w2, _CombinedWait)
    assert w2.next() == 2.0
    assert w2.next() == 12.0


def test_combined_wait_add_combined_flattens():
    a = wait_combine(constant(1))
    b = wait_combine(constant(2))
    c = a + b
    assert isinstance(c, _CombinedWait)
    assert len(c._waits) == 2
    assert c.next() == 3.0


def test_combined_wait_add_single_wait():
    a = wait_combine(constant(1))
    c = a + constant(4)
    assert len(c._waits) == 2
    assert c.next() == 5.0


# --------------------------------------------------------------------------- #
# wait_chain
# --------------------------------------------------------------------------- #
def test_wait_chain_returns_chain():
    w = wait_chain(constant(1), constant(2))
    assert isinstance(w, _WaitChain)


def test_wait_chain_cycles_through_waits():
    w = wait_chain(constant(1), constant(2), constant(3))
    assert w.next() == 1.0
    assert w.next() == 2.0
    assert w.next() == 3.0
    assert w.next() == 1.0
    assert w.next() == 2.0


def test_wait_chain_empty_returns_zero():
    w = wait_chain()
    assert w.next() == 0.0
    assert w.next() == 0.0


def test_wait_chain_single_wait():
    w = wait_chain(constant(5))
    assert w.next() == 5.0
    assert w.next() == 5.0


def test_wait_chain_with_stateful_waits():
    w = wait_chain(wait_incrementing(start=0, increment=10), constant(100))
    assert w.next() == 0.0
    assert w.next() == 100.0
    assert w.next() == 10.0
    assert w.next() == 100.0


def test_wait_chain_call_kwargs_propagates():
    w = wait_chain(wait_incrementing(start=1, increment=1))
    w2 = w(increment=5)
    assert isinstance(w2, _WaitChain)
    assert w2.next() == 1.0
    assert w2.next() == 6.0


def test_wait_chain_passes_send_value():
    def fn(v):
        return float(v) * 2

    w = wait_chain(runtime(value=fn))
    assert w.next(3) == 6.0


# --------------------------------------------------------------------------- #
# Mutable combinators - each next advances underlying waits
# --------------------------------------------------------------------------- #
def test_combined_wait_advances_underlying_state():
    inc = wait_incrementing(start=1, increment=1)
    w = wait_combine(inc)
    assert w.next() == 1.0
    assert w.next() == 2.0
    assert w.next() == 3.0


def test_chain_advances_underlying_state():
    inc = wait_incrementing(start=1, increment=1)
    w = wait_chain(inc)
    assert w.next() == 1.0
    assert w.next() == 2.0


# --------------------------------------------------------------------------- #
# Error handling / NotImplementedError
# --------------------------------------------------------------------------- #
def test_base_wait_next_not_implemented():
    w = _Wait()
    with pytest.raises(NotImplementedError):
        w.next()


# --------------------------------------------------------------------------- #
# Factory callable behavior used by combinators
# --------------------------------------------------------------------------- #
def test_wait_factory_callable():
    w = constant(2.5)
    assert isinstance(w, _Wait)
    assert w.next() == 2.5


def test_wait_factory_add_creates_combined():
    w = constant(1) + wait_none()
    assert isinstance(w, _CombinedWait)
    assert w.next() == 1.0


# --------------------------------------------------------------------------- #
# Pure wrappers - chain over none
# --------------------------------------------------------------------------- #
def test_chain_of_none():
    w = wait_chain(wait_none())
    assert w.next() == 0.0


def test_combine_of_none():
    w = wait_combine(wait_none(), wait_none())
    assert w.next() == 0.0


# --------------------------------------------------------------------------- #
# Decorator-style call forwarding
# --------------------------------------------------------------------------- #
def test_wait_chain_call_replaces_positional_args():
    w = wait_chain(wait_incrementing(start=1))
    # calling with positional-style kwargs should rebuild inner waits
    w2 = w(start=10)
    assert w2.next() == 10.0


def test_combined_call_forwards_to_children():
    w = wait_combine(wait_random(min=0, max=0))
    w2 = w(min=0, max=0)
    assert w2.next() == 0.0


# --------------------------------------------------------------------------- #
# Combined wait with mixed deterministic waits (no randomness)
# --------------------------------------------------------------------------- #
def test_combined_mixed_deterministic():
    w = wait_combine(expo(base=2, factor=1), constant(0.5))
    assert math.isclose(w.next(), 1.5)
    assert math.isclose(w.next(), 2.5)
    assert math.isclose(w.next(), 4.5)


def test_chain_mixed_deterministic():
    w = wait_chain(expo(base=2, factor=1), constant(0.5))
    assert math.isclose(w.next(), 1.0)
    assert math.isclose(w.next(), 0.5)
    assert math.isclose(w.next(), 2.0)
    assert math.isclose(w.next(), 0.5)


# =========================================================================== #
# ADDED TESTS to improve coverage
# =========================================================================== #

# --------------------------------------------------------------------------- #
# _Wait.__call__ with positional args (lines 21-25)
# --------------------------------------------------------------------------- #
def test_wait_call_with_positional_args_drops_overridden_kwargs():
    # _Constant(interval=...) - call with positional arg overrides interval
    w = constant(1)
    w2 = w(5)
    assert w2.next() == 5.0
    # original unaffected
    assert w.next() == 1.0


def test_wait_call_positional_more_args_than_kwargs():
    # More positional args than known keyword params
    w = constant(1)
    w2 = w(2)
    assert w2.next() == 2.0


def test_wait_call_multiple_positional_args():
    # _WaitIncrementing has start, increment, max_value as kwargs.
    # Passing 2 positional args should override start and increment.
    w = wait_incrementing(start=1, increment=1)
    w2 = w(10, 3)
    assert w2.next() == 10.0
    assert w2.next() == 13.0


# --------------------------------------------------------------------------- #
# _Wait.__add__ / __radd__ with _CombinedWait (lines 29-30, 33-34)
# --------------------------------------------------------------------------- #
def test_wait_add_combined_wait_prepends():
    a = constant(1)
    b = wait_combine(constant(2), constant(3))
    c = a + b
    assert isinstance(c, _CombinedWait)
    assert len(c._waits) == 3
    assert c.next() == 6.0


def test_wait_radd_combines():
    a = constant(1)
    # Force use of __radd__: a _CombinedWait has __radd__ inherited from _Wait
    # but for a plain _Wait on the right, __add__ of left combinator
    # would kick in. Use __radd__ directly via _Wait.__radd__.
    c = _Wait.__radd__(a, constant(2))
    assert isinstance(c, _CombinedWait)
    assert c.next() == 3.0


# --------------------------------------------------------------------------- #
# _WaitFactory.__add__ and __radd__ (lines 63-71)
# --------------------------------------------------------------------------- #
def test_wait_factory_add_combined_wait():
    combined = wait_combine(constant(2))
    c = constant.__add__(combined)
    assert isinstance(c, _CombinedWait)
    assert len(c._waits) == 2
    assert c.next() == 3.0


def test_wait_factory_add_plain_wait():
    c = constant.__add__(wait_none())
    assert isinstance(c, _CombinedWait)
    assert c.next() == 1.0


def test_wait_factory_radd_combined():
    combined = wait_combine(constant(2))
    c = constant.__radd__(combined)
    assert isinstance(c, _CombinedWait)
    assert len(c._waits) == 2
    assert c.next() == 3.0


def test_wait_factory_radd_plain():
    c = constant.__radd__(wait_none())
    assert isinstance(c, _CombinedWait)
    assert c.next() == 1.0


# --------------------------------------------------------------------------- #
# _Expo max_value branch (lines 89, 92)
# --------------------------------------------------------------------------- #
def test_expo_with_max_value():
    w = expo(base=2, factor=1, max_value=5)
    assert w.next() == 1.0
    assert w.next() == 2.0
    assert w.next() == 4.0
    # 8 >= 5 so returns max_value and does not advance
    assert w.next() == 5.0
    assert w.next() == 5.0


# --------------------------------------------------------------------------- #
# _Decay (lines 98-120)
# --------------------------------------------------------------------------- #
def test_decay_default():
    w = decay()
    assert math.isclose(w.next(), 1.0)
    assert math.isclose(w.next(), math.e ** -1)
    assert math.isclose(w.next(), math.e ** -2)


def test_decay_with_min_value():
    w = decay(initial_value=1, decay_factor=10, min_value=0.1)
    a = w.next()
    assert math.isclose(a, 1.0)
    b = w.next()
    # e^-10 ~= 4.5e-5 which is <= min_value 0.1 -> returns min_value
    assert b == 0.1
    assert w.next() == 0.1
    assert w.next() == 0.1


def test_decay_custom_params():
    w = decay(initial_value=2, decay_factor=1)
    assert math.isclose(w.next(), 2.0)
    assert math.isclose(w.next(), 2 * math.e ** -1)


# --------------------------------------------------------------------------- #
# _Fibo (lines 126-138)
# --------------------------------------------------------------------------- #
def test_fibo_sequence():
    w = fibo()
    assert w.next() == 1.0
    assert w.next() == 1.0
    assert w.next() == 2.0
    assert w.next() == 3.0
    assert w.next() == 5.0
    assert w.next() == 8.0


def test_fibo_with_max_value():
    w = fibo(max_value=3)
    assert w.next() == 1.0
    assert w.next() == 1.0
    assert w.next() == 2.0
    # next a=3 >= 3 -> return max_value saturated
    assert w.next() == 3.0
    assert w.next() == 3.0


# --------------------------------------------------------------------------- #
# _Constant with sequence (line 148 else branch, line 151)
# --------------------------------------------------------------------------- #
def test_constant_with_sequence():
    w = constant([1.5, 2.5, 3.5])
    assert w.next() == 1.5
    assert w.next() == 2.5
    assert w.next() == 3.5
    with pytest.raises(StopIteration):
        w.next()


def test_constant_with_int():
    w = constant(3)
    assert w.next() == 3.0
    assert isinstance(w.next(), float)


# --------------------------------------------------------------------------- #
# _Runtime (lines 160-166)
# --------------------------------------------------------------------------- #
def test_runtime_calls_function_with_send_value():
    w = runtime(value=lambda v: (v or 0) * 2.0)
    assert w.next() == 0.0
    assert w.next(5) == 10.0


# --------------------------------------------------------------------------- #
# _WaitRandomExponential (lines 172-199)
# --------------------------------------------------------------------------- #
def test_wait_random_exponential_within_bounds():
    random.seed(12345)
    w = wait_random_exponential(multiplier=1, max_value=4, exp_base=2, min_value=0)
    for expected_hi in (1.0, 2.0, 4.0, 4.0, 4.0):
        v = w.next()
        assert 0.0 <= v <= expected_hi


def test_wait_random_exponential_min_value_clamps():
    random.seed(1)
    w = wait_random_exponential(multiplier=1, max_value=2, exp_base=2, min_value=10)
    # a gets clamped up to min_value 10, then uniform(0, 10)
    v = w.next()
    assert 0.0 <= v <= 10.0


def test_wait_random_exponential_no_max():
    random.seed(0)
    w = wait_random_exponential(multiplier=1, max_value=None, exp_base=2)
    assert 0.0 <= w.next() <= 1.0
    assert 0.0 <= w.next() <= 2.0
    assert 0.0 <= w.next() <= 4.0


# --------------------------------------------------------------------------- #
# _WaitIncrementing (lines 205-223)
# --------------------------------------------------------------------------- #
def test_wait_incrementing_with_max_value():
    w = wait_incrementing(start=1, increment=2, max_value=5)
    assert w.next() == 1.0
    assert w.next() == 3.0
    assert w.next() == 5.0
    # next a = 7 > 5 -> returns 5
    assert w.next() == 5.0
    assert w.next() == 5.0


def test_wait_incrementing_defaults():
    w = wait_incrementing()
    assert w.next() == 1.0
    assert w.next() == 2.0
    assert w.next() == 3.0


# --------------------------------------------------------------------------- #
# _WaitRandom (lines 266-273)
# --------------------------------------------------------------------------- #
def test_wait_random_bounds():
    random.seed(42)
    w = wait_random(min=2, max=3)
    for _ in range(20):
        v = w.next()
        assert 2.0 <= v <= 3.0


def test_wait_random_identical_bounds():
    w = wait_random(min=1.5, max=1.5)
    assert w.next() == 1.5


# --------------------------------------------------------------------------- #
# _WaitExponentialJitter (lines 279-298)
# --------------------------------------------------------------------------- #
def test_wait_exponential_jitter_bounds():
    random.seed(7)
    w = wait_exponential_jitter(initial=1, max=10, exp_base=2, jitter=1)
    # n starts at 1 -> delay = 1*2^1 = 2, plus [0,1]
    v = w.next()
    assert 2.0 <= v <= 3.0
    # next n=2 -> 4, plus [0,1]
    v = w.next()
    assert 4.0 <= v <= 5.0
    # next n=3 -> 8, plus [0,1]
    v = w.next()
    assert 8.0 <= v <= 9.0
    # cap at max=10 plus jitter
    v = w.next()
    assert 10.0 <= v <= 11.0


def test_wait_exponential_jitter_zero_jitter():
    w = wait_exponential_jitter(initial=2, max=100, exp_base=3, jitter=0)
    assert w.next() == 6.0
    assert w.next() == 18.0


# --------------------------------------------------------------------------- #
# _WaitNone (lines 304-306)
# --------------------------------------------------------------------------- #
def test_wait_none_returns_zero():
    w = wait_none()
    assert w.next() == 0.0
    assert w.next(10) == 0.0


# --------------------------------------------------------------------------- #
# _WaitException (lines 254-263)
# --------------------------------------------------------------------------- #
def test_wait_exception_class_returns_value():
    from backon._wait_gen import _WaitException
    w = _WaitException(value=lambda v: (v or 0) + 1.0)
    assert w.next() == 1.0
    assert w.next(4) == 5.0


# --------------------------------------------------------------------------- #
# _CombinedWait.next sums correctly with send_value
# --------------------------------------------------------------------------- #
def test_combined_wait_next_send_value_multiple():
    def fn(v):
        return float(v)

    w = wait_combine(runtime(value=fn), constant(1))
    assert w.next(3) == 4.0


# --------------------------------------------------------------------------- #
# _WaitChain.__call__ rebuilds independent chain
# --------------------------------------------------------------------------- #
def test_wait_chain_call_rebuild():
    w = wait_chain(wait_incrementing(start=1), wait_incrementing(start=100))
    w2 = w(start=5)
    assert isinstance(w2, _WaitChain)
    # Both children receive the same kwargs; start=5 overrides both starts.
    assert w2.next() == 5.0
    assert w2.next() == 5.0
    assert w2.next() == 6.0
    assert w2.next() == 6.0


# =========================================================================== #
# ADDED TESTS for previously uncovered branch (line 24)
# =========================================================================== #
def test_wait_call_more_positional_args_than_params():
    # _Constant stores a single kwarg ("interval"). Passing more positional
    # args than known kwargs hits the ``i < len(param_order)`` false branch.
    # The extra positional arg is forwarded to the constructor, which raises
    # TypeError because _Constant only accepts one positional argument.
    w = constant(1)
    with pytest.raises(TypeError):
        w(2, 3)


def test_wait_call_fewer_positional_args_than_params():
    # _WaitIncrementing has 3 kwargs (start, increment, max_value). Passing
    # exactly 2 positional args exercises the true branch for i=0,1 and then
    # leaves the loop before i=2. The result must drop start/increment but
    # keep max_value.
    w = wait_incrementing(start=1, increment=2, max_value=100)
    w2 = w(10, 20)
    assert w2.next() == 10.0
    assert w2.next() == 30.0
    assert w2.next() == 50.0


def test_wait_call_positional_exceeds_params_raises():
    # Passing 4 positional args to a wait that knows 3 kwargs: after the
    # loop, all positional args are forwarded. _WaitIncrementing accepts
    # exactly 3 positional args, so a 4th raises TypeError. This also
    # guarantees the false branch of ``i < len(param_order)`` is executed.
    w = wait_incrementing(start=0, increment=1, max_value=10)
    with pytest.raises(TypeError):
        w(1, 2, 3, 4)
