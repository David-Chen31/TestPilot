from __future__ import annotations

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
    _WaitIncrementing,
    _WaitRandom,
    _WaitRandomExponential,
    _WaitNone,
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
# wait_chain
# ---------------------------------------------------------------------------

class TestWaitChain:
    def test_returns_wait_chain_instance(self):
        chain = wait_chain(expo())
        assert isinstance(chain, _WaitChain)

    def test_empty_chain_returns_zero(self):
        chain = wait_chain()
        assert chain.next() == 0.0
        assert chain.next() == 0.0

    def test_cycles_through_waits(self):
        chain = wait_chain(constant(1), constant(2), constant(3))
        assert [chain.next() for _ in range(7)] == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]

    def test_index_wraps_correctly(self):
        chain = wait_chain(constant(1), constant(2))
        assert chain._idx == 0
        chain.next()
        assert chain._idx == 1
        chain.next()
        assert chain._idx == 0

    def test_single_wait_repeats(self):
        chain = wait_chain(constant(5))
        assert [chain.next() for _ in range(3)] == [5.0, 5.0, 5.0]

    def test_send_value_forwarded(self):
        received = []

        def recorder(v):
            received.append(v)
            return 0.5

        chain = wait_chain(runtime(value=recorder))
        assert chain.next("hello") == 0.5
        assert received == ["hello"]

    def test_spreads_multiple_calls_across_waits(self):
        chain = wait_chain(wait_incrementing(start=10, increment=1),
                           wait_incrementing(start=100, increment=10))
        results = [chain.next() for _ in range(4)]
        assert results == [10.0, 100.0, 11.0, 110.0]

    def test_call_returns_new_chain(self):
        chain = wait_chain(expo(), fibo())
        new_chain = chain(**{})
        assert isinstance(new_chain, _WaitChain)
        assert new_chain is not chain
        assert new_chain._waits is not chain._waits

    def test_call_reconfigures_waits(self):
        chain = wait_chain(expo(base=2), expo(base=3))
        new_chain = chain(base=5)
        assert new_chain.next() == 1.0
        assert new_chain.next() == 1.0
        # After one step each, n=1 -> factor * base**1 = 5
        assert new_chain.next() == 5.0


# ---------------------------------------------------------------------------
# wait_combine
# ---------------------------------------------------------------------------

class TestWaitCombine:
    def test_returns_combined_wait(self):
        combined = wait_combine(expo(), expo())
        assert isinstance(combined, _CombinedWait)

    def test_empty_combine_returns_zero(self):
        combined = wait_combine()
        assert combined.next() == 0.0

    def test_sums_single_next_calls(self):
        combined = wait_combine(constant(1), constant(2), constant(3))
        assert combined.next() == 6.0

    def test_sums_repeatedly(self):
        combined = wait_combine(constant(1), constant(2))
        assert [combined.next() for _ in range(4)] == [3.0, 3.0, 3.0, 3.0]

    def test_mixed_wait_types(self):
        combined = wait_combine(expo(), constant(1))
        # expo: 1, 2, 4 ... constant: 1, 1, 1
        assert [combined.next() for _ in range(3)] == [2.0, 3.0, 5.0]

    def test_combined_waits_are_tuple(self):
        w1 = constant(1)
        w2 = constant(2)
        combined = wait_combine(w1, w2)
        assert combined._waits == (w1, w2)

    def test_call_returns_new_combined(self):
        combined = wait_combine(expo(base=2), expo(base=3))
        new_combined = combined(base=4)
        assert isinstance(new_combined, _CombinedWait)
        assert new_combined is not combined
        # each expo now has base 4: 1 + 1 = 2, then 4 + 4 = 8
        assert new_combined.next() == 2.0
        assert new_combined.next() == 8.0

    def test_send_value_forwarded(self):
        values = []

        def rec(v):
            values.append(v)
            return 1.0

        combined = wait_combine(runtime(value=rec), runtime(value=rec))
        assert combined.next("x") == 2.0
        assert values == ["x", "x"]


# ---------------------------------------------------------------------------
# Edge cases / behavior of underlying wait classes used by public API
# ---------------------------------------------------------------------------

class TestUnderlyingWaits:
    def test_expo_progression(self):
        w = expo()
        assert [w.next() for _ in range(5)] == [1.0, 2.0, 4.0, 8.0, 16.0]

    def test_expo_respects_max_value(self):
        w = expo(base=2, factor=1, max_value=5)
        assert [w.next() for _ in range(6)] == [1.0, 2.0, 4.0, 5.0, 5.0, 5.0]

    def test_fibo_progression(self):
        w = fibo()
        assert [w.next() for _ in range(8)] == [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0]

    def test_fibo_max_value(self):
        w = fibo(max_value=5)
        assert [w.next() for _ in range(5)] == [1.0, 1.0, 2.0, 3.0, 5.0]

    def test_constant_sequence(self):
        w = constant([1, 2, 3])
        assert [w.next() for _ in range(5)] == [1.0, 2.0, 3.0, 1.0, 2.0]

    def test_constant_scalar_repeats(self):
        w = constant(2.5)
        assert [w.next() for _ in range(3)] == [2.5, 2.5, 2.5]

    def test_constant_exhausted_raises(self):
        w = constant([1])
        assert w.next() == 1.0
        with pytest.raises(StopIteration):
            w.next()

    def test_wait_incrementing_progression(self):
        w = wait_incrementing(start=1, increment=2)
        assert [w.next() for _ in range(4)] == [1.0, 3.0, 5.0, 7.0]

    def test_wait_incrementing_max_value(self):
        w = wait_incrementing(start=1, increment=2, max_value=5)
        assert [w.next() for _ in range(4)] == [1.0, 3.0, 5.0, 5.0]

    def test_decay_progression(self):
        w = decay()
        import math
        expected = [math.e ** (-t) for t in range(4)]
        assert [w.next() for _ in range(4)] == pytest.approx(expected)

    def test_decay_min_value(self):
        w = decay(initial_value=1, decay_factor=5, min_value=0.1)
        vals = [w.next() for _ in range(4)]
        assert vals[-1] == 0.1

    def test_wait_none_returns_zero(self):
        w = wait_none()
        assert w.next() == 0.0
        assert w.next() == 0.0

    def test_wait_random_within_bounds(self):
        w = wait_random(min=2, max=2)
        # min == max -> uniform returns exactly 2
        assert w.next() == 2.0

    def test_wait_random_exponential_bounded(self):
        w = wait_random_exponential(multiplier=1, max_value=4, exp_base=2, min_value=0)
        for _ in range(10):
            v = w.next()
            assert 0 <= v <= 4

    def test_runtime_uses_send_value(self):
        w = runtime(value=lambda x: (x or 0) * 2)
        assert w.next(3) == 6
        assert w.next(0) == 0

    def test_wait_exception_calls_value(self):
        w = wait_exception(value=lambda x: 42.0)
        assert w.next() == 42.0


# ---------------------------------------------------------------------------
# Operator overloading behavior
# ---------------------------------------------------------------------------

class TestOperators:
    def test_wait_add_wait(self):
        combined = expo() + expo()
        assert isinstance(combined, _CombinedWait)
        assert combined.next() == 2.0

    def test_factory_add_wait(self):
        combined = expo + expo()
        assert isinstance(combined, _CombinedWait)
        assert combined.next() == 2.0

    def test_wait_add_combined(self):
        w = expo()
        c = expo() + expo()
        result = w + c
        assert isinstance(result, _CombinedWait)
        assert len(result._waits) == 3
        assert result.next() == 3.0

    def test_combined_add_wait(self):
        c = expo() + expo()
        result = c + expo()
        assert len(result._waits) == 3
        assert result.next() == 3.0

    def test_combined_add_combined(self):
        c1 = expo() + expo()
        c2 = expo() + expo()
        result = c1 + c2
        assert len(result._waits) == 4
        assert result.next() == 4.0

    def test_wait_radd(self):
        w = expo()
        result = w.__radd__(expo())
        assert isinstance(result, _CombinedWait)
        assert len(result._waits) == 2

    def test_factory_radd(self):
        result = expo.__radd__(expo())
        assert isinstance(result, _CombinedWait)
        assert len(result._waits) == 2

    def test_factory_radd_with_combined(self):
        c = expo() + expo()
        result = expo.__radd__(c)
        assert len(result._waits) == 3


# ---------------------------------------------------------------------------
# _WaitFactory behavior (via public factory objects)
# ---------------------------------------------------------------------------

class TestWaitFactory:
    def test_factory_creates_instance(self):
        w = expo(base=3)
        assert isinstance(w, _Expo)
        assert w._base == 3

    def test_factory_call_with_args(self):
        w = constant(5)
        assert isinstance(w, _Constant)
        assert w.next() == 5.0

    def test_factory_slots(self):
        assert not hasattr(expo, "__dict__")


# ---------------------------------------------------------------------------
# _Wait.__call__ reconfiguration (kwargs only for normal waits)
# ---------------------------------------------------------------------------

class TestWaitCall:
    def test_call_with_kwargs(self):
        w = expo()
        w2 = w(base=10)
        assert isinstance(w2, _Expo)
        assert w2._base == 10
        assert w is not w2

    def test_call_with_positional_args_overrides(self):
        w = expo()
        w2 = w(3)  # base=3
        assert w2._base == 3

    def test_call_positional_then_kwargs(self):
        w = expo()
        w2 = w(3, factor=2)
        assert w2._base == 3
        assert w2._factor == 2

    def test_call_no_args_returns_new_instance(self):
        w = expo()
        w2 = w()
        assert w2 is not w
        assert w2._base == w._base


# ---------------------------------------------------------------------------
# _Wait base class
# ---------------------------------------------------------------------------

class TestWaitBase:
    def test_next_raises_not_implemented(self):
        w = _Wait()
        with pytest.raises(NotImplementedError):
            w.next()

    def test_base_init_stores_args(self):
        w = _Wait(1, 2, a=3)
        assert w._args == (1, 2)
        assert w._kwargs == {"a": 3}
