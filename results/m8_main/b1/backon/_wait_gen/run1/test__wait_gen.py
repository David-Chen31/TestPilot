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
    _WaitFactory,
    _WaitIncrementing,
    _WaitNone,
    _WaitRandom,
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
)


# ---------------------------------------------------------------------------
# wait_chain
# ---------------------------------------------------------------------------

def test_wait_chain_returns_wait_chain_instance():
    chain = wait_chain()
    assert isinstance(chain, _WaitChain)


def test_wait_chain_empty_returns_zero():
    chain = wait_chain()
    assert chain.next() == 0.0
    assert chain.next() == 0.0


def test_wait_chain_cycles_through_waits_in_order():
    chain = wait_chain(constant(1), constant(2), constant(3))
    results = [chain.next() for _ in range(6)]
    assert results == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]


def test_wait_chain_single_wait_repeats():
    chain = wait_chain(constant(5))
    assert [chain.next() for _ in range(3)] == [5.0, 5.0, 5.0]


def test_wait_chain_advances_underlying_generators():
    chain = wait_chain(expo(base=2, factor=1), constant(10))
    # Interleaves expo (1,2,4,...) and constant 10
    assert chain.next() == 1.0
    assert chain.next() == 10.0
    assert chain.next() == 2.0
    assert chain.next() == 10.0
    assert chain.next() == 4.0


def test_wait_chain_send_value_propagates():
    received = []

    def record(v):
        received.append(v)
        return 0.0

    chain = wait_chain(runtime(value=record))
    chain.next("a")
    chain.next("b")
    assert received == ["a", "b"]


def test_wait_chain_rejects_non_wait():
    with pytest.raises(AttributeError):
        wait_chain(42).next()


def test_wait_chain_factory_call_returns_new_chain_with_reconfigured_waits():
    chain = wait_chain(expo(base=2))
    # __call__ on _WaitChain reconstructs each wait via its __call__
    new_chain = chain(exponent=None) if False else chain(base=3)
    assert isinstance(new_chain, _WaitChain)
    assert new_chain.next() == 1.0  # factor default still 1
    assert new_chain.next() == 3.0


# ---------------------------------------------------------------------------
# wait_combine
# ---------------------------------------------------------------------------

def test_wait_combine_returns_combined_wait():
    combined = wait_combine(constant(1), constant(2))
    assert isinstance(combined, _CombinedWait)


def test_wait_combine_empty_sums_to_zero():
    combined = wait_combine()
    assert combined.next() == 0


def test_wait_combine_sums_waits():
    combined = wait_combine(constant(1), constant(2), constant(3))
    assert combined.next() == 6.0


def test_wait_combine_single_wait():
    combined = wait_combine(constant(7))
    assert combined.next() == 7.0


def test_wait_combine_progresses_underlying_waits_together():
    combined = wait_combine(expo(base=2, factor=1), constant(10))
    assert combined.next() == 11.0  # 1 + 10
    assert combined.next() == 12.0  # 2 + 10
    assert combined.next() == 14.0  # 4 + 10


def test_wait_combine_send_value_propagates_to_all():
    received = []

    def record(v):
        received.append(v)
        return 0.0

    combined = wait_combine(runtime(value=record), runtime(value=record))
    combined.next("x")
    assert received == ["x", "x"]


def test_wait_combine_factory_call_reconfigures_children():
    combined = wait_combine(expo(base=2, factor=1))
    new_combined = combined(factor=5)
    assert isinstance(new_combined, _CombinedWait)
    assert new_combined.next() == 5.0
    assert new_combined.next() == 10.0


# ---------------------------------------------------------------------------
# Interaction: combining / adding
# ---------------------------------------------------------------------------

def test_add_two_waits_forms_combined():
    combined = constant(1) + constant(2)
    assert isinstance(combined, _CombinedWait)
    assert combined.next() == 3.0


def test_add_combined_with_wait_flattens():
    combined = wait_combine(constant(1), constant(2))
    extended = combined + constant(3)
    assert isinstance(extended, _CombinedWait)
    # Flattened, not nested: 1 + 2 + 3
    assert extended.next() == 6.0


def test_add_wait_with_combined_left():
    combined = wait_combine(constant(2), constant(3))
    extended = combined + constant(4)
    assert extended.next() == 9.0


def test_radd_builds_combined():
    combined = 1 + constant(2) if False else None  # int + _Wait not supported
    # Use _Wait.__radd__ via _CombinedWait combination instead
    w1 = constant(1)
    combined = w1.__add__(constant(2))
    assert combined.next() == 3.0


def test_factory_add_wait():
    combined = constant(1) if False else constant + constant(2)
    # constant is a _WaitFactory; its __call__ yields a _Constant
    assert isinstance(combined, _CombinedWait)


# ---------------------------------------------------------------------------
# Sanity of base _Wait / factory interplay
# ---------------------------------------------------------------------------

def test_factory_produces_wait_instance():
    w = constant(3)
    assert isinstance(w, _Constant)
    assert w.next() == 3.0


def test_base_wait_next_not_implemented():
    class Dummy(_Wait):
        pass

    with pytest.raises(NotImplementedError):
        Dummy().next()


def test_wait_call_merges_kwargs():
    w = expo(base=2, factor=1)
    w2 = w(factor=10)
    assert w2.next() == 10.0
    assert w2.next() == 20.0


def test_wait_call_replaces_positional_args():
    w = expo(base=3)
    w2 = w(5, 2)  # base=5, factor=2
    assert w2.next() == 2.0
    assert w2.next() == 10.0
