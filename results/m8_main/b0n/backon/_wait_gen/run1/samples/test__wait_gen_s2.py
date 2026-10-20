import math

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
    wait_exponential_jitter,
    wait_incrementing,
    wait_none,
    wait_random,
    wait_random_exponential,
)


# ---------------------------------------------------------------------------
# wait_chain
# ---------------------------------------------------------------------------

def test_wait_chain_returns_wait_chain_instance():
    chain = wait_chain(constant(1))
    assert isinstance(chain, _WaitChain)


def test_wait_chain_empty_returns_zero():
    chain = wait_chain()
    assert chain.next() == 0.0
    assert chain.next() == 0.0


def test_wait_chain_cycles_through_waits():
    chain = wait_chain(constant(1), constant(2), constant(3))
    assert [chain.next() for _ in range(6)] == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]


def test_wait_chain_single_wait_always_same():
    chain = wait_chain(constant(5))
    assert [chain.next() for _ in range(3)] == [5.0, 5.0, 5.0]


def test_wait_chain_with_incrementing():
    chain = wait_chain(wait_incrementing(start=1, increment=1))
    assert [chain.next() for _ in range(3)] == [1.0, 2.0, 3.0]


def test_wait_chain_alternates_two_stateful_waits():
    a = wait_incrementing(start=0, increment=10)
    b = wait_incrementing(start=100, increment=1)
    chain = wait_chain(a, b)
    assert [chain.next() for _ in range(4)] == [0.0, 100.0, 10.0, 101.0]


def test_wait_chain_send_value_forwarded():
    received = []

    def record(v):
        received.append(v)
        return 7.0

    chain = wait_chain(runtime(value=record))
    assert chain.__class__ is _WaitChain
    assert chain.next(send_value="hi") == 7.0
    assert received == ["hi"]


def test_wait_chain_call_returns_new_chain():
    chain = wait_chain(constant(1), constant(2))
    new_chain = chain()
    assert isinstance(new_chain, _WaitChain)
    assert new_chain is not chain
    assert [new_chain.next() for _ in range(4)] == [1.0, 2.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# wait_combine
# ---------------------------------------------------------------------------

def test_wait_combine_returns_combined_instance():
    combined = wait_combine(constant(1), constant(2))
    assert isinstance(combined, _CombinedWait)


def test_wait_combine_empty_returns_zero():
    combined = wait_combine()
    assert combined.next() == 0.0


def test_wait_combine_sums_waits():
    combined = wait_combine(constant(1), constant(2), constant(3))
    assert combined.next() == 6.0
    assert combined.next() == 6.0


def test_wait_combine_sum_of_incrementing():
    combined = wait_combine(
        wait_incrementing(start=1, increment=1),
        wait_incrementing(start=10, increment=10),
    )
    assert [combined.next() for _ in range(3)] == [11.0, 22.0, 33.0]


def test_wait_combine_send_value_forwarded_to_all():
    received = []

    def r1(v):
        received.append(("r1", v))
        return 1.0

    def r2(v):
        received.append(("r2", v))
        return 2.0

    combined = wait_combine(runtime(value=r1), runtime(value=r2))
    assert combined.next(send_value=42) == 3.0
    assert received == [("r1", 42), ("r2", 42)]


def test_wait_combine_call_returns_new_combined():
    combined = wait_combine(constant(1), constant(2))
    new_combined = combined()
    assert isinstance(new_combined, _CombinedWait)
    assert new_combined is not combined
    assert new_combined.next() == 3.0


def test_wait_combine_single():
    combined = wait_combine(constant(5))
    assert combined.next() == 5.0


# ---------------------------------------------------------------------------
# wait_combine / wait_chain are equivalent factories
# ---------------------------------------------------------------------------

def test_wait_combine_with_fibo():
    combined = wait_combine(fibo(), fibo())
    assert [combined.next() for _ in range(5)] == [2.0, 2.0, 4.0, 6.0, 10.0]


# ---------------------------------------------------------------------------
# _CombinedWait internals / branches
# ---------------------------------------------------------------------------

def test_combined_wait_stores_waits():
    a = constant(1)
    b = constant(2)
    combined = _CombinedWait(a, b)
    assert combined._waits == (a, b)


def test_combined_wait_add_plain_wait():
    c1 = _CombinedWait(constant(1))
    result = c1 + constant(2)
    assert isinstance(result, _CombinedWait)
    assert result.next() == 3.0


def test_combined_wait_add_combined_flattens():
    c1 = _CombinedWait(constant(1))
    c2 = _CombinedWait(constant(2), constant(3))
    result = c1 + c2
    assert isinstance(result, _CombinedWait)
    assert len(result._waits) == 3
    assert result.next() == 6.0


def test_combined_wait_add_chain_order():
    c1 = _CombinedWait(constant(10))
    result = c1 + constant(20) + constant(30)
    assert result.next() == 60.0


# ---------------------------------------------------------------------------
# _WaitFactory add operations
# ---------------------------------------------------------------------------

def test_factory_add_creates_combined():
    result = constant(1) + constant(2)
    assert isinstance(result, _CombinedWait)
    assert result.next() == 3.0


def test_factory_add_combined():
    c = _CombinedWait(constant(2), constant(3))
    result = constant(1) + c
    assert isinstance(result, _CombinedWait)
    assert len(result._waits) == 3
    assert result.next() == 6.0


def test_factory_radd_plain():
    result = constant(1).__radd__(constant(2))
    assert isinstance(result, _CombinedWait)
    assert result.next() == 3.0


def test_factory_radd_combined():
    c = _CombinedWait(constant(2), constant(3))
    result = constant(1).__radd__(c)
    assert isinstance(result, _CombinedWait)
    assert len(result._waits) == 3
    assert result.next() == 6.0


def test_factory_add_uses_default_instantiation():
    # constant() with no args -> default interval 1
    factory = _WaitFactory(_Constant)
    result = factory + constant(2)
    assert result.next() == 3.0


# ---------------------------------------------------------------------------
# Chain + combine equality of behavior via public functions
# ---------------------------------------------------------------------------

def test_chain_of_combined():
    combined = wait_combine(constant(1), constant(2))
    chain = wait_chain(combined, constant(10))
    assert [chain.next() for _ in range(4)] == [3.0, 10.0, 3.0, 10.0]


def test_combine_of_chains():
    chain = wait_chain(constant(1), constant(2))
    combined = wait_combine(chain, constant(10))
    assert [combined.next() for _ in range(4)] == [11.0, 12.0, 11.0, 12.0]


def test_combine_chain_empty():
    combined = wait_combine(wait_chain(), wait_chain())
    assert combined.next() == 0.0
