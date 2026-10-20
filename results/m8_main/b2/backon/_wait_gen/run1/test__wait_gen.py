from __future__ import annotations

import math

import pytest

from backon._wait_gen import (
    wait_chain,
    wait_combine,
    constant,
    expo,
    fibo,
    decay,
    wait_incrementing,
    wait_none,
    wait_random,
    wait_exception,
    runtime,
    wait_random_exponential,
    wait_exponential_jitter,
    _Wait,
    _CombinedWait,
    _WaitChain,
    _Constant,
    _Expo,
    _Fibo,
    _WaitFactory,
    _WaitRandomExponential,
    _WaitIncrementing,
    _WaitRandom,
    _WaitExponentialJitter,
    _WaitNone,
    _WaitException,
    _Runtime,
    _Decay,
)


# ---------------------------------------------------------------------------
# wait_chain
# ---------------------------------------------------------------------------

def test_wait_chain_returns_wait_chain_instance():
    chain = wait_chain(constant(1))
    assert isinstance(chain, _WaitChain)


def test_wait_chain_cycles_through_waits():
    chain = wait_chain(constant(1), constant(2), constant(3))
    assert [chain.next() for _ in range(7)] == [1, 2, 3, 1, 2, 3, 1]


def test_wait_chain_single_wait_repeats():
    chain = wait_chain(constant(5))
    assert [chain.next() for _ in range(3)] == [5, 5, 5]


def test_wait_chain_empty_returns_zero():
    chain = wait_chain()
    assert isinstance(chain, _WaitChain)
    assert chain.next() == 0.0
    assert [chain.next() for _ in range(3)] == [0.0, 0.0, 0.0]


def test_wait_chain_with_stateful_waits():
    e1 = expo(factor=1)
    e2 = expo(factor=10)
    chain = wait_chain(e1, e2)
    results = [chain.next() for _ in range(6)]
    assert results[0] == 1
    assert results[1] == 10
    assert results[2] == 2
    assert results[3] == 20
    assert results[4] == 4
    assert results[5] == 40


def test_wait_chain_forwards_send_value():
    captured = []

    def recorder(value):
        captured.append(value)
        return float(value if isinstance(value, (int, float)) else 0)

    chain = wait_chain(runtime(value=recorder))
    chain.next(42)
    chain.next(99)
    assert captured == [42, 99]
    assert chain.next() == 0.0
    assert captured == [42, 99, None]


def test_wait_chain_call_resets_and_clones():
    chain = wait_chain(constant(1), constant(2))
    chain.next()  # idx -> 1
    clone = chain(**{})
    assert isinstance(clone, _WaitChain)
    assert clone.next() == 1
    assert clone.next() == 2
    assert clone.next() == 1
    assert chain.next() == 2


# ---------------------------------------------------------------------------
# wait_combine
# ---------------------------------------------------------------------------

def test_wait_combine_returns_combined_instance():
    combined = wait_combine(constant(1), constant(2))
    assert isinstance(combined, _CombinedWait)


def test_wait_combine_sums_waits():
    combined = wait_combine(constant(1), constant(2), constant(3))
    assert combined.next() == 6
    assert combined.next() == 6


def test_wait_combine_empty_is_zero():
    combined = wait_combine()
    assert isinstance(combined, _CombinedWait)
    assert combined.next() == 0.0


def test_wait_combine_advances_stateful_waits():
    e1 = expo(factor=1)
    e2 = expo(factor=10)
    combined = wait_combine(e1, e2)
    assert combined.next() == 11
    assert combined.next() == 22
    assert combined.next() == 44


def test_wait_combine_forwards_send_value():
    captured = []

    def recorder(value):
        captured.append(value)
        return 1.0

    combined = wait_combine(runtime(value=recorder), runtime(value=recorder))
    combined.next(7)
    assert captured == [7, 7]


def test_wait_combine_call_clones_children():
    e = expo(factor=1)
    combined = wait_combine(e)
    combined.next()  # e advances to n=1
    clone = combined(factor=100)
    assert isinstance(clone, _CombinedWait)
    assert clone.next() == 100
    assert combined.next() == 2


# ---------------------------------------------------------------------------
# Interaction between chain and combine
# ---------------------------------------------------------------------------

def test_chain_with_combined_wait():
    combined = wait_combine(constant(1), constant(2))
    chain = wait_chain(combined, constant(10))
    assert chain.next() == 3
    assert chain.next() == 10
    assert chain.next() == 3
    assert chain.next() == 10


def test_combine_with_chain_object():
    chain = wait_chain(constant(1), constant(2))
    combined = wait_combine(chain, constant(10))
    assert combined.next() == 11
    assert combined.next() == 12
    assert combined.next() == 11


def test_combined_wait_add_returns_combined():
    combined = wait_combine(constant(1))
    other = constant(2)
    result = combined + other
    assert isinstance(result, _CombinedWait)
    assert result.next() == 3


def test_combined_wait_add_another_combined():
    c1 = wait_combine(constant(1), constant(2))
    c2 = wait_combine(constant(10), constant(20))
    result = c1 + c2
    assert isinstance(result, _CombinedWait)
    assert len(result._waits) == 4
    assert result.next() == 33


def test_wait_add_returns_combined():
    result = constant(1) + constant(2)
    assert isinstance(result, _CombinedWait)
    assert result.next() == 3


def test_wait_radd():
    result = constant(2).__radd__(constant(1))  # noqa: SLF001
    assert isinstance(result, _CombinedWait)
    assert result.next() == 3


# ---------------------------------------------------------------------------
# _WaitFactory.__add__ and __radd__
# ---------------------------------------------------------------------------

def test_factory_add_produces_instance_and_other():
    # `constant + constant(5)` -> _CombinedWait(self(), other)
    result = constant + constant(5)
    assert isinstance(result, _CombinedWait)
    assert len(result._waits) == 2
    assert result._waits[1].next() == 5
    # Factory instance creates default _Constant(interval=1)
    assert result.next() == 1 + 5


def test_factory_add_combined():
    c = wait_combine(constant(1), constant(2))
    result = constant(10) + c
    assert isinstance(result, _CombinedWait)
    assert len(result._waits) == 3
    assert result.next() == 10 + 1 + 2


def test_factory_radd_wait():
    result = constant(5).__radd__(constant(2))  # noqa: SLF001
    assert isinstance(result, _CombinedWait)
    assert result.next() == 2 + 5


def test_factory_radd_combined():
    # _WaitFactory.__radd__ only special-cases when `other` is _CombinedWait;
    # when other is a plain _Wait, it wraps both: _CombinedWait(other, self()).
    c = wait_combine(constant(1), constant(2))
    result = constant(10).__radd__(c)  # noqa: SLF001
    assert isinstance(result, _CombinedWait)
    # own result has 2 waits: the _CombinedWait and the factory-created one.
    assert len(result._waits) == 2
    assert result._waits[0] is c
    assert result.next() == 1 + 2 + 10


def test_factory_add_with_explicit_args():
    factory = expo(factor=3)
    result = factory + constant(1)
    assert isinstance(result, _CombinedWait)
    assert result.next() == 3 + 1


# ---------------------------------------------------------------------------
# Factories behave as _Wait and can be combined directly
# ---------------------------------------------------------------------------

def test_factory_creation_and_next():
    e = expo(factor=1)
    assert e.next() == 1
    assert e.next() == 2
    assert e.next() == 4


def test_constant_sequence():
    c = constant([1, 2, 3])
    assert [c.next(), c.next(), c.next()] == [1.0, 2.0, 3.0]
    with pytest.raises(StopIteration):
        c.next()


def test_fibo_sequence():
    f = fibo()
    assert [f.next() for _ in range(6)] == [1.0, 1.0, 2.0, 3.0, 5.0, 8.0]


def test_fibo_max_value():
    f = fibo(max_value=3)
    assert [f.next() for _ in range(5)] == [1.0, 1.0, 2.0, 3.0, 3.0]


def test_fibo_max_value_boundary():
    # _Fibo clamps when _a >= max_value.  With max_value=5:
    # sequence 1,1,2,3,5 -> at _a=5 the "5" would NOT be returned as a
    # natural term; instead the clamped 5 is returned.  Confirm boundary.
    f = fibo(max_value=5)
    vals = [f.next() for _ in range(9)]
    assert vals == [1.0, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0]


def test_fibo_max_value_equal_one_step():
    # max_value=2: 1, 1 returned naturally, then 2 clamped.
    f = fibo(max_value=2)
    assert [f.next() for _ in range(4)] == [1.0, 1.0, 2.0, 2.0]


def test_wait_incrementing():
    w = wait_incrementing(start=1, increment=2)
    assert [w.next() for _ in range(4)] == [1, 3, 5, 7]


def test_wait_incrementing_max_value():
    w = wait_incrementing(start=1, increment=2, max_value=4)
    assert [w.next() for _ in range(5)] == [1, 3, 4, 4, 4]


def test_wait_incrementing_max_value_boundary():
    # _WaitIncrementing uses strict > so a == max_value returns a exactly.
    w = wait_incrementing(start=1, increment=1, max_value=3)
    assert [w.next() for _ in range(5)] == [1, 2, 3, 3, 3]


def test_wait_none_always_zero():
    w = wait_none()
    assert [w.next() for _ in range(3)] == [0.0, 0.0, 0.0]


def test_wait_random_in_range(monkeypatch):
    # Deterministic: monkeypatch random.uniform
    calls = []

    def fake_uniform(a, b):
        calls.append((a, b))
        return (a + b) / 2

    monkeypatch.setattr("backon._wait_gen.random.uniform", fake_uniform)
    w = wait_random(min=5, max=10)
    assert w.next() == 7.5
    assert w.next() == 7.5
    assert calls == [(5, 10), (5, 10)]


def test_wait_exception_calls_callable():
    captured = []

    def recorder(value):
        captured.append(value)
        return 2.0

    w = wait_exception(value=recorder)
    assert w.next(1) == 2.0
    assert captured == [1]


def test_decay_sequence():
    d = decay(initial_value=1, decay_factor=1)
    vals = [d.next() for _ in range(3)]
    assert vals[0] == pytest.approx(1.0)
    assert vals[1] == pytest.approx(1 / 2.718281828459045)
    assert vals[2] == pytest.approx(1 / (2.718281828459045 ** 2))


# ---------------------------------------------------------------------------
# _Wait base and _Wait.__call__ overrides
# ---------------------------------------------------------------------------

def test_wait_base_next_not_implemented():
    with pytest.raises(NotImplementedError):
        _Wait().next()


def test_expo_call_override():
    e = _Expo(factor=1)
    e.next()  # n=1
    clone = e(factor=5)
    assert isinstance(clone, _Expo)
    assert clone.next() == 5


def test_constant_call_override():
    c = _Constant(interval=1)
    c.next()
    clone = c(interval=10)
    assert clone.next() == 10.0


# ---------------------------------------------------------------------------
# Additional coverage for wait_random_exponential
# ---------------------------------------------------------------------------

def test_wait_random_exponential_default_multiplier(monkeypatch):
    # Verify default multiplier is 1, exp_base 2 -> uniform(0, 1), (0, 2), ...
    captured = []

    def fake_uniform(a, b):
        captured.append((a, b))
        return a

    monkeypatch.setattr("backon._wait_gen.random.uniform", fake_uniform)
    w = wait_random_exponential()
    assert w.next() == 0
    assert w.next() == 0
    assert w.next() == 0
    assert captured[0] == (0, 1)
    assert captured[1] == (0, 2)
    assert captured[2] == (0, 4)


def test_wait_random_exponential_max_value(monkeypatch):
    captured = []

    def fake_uniform(a, b):
        captured.append((a, b))
        return a

    monkeypatch.setattr("backon._wait_gen.random.uniform", fake_uniform)
    w = wait_random_exponential(multiplier=1, max_value=3, exp_base=2)
    w.next()  # a = 1
    w.next()  # a = 2
    w.next()  # a = 4 -> clamped to 3
    w.next()  # a = 8 -> clamped to 3
    assert captured == [(0, 1), (0, 2), (0, 3), (0, 3)]


def test_wait_random_exponential_max_value_boundary(monkeypatch):
    # Boundary: a == max_value satisfies `a > max_value` being False, so a is
    # not clamped (uses strict >).  With a=2 and max_value=2, uniform(0,2).
    captured = []

    def fake_uniform(a, b):
        captured.append((a, b))
        return a

    monkeypatch.setattr("backon._wait_gen.random.uniform", fake_uniform)
    w = wait_random_exponential(multiplier=2, max_value=2, exp_base=2)
    w.next()  # a = 2, not > 2, so uniform(0, 2)
    assert captured[0] == (0, 2)


def test_wait_random_exponential_min_value(monkeypatch):
    captured = []

    def fake_uniform(a, b):
        captured.append((a, b))
        return b

    monkeypatch.setattr("backon._wait_gen.random.uniform", fake_uniform)
    w = wait_random_exponential(multiplier=0, min_value=5)
    # a = 0*1 = 0, then clamped up to min 5
    assert w.next() == 5
    assert captured[0] == (0, 5)


def test_wait_random_exponential_min_value_boundary(monkeypatch):
    # Boundary: a == min_value -> `a < min_value` is False, so no clamp.
    captured = []

    def fake_uniform(a, b):
        captured.append((a, b))
        return a

    monkeypatch.setattr("backon._wait_gen.random.uniform", fake_uniform)
    w = wait_random_exponential(multiplier=5, exp_base=2, min_value=5)
    w.next()  # a = 5 == min_value, no clamp -> uniform(0, 5)
    assert captured[0] == (0, 5)


def test_wait_random_exponential_default_min_value_is_zero(monkeypatch):
    # Default min_value must be 0 (not 1).  With multiplier=0 -> a=0,
    # min clamp to 0 keeps it 0 (uniform(0,0)); if default were 1 -> uniform(0,1).
    captured = []

    def fake_uniform(a, b):
        captured.append((a, b))
        return a

    monkeypatch.setattr("backon._wait_gen.random.uniform", fake_uniform)
    w = wait_random_exponential(multiplier=0)
    w.next()
    assert captured[0] == (0, 0)


def test_wait_random_exponential_string_arg(monkeypatch):
    # Confirm multiplier/exp_base reach the uniform call via a separate
    # multi-value sequence: 3, 6, 12 with multiplier=3, exp_base=2.
    captured = []

    def fake_uniform(a, b):
        captured.append((a, b))
        return a

    monkeypatch.setattr("backon._wait_gen.random.uniform", fake_uniform)
    w = wait_random_exponential(multiplier=3, exp_base=2)
    w.next()
    w.next()
    w.next()
    assert captured == [(0, 3), (0, 6), (0, 12)]


def test_wait_random_exponential_factory_type():
    w = wait_random_exponential(multiplier=2, max_value=10, exp_base=3, min_value=1)
    assert isinstance(w, _WaitRandomExponential)
    assert w._multiplier == 2
    assert w._max_value == 10
    assert w._exp_base == 3
    assert w._min_value == 1


def test_wait_random_exponential_stores_kwargs():
    # Verify the super().__init__ kwargs are intact (mutation coverage).
    w = wait_random_exponential(multiplier=2, max_value=10, exp_base=3, min_value=1)
    assert w._kwargs == {
        "multiplier": 2,
        "max_value": 10,
        "exp_base": 3,
        "min_value": 1,
    }


def test_wait_random_exponential_default_stored_kwargs():
    w = wait_random_exponential()
    assert w._kwargs == {
        "multiplier": 1,
        "max_value": None,
        "exp_base": 2,
        "min_value": 0,
    }


# ---------------------------------------------------------------------------
# Additional coverage for wait_exponential_jitter
# ---------------------------------------------------------------------------

def test_wait_exponential_jitter_default(monkeypatch):
    calls = []

    def fake_uniform(a, b):
        calls.append((a, b))
        return 0.0

    monkeypatch.setattr("backon._wait_gen.random.uniform", fake_uniform)
    w = wait_exponential_jitter()
    # n starts at 1: delay = 1 * 2^1 = 2, then 4, 8...
    assert w.next() == 2.0
    assert w.next() == 4.0
    assert w.next() == 8.0
    assert calls[0] == (0, 1)


def test_wait_exponential_jitter_max_clamp(monkeypatch):
    monkeypatch.setattr("backon._wait_gen.random.uniform", lambda a, b: 0.0)
    w = wait_exponential_jitter(initial=1, max=5, exp_base=2, jitter=1)
    # n=1: 2, n=2: 4, n=3: 8 -> min(8,5)=5, n=4: 16->5
    assert w.next() == 2.0
    assert w.next() == 4.0
    assert w.next() == 5.0
    assert w.next() == 5.0


def test_wait_exponential_jitter_jitter_added(monkeypatch):
    calls = []

    def fake_uniform(a, b):
        calls.append((a, b))
        return 0.5

    monkeypatch.setattr("backon._wait_gen.random.uniform", fake_uniform)
    w = wait_exponential_jitter(initial=1, max=100, exp_base=2, jitter=2)
    assert w.next() == 2 + 0.5
    assert calls == [(0, 2)]


# ---------------------------------------------------------------------------
# Additional coverage for _Runtime and _WaitException factories
# ---------------------------------------------------------------------------

def test_runtime_factory_type():
    w = runtime(value=lambda v: 3.0)
    assert isinstance(w, _Runtime)
    assert w.next() == 3.0


def test_runtime_stores_value_in_kwargs():
    # super().__init__(value=value) must store the callable (mutation coverage).
    f = lambda v: 3.0  # noqa: E731
    w = runtime(value=f)
    assert w._kwargs == {"value": f}
    assert w._value is f


def test_wait_exception_factory_type():
    w = wait_exception(value=lambda v: 4.0)
    assert isinstance(w, _WaitException)
    assert w.next() == 4.0


def test_wait_exception_stores_value_in_kwargs():
    f = lambda v: 4.0  # noqa: E731
    w = wait_exception(value=f)
    assert w._kwargs == {"value": f}
    assert w._value is f


# ---------------------------------------------------------------------------
# Additional coverage for _Expo default parameters & max
# ---------------------------------------------------------------------------

def test_expo_defaults():
    e = expo()
    assert isinstance(e, _Expo)
    assert e.next() == 1
    assert e.next() == 2
    assert e.next() == 4


def test_expo_max_value():
    e = expo(base=2, factor=1, max_value=5)
    # 1, 2, 4, 8(clamped 5), 5 ...
    assert [e.next() for _ in range(5)] == [1, 2, 4, 5, 5]


def test_expo_max_value_boundary():
    # `a < max_value` is false for a == max_value -> clamp.  With 1,2,4,8 and
    # max_value=8: a=8 is NOT returned naturally, instead clamped 8.  Confirm
    # by checking max_value=4 which equals the third term 4.
    e = expo(base=2, factor=1, max_value=4)
    assert [e.next() for _ in range(5)] == [1, 2, 4, 4, 4]


# ---------------------------------------------------------------------------
# Additional coverage for _Decay defaults and min value
# ---------------------------------------------------------------------------

def test_decay_defaults():
    d = decay()
    assert isinstance(d, _Decay)
    assert d.next() == pytest.approx(1.0)


def test_decay_min_value():
    # _Decay clamps on the *current* value: t=0 -> 1.0 > 0.5, so returns 1.0
    # and increments. t=1 -> 1/e ~= 0.3679 which is NOT > 0.5, so returns 0.5.
    d = decay(initial_value=1, decay_factor=1, min_value=0.5)
    vals = [d.next() for _ in range(4)]
    assert vals[0] == pytest.approx(1.0)
    assert vals[1] == 0.5
    assert vals[2] == 0.5
    assert vals[3] == 0.5


def test_decay_min_value_boundary():
    # Boundary: a == min_value -> `a > min_value` is False -> clamp.
    # At t=0, a=1.0.  Set min_value=1.0: a > 1.0 is False -> clamp to 1.0.
    # The clamp returns min_value (1.0) and does NOT increment _t, so it
    # keeps returning 1.0 (which is the same numeric value, so test _t stays 0).
    d = decay(initial_value=1, decay_factor=1, min_value=1.0)
    v1 = d.next()
    v2 = d.next()
    assert v1 == 1.0
    assert v2 == 1.0
    assert d._t == 0


# ---------------------------------------------------------------------------
# Additional coverage for _Fibo defaults
# ---------------------------------------------------------------------------

def test_fibo_default_max_none():
    f = fibo()
    assert isinstance(f, _Fibo)
    assert f._max_value is None


def test_fibo_stores_max_value_kwarg():
    # super().__init__(max_value=max_value) must pass the actual value.
    f = fibo(max_value=7)
    assert f._kwargs == {"max_value": 7}


# ---------------------------------------------------------------------------
# Additional coverage for _Constant with list identity
# ---------------------------------------------------------------------------

def test_constant_default_interval():
    c = constant()
    assert isinstance(c, _Constant)
    assert c.next() == 1.0
    assert c.next() == 1.0


# ---------------------------------------------------------------------------
# _Wait.__call__ positional args override logic
# ---------------------------------------------------------------------------

def test_wait_call_with_positional_args_overrides_kwargs():
    # _Expo stores base, factor, max_value as kwargs
    e = _Expo(base=2, factor=1, max_value=None)
    e.next()  # advance state
    clone = e(3)  # positional -> base=3
    assert clone.next() == 1
    assert clone.next() == 3
    assert clone.next() == 9


def test_wait_call_merges_kwargs():
    e = _Expo(base=2, factor=1)
    clone = e(base=3, factor=2)
    assert clone.next() == 2
    assert clone.next() == 6


def test_wait_call_more_positional_than_params_raises_typeerror():
    # _Expo registers exactly 3 kwarg params (base, factor, max_value).
    # Passing more positional args than stored params must not raise
    # IndexError in the pop loop; the type error comes from the constructor.
    e = expo(base=1)
    with pytest.raises(TypeError):
        e(2, 3, 4, 5)


def test_expo_stores_all_kwargs():
    e = _Expo(base=5, factor=3, max_value=100)
    assert e._kwargs == {"base": 5, "factor": 3, "max_value": 100}


def test_expo_clone_reuses_all_stored_kwargs():
    e = _Expo(base=5, factor=3, max_value=100)
    clone = e()
    assert isinstance(clone, _Expo)
    # Without args, all stored kwargs must be preserved in the clone.
    assert clone.next() == 3
    assert clone.next() == 15
    assert clone.next() == 75


def test_expo_default_stores_max_value_none():
    e = _Expo()
    assert e._kwargs == {"base": 2, "factor": 1, "max_value": None}


# ---------------------------------------------------------------------------
# _Decay stored kwargs / defaults / clone
# ---------------------------------------------------------------------------

def test_decay_stores_all_kwargs():
    d = _Decay(initial_value=5, decay_factor=2, min_value=0.1)
    assert d._kwargs == {
        "initial_value": 5,
        "decay_factor": 2,
        "min_value": 0.1,
    }


def test_decay_default_kwargs():
    d = _Decay()
    assert d._kwargs == {
        "initial_value": 1,
        "decay_factor": 1,
        "min_value": None,
    }


def test_decay_default_decay_factor_is_one():
    # If the default decay_factor were anything but 1, the second value
    # would not be e^-1.
    d = decay()
    first = d.next()
    second = d.next()
    assert first == pytest.approx(1.0)
    assert second == pytest.approx(math.e ** -1)


def test_decay_clone_uses_stored_kwargs():
    d = _Decay(initial_value=5, decay_factor=2, min_value=0.1)
    clone = d()
    assert isinstance(clone, _Decay)
    # initial_value=5 must be preserved by the clone.
    assert clone.next() == pytest.approx(5.0)
    assert clone.next() == pytest.approx(5.0 * math.e ** -2)


# ---------------------------------------------------------------------------
# _CombinedWait identity / length / sum semantics
# ---------------------------------------------------------------------------

def test_combined_wait_empty_sum_zero():
    combined = wait_combine()
    assert combined.next() == 0.0
    assert combined.next() == 0.0


def test_combined_wait_send_value_forwarded_to_all_children():
    captured = []

    def rec(v):
        captured.append(v)
        return 1.0

    combined = wait_combine(runtime(value=rec), runtime(value=rec), runtime(value=rec))
    combined.next("payload")
    assert captured == ["payload", "payload", "payload"]


# ---------------------------------------------------------------------------
# _WaitNone returns float 0.0
# ---------------------------------------------------------------------------

def test_wait_none_value_and_type():
    w = wait_none()
    assert isinstance(w, _WaitNone)
    v = w.next()
    assert v == 0.0
    assert isinstance(v, float)


# ---------------------------------------------------------------------------
# _WaitRandom default parameters
# ---------------------------------------------------------------------------

def test_wait_random_default_parameters(monkeypatch):
    calls = []

    def fake_uniform(a, b):
        calls.append((a, b))
        return a

    monkeypatch.setattr("backon._wait_gen.random.uniform", fake_uniform)
    w = wait_random()
    assert isinstance(w, _WaitRandom)
    w.next()
    assert calls == [(0, 1)]


# ---------------------------------------------------------------------------
# _WaitIncrementing defaults
# ---------------------------------------------------------------------------

def test_wait_incrementing_defaults():
    w = wait_incrementing()
    assert isinstance(w, _WaitIncrementing)
    assert [w.next() for _ in range(3)] == [1, 2, 3]


# ---------------------------------------------------------------------------
# _WaitFactory is a tiny wrapper
# ---------------------------------------------------------------------------

def test_wait_factory_creates_instances():
    f = _WaitFactory(_Constant)
    w = f(interval=7)
    assert isinstance(w, _Constant)
    assert w.next() == 7.0
