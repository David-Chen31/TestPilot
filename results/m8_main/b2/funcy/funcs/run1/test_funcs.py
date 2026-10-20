import pytest
from funcy.funcs import (
    identity, constantly, caller, func_partial, rpartial,
    curry, rcurry, autocurry, iffy, compose, rcompose,
    complement, ljuxt, juxt,
)


# ---- identity ----

def test_identity_returns_argument():
    obj = object()
    assert identity(obj) is obj
    assert identity(5) == 5
    assert identity(None) is None


# ---- constantly ----

def test_constantly_returns_constant():
    f = constantly(42)
    assert f() == 42
    assert f(1, 2, 3) == 42
    assert f(a=1, b=2) == 42
    assert f(1, a=2) == 42


def test_constantly_returns_same_object():
    obj = object()
    f = constantly(obj)
    assert f() is obj
    assert f(1) is obj


# ---- caller ----

def test_caller_calls_with_args():
    f = caller(1, 2)
    assert f(lambda a, b: a + b) == 3


def test_caller_with_kwargs():
    f = caller(x=10)
    assert f(lambda x: x * 2) == 20


def test_caller_with_both():
    f = caller(1, y=2)
    assert f(lambda a, y: (a, y)) == (1, 2)


# ---- func_partial ----

def test_func_partial_basic():
    add = lambda a, b: a + b
    inc = func_partial(add, 1)
    assert inc(2) == 3


def test_func_partial_extends_args():
    f = func_partial(lambda *a: a, 1, 2)
    assert f(3, 4) == (1, 2, 3, 4)


def test_func_partial_kwargs_override():
    f = func_partial(lambda **kw: kw, a=1, b=2)
    assert f(b=20, c=30) == {'a': 1, 'b': 20, 'c': 30}


def test_func_partial_returns_real_function():
    f = func_partial(lambda x: x, 1)
    assert callable(f)
    assert not hasattr(f, 'func')  # not a functools.partial


# ---- rpartial ----

def test_rpartial_basic():
    sub = lambda a, b: a - b
    f = rpartial(sub, 1)  # a - 1
    assert f(5) == 4


def test_rpartial_appends_after():
    f = rpartial(lambda *a: a, 3, 4)
    assert f(1, 2) == (1, 2, 3, 4)


def test_rpartial_kwargs_override():
    f = rpartial(lambda **kw: kw, a=1)
    assert f(a=99, b=2) == {'a': 99, 'b': 2}


# ---- curry ----

def test_curry_default_arity():
    f = curry(lambda a, b: a + b)
    assert f(1)(2) == 3


def test_curry_n_one_returns_func():
    g = lambda a: a
    assert curry(g, n=1) is g


def test_curry_n_zero_returns_func():
    g = lambda: 5
    assert curry(g, n=0) is g


def test_curry_three_args():
    f = curry(lambda a, b, c: a + b + c)
    assert f(1)(2)(3) == 6


def test_curry_four_args():
    f = curry(lambda a, b, c, d: a + b + c + d)
    assert f(1)(2)(3)(4) == 10


def test_curry_explicit_n():
    # n=2 wraps only first two args; the underlying func is called with (1, 2)
    # which is missing 'c', so TypeError is raised.
    f = curry(lambda a, b, c: a + b + c, n=2)
    with pytest.raises(TypeError):
        f(1)(2)


def test_curry_explicit_n_correct_arity():
    # with a two-arg function and n=2, it curries into two one-arg functions
    f = curry(lambda a, b: a + b, n=2)
    assert f(1)(2) == 3


# ---- rcurry ----

def test_rcurry_default():
    # right to left: f(2)(1) => func(1, 2)
    sub = lambda a, b: a - b
    f = rcurry(sub)
    assert f(2)(10) == 8  # 10 - 2


def test_rcurry_n_one():
    g = lambda a: a
    assert rcurry(g, n=1) is g


def test_rcurry_three():
    f = rcurry(lambda a, b, c: (a, b, c))
    # args passed right to left
    assert f(3)(2)(1) == (1, 2, 3)


def test_rcurry_four():
    f = rcurry(lambda a, b, c, d: (a, b, c, d))
    assert f(4)(3)(2)(1) == (1, 2, 3, 4)


# ---- autocurry ----

def test_autocurry_single_call():
    f = autocurry(lambda a, b: a + b)
    assert f(1, 2) == 3


def test_autocurry_partial_then_call():
    f = autocurry(lambda a, b: a + b)
    g = f(1)
    assert g(2) == 3


def test_autocurry_stepwise():
    f = autocurry(lambda a, b, c: a + b + c)
    assert f(1)(2)(3) == 6


def test_autocurry_mixed_args():
    f = autocurry(lambda a, b, c: a + b + c)
    assert f(1, 2)(3) == 6
    assert f(1)(2, 3) == 6
    assert f(1, 2, 3) == 6


def test_autocurry_kwargs():
    f = autocurry(lambda a, b: a + b)
    assert f(a=1)(b=2) == 3
    assert f(a=1, b=2) == 3


def test_autocurry_explicit_n():
    f = autocurry(lambda a, b, c: a + b + c, n=2)
    # After 2 args, calls func with those two; missing third raises.
    f2 = f(1)
    with pytest.raises(TypeError):
        f2(2)


def test_autocurry_optional_args():
    def fn(a, b=10):
        return a + b
    f = autocurry(fn)
    assert f(1) == 11


def test_autocurry_preserves_name():
    def my_func(a, b):
        return a + b
    f = autocurry(my_func)
    # wraps should keep __name__ (or at least not crash); basic invocation
    assert f(1)(2) == 3
    assert f.__name__ == 'my_func'


# ---- iffy ----

def test_iffy_single_arg():
    f = iffy(lambda x: x > 0)
    assert f(5) is True
    assert f(-5) is False


def test_iffy_with_action_and_default():
    f = iffy(lambda x: x > 0, lambda x: 'pos', lambda x: 'neg')
    assert f(5) == 'pos'
    assert f(-5) == 'neg'


def test_iffy_default_constant():
    f = iffy(lambda x: x > 0, lambda x: 'pos', 'other')
    assert f(5) == 'pos'
    assert f(-5) == 'other'


def test_iffy_default_identity():
    f = iffy(lambda x: x > 0, lambda x: 'pos')
    assert f(5) == 'pos'
    assert f(-5) == -5


def test_iffy_pred_made_from_value():
    # make_pred allows non-callable preds like sets
    f = iffy({1, 2, 3})
    assert f(1) is True
    assert f(5) is False


def test_iffy_action_made_from_value():
    # make_func uses mapping.__getitem__; missing keys raise KeyError
    f = iffy(lambda x: x > 0, {1: 'one', 2: 'two'}, 'other')
    assert f(1) == 'one'
    assert f(2) == 'two'
    with pytest.raises(KeyError):
        f(3)


def test_iffy_action_mapping_negative_default():
    # non-callable default is returned as-is for falsy pred
    f = iffy(lambda x: x > 0, {1: 'one'}, 'neg')
    assert f(1) == 'one'
    assert f(-1) == 'neg'


# ---- compose ----

def test_compose_with_functions():
    f = compose(lambda x: x + 1, lambda x: x * 2)
    assert f(3) == 7  # (3*2)+1


def test_compose_empty_returns_identity():
    f = compose()
    assert f is identity
    assert f(42) == 42


def test_compose_single():
    f = compose(lambda x: x + 1)
    assert f(1) == 2


def test_compose_three():
    f = compose(lambda x: x + 1, lambda x: x * 2, lambda x: x - 3)
    assert f(5) == 5  # ((5-3)*2)+1


def test_compose_with_non_callable():
    # make_func turns strings into re_finder etc.  Use None -> identity
    f = compose(lambda x: x + 1, None)
    assert f(1) == 2


# ---- rcompose ----

def test_rcompose_left_to_right():
    f = rcompose(lambda x: x + 1, lambda x: x * 2)
    assert f(3) == 8  # (3+1)*2


def test_rcompose_empty():
    f = rcompose()
    assert f(7) == 7


def test_rcompose_single():
    f = rcompose(lambda x: x + 1)
    assert f(1) == 2


# ---- complement ----

def test_complement():
    f = complement(lambda x: x > 0)
    assert f(5) is False
    assert f(-5) is True


def test_complement_with_non_bool_pred():
    # non-bool predicate gets negated with operator.__not__
    f = complement(lambda x: x)
    assert f(0) is True
    assert f(1) is False


# ---- ljuxt ----

def test_ljuxt_returns_list():
    f = ljuxt(lambda x: x + 1, lambda x: x * 2)
    result = f(3)
    assert isinstance(result, list)
    assert result == [4, 6]


def test_ljuxt_empty():
    f = ljuxt()
    assert f(1, 2, 3) == []


def test_ljuxt_reusable():
    f = ljuxt(lambda x: x + 1, lambda x: x * 2)
    assert f(1) == [2, 2]
    assert f(2) == [3, 4]


def test_ljuxt_kwargs():
    # kwargs are forwarded to each function; each must accept them
    f = ljuxt(lambda a, b: a + b)
    assert f(a=1, b=2) == [3]


def test_ljuxt_kwargs_mismatch_raises():
    f = ljuxt(lambda a: a, lambda b: b)
    with pytest.raises(TypeError):
        f(a=1, b=2)


# ---- juxt ----

def test_juxt_returns_iterator():
    f = juxt(lambda x: x + 1, lambda x: x * 2)
    result = f(3)
    assert not isinstance(result, list)
    assert list(result) == [4, 6]


def test_juxt_empty():
    f = juxt()
    assert list(f(1)) == []


def test_juxt_reusable():
    f = juxt(lambda x: x + 1, lambda x: x * 2)
    assert list(f(1)) == [2, 2]
    assert list(f(2)) == [3, 4]


def test_juxt_kwargs():
    # kwargs are forwarded to each function; each must accept them
    f = juxt(lambda a, b: a + b)
    assert list(f(a=1, b=2)) == [3]


def test_juxt_kwargs_mismatch_is_lazy():
    # juxt is lazy: each function is called independently, so a mismatch
    # does not raise at call time. The first function receives b=2 (TypeError),
    # which surfaces only when the generator is consumed.
    f = juxt(lambda a: a, lambda b: b)
    result = f(a=1, b=2)
    with pytest.raises(TypeError):
        list(result)
