import re

import pytest

from funcy.funcs import (
    identity,
    constantly,
    caller,
    func_partial,
    rpartial,
    curry,
    rcurry,
    autocurry,
    iffy,
    compose,
    rcompose,
    complement,
    ljuxt,
    juxt,
    reduce,
    partial,
)


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def test_identity_returns_argument():
    assert identity(42) == 42
    obj = object()
    assert identity(obj) is obj
    assert identity(None) is None


# ---------------------------------------------------------------------------
# constantly
# ---------------------------------------------------------------------------

def test_constantly_ignores_args():
    f = constantly(7)
    assert f() == 7
    assert f(1, 2, 3) == 7
    assert f(a=1, b=2) == 7
    assert f(1, b=2) == 7


def test_constantly_fixed_value_identity():
    marker = object()
    f = constantly(marker)
    assert f(1) is marker


# ---------------------------------------------------------------------------
# caller
# ---------------------------------------------------------------------------

def test_caller_calls_with_args():
    f = caller(1, 2, c=3)

    def target(*a, **kw):
        return (a, kw)

    assert f(target) == ((1, 2), {"c": 3})


def test_caller_no_args():
    f = caller()
    assert f(lambda: "hi") == "hi"


def test_caller_with_method():
    f = caller(2, 3)
    assert f(lambda a, b: a * b) == 6


# ---------------------------------------------------------------------------
# func_partial
# ---------------------------------------------------------------------------

def test_func_partial_basic():
    def add(a, b):
        return a + b

    p = func_partial(add, 1)
    assert p(2) == 3


def test_func_partial_extends_args():
    def f(a, b, c):
        return (a, b, c)

    p = func_partial(f, 1, 2)
    assert p(3) == (1, 2, 3)


def test_func_partial_kwargs_override():
    def f(a, b=0):
        return (a, b)

    p = func_partial(f, 1, b=2)
    assert p() == (1, 2)
    assert p(b=5) == (1, 5)


def test_func_partial_is_real_function():
    def f(a):
        return a

    p = func_partial(f, 1)
    assert callable(p)


def test_func_partial_as_method():
    class C:
        def __init__(self, n):
            self.n = n

        method = func_partial(lambda self, x: self.n + x)

    c = C(10)
    # func_partial-built "method": passed instance explicitly
    assert C.method(c, 5) == 15


# ---------------------------------------------------------------------------
# rpartial
# ---------------------------------------------------------------------------

def test_rpartial_basic():
    def sub(a, b):
        return a - b

    p = rpartial(sub, 1)
    assert p(10) == 9


def test_rpartial_args_order():
    def f(a, b, c):
        return (a, b, c)

    p = rpartial(f, 2, 3)
    assert p(1) == (1, 2, 3)


def test_rpartial_kwargs():
    def f(a, b=0):
        return (a, b)

    p = rpartial(f, 1, b=2)
    assert p() == (1, 2)
    assert p(b=5) == (1, 5)


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_three_args():
    def f(a, b, c):
        return (a, b, c)

    cf = curry(f)
    assert cf(1)(2)(3) == (1, 2, 3)


def test_curry_two_args_n():
    def f(a, b):
        return a + b

    cf = curry(f, 2)
    assert cf(1)(2) == 3


def test_curry_single_arg_returns_func():
    def f(a):
        return a

    cf = curry(f)
    assert cf is f


def test_curry_n_zero():
    def f():
        return 5

    assert curry(f, 0) is f


def test_curry_explicit_n():
    def f(a, b, c):
        return a + b + c

    cf = curry(f, 3)
    assert cf(1)(2)(3) == 6


def test_curry_partial_used():
    def f(a, b, c):
        return (a, b, c)

    cf = curry(f)
    step1 = cf(1)
    step2 = step1(2)
    assert step2(3) == (1, 2, 3)


# ---------------------------------------------------------------------------
# rcurry
# ---------------------------------------------------------------------------

def test_rcurry_basic():
    def f(a, b):
        return (a, b)

    rf = rcurry(f)
    # first arg goes to the rightmost position
    assert rf(2)(1) == (1, 2)


def test_rcurry_three_args():
    def f(a, b, c):
        return (a, b, c)

    rf = rcurry(f)
    assert rf(3)(2)(1) == (1, 2, 3)


def test_rcurry_single_arg():
    def f(a):
        return a

    assert rcurry(f, 1) is f


def test_rcurry_n_two():
    def f(a, b):
        return (a, b)

    rf = rcurry(f, 2)
    assert rf(2)(1) == (1, 2)


# ---------------------------------------------------------------------------
# autocurry
# ---------------------------------------------------------------------------

def test_autocurry_basic():
    def f(a, b, c):
        return (a, b, c)

    af = autocurry(f)
    assert af(1)(2)(3) == (1, 2, 3)


def test_autocurry_multiple_args_at_once():
    def f(a, b, c):
        return (a, b, c)

    af = autocurry(f)
    assert af(1, 2)(3) == (1, 2, 3)
    assert af(1)(2, 3) == (1, 2, 3)
    assert af(1, 2, 3) == (1, 2, 3)


def test_autocurry_all_at_once():
    def f(a, b):
        return a + b

    af = autocurry(f)
    assert af(1, 2) == 3


def test_autocurry_with_kwargs():
    def f(a, b, c=10):
        return a + b + c

    af = autocurry(f)
    assert af(1)(2) == 13
    assert af(1)(2, c=0) == 3


def test_autocurry_explicit_n():
    def f(*args):
        return args

    af = autocurry(f, 2)
    assert af(1)(2) == (1, 2)


def test_autocurry_preserves_wraps():
    def f(a, b):
        """docstring here"""
        return a + b

    af = autocurry(f)
    assert af.__name__ == "f"
    assert af.__doc__ == "docstring here"


def test_autocurry_varkw():
    def f(**kw):
        return kw

    af = autocurry(f)
    # With varkw and no required args, calling with kwargs should work
    assert af(a=1) == {"a": 1}


# ---------------------------------------------------------------------------
# iffy
# ---------------------------------------------------------------------------

def test_iffy_with_action_and_default():
    is_even = lambda x: x % 2 == 0
    f = iffy(is_even, lambda x: x * 10, lambda x: x + 1)
    assert f(4) == 40
    assert f(3) == 4


def test_iffy_default_identity():
    f = iffy(lambda x: x > 0)
    assert f(5) == 5
    assert f(-5) == -5


def test_iffy_default_constant():
    f = iffy(lambda x: x > 0, lambda x: "yes", "no")
    assert f(5) == "yes"
    assert f(-5) == "no"


def test_iffy_pred_as_bool_via_partial():
    # iffy(pred) makes iffy(bool, pred, identity)
    f = iffy(lambda x: x)
    assert f(1) == 1
    assert f(0) == 0


def test_iffy_pred_string():
    # make_pred converts string pattern to regex tester
    f = iffy("ab", lambda x: "matched", "nope")
    assert f("xxabxx") == "matched"
    assert f("xxxxx") == "nope"


def test_iffy_pred_true():
    f = iffy(lambda x: True, lambda x: x * 2)
    assert f(3) == 6


# ---------------------------------------------------------------------------
# compose / rcompose
# ---------------------------------------------------------------------------

def test_compose_empty_is_identity():
    f = compose()
    assert f is identity
    assert f(5) == 5


def test_compose_single():
    f = compose(lambda x: x + 1)
    assert f(1) == 2


def test_compose_two():
    f = compose(lambda x: x * 2, lambda x: x + 1)
    # compose applies right-to-left: (x+1)*2
    assert f(3) == 8


def test_compose_three():
    f = compose(lambda x: x - 1, lambda x: x * 2, lambda x: x + 1)
    # ((x+1)*2)-1
    assert f(3) == 7


def test_compose_with_kwargs():
    f = compose(lambda x: x * 2, lambda a, b: a + b)
    assert f(1, b=2) == 6


def test_compose_non_callable_operand():
    # make_func converts non-callables
    f = compose(lambda x: x, lambda x: x)
    assert f(5) == 5


def test_rcompose():
    f = rcompose(lambda x: x + 1, lambda x: x * 2)
    # left-to-right: (x+1)*2
    assert f(3) == 8


def test_rcompose_empty():
    f = rcompose()
    assert f(7) == 7


def test_rcompose_three():
    f = rcompose(lambda x: x + 1, lambda x: x * 2, lambda x: x - 1)
    # ((x+1)*2)-1
    assert f(3) == 7


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------

def test_complement_true():
    is_even = lambda x: x % 2 == 0
    is_odd = complement(is_even)
    assert is_odd(3) is True
    assert is_odd(4) is False


def test_complement_builtin_pred():
    is_odd = complement(bool)
    assert is_odd(0) is True
    assert is_odd(1) is False


def test_complement_string_pattern():
    # make_func converts string to regex finder -> truthy match
    not_digit = complement("digit")
    assert not_digit("abc") is True
    assert not_digit("digit") is False


# ---------------------------------------------------------------------------
# ljuxt / juxt
# ---------------------------------------------------------------------------

def test_ljuxt_returns_list():
    f = ljuxt(lambda x: x + 1, lambda x: x * 2)
    result = f(3)
    assert isinstance(result, list)
    assert result == [4, 6]


def test_ljuxt_kwargs():
    f = ljuxt(lambda a, b: a + b, lambda a, b: a - b)
    assert f(5, b=3) == [8, 2]


def test_ljuxt_empty():
    f = ljuxt()
    assert f() == []


def test_ljuxt_multiple_calls_independent():
    f = ljuxt(lambda x: x, lambda x: x + 1)
    assert f(1) == [1, 2]
    assert f(1) == [1, 2]


def test_juxt_returns_iterator():
    f = juxt(lambda x: x + 1, lambda x: x * 2)
    result = f(3)
    assert not isinstance(result, list)
    assert list(result) == [4, 6]


def test_juxt_empty():
    f = juxt()
    assert list(f()) == []


def test_juxt_multiple_calls_independent():
    f = juxt(lambda x: x, lambda x: x + 1)
    assert list(f(1)) == [1, 2]
    assert list(f(1)) == [1, 2]


def test_juxt_lazy():
    called = []

    def side(n):
        called.append(n)
        return n

    f = juxt(lambda: side(1), lambda: side(2))
    gen = f()
    assert called == []
    assert next(gen) == 1
    assert called == [1]


def test_ljuxt_non_callable_used_as_func():
    # mapping -> __getitem__
    f = ljuxt(lambda k: k, {"a": 1})
    assert f("a") == ["a", 1]


def test_juxt_with_mapping():
    f = juxt({"a": 1})
    assert list(f("a")) == [1]


# ---------------------------------------------------------------------------
# Reexported reduce / partial
# ---------------------------------------------------------------------------

def test_reduce_reexport():
    assert reduce(lambda a, b: a + b, [1, 2, 3]) == 6


def test_partial_reexport():
    add_one = partial(lambda a, b: a + b, 1)
    assert add_one(2) == 3
