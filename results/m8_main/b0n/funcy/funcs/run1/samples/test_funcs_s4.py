"""Unit tests for funcy.funcs."""

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
)


# ---------------------------------------------------------------------------
# identity / constantly / caller
# ---------------------------------------------------------------------------

def test_identity():
    assert identity(42) == 42
    obj = object()
    assert identity(obj) is obj
    assert identity(None) is None


def test_constantly():
    f = constantly(10)
    assert f() == 10
    assert f(1, 2, 3) == 10
    assert f(a=1, b=2) == 10
    assert f(1, 2, a=3) == 10

    obj = object()
    g = constantly(obj)
    assert g(1) is obj


def test_caller():
    def add(a, b):
        return a + b

    f = caller(1, 2)
    assert f(add) == 3

    def kw_func(a=0, b=0):
        return a - b

    g = caller(a=10, b=4)
    assert g(kw_func) == 6

    h = caller()
    assert h(lambda: "hi") == "hi"


# ---------------------------------------------------------------------------
# func_partial / rpartial
# ---------------------------------------------------------------------------

def test_func_partial_is_real_function():
    def add(a, b):
        return a + b

    p = func_partial(add, 1)
    assert callable(p)
    # A real function, not a functools.partial object.
    assert not isinstance(p, type(func_partial))
    assert p(2) == 3
    assert p(10) == 11


def test_func_partial_kwargs_extend_and_override():
    def f(*args, **kwargs):
        return args, kwargs

    p = func_partial(f, 1, 2, x=1, y=2)
    args, kwargs = p(3, x=10, z=3)
    assert args == (1, 2, 3)
    assert kwargs == {"x": 10, "y": 2, "z": 3}


def test_func_partial_as_method():
    class A:
        def __init__(self, base):
            self.base = base

        def add(self, a, b):
            return self.base + a + b

    a = A(100)
    method = func_partial(a.add, 5)
    assert method(3) == 108


def test_rpartial():
    def f(*args, **kwargs):
        return args, kwargs

    p = rpartial(f, 2, 3, x=1)
    args, kwargs = p(1, y=2, x=99)
    assert args == (1, 2, 3)
    assert kwargs == {"x": 99, "y": 2}


def test_rpartial_simple():
    def sub(a, b):
        return a - b

    p = rpartial(sub, 3)
    assert p(10) == 7


# ---------------------------------------------------------------------------
# curry / rcurry
# ---------------------------------------------------------------------------

def test_curry_single_arg_returns_func():
    def f(x):
        return x * 2

    assert curry(f, 1) is f


def test_curry_n_zero_returns_func():
    def f():
        return 5

    assert curry(f, 0) is f


def test_curry_two():
    def add(a, b):
        return a + b

    curried = curry(add, 2)
    assert curried(1)(2) == 3


def test_curry_three():
    def add3(a, b, c):
        return a + b + c

    curried = curry(add3, 3)
    assert curried(1)(2)(3) == 6


def test_curry_infers_from_spec():
    def add3(a, b, c):
        return a + b + c

    curried = curry(add3)
    assert curried(1)(2)(3) == 6


def test_rcurry_two():
    def sub(a, b):
        return a - b

    curried = rcurry(sub, 2)
    # Arguments passed right-to-left: curried(x)(y) == sub(y, x)
    assert curried(3)(10) == 7


def test_rcurry_three():
    def f(a, b, c):
        return (a, b, c)

    curried = rcurry(f, 3)
    # curried(z)(y)(x) == f(x, y, z)
    assert curried("c")("b")("a") == ("a", "b", "c")


def test_rcurry_single_returns_func():
    def f(x):
        return x

    assert rcurry(f, 1) is f


# ---------------------------------------------------------------------------
# autocurry
# ---------------------------------------------------------------------------

def test_autocurry_calls_when_enough_args():
    def add(a, b):
        return a + b

    curried = autocurry(add)
    assert curried(1, 2) == 3


def test_autocurry_partial_then_complete():
    def add(a, b, c):
        return a + b + c

    curried = autocurry(add)
    partial = curried(1)
    assert callable(partial)
    assert partial(2)(3) == 6


def test_autocurry_kwargs():
    def f(a, b):
        return a - b

    curried = autocurry(f)
    assert curried(a=10)(b=3) == 7
    assert curried(10)(b=3) == 7


def test_autocurry_explicit_n():
    def add(a, b):
        return a + b

    curried = autocurry(add, n=2)
    assert curried(1)(2) == 3


def test_autocurry_preserves_name():
    def add(a, b):
        return a + b

    curried = autocurry(add)
    assert curried.__name__ == "add"


def test_autocurry_with_varkw():
    def f(a, b, **kw):
        return a + b + sum(kw.values())

    curried = autocurry(f)
    # With varkw, reaching max_n with all args works.
    assert curried(1, 2, c=3) == 6


# ---------------------------------------------------------------------------
# iffy
# ---------------------------------------------------------------------------

def test_iffy_basic():
    f = iffy(lambda x: x > 0, lambda x: x * 10)
    assert f(5) == 50
    assert f(-1) == -1


def test_iffy_with_default_value():
    f = iffy(lambda x: x > 0, lambda x: x * 10, default="neg")
    assert f(5) == 50
    assert f(-1) == "neg"


def test_iffy_with_default_callable():
    f = iffy(lambda x: x > 0, lambda x: x * 10, default=lambda x: x - 100)
    assert f(5) == 50
    assert f(-1) == -101


def test_iffy_one_arg():
    # Single arg: predicate becomes bool, action is the passed arg
    f = iffy(lambda x: x * 2)
    assert f(3) == 6
    assert f(0) == 0


def test_iffy_pred_mapping():
    # Non-callable pred gets converted via make_pred (e.g. dict -> contains)
    f = iffy({"a": 1}, lambda x: "in", default="out")
    assert f("a") == "in"
    assert f("b") == "out"


# ---------------------------------------------------------------------------
# compose / rcompose
# ---------------------------------------------------------------------------

def test_compose_empty():
    assert compose() is identity
    assert compose()(5) == 5


def test_compose_single():
    def f(x):
        return x + 1

    c = compose(f)
    assert c(1) == 2


def test_compose_order():
    def inc(x):
        return x + 1

    def dbl(x):
        return x * 2

    # compose(f, g)(x) == f(g(x))
    assert compose(inc, dbl)(3) == 7  # inc(dbl(3)) = inc(6) = 7
    assert compose(dbl, inc)(3) == 8  # dbl(inc(3)) = dbl(4) = 8


def test_compose_args_kwargs():
    def collect(*a, **kw):
        return a, kw

    def wrap(x):
        return [x]

    c = compose(wrap, collect)
    assert c(1, 2, k=3) == [((1, 2), {"k": 3})]


def test_compose_with_noncallables():
    # map(make_func, fs) converts e.g. strings to re-finders
    c = compose(str, "a")
    assert c("abc") == "a"


def test_rcompose_order():
    def inc(x):
        return x + 1

    def dbl(x):
        return x * 2

    # rcompose(f, g)(x) == g(f(x))
    assert rcompose(inc, dbl)(3) == 8  # dbl(inc(3)) = dbl(4) = 8
    assert rcompose(dbl, inc)(3) == 7  # inc(dbl(3)) = inc(6) = 7


def test_rcompose_empty():
    assert rcompose() is identity


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------

def test_complement():
    is_even = lambda x: x % 2 == 0
    is_odd = complement(is_even)
    assert is_odd(1) is True
    assert is_odd(2) is False


def test_complement_with_noncallable():
    # "a" is converted to re-finder by make_func
    has_a = complement("a")
    assert has_a("xyz") is True
    assert has_a("abc") is False


# ---------------------------------------------------------------------------
# juxt / ljuxt
# ---------------------------------------------------------------------------

def test_juxt_returns_iterator():
    j = juxt(lambda x: x + 1, lambda x: x * 2)
    result = j(3)
    assert iter(result) is result  # is an iterator
    assert list(result) == [4, 6]


def test_juxt_passes_args_kwargs():
    def f(*a, **kw):
        return a, kw

    def g(*a, **kw):
        return len(a) + len(kw)

    j = juxt(f, g)
    res = list(j(1, 2, x=3))
    assert res[0] == ((1, 2), {"x": 3})
    assert res[1] == 3


def test_juxt_with_noncallables():
    j = juxt("a", "b")
    assert list(j("ab")) == ["a", "b"]


def test_juxt_reusable():
    # Must not deplete its internal functions across calls.
    j = juxt(lambda x: x, lambda x: x * 2)
    assert list(j(2)) == [2, 4]
    assert list(j(3)) == [3, 6]


def test_ljuxt_returns_list():
    j = ljuxt(lambda x: x + 1, lambda x: x * 2)
    result = j(3)
    assert isinstance(result, list)
    assert result == [4, 6]


def test_ljuxt_passes_args():
    def f(*a, **kw):
        return a, kw

    j = ljuxt(f, lambda x: x)
    res = j(5, k=1)
    assert res[0] == ((5,), {"k": 1})
    assert res[1] == 5


def test_ljuxt_reusable():
    j = ljuxt(lambda x: x, lambda x: x * 2)
    assert j(2) == [2, 4]
    assert j(3) == [3, 6]


def test_juxt_and_ljuxt_same_results():
    a = juxt(lambda x: x + 1, lambda x: x - 1)
    b = ljuxt(lambda x: x + 1, lambda x: x - 1)
    assert list(a(10)) == b(10) == [11, 9]
