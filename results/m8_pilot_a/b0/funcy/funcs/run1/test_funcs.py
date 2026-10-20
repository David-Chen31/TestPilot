import pytest
from functools import partial
from operator import add

import funcy.funcs as funcs
from funcy.funcs import (
    identity, constantly, caller,
    func_partial, rpartial,
    curry, rcurry, autocurry,
    iffy,
    compose, rcompose, complement, ljuxt, juxt,
)


# ---------------------------------------------------------------------------
# identity / constantly / caller
# ---------------------------------------------------------------------------

def test_identity_basic():
    assert identity(42) == 42
    obj = object()
    assert identity(obj) is obj


def test_constantly_any_args():
    f = constantly("x")
    assert f() == "x"
    assert f(1, 2, 3, a=4) == "x"
    # Always returns same object
    marker = object()
    g = constantly(marker)
    assert g(1) is marker
    assert g(2) is marker


def test_caller():
    f = caller(1, 2, c=3)
    assert f(lambda a, b, c: (a, b, c)) == (1, 2, 3)

    g = caller()
    assert g(lambda: "result") == "result"


# ---------------------------------------------------------------------------
# func_partial / rpartial
# ---------------------------------------------------------------------------

def test_func_partial_positional():
    p = func_partial(pow, 2)
    assert p(3) == 8
    assert p(4) == 16


def test_func_partial_returns_real_function():
    p = func_partial(add, 1)
    assert callable(p)
    # A real function has __call__, not a functools.partial
    assert not isinstance(p, partial)


def test_func_partial_kwargs():
    def f(a, b, c=10):
        return a + b + c

    p = func_partial(f, 1, c=100)
    assert p(2) == 103
    # New kwargs override bound kwargs
    assert p(2, c=200) == 203


def test_func_partial_can_become_method():
    class C:
        def __init__(self, base):
            self.base = base

    def method(self, x):
        return self.base + x

    C.method = func_partial(method, __import__("builtins").object())  # not used
    # Simpler: construct a bound-style method via func_partial
    inst = C(10)
    assert func_partial(method, inst)(5) == 15


def test_rpartial_positional():
    p = rpartial(pow, 3)  # pow(x, 3)
    assert p(2) == 8


def test_rpartial_extends_positional():
    def f(*args, **kw):
        return args, kw

    p = rpartial(f, 3, 4)
    args, kw = p(1, 2)
    assert args == (1, 2, 3, 4)
    assert kw == {}


def test_rpartial_kwargs_extend_and_override():
    def f(*args, **kw):
        return args, kw

    p = rpartial(f, 5, a=1)
    assert p() == ((5,), {"a": 1})
    assert p(1, 2) == ((1, 2, 5), {"a": 1})
    assert p(a=2, b=3) == ((5,), {"a": 2, "b": 3})


# ---------------------------------------------------------------------------
# curry / rcurry
# ---------------------------------------------------------------------------

def test_curry_with_n():
    c = curry(add, 2)
    assert c(2)(3) == 5


def test_curry_three_args():
    def f(a, b, c):
        return (a, b, c)

    c = curry(f, 3)
    assert c(1)(2)(3) == (1, 2, 3)


def test_curry_n_one_returns_func():
    def f(a):
        return a

    assert curry(f, 1) is f


def test_curry_infers_n():
    def f(a, b):
        return a + b

    assert curry(f)(2)(3) == 5


def test_rcurry_with_n():
    def sub(a, b):
        return a - b

    # Arguments passed right to left: rcurry(sub, 2)(x) => sub(?, x)
    rc = rcurry(sub, 2)
    # lambda x: lambda y: func(y, x) is reversed below:
    assert rc(3)(10) == 10 - 3


def test_rcurry_three_args():
    def f(a, b, c):
        return (a, b, c)

    rc = rcurry(f, 3)
    # First passed arg becomes last positional, etc.
    assert rc(1)(2)(3) == (3, 2, 1)


def test_rcurry_n_one_returns_func():
    def f(a):
        return a

    assert rcurry(f, 1) is f


def test_rcurry_infers_n():
    def sub(a, b):
        return a - b

    rc = rcurry(sub)
    assert rc(3)(10) == 7


# ---------------------------------------------------------------------------
# autocurry
# ---------------------------------------------------------------------------

def test_autocurry_collects_until_enough():
    def f(a, b, c):
        return a + b + c

    ac = autocurry(f)
    p = ac(1)
    assert callable(p)
    p2 = p(2)
    assert callable(p2)
    assert p2(3) == 6


def test_autocurry_can_pass_several_at_once():
    def f(a, b, c):
        return a + b + c

    ac = autocurry(f)
    assert ac(1, 2)(3) == 6
    assert ac(1)(2, 3) == 6
    assert ac(1, 2, 3) == 6


def test_autocurry_explicit_n():
    def f(*args):
        return sum(args)

    ac = autocurry(f, 2)
    assert ac(1)(2) == 3


def test_autocurry_kwargs():
    def f(a, b):
        return a + b

    ac = autocurry(f)
    assert ac(a=1)(b=2) == 3
    assert ac(b=2)(a=1) == 3


def test_autocurry_returns_partial_when_not_enough():
    def f(a, b):
        return a + b

    ac = autocurry(f)
    p = ac(1)
    assert callable(p)
    assert p(5) == 6


def test_autocurry_wraps_name():
    def foo(a, b):
        return a + b

    ac = autocurry(foo)
    assert ac.__name__ == "foo"


# ---------------------------------------------------------------------------
# iffy
# ---------------------------------------------------------------------------

def test_iffy_with_action():
    f = iffy(lambda v: v > 0, lambda v: v * 2)
    assert f(3) == 6
    assert f(-1) == -1


def test_iffy_no_action_delegates_to_bool_pred():
    # iffy(pred) -> iffy(bool, pred, identity)
    f = iffy(lambda v: v)
    # pred becomes bool predicate applied to v; action becomes pred (from make_func)
    assert f(5) == 5
    assert f(0) == 0


def test_iffy_default_non_callable():
    f = iffy(lambda v: v > 0, lambda v: v * 2, default=0)
    assert f(3) == 6
    assert f(-1) == 0


def test_iffy_default_callable():
    f = iffy(lambda v: v > 0, lambda v: v * 2, default=lambda v: -v)
    assert f(3) == 6
    assert f(-1) == 1


# ---------------------------------------------------------------------------
# compose / rcompose
# ---------------------------------------------------------------------------

def test_compose_empty_is_identity():
    assert compose() is identity


def test_compose_single():
    f = compose(lambda x: x + 1)
    assert f(1) == 2


def test_compose_multiple():
    f = compose(str, lambda x: x + 1)
    assert f(1) == "2"


def test_compose_order():
    calls = []

    def a(x):
        calls.append("a")
        return x + 1

    def b(x):
        calls.append("b")
        return x * 2

    f = compose(a, b)
    assert f(3) == 7  # a(b(3)) = a(6) = 7
    assert calls == ["b", "a"]


def test_compose_args_kwargs():
    def f(*a, **kw):
        return (a, kw)

    g = compose(f, lambda *a, **kw: (a, kw))
    assert g(1, 2, x=3) == (((1, 2), {"x": 3}), {})


def test_rcompose_order():
    calls = []

    def a(x):
        calls.append("a")
        return x + 1

    def b(x):
        calls.append("b")
        return x * 2

    f = rcompose(a, b)
    # rcompose(a, b) == compose(b, a) => b(a(x))
    assert f(3) == 8  # b(a(3)) = b(4) = 8
    assert calls == ["a", "b"]


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------

def test_complement():
    is_even = lambda x: x % 2 == 0
    is_odd = complement(is_even)
    assert is_odd(1) is True
    assert is_odd(2) is False


# ---------------------------------------------------------------------------
# ljuxt / juxt
# ---------------------------------------------------------------------------

def test_ljuxt_returns_list():
    f = ljuxt(lambda x: x + 1, lambda x: x * 2)
    result = f(3)
    assert result == [4, 6]
    assert isinstance(result, list)


def test_ljuxt_reusable():
    f = ljuxt(lambda x: x + 1, lambda x: x * 2)
    assert f(1) == [2, 2]
    assert f(2) == [3, 4]


def test_juxt_returns_iterator():
    f = juxt(lambda x: x + 1, lambda x: x * 2)
    result = f(3)
    assert not isinstance(result, list)
    assert list(result) == [4, 6]


def test_juxt_reusable():
    f = juxt(lambda x: x + 1, lambda x: x * 2)
    assert list(f(1)) == [2, 2]
    assert list(f(2)) == [3, 4]


def test_juxt_args_kwargs():
    f = juxt(lambda *a, **kw: (a, kw))
    assert list(f(1, 2, x=3)) == [((1, 2), {"x": 3})]


def test_ljuxt_args_kwargs():
    f = ljuxt(lambda *a, **kw: (a, kw))
    assert f(1, 2, x=3) == [((1, 2), {"x": 3})]


# ---------------------------------------------------------------------------
# make_func path integration (strings / mappings used as functions)
# ---------------------------------------------------------------------------

def test_compose_with_mapping_as_func():
    f = compose(lambda v: v["name"], {"name": "bob"})
    # compose applies mapping lookup after identity, then extracts name
    assert f("anything") == "bob"


def test_iffy_with_string_pred():
    f = iffy("a", lambda v: "has-a", default="no-a")
    assert f("cat") == "has-a"
    assert f("dog") == "no-a"
