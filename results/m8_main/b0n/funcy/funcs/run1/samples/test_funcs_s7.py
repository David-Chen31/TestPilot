import pytest
from functools import partial

from funcy.funcs import (
    identity, constantly, caller, func_partial, rpartial,
    curry, rcurry, autocurry, iffy, compose, rcompose, complement,
    ljuxt, juxt, reduce, partial,
)


# ---------------------------------------------------------------------------
# identity / constantly / caller
# ---------------------------------------------------------------------------

def test_identity():
    assert identity(5) == 5
    obj = object()
    assert identity(obj) is obj
    assert identity(None) is None


def test_constantly():
    f = constantly(42)
    assert f() == 42
    assert f(1, 2, 3) == 42
    assert f(a=1, b=2) == 42
    # value captured, not copied; same value returned each time
    marker = object()
    g = constantly(marker)
    assert g() is marker
    assert g() is marker


def test_caller():
    f = caller(1, 2, x=3)
    def target(*a, **kw):
        return (a, kw)
    assert f(target) == ((1, 2), {'x': 3})


def test_caller_no_args():
    f = caller()
    assert f(lambda: 99) == 99


# ---------------------------------------------------------------------------
# func_partial / rpartial
# ---------------------------------------------------------------------------

def _three(a, b, c):
    return (a, b, c)


def test_func_partial_basic():
    p = func_partial(_three, 1)
    assert p(2, 3) == (1, 2, 3)


def test_func_partial_kwargs_override():
    def f(a, b, c=0):
        return (a, b, c)
    p = func_partial(f, 1, c=10)
    assert p(2) == (1, 2, 10)
    # new kwargs override stored ones
    assert p(2, c=20) == (1, 2, 20)


def test_func_partial_returns_real_function():
    # Should be a function, not a functools.partial
    p = func_partial(_three, 1, 2, 3)
    assert callable(p)
    assert p() == (1, 2, 3)


def test_rpartial_basic():
    def f(a, b, c):
        return (a, b, c)
    p = rpartial(f, 3)
    assert p(1, 2) == (1, 2, 3)
    p2 = rpartial(f, 2, 3)
    assert p2(1) == (1, 2, 3)


def test_rpartial_kwargs():
    def f(a, b, c=0):
        return (a, b, c)
    p = rpartial(f, 1, c=10)
    # positional args come first, then stored positional (only c=10 stored as kw)
    assert p(2, 1) == (2, 1, 10)


# ---------------------------------------------------------------------------
# curry / rcurry
# ---------------------------------------------------------------------------

def test_curry_n2():
    def add(a, b):
        return a + b
    curried = curry(add, 2)
    assert curried(1)(2) == 3


def test_curry_n3():
    def add3(a, b, c):
        return a + b + c
    curried = curry(add3, 3)
    # one argument at a time
    assert curried(1)(2)(3) == 6


def test_curry_n1_returns_func():
    def f(a):
        return a
    assert curry(f, 1) is f


def test_curry_n0_returns_func():
    f = lambda: 1
    assert curry(f, 0) is f


def test_curry_infers_arity():
    def add(a, b):
        return a + b
    assert curry(add)(1)(2) == 3


def test_rcurry_n2():
    def sub(a, b):
        return a - b
    # right-to-left: first passed arg goes last
    rc = rcurry(sub, 2)
    assert rc(1)(5) == 5 - 1  # func(y, x) = sub(5, 1)


def test_rcurry_n3():
    def f(a, b, c):
        return (a, b, c)
    rc = rcurry(f, 3)
    assert rc(3)(2)(1) == (1, 2, 3)


def test_rcurry_n1_returns_func():
    def f(a):
        return a
    assert rcurry(f, 1) is f


def test_rcurry_infers_arity():
    def sub(a, b):
        return a - b
    rc = rcurry(sub)
    assert rc(1)(5) == 4


# ---------------------------------------------------------------------------
# autocurry
# ---------------------------------------------------------------------------

def test_autocurry_partial_until_enough():
    def add(a, b, c):
        return a + b + c
    ac = autocurry(add)
    part = ac(1)
    assert callable(part)
    part2 = part(2)
    assert callable(part2)
    assert part2(3) == 6


def test_autocurry_all_at_once():
    def add(a, b):
        return a + b
    assert autocurry(add)(1, 2) == 3


def test_autocurry_with_explicit_n():
    def add(a, b, c):
        return a + b + c
    ac = autocurry(add, 3)
    assert ac(1)(2)(3) == 6


def test_autocurry_kwargs():
    def f(a, b):
        return (a, b)
    ac = autocurry(f)
    assert ac(a=1)(b=2) == (1, 2)


def test_autocurry_mixed_kwargs():
    def f(a, b):
        return (a, b)
    ac = autocurry(f)
    assert ac(1)(b=2) == (1, 2)


def test_autocurry_preserves_name():
    def myfunc(a, b):
        return a + b
    ac = autocurry(myfunc)
    assert ac.__name__ == 'myfunc'


# ---------------------------------------------------------------------------
# iffy
# ---------------------------------------------------------------------------

def test_iffy_with_action():
    def double(x):
        return x * 2
    f = iffy(lambda x: x > 0, double)
    assert f(5) == 10
    # default is identity
    assert f(-5) == -5


def test_iffy_default_callable():
    f = iffy(lambda x: x > 0, lambda x: 'pos', lambda x: 'nonpos')
    assert f(1) == 'pos'
    assert f(-1) == 'nonpos'


def test_iffy_default_constant():
    f = iffy(lambda x: x > 0, lambda x: 'pos', 'neg')
    assert f(1) == 'pos'
    assert f(-1) == 'neg'


def test_iffy_single_arg_is_predicate():
    # iffy(pred) -> iffy(bool, pred, identity)
    f = iffy(lambda x: x > 0)
    assert f(5) == 5
    assert f(0) == 0
    assert f(-3) == -3


def test_iffy_with_str_pred_and_action():
    # make_pred / make_func handle strings as regex testers / finders
    f = iffy(r'\d+', str.upper, 'none')
    assert f('abc123') == 'ABC123'
    assert f('abc') == 'none'


# ---------------------------------------------------------------------------
# compose / rcompose / complement
# ---------------------------------------------------------------------------

def test_compose_empty_is_identity():
    assert compose() is identity
    assert compose()(5) == 5


def test_compose_single():
    def inc(x):
        return x + 1
    assert compose(inc)(5) == 6


def test_compose_multiple():
    def inc(x):
        return x + 1
    def double(x):
        return x * 2
    # compose(f, g) -> f(g(x))
    assert compose(inc, double)(5) == 11  # inc(double(5))
    assert compose(double, inc)(5) == 12  # double(inc(5))


def test_compose_with_non_callable():
    # strings become finders via make_func
    composed = compose(len, r'\d+')
    assert composed('abc123') == 3


def test_rcompose():
    def inc(x):
        return x + 1
    def double(x):
        return x * 2
    # rcompose(f, g) -> g(f(x))
    assert rcompose(inc, double)(5) == 12  # double(inc(5))
    assert rcompose(double, inc)(5) == 11  # inc(double(5))


def test_rcompose_empty():
    assert rcompose()(7) == 7


def test_complement():
    is_pos = complement(lambda x: x > 0)
    assert is_pos(1) is False
    assert is_pos(-1) is True


def test_complement_with_builtin():
    is_even = lambda x: x % 2 == 0
    is_odd = complement(is_even)
    assert is_odd(3) is True
    assert is_odd(4) is False


# ---------------------------------------------------------------------------
# juxt / ljuxt
# ---------------------------------------------------------------------------

def test_juxt_returns_iterator():
    def inc(x):
        return x + 1
    def double(x):
        return x * 2
    j = juxt(inc, double)
    result = j(3)
    # lazy -> generator
    assert not isinstance(result, list)
    assert list(result) == [4, 6]


def test_juxt_ reusable():
    j = juxt(lambda x: x + 1, lambda x: x * 2)
    assert list(j(3)) == [4, 6]
    assert list(j(5)) == [6, 10]


def test_juxt_non_callable():
    j = juxt(0)
    assert list(j([10, 20, 30])) == [10]


def test_juxt_empty():
    j = juxt()
    assert list(j(1, 2, 3)) == []


def test_ljuxt_returns_list():
    def inc(x):
        return x + 1
    def double(x):
        return x * 2
    j = ljuxt(inc, double)
    result = j(3)
    assert isinstance(result, list)
    assert result == [4, 6]


def test_ljuxt_reusable():
    j = ljuxt(lambda x: x + 1, lambda x: x * 2)
    assert j(3) == [4, 6]
    assert j(5) == [6, 10]


def test_ljuxt_with_kwargs():
    j = ljuxt(lambda a, b: a + b, lambda a, b: a - b)
    assert j(a=5, b=3) == [8, 2]


def test_ljuxt_empty():
    j = ljuxt()
    assert j(1, 2) == []


# ---------------------------------------------------------------------------
# reexports
# ---------------------------------------------------------------------------

def test_reexported_reduce():
    assert reduce(lambda a, b: a + b, [1, 2, 3, 4]) == 10


def test_reexported_partial():
    def f(a, b):
        return a + b
    p = partial(f, 1)
    assert p(2) == 3
