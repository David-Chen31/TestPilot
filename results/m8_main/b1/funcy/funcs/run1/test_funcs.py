import pytest
from funcy.funcs import (
    identity, constantly, caller, func_partial, rpartial,
    curry, rcurry, autocurry, iffy,
    compose, rcompose, complement, ljuxt, juxt,
)


# ---------- identity / constantly / caller ----------

def test_identity():
    assert identity(5) == 5
    assert identity(None) is None
    obj = object()
    assert identity(obj) is obj


def test_constantly():
    f = constantly(42)
    assert f() == 42
    assert f(1, 2, 3, x=4) == 42
    z = constantly(None)
    assert z(1) is None


def test_caller():
    def add(a, b):
        return a + b
    f = caller(1, 2)
    assert f(add) == 3

    def greet(name, greeting="hi"):
        return greeting + " " + name
    assert caller("bob", greeting="hello")(greet) == "hello bob"
    assert caller("bob")(greet) == "hi bob"


# ---------- func_partial / rpartial ----------

def test_func_partial():
    def f(a, b, c=0):
        return (a, b, c)
    p = func_partial(f, 1)
    assert p(2) == (1, 2, 0)
    assert p(2, 3) == (1, 2, 3)
    p2 = func_partial(f, 1, 2)
    assert p2() == (1, 2, 0)


def test_func_partial_kwargs_override():
    def f(a, b, c=0, d=0):
        return (a, b, c, d)
    p = func_partial(f, 1, c=3)
    assert p(2) == (1, 2, 3, 0)
    # new kw extends/overrides
    assert p(2, c=9) == (1, 2, 9, 0)
    assert p(2, d=5) == (1, 2, 3, 5)


def test_func_partial_is_function():
    def f(a):
        return a
    p = func_partial(f, 1)
    assert callable(p)


def test_rpartial():
    def f(a, b, c=0):
        return (a, b, c)
    p = rpartial(f, 3)
    assert p(1) == (1, 3, 0)
    p2 = rpartial(f, 2, 3)
    assert p2(1) == (1, 2, 3)


def test_rpartial_kwargs_override():
    def f(a, b, c=0, d=0):
        return (a, b, c, d)
    p = rpartial(f, c=3)
    assert p(1, 2) == (1, 2, 3, 0)
    assert p(1, 2, c=9) == (1, 2, 9, 0)


# ---------- curry / rcurry ----------

def test_curry_explicit_n():
    def f(a, b, c):
        return (a, b, c)
    c = curry(f, 3)
    assert c(1)(2)(3) == (1, 2, 3)


def test_curry_n_1_returns_func():
    def f(a):
        return a
    assert curry(f, 1) is f


def test_curry_n_2():
    def f(a, b):
        return (a, b)
    c = curry(f, 2)
    assert c(1)(2) == (1, 2)


def test_curry_inferred():
    def f(a, b, c):
        return (a, b, c)
    c = curry(f)
    assert c(1)(2)(3) == (1, 2, 3)


def test_rcurry_explicit_n():
    def f(a, b, c):
        return (a, b, c)
    c = rcurry(f, 3)
    # args passed right to left
    assert c(3)(2)(1) == (1, 2, 3)


def test_rcurry_n_1_returns_func():
    def f(a):
        return a
    assert rcurry(f, 1) is f


def test_rcurry_n_2():
    def f(a, b):
        return (a, b)
    c = rcurry(f, 2)
    assert c(2)(1) == (1, 2)


# ---------- autocurry ----------

def test_autocurry_full_call():
    def f(a, b):
        return (a, b)
    ac = autocurry(f)
    assert ac(1, 2) == (1, 2)


def test_autocurry_partial():
    def f(a, b):
        return (a, b)
    ac = autocurry(f)
    g = ac(1)
    assert callable(g)
    assert g(2) == (1, 2)


def test_autocurry_three_args():
    def f(a, b, c):
        return (a, b, c)
    ac = autocurry(f)
    assert ac(1, 2, 3) == (1, 2, 3)
    assert ac(1)(2)(3) == (1, 2, 3)
    assert ac(1, 2)(3) == (1, 2, 3)


def test_autocurry_kwargs():
    def f(a, b, c=0):
        return (a, b, c)
    ac = autocurry(f)
    assert ac(1, 2) == (1, 2, 0)
    assert ac(1)(b=2) == (1, 2, 0)


def test_autocurry_explicit_n():
    def f(a, b, c):
        return (a, b, c)
    ac = autocurry(f, 3)
    assert ac(1, 2, 3) == (1, 2, 3)


def test_autocurry_with_varkw():
    def f(a, b=0, **kw):
        return (a, b, kw)
    ac = autocurry(f)
    assert ac(1) == (1, 0, {})
    assert ac(1, b=2) == (1, 2, {})
    assert ac(1, z=5) == (1, 0, {'z': 5})


# ---------- iffy ----------

def test_iffy_basic():
    f = iffy(lambda x: x > 0, lambda x: x * 10)
    assert f(5) == 50
    assert f(-5) == -5  # default identity


def test_iffy_default_callable():
    f = iffy(lambda x: x > 0, lambda x: x * 10, lambda x: x - 1)
    assert f(5) == 50
    assert f(-5) == -6


def test_iffy_default_value():
    f = iffy(lambda x: x > 0, lambda x: x * 10, "neg")
    assert f(5) == 50
    assert f(-5) == "neg"


def test_iffy_two_arg_form():
    # iffy(pred) means iffy(bool, pred)
    f = iffy(lambda x: x * 2)
    assert f(5) == 10
    assert f(0) == False


def test_iffy_with_noncallable_pred():
    # make_pred handles e.g. set membership
    f = iffy({1, 2, 3}, lambda x: x * 10)
    assert f(2) == 20
    assert f(5) == 5


def test_iffy_with_mapping_action():
    f = iffy(lambda x: True, {1: "a", 2: "b"})
    assert f(1) == "a"
    assert f(2) == "b"


# ---------- compose / rcompose ----------

def test_compose_empty():
    assert compose() is identity


def test_compose_single():
    f = compose(lambda x: x + 1)
    assert f(1) == 2


def test_compose_two():
    f = compose(lambda x: x + 1, lambda x: x * 2)
    # compose(f, g)(x) == f(g(x))
    assert f(3) == 7


def test_compose_three():
    f = compose(lambda x: x + 100, lambda x: x * 2, lambda x: x - 1)
    # ((x-1)*2)+100
    assert f(5) == 108


def test_rcompose():
    f = rcompose(lambda x: x - 1, lambda x: x * 2, lambda x: x + 100)
    # rcompose reuses compose(*reversed(fs)):
    # compose((x+100), (x*2), (x-1))(5) == ((5-1)*2)+100 == 108
    assert f(5) == 108


def test_rcompose_simple():
    f = rcompose(lambda x: x + 1, lambda x: x * 2)
    # compose(x*2, x+1)(3) == (3+1)*2 == 8
    assert f(3) == 8


def test_rcompose_empty():
    assert rcompose() is identity


def test_compose_with_noncallable():
    # make_func converts non-callables
    f = compose({1: "a", 2: "b"})
    assert f(1) == "a"


# ---------- complement ----------

def test_complement():
    is_odd = lambda x: x % 2 == 1
    is_even = complement(is_odd)
    assert is_even(2) is True
    assert is_even(3) is False


def test_complement_returns_bool():
    c = complement(lambda x: x)
    assert c(0) is True
    assert c(1) is False


# ---------- ljuxt / juxt ----------

def test_ljuxt():
    f = ljuxt(lambda x: x + 1, lambda x: x * 2)
    assert f(3) == [4, 6]


def test_ljuxt_no_args():
    f = ljuxt()
    assert f(1) == []


def test_ljuxt_kwargs():
    f = ljuxt(lambda a, b: a + b, lambda a, b: a * b)
    assert f(2, 3) == [5, 6]


def test_juxt_returns_iterator():
    f = juxt(lambda x: x + 1, lambda x: x * 2)
    result = f(3)
    assert not isinstance(result, list)
    assert list(result) == [4, 6]


def test_juxt_reusable():
    f = juxt(lambda x: x + 1, lambda x: x * 2)
    assert list(f(3)) == [4, 6]
    assert list(f(5)) == [6, 10]


def test_juxt_no_args():
    f = juxt()
    assert list(f(1)) == []


def test_juxt_with_noncallable():
    f = juxt({1: "a"}, lambda x: x + 10)
    assert list(f(1)) == ["a", 11]
