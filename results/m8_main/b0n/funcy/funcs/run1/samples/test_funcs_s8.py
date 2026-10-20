import pytest

from funcy.funcs import (
    identity, constantly, caller, func_partial, rpartial,
    curry, rcurry, autocurry, iffy,
    compose, rcompose, complement, ljuxt, juxt,
)


# ---------- identity ----------

def test_identity():
    assert identity(5) == 5
    obj = object()
    assert identity(obj) is obj
    assert identity(None) is None


# ---------- constantly ----------

def test_constantly():
    f = constantly(42)
    assert f() == 42
    assert f(1, 2, 3) == 42
    assert f(a=1, b=2) == 42


def test_constantly_with_none():
    f = constantly(None)
    assert f(1, 2) is None


# ---------- caller ----------

def test_caller():
    f = caller(1, 2, x=3)

    def target(*a, **kw):
        return (a, kw)

    assert f(target) == ((1, 2), {'x': 3})


def test_caller_no_args():
    f = caller()
    assert f(lambda: 7) == 7


# ---------- func_partial ----------

def test_func_partial():
    def add(a, b, c=0):
        return a + b + c

    p = func_partial(add, 1)
    assert p(2) == 3
    assert p(2, 3) == 6
    assert p(2, c=10) == 13


def test_func_partial_extends_kwargs():
    def show(a, b=1):
        return (a, b)

    p = func_partial(show, 10, b=2)
    assert p() == (10, 2)
    assert p(b=5) == (10, 5)


def test_func_partial_is_function():
    p = func_partial(lambda x: x, 1)

    def m(self):
        return 1

    # Real function, usable as a method
    assert callable(p)
    assert func_partial(m, None)() == 1


# ---------- rpartial ----------

def test_rpartial():
    def sub(a, b):
        return a - b

    p = rpartial(sub, 5)
    assert p(10) == 5


def test_rpartial_kwargs_extend_and_override():
    def show(a=0, b=0, c=0):
        return (a, b, c)

    p = rpartial(show, b=1)
    assert p(a=10) == (10, 1, 0)
    assert p(b=9) == (0, 9, 0)


def test_rpartial_positional():
    def join(*args):
        return args

    p = rpartial(join, 3, 4)
    assert p(1, 2) == (1, 2, 3, 4)


# ---------- curry ----------

def test_curry_simple():
    def add(a, b):
        return a + b

    assert curry(add)(1)(2) == 3


def test_curry_three_args():
    def f(a, b, c):
        return a + b + c

    assert curry(f)(1)(2)(3) == 6


def test_curry_explicit_n():
    def f(*a):
        return a

    assert curry(f, 3)(1)(2)(3) == (1, 2, 3)


def test_curry_n_leq_1_returns_func():
    def f(a):
        return a

    assert curry(f, 1) is f
    assert curry(f, 0) is f


# ---------- rcurry ----------

def test_rcurry_simple():
    def sub(a, b):
        return a - b

    # rcurry: argument passed first ends up last
    assert rcurry(sub)(5)(10) == 5   # sub(10, 5)


def test_rcurry_three_args():
    def f(a, b, c):
        return (a, b, c)

    assert rcurry(f)(1)(2)(3) == (3, 2, 1)


def test_rcurry_n_leq_1_returns_func():
    def f(a):
        return a

    assert rcurry(f, 1) is f


# ---------- autocurry ----------

def test_autocurry_until_enough_args():
    def add(a, b, c):
        return a + b + c

    ac = autocurry(add)
    assert ac(1, 2, 3) == 6
    assert ac(1)(2, 3) == 6
    assert ac(1, 2)(3) == 6
    assert ac(1)(2)(3) == 6


def test_autocurry_with_kwargs():
    def f(a, b=None):
        return (a, b)

    ac = autocurry(f)
    assert ac(1) == (1, None)
    assert ac(1, b=2) == (1, 2)


def test_autocurry_explicit_n():
    def f(*a):
        return a

    ac = autocurry(f, 2)
    assert ac(1)(2) == (1, 2)
    assert ac(1, 2, 3, 4) == (1, 2, 3, 4)


def test_autocurry_provides_name():
    def myfunc(a, b):
        return a + b

    ac = autocurry(myfunc)
    assert ac.__name__ == 'myfunc'


# ---------- iffy ----------

def test_iffy_basic():
    f = iffy(lambda x: x > 0, lambda x: x * 2)
    assert f(3) == 6
    assert f(-1) == -1


def test_iffy_custom_default():
    f = iffy(lambda x: x > 0, lambda x: x * 2, lambda x: -x)
    assert f(3) == 6
    assert f(-3) == 3


def test_iffy_constant_default():
    f = iffy(lambda x: x > 0, lambda x: x * 2, 'no')
    assert f(3) == 6
    assert f(-3) == 'no'


def test_iffy_single_arg_treats_first_as_action():
    # calling iffy with one arg makes it `iffy(bool, pred, default=identity)`
    f = iffy(lambda x: x * 2)
    assert f(3) == 6
    assert f(0) == 0


def test_iffy_pred_shortcut():
    # pred may be a non-callable (e.g. a set) via make_pred
    f = iffy({1, 2, 3}, lambda x: x * 10)
    assert f(1) == 10
    assert f(9) == 9


# ---------- compose ----------

def test_compose():
    f = compose(lambda x: x + 1, lambda x: x * 2)
    assert f(3) == 7  # (3*2)+1


def test_compose_empty():
    assert compose() is identity
    assert compose()(5) == 5


def test_compose_single():
    f = compose(lambda x: x + 1)
    assert f(1) == 2


def test_compose_with_kwargs():
    f = compose(lambda x: x + 1, lambda a, b: a + b)
    assert f(a=1, b=2) == 4


def test_compose_with_non_callable():
    # make_func allows e.g. an int as itemgetter
    f = compose(lambda x: x + 1, 0)
    assert f([10, 20]) == 11


# ---------- rcompose ----------

def test_rcompose():
    f = rcompose(lambda x: x + 1, lambda x: x * 2)
    assert f(3) == 8  # (3+1)*2


def test_rcompose_empty():
    assert rcompose()(5) == 5


# ---------- complement ----------

def test_complement():
    is_positive = lambda x: x > 0
    not_positive = complement(is_positive)
    assert not_positive(1) is False
    assert not_positive(-1) is True


def test_complement_with_non_callable():
    # make_func converts a set into __contains__
    not_in = complement({1, 2, 3})
    assert not_in(1) is False
    assert not_in(9) is True


# ---------- ljuxt ----------

def test_ljuxt():
    f = ljuxt(lambda x: x + 1, lambda x: x * 2)
    result = f(3)
    assert result == [4, 6]
    assert isinstance(result, list)


def test_ljuxt_reusable():
    f = ljuxt(lambda x: x + 1, lambda x: x * 2)
    assert f(1) == [2, 2]
    assert f(1) == [2, 2]


def test_ljuxt_no_funcs():
    f = ljuxt()
    assert f(1) == []


def test_ljuxt_kwargs():
    f = ljuxt(lambda a, b: a + b, lambda a, b: a - b)
    assert f(10, 3) == [13, 7]


# ---------- juxt ----------

def test_juxt():
    f = juxt(lambda x: x + 1, lambda x: x * 2)
    result = f(3)
    # returns an iterator
    assert not isinstance(result, list)
    assert list(result) == [4, 6]


def test_juxt_reusable():
    f = juxt(lambda x: x + 1, lambda x: x * 2)
    assert list(f(1)) == [2, 2]
    assert list(f(1)) == [2, 2]


def test_juxt_no_funcs():
    f = juxt()
    assert list(f(1)) == []


def test_juxt_with_non_callable():
    # make_func turns int into itemgetter
    f = juxt(0, 1)
    assert list(f([10, 20])) == [10, 20]
