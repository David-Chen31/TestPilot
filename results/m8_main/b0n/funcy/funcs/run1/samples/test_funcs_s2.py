import pytest

from funcy.funcs import (
    identity, constantly, caller, func_partial, rpartial,
    curry, rcurry, autocurry, iffy,
    compose, rcompose, complement, ljuxt, juxt,
)


# ---------- identity ----------

def test_identity_returns_argument():
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
    assert f(1, key="value") == 42


def test_constantly_none():
    f = constantly(None)
    assert f() is None


# ---------- caller ----------

def test_caller_positional():
    f = caller(1, 2)
    assert f(lambda a, b: a + b) == 3


def test_caller_kwargs():
    f = caller(x=10)
    assert f(lambda x: x * 2) == 20


def test_caller_mixed():
    f = caller(1, y=2)
    assert f(lambda a, y: a + y) == 3


# ---------- func_partial ----------

def test_func_partial_basic():
    def add(a, b):
        return a + b

    p = func_partial(add, 1)
    assert p(2) == 3


def test_func_partial_with_kwargs():
    def f(a, b=10, c=20):
        return a + b + c

    p = func_partial(f, 1, b=2)
    assert p() == 23
    assert p(b=3) == 24


def test_func_partial_returns_real_function():
    def f(a, b):
        return a + b

    p = func_partial(f, 1)
    # must be a real function, not a functools.partial object
    assert callable(p)
    assert not hasattr(p, 'func')


def test_func_partial_override_kwargs():
    def f(a, b):
        return a - b

    p = func_partial(f, 5, b=1)
    assert p() == 4
    assert p(b=2) == 3


# ---------- rpartial ----------

def test_rpartial_basic():
    def sub(a, b):
        return a - b

    p = rpartial(sub, 1)
    assert p(10) == 9


def test_rpartial_with_kwargs():
    def f(a, b, c=0):
        return a + b + c

    p = rpartial(f, 2, c=3)
    assert p(1) == 6


def test_rpartial_returns_real_function():
    def f(a, b):
        return a + b

    p = rpartial(f, 1)
    assert callable(p)
    assert not hasattr(p, 'func')


# ---------- curry ----------

def test_curry_default_arity():
    def f(a, b, c):
        return a + b + c

    c = curry(f)
    assert c(1)(2)(3) == 6


def test_curry_explicit_arity():
    def f(a, b):
        return a * b

    c = curry(f, 2)
    assert c(3)(4) == 12


def test_curry_one_arg_returns_func_itself():
    def f(a):
        return a

    assert curry(f) is f
    assert curry(f, 1) is f
    assert curry(f, 0) is f


def test_curry_two_arg():
    def f(a, b):
        return a - b

    c = curry(f, 2)
    assert c(10)(3) == 7


def test_curry_three_arg_chain():
    def f(a, b, c, d):
        return a + b + c + d

    c = curry(f, 4)
    assert c(1)(2)(3)(4) == 10


# ---------- rcurry ----------

def test_rcurry_default_arity():
    def f(a, b, c):
        return (a, b, c)

    rc = rcurry(f)
    assert rc(3)(2)(1) == (1, 2, 3)


def test_rcurry_two_arg():
    def sub(a, b):
        return a - b

    rc = rcurry(sub, 2)
    # rc(x)(y) == sub(y, x)
    assert rc(3)(10) == 7


def test_rcurry_one_arg():
    def f(a):
        return a

    assert rcurry(f) is f
    assert rcurry(f, 1) is f


def test_rcurry_three_arg():
    def f(a, b, c):
        return (a, b, c)

    rc = rcurry(f, 3)
    assert rc(3)(2)(1) == (1, 2, 3)


# ---------- autocurry ----------

def test_autocurry_full_call():
    def f(a, b, c):
        return a + b + c

    ac = autocurry(f)
    assert ac(1, 2, 3) == 6


def test_autocurry_partial_chain():
    def f(a, b, c):
        return a + b + c

    ac = autocurry(f)
    assert ac(1)(2)(3) == 6


def test_autocurry_mixed():
    def f(a, b, c):
        return a + b + c

    ac = autocurry(f)
    assert ac(1, 2)(3) == 6
    assert ac(1)(2, 3) == 6


def test_autocurry_with_kwargs():
    def f(a, b, c):
        return a + b + c

    ac = autocurry(f)
    assert ac(1, b=2)(3) == 6
    assert ac(1)(b=2, c=3) == 6


def test_autocurry_explicit_n():
    def f(a, b):
        return a + b

    ac = autocurry(f, 2)
    assert ac(1)(2) == 3


def test_autocurry_preserves_metadata():
    def f(a, b):
        "docstring"
        return a + b

    ac = autocurry(f)
    assert ac.__name__ == 'f'
    assert ac.__doc__ == 'docstring'


def test_autocurry_typeerror_retry():
    def f(a, b, c):
        return a + b  # raises TypeError if c provided but also b needed

    ac = autocurry(f)
    # Providing a and c but not b should not immediately call f successfully
    partial_result = ac(a=1, c=3)
    assert callable(partial_result)
    assert partial_result(2) == 3


# ---------- iffy ----------

def test_iffy_basic():
    f = iffy(lambda x: x > 0, lambda x: x * 2, lambda x: 0)
    assert f(5) == 10
    assert f(-1) == 0


def test_iffy_default_identity():
    f = iffy(lambda x: x > 0)
    assert f(5) == 5
    assert f(-1) is False


def test_iffy_with_non_callable_default():
    f = iffy(lambda x: x > 0, lambda x: x * 2, 'no')
    assert f(5) == 10
    assert f(-1) == 'no'


def test_iffy_pred_truthiness():
    f = iffy(lambda x: x > 0, lambda x: 'pos', 'non-pos')
    assert f(1) == 'pos'
    assert f(0) == 'non-pos'
    assert f(-1) == 'non-pos'


# ---------- compose ----------

def test_compose_basic():
    f = compose(lambda x: x + 1, lambda x: x * 2)
    assert f(3) == 7  # (3*2)+1


def test_compose_single():
    f = compose(lambda x: x + 1)
    assert f(5) == 6


def test_compose_empty():
    assert compose() is identity


def test_compose_args_passed_to_rightmost():
    f = compose(lambda x: x, lambda a, b: a + b)
    assert f(1, 2) == 3


# ---------- rcompose ----------

def test_rcompose_basic():
    f = rcompose(lambda x: x + 1, lambda x: x * 2)
    assert f(3) == 8  # (3+1)*2


def test_rcompose_empty():
    assert rcompose() is identity


# ---------- complement ----------

def test_complement():
    is_positive = lambda x: x > 0
    not_positive = complement(is_positive)
    assert not_positive(5) is False
    assert not_positive(-1) is True
    assert not_positive(0) is True


# ---------- ljuxt ----------

def test_ljuxt():
    f = ljuxt(lambda x: x + 1, lambda x: x * 2)
    result = f(4)
    assert result == [5, 8]
    assert isinstance(result, list)


def test_ljuxt_single():
    f = ljuxt(lambda x: x * 3)
    assert f(2) == [6]


def test_ljuxt_empty():
    f = ljuxt()
    assert f() == []
    assert f(1, 2) == []


def test_ljuxt_multiple_calls():
    f = ljuxt(lambda x: x + 1, lambda x: x * 2)
    assert f(1) == [2, 2]
    assert f(2) == [3, 4]


# ---------- juxt ----------

def test_juxt():
    f = juxt(lambda x: x + 1, lambda x: x * 2)
    result = f(4)
    assert list(result) == [5, 8]


def test_juxt_returns_iterator():
    f = juxt(lambda x: x)
    result = f(1)
    # should be an iterator (generator)
    assert iter(result) is result


def test_juxt_single():
    f = juxt(lambda x: x * 3)
    assert list(f(2)) == [6]


def test_juxt_empty():
    f = juxt()
    assert list(f()) == []
    assert list(f(1, 2)) == []


def test_juxt_multiple_calls():
    f = juxt(lambda x: x + 1, lambda x: x * 2)
    assert list(f(1)) == [2, 2]
    assert list(f(2)) == [3, 4]
