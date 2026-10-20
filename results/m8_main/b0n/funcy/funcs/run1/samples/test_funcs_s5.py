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
# identity
# ---------------------------------------------------------------------------
def test_identity_returns_argument():
    obj = object()
    assert identity(obj) is obj
    assert identity(5) == 5
    assert identity(None) is None


# ---------------------------------------------------------------------------
# constantly
# ---------------------------------------------------------------------------
def test_constantly_ignores_arguments():
    c = constantly(42)
    assert c() == 42
    assert c(1, 2, 3) == 42
    assert c(a=1, b=2) == 42


def test_constantly_returns_same_object():
    sentinel = object()
    c = constantly(sentinel)
    assert c(1) is sentinel
    assert c("x", y="z") is sentinel


# ---------------------------------------------------------------------------
# caller
# ---------------------------------------------------------------------------
def test_caller_calls_with_given_args():
    f = caller(1, 2, x=3)
    calls = []

    def target(*a, **kw):
        calls.append((a, kw))
        return "result"

    assert f(target) == "result"
    assert calls == [((1, 2), {"x": 3})]


def test_caller_no_args():
    f = caller()
    assert f(lambda: 7) == 7


# ---------------------------------------------------------------------------
# func_partial
# ---------------------------------------------------------------------------
def test_func_partial_prepends_args():
    def f(*a, **kw):
        return a, kw

    p = func_partial(f, 1, 2, x=10)
    assert p(3, y=20) == ((1, 2, 3), {"x": 10, "y": 20})


def test_func_partial_kwargs_override():
    def f(*a, **kw):
        return a, kw

    p = func_partial(f, 1, x=10)
    assert p(x=99) == ((1,), {"x": 99})


def test_func_partial_is_real_function():
    def f(a, b):
        return a + b

    p = func_partial(f, 1)
    assert p(2) == 3
    # lambdas are plain functions
    assert callable(p)


# ---------------------------------------------------------------------------
# rpartial
# ---------------------------------------------------------------------------
def test_rpartial_appends_args():
    def f(*a, **kw):
        return a, kw

    p = rpartial(f, 2, 3, x=10)
    assert p(1, y=20) == ((1, 2, 3), {"x": 10, "y": 20})


def test_rpartial_kwargs_extend_and_override():
    def f(*a, **kw):
        return a, kw

    p = rpartial(f, x=10, y=5)
    assert p(x=99) == ((), {"x": 99, "y": 5})


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------
def test_curry_basic():
    def add3(a, b, c):
        return a + b + c

    curried = curry(add3)
    assert curried(1)(2)(3) == 6


def test_curry_with_n():
    def f(a, b):
        return a - b

    curried = curry(f, 2)
    assert curried(10)(3) == 7


def test_curry_n_leq_1_returns_func():
    def f(a):
        return a * 2

    assert curry(f, 1) is f
    assert curry(f, 0) is f


def test_curry_n_2():
    def f(a, b):
        return a * b

    c = curry(f, 2)
    assert c(3)(4) == 12


def test_curry_with_arg_count_from_spec():
    def f(a, b, c, d, e=None):
        return (a, b, c, d, e)

    curried = curry(f)
    # 5 max_n -> chain of 5 lambdas
    assert curried(1)(2)(3)(4)(5) == (1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# rcurry
# ---------------------------------------------------------------------------
def test_rcurry_basic():
    def sub(a, b, c):
        return a - b - c

    curried = rcurry(sub)
    # arguments passed right-to-left: rcurry(f)(c)(b)(a) == f(a, b, c)
    assert curried(3)(2)(1) == 1 - 2 - 3


def test_rcurry_n_leq_1():
    def f(a):
        return a

    assert rcurry(f, 1) is f
    assert rcurry(f, 0) is f


def test_rcurry_n_2():
    def f(a, b):
        return a - b

    c = rcurry(f, 2)
    # curried(x)(y) == f(y, x)
    assert c(3)(10) == 7


# ---------------------------------------------------------------------------
# autocurry
# ---------------------------------------------------------------------------
def test_autocurry_full_args_at_once():
    def add(a, b, c):
        return a + b + c

    ac = autocurry(add)
    assert ac(1, 2, 3) == 6


def test_autocurry_partial_chain():
    def add(a, b, c):
        return a + b + c

    ac = autocurry(add)
    assert ac(1)(2)(3) == 6
    assert ac(1, 2)(3) == 6
    assert ac(1)(2, 3) == 6


def test_autocurry_with_kwargs():
    def f(a, b, c):
        return a + b + c

    ac = autocurry(f)
    assert ac(1, c=3)(2) == 6
    assert ac(a=1)(b=2)(c=3) == 6


def test_autocurry_n():
    def f(a, b):
        return a + b

    ac = autocurry(f, 2)
    assert ac(1)(2) == 3
    assert ac(1, 2) == 3


def test_autocurry_returns_wraps_name():
    def named_func(a, b):
        """docstring here"""
        return a + b

    ac = autocurry(named_func)
    assert ac.__name__ == "named_func"
    assert ac.__doc__ == "docstring here"


# ---------------------------------------------------------------------------
# iffy
# ---------------------------------------------------------------------------
def test_iffy_basic():
    f = iffy(lambda x: x > 0, lambda x: x * 10)
    assert f(5) == 50
    assert f(-1) == -1  # default is identity


def test_iffy_custom_default():
    f = iffy(lambda x: x > 0, lambda x: x * 10, default=lambda x: 0)
    assert f(5) == 50
    assert f(-1) == 0


def test_iffy_noncallable_default():
    f = iffy(lambda x: x > 0, lambda x: x * 10, default="neg")
    assert f(5) == 50
    assert f(-1) == "neg"


def test_iffy_single_arg_treats_as_pred_with_bool():
    # iffy(pred) == iffy(bool, pred, identity)
    f = iffy(lambda x: x > 0)
    assert f(5) is True
    assert f(-1) is False


def test_iffy_makes_pred_and_func_from_values():
    f = iffy("yes", lambda x: x)
    assert f("yes") == "yes"


# ---------------------------------------------------------------------------
# compose
# ---------------------------------------------------------------------------
def test_compose_chain():
    f = compose(lambda x: x + 1, lambda x: x * 2)
    # compose(f, g)(x) == f(g(x))
    assert f(3) == 7


def test_compose_empty_is_identity():
    f = compose()
    assert f is identity
    assert f(5) == 5


def test_compose_passes_args_to_first_called():
    # outermost (last in reduce) gets args
    f = compose(str, lambda a, b: a + b)
    assert f(1, 2) == "3"


def test_compose_with_make_func_conversion():
    f = compose(len)
    assert f([1, 2, 3]) == 3


def test_compose_with_keyword_args():
    def g(a, b=0, **kw):
        return a + b + sum(kw.values())

    f = compose(lambda x: x * 2, g)
    assert f(1, b=2, c=3) == 12


# ---------------------------------------------------------------------------
# rcompose
# ---------------------------------------------------------------------------
def test_rcompose_chain():
    f = rcompose(lambda x: x * 2, lambda x: x + 1)
    # rcompose(f, g)(x) == g(f(x))
    assert f(3) == 7


def test_rcompose_empty_is_identity():
    f = rcompose()
    assert f is identity


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------
def test_complement():
    is_even = lambda x: x % 2 == 0
    is_odd = complement(is_even)
    assert is_odd(3) is True
    assert is_odd(4) is False


def test_complement_with_make_func():
    # using a mapping as pred
    pred = complement({1: "one"})
    assert pred(1) is False
    assert pred(2) is True


# ---------------------------------------------------------------------------
# ljuxt
# ---------------------------------------------------------------------------
def test_ljuxt_returns_list():
    f = ljuxt(lambda x: x + 1, lambda x: x * 2)
    result = f(3)
    assert result == [4, 6]
    assert isinstance(result, list)


def test_ljuxt_with_make_func():
    f = ljuxt(len, lambda x: x)
    assert f([1, 2]) == [2, [1, 2]]


def test_ljuxt_reusable():
    f = ljuxt(lambda x: x + 1)
    assert f(1) == [2]
    assert f(2) == [3]


def test_ljuxt_kwargs():
    f = ljuxt(lambda a, b: a + b, lambda a, b: a * b)
    assert f(2, b=3) == [5, 6]


# ---------------------------------------------------------------------------
# juxt
# ---------------------------------------------------------------------------
def test_juxt_returns_iterator():
    f = juxt(lambda x: x + 1, lambda x: x * 2)
    result = f(3)
    assert not isinstance(result, list)
    assert list(result) == [4, 6]


def test_juxt_with_make_func():
    f = juxt(len)
    assert list(f([1, 2, 3])) == [3]


def test_juxt_kwargs():
    f = juxt(lambda a, b: a - b, lambda a, b: a + b)
    assert list(f(5, 2)) == [3, 7]
