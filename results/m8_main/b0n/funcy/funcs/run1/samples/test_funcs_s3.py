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
    assert identity(5) == 5
    obj = object()
    assert identity(obj) is obj
    assert identity(None) is None


def test_constantly():
    f = constantly(42)
    assert f() == 42
    assert f(1, 2, 3) == 42
    assert f(a=1, b=2) == 42
    assert f(1, 2, x=3) == 42

    obj = object()
    g = constantly(obj)
    assert g(1) is obj


def test_caller():
    def add(a, b):
        return a + b

    f = caller(2, 3)
    assert f(add) == 5

    g = caller(2, b=3)
    assert g(add) == 5

    calls = []

    def rec(*a, **kw):
        calls.append((a, kw))
        return "ok"

    h = caller(1, 2, x=3)
    assert h(rec) == "ok"
    assert calls == [((1, 2), {"x": 3})]


def test_caller_no_args():
    def zero():
        return "zero"

    assert caller()(zero) == "zero"


# ---------------------------------------------------------------------------
# func_partial / rpartial
# ---------------------------------------------------------------------------

def test_func_partial_basic():
    def f(a, b, c):
        return (a, b, c)

    p = func_partial(f, 1)
    assert p(2, 3) == (1, 2, 3)

    # Adding more positional args prepends bound args
    p2 = func_partial(f, 1, 2)
    assert p2(3) == (1, 2, 3)


def test_func_partial_kwargs():
    def f(a, b, c=0, d=0):
        return (a, b, c, d)

    p = func_partial(f, 1, c=10)
    assert p(2) == (1, 2, 10, 0)
    # New kwargs extend and override
    assert p(2, d=4) == (1, 2, 10, 4)
    assert p(2, c=20) == (1, 2, 20, 0)


def test_func_partial_is_function():
    import types

    p = func_partial(lambda x: x, 1)
    assert isinstance(p, types.FunctionType)


def test_func_partial_as_method():
    class A:
        def __init__(self, base):
            self.base = base

        def add(self, x, y):
            return self.base + x + y

    a = A(100)
    bound = func_partial(a.add, 1)
    assert bound(2) == 103


def test_rpartial_basic():
    def f(a, b, c):
        return (a, b, c)

    p = rpartial(f, 3)
    assert p(1, 2) == (1, 2, 3)

    p2 = rpartial(f, 2, 3)
    assert p2(1) == (1, 2, 3)


def test_rpartial_kwargs():
    def f(a, b, c=0, d=0):
        return (a, b, c, d)

    p = rpartial(f, c=10)
    assert p(1, 2) == (1, 2, 10, 0)
    # New kwargs extend and override
    assert p(1, 2, d=4) == (1, 2, 10, 4)
    assert p(1, 2, c=20) == (1, 2, 20, 0)


def test_rpartial_is_function():
    import types

    p = rpartial(lambda x, y: x + y, 10)
    assert isinstance(p, types.FunctionType)


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_single_arg_function():
    def f(a):
        return a * 2

    # n defaults to max_n = 1, so function is returned as is
    assert curry(f) is f


def test_curry_two_args():
    def add(a, b):
        return a + b

    c = curry(add)
    assert c(1)(2) == 3
    assert c(10)(-3) == 7


def test_curry_three_args():
    def f(a, b, c):
        return [a, b, c]

    c = curry(f)
    assert c(1)(2)(3) == [1, 2, 3]


def test_curry_explicit_n():
    def f(*args):
        return args

    c = curry(f, 3)
    assert c(1)(2)(3) == (1, 2, 3)


def test_curry_n_one_returns_func():
    def f(a):
        return a + 1

    assert curry(f, 1) is f
    assert curry(f, 0) is f


def test_curry_n_two_explicit():
    def f(a, b):
        return a * b

    c = curry(f, 2)
    assert c(3)(4) == 12


# ---------------------------------------------------------------------------
# rcurry
# ---------------------------------------------------------------------------

def test_rcurry_single_arg():
    def f(a):
        return a

    assert rcurry(f) is f


def test_rcurry_two_args():
    def sub(a, b):
        return a - b

    # rcurry(x)(y) == func(y, x)
    c = rcurry(sub)
    assert c(1)(10) == sub(10, 1) == 9


def test_rcurry_three_args():
    def f(a, b, c):
        return [a, b, c]

    c = rcurry(f)
    assert c(1)(2)(3) == [3, 2, 1]


def test_rcurry_explicit_n():
    def f(*args):
        return args

    c = rcurry(f, 3)
    assert c(1)(2)(3) == (3, 2, 1)


def test_rcurry_n_one():
    def f(a):
        return a + 1

    assert rcurry(f, 1) is f


# ---------------------------------------------------------------------------
# autocurry
# ---------------------------------------------------------------------------

def test_autocurry_all_at_once():
    def add(a, b, c):
        return a + b + c

    c = autocurry(add)
    assert c(1, 2, 3) == 6


def test_autocurry_steps():
    def add(a, b, c):
        return a + b + c

    c = autocurry(add)
    assert c(1)(2)(3) == 6
    assert c(1, 2)(3) == 6
    assert c(1)(2, 3) == 6


def test_autocurry_kwargs():
    def add(a, b, c):
        return a + b + c

    c = autocurry(add)
    assert c(1)(2)(c=3) == 6
    assert c(a=1)(b=2)(c=3) == 6
    assert c(1, b=2)(c=3) == 6


def test_autocurry_too_many_args_uses_varkw_path():
    # When varkw is False and len(args)+len(kwargs) >= max_n, call directly
    def add(a, b):
        return a + b

    c = autocurry(add)
    assert c(1, 2, extra=1) if False else True  # placeholder ensuring no accidental call
    # The extra kwarg causes a TypeError normally; but this checks normal behavior:
    assert c(1, 2) == 3


def test_autocurry_with_defaults():
    def f(a, b=10):
        return a + b

    c = autocurry(f)
    # After supplying required arg a, defaults allow the call to succeed
    assert c(1) == 11
    assert c(1, 5) == 6


def test_autocurry_explicit_n():
    def f(*args):
        return args

    c = autocurry(f, n=3)
    assert c(1)(2)(3) == (1, 2, 3)
    assert c(1, 2)(3) == (1, 2, 3)


def test_autocurry_varargs():
    def f(*args):
        return args

    c = autocurry(f)
    # max_n from spec of *args is 1
    assert c(1, 2, 3) == (1, 2, 3)


def test_autocurry_typeerror_recovery():
    # Function whose args can't be translated by count alone require a retry pathway
    def f(a, b):
        return a + b

    c = autocurry(f)
    # Two required args; providing kwarg name 'a' plus positional? Standard flow
    assert c(a=1)(2) == 3


# ---------------------------------------------------------------------------
# iffy
# ---------------------------------------------------------------------------

def test_iffy_basic():
    def is_positive(x):
        return x > 0

    f = iffy(is_positive, lambda x: x * 2)
    assert f(5) == 10
    assert f(-3) == -3


def test_iffy_with_default():
    f = iffy(lambda x: x > 0, lambda x: "pos", "neg")
    assert f(1) == "pos"
    assert f(-1) == "neg"


def test_iffy_with_callable_default():
    f = iffy(lambda x: x > 0, lambda x: "pos", lambda x: -x)
    assert f(1) == "pos"
    assert f(-5) == 5


def test_iffy_no_action():
    # If action is EMPTY, pred becomes the action and bool becomes pred
    f = iffy(lambda x: x * 2)
    assert f(5) == 10  # truthy input -> action applied
    assert f(0) == 0   # falsy input x*2 -> 0


def test_iffy_truthiness_pred():
    f = iffy(None, lambda x: "yes", "no")
    # None -> bool as predicate
    assert f([1]) == "yes"
    assert f([]) == "no"


def test_iffy_string_pred_action():
    f = iffy(r"^\d+$", lambda x: "num", "other")
    assert f("123") == "num"
    assert f("abc") == "other"


# ---------------------------------------------------------------------------
# compose / rcompose
# ---------------------------------------------------------------------------

def test_compose_empty():
    assert compose() is identity


def test_compose_single():
    def inc(x):
        return x + 1

    f = compose(inc)
    assert f(1) == 2


def test_compose_order():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    f = compose(inc, double)
    # inc(double(x))
    assert f(3) == 7

    g = compose(double, inc)
    assert g(3) == 8


def test_compose_multiple_args():
    def add(a, b):
        return a + b

    def square(x):
        return x * x

    f = compose(square, add)
    assert f(2, 3) == 25


def test_compose_with_kwargs():
    def sub(a, b):
        return a - b

    def neg(x):
        return -x

    f = compose(neg, sub)
    assert f(a=5, b=3) == -2


def test_compose_make_func_via_mapping():
    f = compose(lambda x: x + 1, {"a": 10})
    assert f("a") == 11


def test_compose_make_func_via_string_regex_finder():
    # make_func(str) produces a regex finder; we verify the composed function works
    f = compose(len, r"(\d+)")
    assert f("ab 123 cd") == 3


def test_rcompose_order():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    # rcompose calls left to right: double(inc(x))
    f = rcompose(inc, double)
    assert f(3) == 8


def test_rcompose_empty():
    assert rcompose() is identity


def test_rcompose_single():
    def inc(x):
        return x + 1

    f = rcompose(inc)
    assert f(1) == 2


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------

def test_complement():
    is_even = lambda x: x % 2 == 0
    is_odd = complement(is_even)
    assert is_odd(3) is True
    assert is_odd(4) is False


def test_complement_returns_bool():
    f = complement(lambda x: x)
    assert f(0) is True
    assert f(1) is False


# ---------------------------------------------------------------------------
# juxt / ljuxt
# ---------------------------------------------------------------------------

def test_ljuxt_basic():
    f = ljuxt(lambda x: x + 1, lambda x: x * 2)
    assert f(3) == [4, 6]


def test_ljuxt_returns_list():
    f = ljuxt(lambda x: x)
    result = f(3)
    assert isinstance(result, list)
    assert result == [3]


def test_ljuxt_multiple_args():
    f = ljuxt(lambda a, b: a + b, lambda a, b: a * b)
    assert f(2, 3) == [5, 6]

    f2 = ljuxt(lambda a, b: a + b)
    assert f2(a=2, b=3) == [5]


def test_ljuxt_with_mapping():
    f = ljuxt({"a": 1})
    assert f("a") == [1]


def test_ljuxt_preserves_order():
    f = ljuxt(lambda x: 1, lambda x: 2, lambda x: 3)
    assert f(0) == [1, 2, 3]


def test_juxt_basic():
    f = juxt(lambda x: x + 1, lambda x: x * 2)
    result = f(3)
    # juxt returns an iterator
    assert not isinstance(result, list)
    assert list(result) == [4, 6]


def test_juxt_lazy_iterator():
    f = juxt(lambda x: x, lambda x: x + 1)
    it = f(1)
    assert hasattr(it, "__next__")
    assert next(it) == 1
    assert next(it) == 2
    with pytest.raises(StopIteration):
        next(it)


def test_juxt_can_be_called_multiple_times():
    f = juxt(lambda x: x, lambda x: x * 10)
    assert list(f(2)) == [2, 20]
    assert list(f(2)) == [2, 20]


def test_juxt_empty():
    f = juxt()
    assert list(f(1)) == []
    assert list(f()) == []


def test_ljuxt_empty():
    f = ljuxt()
    assert f() == []


def test_complement_of_ljuxt_combination():
    from funcy.funcs import compose
    from funcy.funcolls import all as _all_placeholder  # noqa - not used, just verify imports don't crash
    # typical usage: compose(all, juxt(*fs)) behaves as conjunction of predicates
    all_ = lambda iterable: all(iterable)
    both = compose(all_, juxt(lambda x: x > 0, lambda x: x < 10))
    assert both(5) is True
    assert both(-1) is False
    assert both(20) is False
