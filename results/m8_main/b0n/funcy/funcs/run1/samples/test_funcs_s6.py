import pytest
from funcy.funcs import (
    identity, constantly, caller,
    reduce, partial,
    func_partial, rpartial,
    curry, rcurry, autocurry,
    iffy,
    compose, rcompose, complement,
    ljuxt, juxt,
)


# ---------- identity / constantly / caller ----------

def test_identity():
    assert identity(5) == 5
    obj = object()
    assert identity(obj) is obj
    assert identity(None) is None


def test_constantly():
    f = constantly(42)
    assert f() == 42
    assert f(1, 2, 3) == 42
    assert f(x=1, y=2) == 42
    assert f(1, key="val") == 42

    sentinel = object()
    g = constantly(sentinel)
    assert g() is sentinel


def test_caller_noargs():
    f = caller()
    assert f(lambda: 7) == 7


def test_caller_args_kwargs():
    f = caller(1, 2, x=3)
    seen = {}

    def target(a, b, x):
        seen["a"] = a
        seen["b"] = b
        seen["x"] = x
        return a + b + x

    assert f(target) == 6
    assert seen == {"a": 1, "b": 2, "x": 3}


# ---------- func_partial / rpartial ----------

def test_func_partial_applies_leading_args():
    def f(a, b, c):
        return a, b, c

    p = func_partial(f, 1, 2)
    assert p(3) == (1, 2, 3)


def test_func_partial_returns_real_function():
    p = func_partial(lambda a, b: a - b, 10)
    assert type(p).__name__ == "function"
    assert p(4) == 6


def test_func_partial_kwargs_merge():
    def f(a, b, c=0, d=0):
        return a, b, c, d

    p = func_partial(f, 1, c=3)
    # new kw overrides existing
    assert p(b=2, d=4) == (1, 2, 3, 4)
    assert p(b=2, c=9) == (1, 2, 9, 0)


def test_func_partial_as_method():
    class C:
        def __init__(self, base):
            self.base = base
            self.add = func_partial(self._add, 10)

        def _add(self, x, y):
            return self.base + x + y

    c = C(100)
    assert c.add(5) == 115


def test_rpartial_applies_trailing_args():
    def f(a, b, c):
        return a, b, c

    p = rpartial(f, 2, 3)
    assert p(1) == (1, 2, 3)


def test_rpartial_kwargs_merge():
    def f(a, b, c=0):
        return a, b, c

    p = rpartial(f, 2, c=3)
    assert p(1) == (1, 2, 3)
    # new kw overrides
    assert p(1, c=9) == (1, 2, 9)


def test_rpartial_returns_real_function():
    p = rpartial(lambda a, b: a - b, 4)
    assert type(p).__name__ == "function"
    assert p(10) == 6


# ---------- curry / rcurry ----------

def test_curry_auto_arity():
    def f(a, b, c):
        return a + b + c

    c = curry(f)
    assert c(1)(2)(3) == 6


def test_curry_explicit_n():
    def f(a, b):
        return a - b

    c = curry(f, 2)
    assert c(10)(3) == 7


def test_curry_n_leq_one_returns_func():
    def f(a):
        return a

    assert curry(f, 1) is f
    assert curry(f, 0) is f


def test_rcurry_auto_arity():
    def f(a, b, c):
        return (a, b, c)

    c = rcurry(f)
    # args passed right-to-left, so final call order: c(3)(2)(1) -> f(1, 2, 3)
    assert c(3)(2)(1) == (1, 2, 3)


def test_rcurry_explicit_n():
    def f(a, b):
        return a - b

    c = rcurry(f, 2)
    # first arg becomes second param: c(3)(10) -> f(10, 3)
    assert c(3)(10) == 7


def test_rcurry_n_leq_one_returns_func():
    def f(a):
        return a

    assert rcurry(f, 1) is f


def test_curry_three_plus_steps():
    def f(a, b, c, d):
        return a * 1000 + b * 100 + c * 10 + d

    c = curry(f)
    assert c(1)(2)(3)(4) == 1234


def test_rcurry_three_plus_steps():
    def f(a, b, c, d):
        return (a, b, c, d)

    c = rcurry(f)
    # passing d, c, b, a  -> f(a, b, c, d)
    assert c(4)(3)(2)(1) == (1, 2, 3, 4)


# ---------- autocurry ----------

def test_autocurry_returns_partial_until_enough_args():
    def f(a, b, c):
        return a + b + c

    ac = autocurry(f)
    p = ac(1)
    # partial application returns a new autocurried function
    assert callable(p)
    assert p(2)(3) == 6


def test_autocurry_all_at_once():
    def f(a, b, c):
        return a + b + c

    ac = autocurry(f)
    assert ac(1, 2, 3) == 6


def test_autocurry_with_kwargs():
    def f(a, b, c):
        return a + b + c

    ac = autocurry(f)
    assert ac(1)(b=2)(c=3) == 6


def test_autocurry_mixed_positional_and_kwargs():
    def f(a, b, c):
        return (a, b, c)

    ac = autocurry(f)
    assert ac(1, b=2)(c=3) == (1, 2, 3)


def test_autocurry_explicit_n():
    def f(*args):
        return sum(args)

    ac = autocurry(f, 3)
    assert ac(1)(2)(3) == 6


def test_autocurry_explicit_n_all_at_once():
    def f(*args):
        return sum(args)

    ac = autocurry(f, 2)
    assert ac(5, 7) == 12


def test_autocurry_preserves_name():
    def myfunc(a, b):
        return a + b

    ac = autocurry(myfunc)
    assert ac.__name__ == "myfunc"


def test_autocurry_handles_defaults():
    def f(a, b=10):
        return a + b

    ac = autocurry(f)
    # required args already satisfied, additional call executes
    assert ac(5) == 15
    assert ac(5, 6) == 11


# ---------- iffy ----------

def test_iffy_basic():
    f = iffy(lambda x: x > 0, lambda x: x * 2)
    assert f(5) == 10
    assert f(-3) == -3  # default identity


def test_iffy_with_default_value():
    f = iffy(lambda x: x > 0, lambda x: x * 2, default=0)
    assert f(5) == 10
    assert f(-3) == 0


def test_iffy_with_default_callable():
    f = iffy(lambda x: x > 0, lambda x: x * 2, default=lambda x: -x)
    assert f(5) == 10
    assert f(-3) == 3


def test_iffy_swapped_when_action_missing():
    # iffy(pred) -> iffy(bool, pred, identity)
    f = iffy(lambda x: "truthy" if x else "falsy")
    assert f(1) == "truthy"
    assert f(0) == "falsy"


def test_iffy_string_pred():
    f = iffy("a", lambda x: "has-a", default="no-a")
    assert f("cat") == "has-a"
    assert f("dog") == "no-a"


# ---------- compose / rcompose ----------

def test_compose_empty_is_identity():
    f = compose()
    assert f is identity
    assert f(5) == 5


def test_compose_order():
    f = compose(lambda x: x + 1, lambda x: x * 2)
    # f(x) = (x*2) + 1
    assert f(3) == 7


def test_compose_single():
    f = compose(lambda x: x + 1)
    assert f(3) == 4


def test_compose_multiple_args_to_innermost():
    f = compose(lambda x: x * 10, lambda a, b: a + b)
    assert f(2, 3) == 50


def test_compose_three():
    f = compose(lambda x: x + 1, lambda x: x * 2, lambda x: x - 3)
    # x=5: (5-3)=2 -> *2=4 -> +1=5
    assert f(5) == 5


def test_rcompose_order():
    f = rcompose(lambda x: x + 1, lambda x: x * 2)
    # left to right: (x+1) then *2
    assert f(3) == 8


def test_rcompose_empty():
    f = rcompose()
    assert f is identity


def test_compose_with_dict_as_func():
    # make_func turns a mapping into __getitem__
    f = compose(lambda x: x + 1, {"a": 1})
    assert f("a") == 2


# ---------- complement ----------

def test_complement_basic():
    is_even = complement(lambda x: x % 2 == 0)
    assert is_even(2) is False
    assert is_even(3) is True


def test_complement_returns_bool():
    c = complement(lambda x: x > 0)
    result = c(-1)
    assert result is True


def test_complement_with_string():
    c = complement("a")
    assert c("cat") is False
    assert c("dog") is True


# ---------- ljuxt / juxt ----------

def test_ljuxt_returns_list():
    f = ljuxt(lambda x: x + 1, lambda x: x * 2)
    result = f(3)
    assert isinstance(result, list)
    assert result == [4, 6]


def test_juxt_returns_iterator():
    f = juxt(lambda x: x + 1, lambda x: x * 2)
    result = f(3)
    assert list(result) == [4, 6]


def test_juxt_reusable():
    f = juxt(lambda x: x + 1, lambda x: x * 2)
    assert list(f(3)) == [4, 6]
    assert list(f(5)) == [6, 10]


def test_ljuxt_reusable():
    f = ljuxt(lambda x: x + 1, lambda x: x * 2)
    assert f(3) == [4, 6]
    assert f(5) == [6, 10]


def test_ljuxt_empty():
    f = ljuxt()
    assert f(1) == []


def test_juxt_empty():
    f = juxt()
    assert list(f(1)) == []


def test_ljuxt_multiple_args():
    f = ljuxt(lambda a, b: a + b, lambda a, b: a * b)
    assert f(2, 3) == [5, 6]


def test_juxt_with_string_func():
    f = juxt("a")
    assert list(f("cat")) == ["a"]


def test_ljuxt_with_mapping():
    f = ljuxt({"x": 1})
    assert f("x") == [1]


# ---------- reexports ----------

def test_reexported_reduce_and_partial():
    assert reduce(lambda a, b: a + b, [1, 2, 3]) == 6
    p = partial(lambda a, b: a - b, 10)
    assert p(4) == 6
