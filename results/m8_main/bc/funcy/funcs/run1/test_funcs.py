import pytest

from funcy.funcs import (
    identity, constantly, caller,
    func_partial, rpartial,
    curry, rcurry, autocurry,
    iffy,
    compose, rcompose, complement,
    ljuxt, juxt,
    partial, reduce,
)


# ---------- identity / constantly / caller ----------

def test_identity_returns_argument():
    obj = object()
    assert identity(obj) is obj
    assert identity(42) == 42
    assert identity(None) is None


def test_constantly_returns_constant():
    f = constantly(7)
    assert f() == 7
    assert f(1, 2, 3) == 7
    assert f(a=1, b=2) == 7


def test_constantly_with_mutable():
    sentinel = []
    f = constantly(sentinel)
    assert f() is sentinel


def test_caller():
    def add(a, b):
        return a + b

    f = caller(2, 3)
    assert f(add) == 5

    f = caller(b=10, a=1)
    assert f(add) == 11


# ---------- func_partial / rpartial ----------

def test_func_partial_basic():
    f = func_partial(lambda a, b, c: (a, b, c), 1, 2)
    assert f(3) == (1, 2, 3)


def test_func_partial_kwargs_extension():
    def g(a, b=0, c=0):
        return (a, b, c)

    f = func_partial(g, 1, b=2)
    assert f() == (1, 2, 0)
    # new kwargs override old
    assert f(b=5) == (1, 5, 0)
    assert f(c=9) == (1, 2, 9)


def test_func_partial_returns_function():
    # Must be a real function, usable as method
    class C:
        def __init__(self):
            self.x = 10
        m = func_partial(lambda self, y: self.x + y)

    assert C().m(5) == 15


def test_func_partial_extends_old_kwargs_with_new():
    def g(**kw):
        return kw

    f = func_partial(g, a=1, b=2)
    assert f(c=3) == {'a': 1, 'b': 2, 'c': 3}
    assert f(a=10) == {'a': 10, 'b': 2}


def test_rpartial_basic():
    f = rpartial(lambda a, b, c: (a, b, c), 2, 3)
    assert f(1) == (1, 2, 3)


def test_rpartial_kwargs():
    def g(a, b=0, c=0):
        return (a, b, c)

    f = rpartial(g, 2, 3)
    assert f(1) == (1, 2, 3)
    # Positional args 2, 3 are appended after the new positional 1, so they
    # bind b and c respectively. Supplying b or c as keyword then conflicts
    # with those positional bindings and raises TypeError.
    with pytest.raises(TypeError):
        f(1, b=20)
    with pytest.raises(TypeError):
        f(1, c=30)


def test_rpartial_extends_kwargs():
    def g(**kw):
        return kw

    f = rpartial(g, a=1)
    assert f(b=2) == {'a': 1, 'b': 2}
    assert f(a=10) == {'a': 10}


# ---------- curry / rcurry ----------

def test_curry_explicit_n():
    f = curry(lambda a, b, c: a + b + c, 3)
    assert callable(f)
    g = f(1)
    h = g(2)
    assert h(3) == 6


def test_curry_n_le_1_returns_func():
    def f(a):
        return a

    assert curry(f, 1) is f
    assert curry(f, 0) is f


def test_curry_n_2():
    f = curry(lambda a, b: a - b, 2)
    assert f(10)(3) == 7


def test_curry_inference():
    def add3(a, b, c):
        return a + b + c

    f = curry(add3)
    assert f(1)(2)(3) == 6


def test_rcurry_explicit_n():
    f = rcurry(lambda a, b, c: (a, b, c), 3)
    # x is rightmost, then next, etc.
    g = f(3)  # rightmost arg
    h = g(2)  # middle
    assert h(1) == (1, 2, 3)


def test_rcurry_n_2():
    f = rcurry(lambda a, b: (a, b), 2)
    # x is b, y is a
    assert f(2)(1) == (1, 2)


def test_rcurry_n_le_1():
    def f(a):
        return a

    assert rcurry(f, 1) is f
    assert rcurry(f, 0) is f


def test_rcurry_inference():
    def sub(a, b):
        return a - b

    f = rcurry(sub)
    # f(3) -> returns function taking y; func(y, 3) = y - 3
    assert f(3)(10) == 7


# ---------- autocurry ----------

def test_autocurry_basic():
    @autocurry
    def add(a, b, c):
        return a + b + c

    assert add(1)(2)(3) == 6
    assert add(1, 2)(3) == 6
    assert add(1)(2, 3) == 6
    assert add(1, 2, 3) == 6
    assert add(a=1)(b=2)(c=3) == 6
    assert add(1, b=2)(c=3) == 6


def test_autocurry_explicit_n():
    def add(a, b):
        return a + b

    ac = autocurry(add, n=2)
    assert ac(1)(2) == 3


def test_autocurry_extra_args():
    @autocurry
    def add(a, b):
        return a + b

    # Once enough args, calls func directly which will fail on too many
    result = add(1, 2)
    assert result == 3


def test_autocurry_preserves_wraps():
    @autocurry
    def myfunc(a, b):
        "docstring"
        return a + b

    assert myfunc.__name__ == 'myfunc'
    assert myfunc.__doc__ == 'docstring'


def test_autocurry_varkw_behavior():
    @autocurry
    def f(a, b, **kw):
        return (a, b, kw)

    # With **kwargs absent from positional, need all positional to call
    r = f(1)(2)
    assert r == (1, 2, {})


def test_autocurry_calls_when_required_args_satisfied():
    # Reaches autocurry's ``req_names`` branch (line 84) and the underlying
    # function is called successfully (lines 85-86).
    def f(a, b=1, c=2):
        return (a, b, c)

    ac = autocurry(f)
    # ``a`` is the only required parameter.  A single positional arg satisfies
    # req_n (=1) even though max_n is 3, so the function is invoked right away
    # rather than returning a partial.
    assert ac(1) == (1, 1, 2)
    # Same happens when the required slot is filled via keyword.
    assert ac(1, b=5) == (1, 5, 2)


def test_autocurry_required_branch_type_error_recurries():
    # Reaches the ``req_names`` branch (line 84) but the underlying call raises
    # TypeError, so the except branch (lines 87-88) returns another partial.
    def f(a, b, c, d=1):
        return (a, b, c, d)

    ac = autocurry(f)
    # Args 1, 2 plus keyword a=3 pass the req_names count (2+1 >= 3), but the
    # underlying call ``f(1, 2, a=3)`` raises TypeError ("multiple values for
    # argument 'a'"), which is caught and turned into a new curried partial.
    result = ac(1, 2, a=3)
    assert callable(result)
    # It is *not* the final result tuple.
    assert not isinstance(result, tuple)
    # Invoking without the duplicated keyword still keeps the duplicate and
    # remains a partial (never resolves to a tuple).
    further = result()
    assert callable(further)
    assert not isinstance(further, tuple)


# ---------- iffy ----------

def test_iffy_basic():
    f = iffy(lambda x: x > 0, lambda x: x * 2)
    assert f(3) == 6
    assert f(-3) == -3  # default identity


def test_iffy_default_non_callable():
    f = iffy(lambda x: x > 0, lambda x: x * 2, default=99)
    assert f(3) == 6
    assert f(-3) == 99


def test_iffy_default_callable():
    f = iffy(lambda x: x > 0, lambda x: x * 2, default=lambda x: -x)
    assert f(-3) == 3


def test_iffy_single_arg_uses_bool():
    # iffy(pred) with no action -> iffy(bool, pred)
    # pred coerced to a function; if pred is truthy -> default identity
    f = iffy(0)  # constant 0 (falsy) as "predicate" via make_pred
    # iffy(bool, 0, identity): action = make_func(0) -> itemgetter(0)
    # So f(v) = v[0] if bool(v) else v
    assert f([10, 20]) == 10
    assert f([]) == []


def test_iffy_pred_true():
    # iffy(True) -> iffy(bool, True, identity)
    # action from True -> itemgetter(1)
    f = iffy(True)
    assert f([0, 'x']) == 'x'


# ---------- compose / rcompose ----------

def test_compose_empty_is_identity():
    assert compose() is identity
    assert compose()(5) == 5


def test_compose_functions():
    f = compose(lambda x: x + 1, lambda x: x * 2)
    assert f(3) == 7  # (3*2)+1


def test_compose_single():
    def g(x):
        return x * 3
    f = compose(g)
    assert f(2) == 6


def test_compose_with_non_callable():
    # make_func supports strings via re_finder, which needs a string input.
    f = compose('a')
    assert f('abc') == 'a'
    assert f('xyz') is None


def test_compose_with_int():
    # ints are supported by make_func -> itemgetter
    f = compose(0)
    assert f([5, 6, 7]) == 5


def test_compose_kwargs_passed_to_inner():
    f = compose(lambda x: x + 1, lambda *args, **kw: args[0] + kw['y'])
    assert f(1, y=10) == 12


def test_rcompose_functions():
    f = rcompose(lambda x: x + 1, lambda x: x * 2)
    assert f(3) == 8  # (3+1)*2


def test_rcompose_empty():
    assert rcompose() is identity


# ---------- complement ----------

def test_complement():
    is_positive = lambda x: x > 0
    is_not_positive = complement(is_positive)
    assert is_not_positive(3) is False
    assert is_not_positive(-1) is True


def test_complement_returns_bool():
    even = complement(lambda x: x % 2)
    assert even(2) is True
    assert even(3) is False


def test_complement_with_non_callable():
    # make_func can produce pred from dict/set
    comp = complement({1, 2, 3})
    assert comp(1) is False
    assert comp(5) is True


# ---------- ljuxt / juxt ----------

def test_ljuxt_basic():
    f = ljuxt(lambda x: x + 1, lambda x: x * 2)
    assert f(3) == [4, 6]


def test_ljuxt_returns_list():
    f = ljuxt(lambda x: x)
    r = f(1)
    assert isinstance(r, list)


def test_juxt_basic():
    f = juxt(lambda x: x + 1, lambda x: x * 2)
    result = list(f(3))
    assert result == [4, 6]


def test_juxt_returns_iterator():
    f = juxt(lambda x: x)
    r = f(1)
    assert not isinstance(r, list)
    assert iter(r) is r


def test_juxt_repeated_calls():
    # The implementation makes a list internally so repeated calls work
    f = juxt(lambda x: x + 1, lambda x: x * 2)
    assert list(f(3)) == [4, 6]
    assert list(f(3)) == [4, 6]


def test_ljuxt_repeated_calls():
    f = ljuxt(lambda x: x + 1, lambda x: x * 2)
    assert f(3) == [4, 6]
    assert f(3) == [4, 6]


def test_ljuxt_with_none():
    f = ljuxt(None, lambda x: x + 1)
    assert f(3) == [3, 4]


def test_juxt_with_string():
    # strings are compiled via re_finder, so they need string input.
    f = juxt('a', 'b')
    assert list(f('xyzab')) == ['a', 'b']
    # no match yields None
    assert list(f('xyz')) == [None, None]


# ---------- reexports ----------

def test_reexports():
    assert partial is not None
    assert reduce is not None
    assert partial(lambda a, b: a + b, 1)(2) == 3
    assert reduce(lambda a, b: a + b, [1, 2, 3]) == 6


# ---------- misc error handling ----------

def test_make_func_error_propagates():
    # compose/make_func should raise on unsupported types
    with pytest.raises(TypeError):
        compose(1.5)


def test_iffy_unsupported_type():
    with pytest.raises(TypeError):
        iffy(1.5)
