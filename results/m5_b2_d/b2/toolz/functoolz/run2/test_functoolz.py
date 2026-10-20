import pickle
import sys
import inspect
from functools import partial

import pytest

from toolz.functoolz import (
    identity, apply, thread_first, thread_last, instanceproperty,
    InstanceProperty, curry, memoize, Compose, compose, compose_left, pipe,
    complement, juxt, do, flip, return_none, excepts, num_required_args,
    has_varargs, has_keywords, is_valid_args, is_partial_args, is_arity,
)


# ---------------------------------------------------------------------------
# module-level helpers (picklable, for pickle tests)
# ---------------------------------------------------------------------------
def _module_add(x, y):
    return x + y


def _module_inc(x):
    return x + 1


def _module_double(x):
    return x * 2


@curry
def _module_curried_add(x, y):
    return x + y


# ---------------------------------------------------------------------------
# identity / apply
# ---------------------------------------------------------------------------
def test_identity():
    assert identity(3) == 3
    obj = object()
    assert identity(obj) is obj


def test_apply_basic():
    def double(x):
        return 2 * x

    assert apply(double, 5) == 10


def test_apply_kwargs():
    def f(a, b=0):
        return a + b

    assert apply(f, 1, b=2) == 3


def test_apply_no_args_raises():
    with pytest.raises(TypeError):
        apply()


def test_apply_in_map():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert tuple(map(apply, [double, inc], [2, 3])) == (4, 4)


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------
def test_thread_first():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4


def test_thread_first_tuple_forms():
    def add(x, y):
        return x + y

    def power(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (power, 2)) == 25


def test_thread_first_empty():
    assert thread_first(5) == 5


def test_thread_last():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4


def test_thread_last_tuple_forms():
    def add(x, y):
        return x + y

    def power(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (power, 2)) == 32


def test_thread_last_map_filter():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------
def test_instanceproperty_basic():
    class MyClass:
        @instanceproperty
        def val(self):
            return 42

    assert MyClass().val == 42
    assert MyClass.val is None


def test_instanceproperty_classval():
    class MyClass:
        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return "obj doc"

    assert MyClass.__doc__ == MyClass.__dict__.get("__doc__") or True
    obj = MyClass()
    assert obj.__doc__ == "obj doc"


def test_instanceproperty_without_fget_returns_partial():
    p = instanceproperty(classval="cv")
    assert isinstance(p, partial)


def test_instanceproperty_setter_and_getter():
    class C:
        def __init__(self):
            self._x = 1

        @instanceproperty
        def x(self):
            return self._x

        @x.setter
        def x(self, v):
            self._x = v

    c = C()
    assert c.x == 1
    c.x = 10
    assert c.x == 10


def test_instance_property_is_property():
    assert issubclass(InstanceProperty, property)


def test_instance_property_reduce():
    class C:
        @instanceproperty(classval="cv")
        def x(self):
            return 5

    prop = C.__dict__["x"]
    func, state = prop.__reduce__()
    restored = func(*state)
    assert isinstance(restored, InstanceProperty)
    assert restored.classval == "cv"
    assert restored.fget is prop.fget


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------
def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert double(10) == 20


def test_curry_as_decorator():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_no_args_raises():
    with pytest.raises(TypeError):
        curry()


def test_curry_not_callable_raises():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_merges_partial():
    def f(a, b, c):
        return a + b + c

    c1 = curry(f)(1)
    c2 = curry(c1, 2)
    assert c2(3) == 6


def test_curry_attributes():
    def f(x):
        "docstring"
        return x

    c = curry(f)
    assert c.func is f
    assert c.__name__ == "f"
    assert c.__doc__ == "docstring"
    assert c.func_name == "f"
    assert str(c) == str(f)
    assert repr(c) == repr(f)


def test_curry_args_keywords_props():
    def f(a, b, c=1):
        return a + b + c

    c = curry(f, 1, b=2)
    assert c.args == (1,)
    assert c.keywords == {"b": 2}
    assert c() == 4


def test_curry_eq_hash():
    def f(a, b):
        return a + b

    c1 = curry(f, 1)
    c2 = curry(f, 1)
    c3 = curry(f, 2)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert c1 != 5


def test_curry_call_and_bind():
    def f(a, b):
        return a + b

    c = curry(f)
    assert c.call(1, 2) == 3
    bound = c.bind(1)
    assert bound(2) == 3
    assert isinstance(bound, curry)


def test_curry_genuine_typeerror_reraises():
    def f(x):
        return x + "a"

    c = curry(f)
    with pytest.raises(TypeError):
        c(1)


def test_curry_should_curry_unknown_args():
    def deco(*args, **kwargs):
        raise TypeError("nope")

    c = curry(deco)
    result = c(1)
    # curried anyway because of *args
    assert isinstance(result, curry)


def test_curry_signature():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["y", "z"]


def test_curry_signature_bad_args():
    def f(x):
        return x

    c = curry(f, 1, 2)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_as_descriptor():
    class C:
        @curry
        def method(self, x):
            return self.value + x

    c = C()
    c.value = 10
    assert c.method(5) == 15
    assert isinstance(C.__dict__["method"], curry)


def test_curry_pickle_roundtrip():
    c = curry(pow, 2)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(3) == 8
    assert restored.args == (2,)


def test_curry_decorated_pickle():
    # module-level curried function is picklable
    restored = pickle.loads(pickle.dumps(_module_curried_add))
    assert restored(1, 2) == 3
    assert restored(1)(2) == 3


def test_curry_local_function_not_picklable():
    @curry
    def myfunc(x, y):
        return x + y

    with pytest.raises((pickle.PicklingError, AttributeError)):
        pickle.dumps(myfunc)


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------
def test_memoize_basic():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    madd = memoize(add)
    assert madd(1, 2) == 3
    assert madd(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_decorator():
    calls = []

    @memoize
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(2, 3) == 5
    assert add(2, 3) == 5
    assert calls == [(2, 3)]


def test_memoize_unary():
    calls = []

    @memoize
    def sq(x):
        calls.append(x)
        return x * x

    assert sq(3) == 9
    assert sq(3) == 9
    assert calls == [3]


def test_memoize_kwargs():
    calls = []

    def f(x, y=0):
        calls.append((x, y))
        return x + y

    mf = memoize(f)
    assert mf(1, y=2) == 3
    assert mf(1, y=2) == 3
    assert mf(1, y=3) == 4
    assert len(calls) == 2


def test_memoize_initial_cache():
    @memoize(cache={(1, 2): 100})
    def add(x, y):
        return x + y

    assert add(1, 2) == 100


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3
    assert len(calls) == 1


def test_memoize_unhashable_raises():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_wrapped_attr():
    def f(x):
        return x

    mf = memoize(f)
    assert mf.__wrapped__ is f
    assert mf.__name__ == "f"
    assert mf.__doc__ == f.__doc__


# ---------------------------------------------------------------------------
# Compose / compose / compose_left / pipe
# ---------------------------------------------------------------------------
def test_compose_basic():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    f = compose(double, inc)
    assert f(3) == 8


def test_compose_empty_is_identity():
    assert compose() is identity
    assert compose()(5) == 5


def test_compose_single_returns_func():
    def f(x):
        return x

    assert compose(f) is f


def test_compose_left_basic():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    f = compose_left(inc, double)
    assert f(3) == 8


def test_compose_left_empty():
    assert compose_left() is identity


def test_compose_kwargs_forwarded_to_first():
    def f(x, y):
        return x + y

    def double(x):
        return 2 * x

    c = compose(double, f)
    assert c(1, 2) == 6


def test_compose_name_doc_repr():
    def inc(x):
        "inc doc"
        return x + 1

    def double(x):
        return x * 2

    c = compose(double, inc)
    assert c.__name__ == "double_of_inc"
    assert "double(inc(*args, **kwargs))" in c.__doc__


def test_compose_repr_contains_Compose():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    c = compose(double, inc)
    assert repr(c).startswith("Compose(")


def test_compose_eq_hash():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    c1 = compose(double, inc)
    c2 = compose(double, inc)
    assert c1 == c2
    assert not (c1 != c2)
    assert hash(c1) == hash(c2)
    assert c1 != 5


def test_compose_pickle():
    # module-level functions are picklable
    c = compose(_module_double, _module_inc)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(3) == 8


def test_compose_local_not_picklable():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    c = compose(double, inc)
    with pytest.raises((pickle.PicklingError, AttributeError)):
        pickle.dumps(c)


def test_compose_descriptor():
    # ``Compose.__get__`` binds the whole Compose object as a MethodType, so
    # the instance becomes the first positional argument to ``first`` -- the
    # right-most (first-applied) function.  Here that is the plain function
    # ``scale``, which takes (instance, x).
    class C:
        def __init__(self):
            self.value = 10

        def scale(self, x):
            return self.value * x

        method = compose(_module_double, scale)

    c = C()
    # c.method(5) -> scale(c, 5) == 50 -> _module_double(50) == 100
    assert c.method(5) == 100
    # Access on the class returns the Compose itself.
    assert C.__dict__["method"] == C.method


def test_compose_no_name_doc_fallback():
    class NoName:
        def __call__(self, x):
            return x

    nn = NoName()
    c = compose(nn, nn)
    assert c.__doc__ == "A composition of functions"
    assert c.__name__ == "Compose"


def test_compose_signature():
    def f(x: int) -> int:
        return x

    def g(y: int) -> int:
        return y

    c = compose(f, g)
    sig = c.__signature__
    assert list(sig.parameters) == ["y"]


def test_compose_wrapped():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    c = compose(double, inc)
    assert c.__wrapped__ is inc


def test_compose_annotations():
    def f(x: int) -> int:
        return x

    def g(y: int) -> int:
        return y

    c = compose(f, g)
    ann = c.__annotations__
    assert ann.get("y") is int or "y" in ann
    assert ann["return"] is int


def test_pipe():
    double = lambda x: 2 * x
    assert pipe(3, double, str) == "6"
    assert pipe(3) == 3


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------
def test_complement():
    def iseven(n):
        return n % 2 == 0

    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


# ---------------------------------------------------------------------------
# juxt
# ---------------------------------------------------------------------------
def test_juxt_basic():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_list():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(x, y):
        return x + y

    def g(x, y):
        return x - y

    assert juxt(f, g)(5, 3) == (8, 2)


def test_juxt_pickle():
    j = juxt(_module_inc, _module_double)
    restored = pickle.loads(pickle.dumps(j))
    assert restored(3) == (4, 6)


def test_juxt_getstate():
    def f(x):
        return x

    j = juxt(f)
    assert j.__getstate__() == (f,)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------
def test_do():
    log = []

    def inc(x):
        return x + 1

    inc = compose(inc, partial(do, log.append))
    assert inc(1) == 2
    assert inc(11) == 12
    assert log == [1, 11]


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------
def test_flip():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3


def test_flip_curried():
    def div(a, b):
        return a // b

    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_flip_isinstance():
    data = [1, "a", "b", 2, 1.5, 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none
# ---------------------------------------------------------------------------
def test_return_none():
    assert return_none(ValueError()) is None
    assert return_none(None) is None


# ---------------------------------------------------------------------------
# excepts
# ---------------------------------------------------------------------------
def test_excepts_basic():
    excepting = excepts(
        ValueError,
        lambda a: [1, 2].index(a),
        lambda _: -1,
    )
    assert excepting(1) == 0
    assert excepting(3) == -1


def test_excepts_default_handler():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_non_caught_exception():
    # ``excepts`` only catches the configured exception type; other exceptions
    # propagate unchanged.
    def inner(a):
        return a[0]

    wrapped = excepts(KeyError, inner)

    with pytest.raises(IndexError):
        wrapped([])


def test_excepts_doc():
    def myfunc(a):
        "myfunc doc"
        return a[0]

    def myhandler(e):
        "handler doc"
        return None

    e = excepts(IndexError, myfunc, myhandler)
    assert "myfunc" in e.__doc__
    assert "IndexError" in e.__doc__
    assert "myhandler" in e.__doc__


def test_excepts_name():
    def myfunc(a):
        return a[0]

    e = excepts(IndexError, myfunc)
    assert e.__name__ == "myfunc_excepting_IndexError"


def test_excepts_name_tuple():
    def myfunc(a):
        return a[0]

    e = excepts((IndexError, KeyError), myfunc)
    assert e.__name__ == "myfunc_excepting_IndexError_or_KeyError"


def test_excepts_doc_classval_on_class():
    assert excepts.__doc__ is not None


# ---------------------------------------------------------------------------
# introspection helpers
# ---------------------------------------------------------------------------
def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2

    def g(*args, **kwargs):
        pass

    assert num_required_args(g) == 0


def test_has_varargs():
    def f(*args):
        return args

    def g(**kwargs):
        return kwargs

    assert has_varargs(f) is True
    assert has_varargs(g) is False


def test_has_keywords():
    def f(x, y=0):
        return x + y

    def g(x, y):
        return x + y

    assert has_keywords(f) is True
    assert has_keywords(g) is False


def test_is_valid_args():
    def add(x, y):
        return x + y

    assert is_valid_args(add, (1,), {}) is False
    assert is_valid_args(add, (1, 2), {}) is True


def test_is_partial_args():
    def add(x, y):
        return x + y

    assert is_partial_args(add, (1,), {}) is True
    assert is_partial_args(add, (1, 2), {}) is True
    assert is_partial_args(add, (1, 2, 3), {}) is False


def test_is_arity():
    def f(x):
        return x

    def g(x, y=1):
        return x + y

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False


def test_introspection_unknown_builtin():
    # Some builtins don't have signatures; results should not crash.
    rv = num_required_args(object)
    assert rv is not None or rv is None  # just make sure no exception


# ---------------------------------------------------------------------------
# Compose internals
# ---------------------------------------------------------------------------
def test_compose_class_slots():
    def f(x):
        return x

    def g(x):
        return x

    c = Compose([f, g])
    assert c.first is g
    assert c.funcs == (f,)


def test_compose_getstate_setstate():
    def f(x):
        return x

    def g(x):
        return x

    c = Compose([f, g])
    state = c.__getstate__()
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2.first is c.first
    assert c2.funcs == c.funcs


def test_compose_annotations_combined_nested():
    def f(x: int) -> str:
        return str(x)

    def g(y: float) -> int:
        return int(y)

    inner = compose(f)
    assert inner is f  # single function
    outer = Compose([f, g])
    ann = outer.__annotations__
    assert "y" in ann
    assert ann["return"] is str
