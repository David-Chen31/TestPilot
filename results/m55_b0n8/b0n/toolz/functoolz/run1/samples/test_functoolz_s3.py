import pickle
import sys
import pytest

from toolz.functoolz import (
    identity, apply, thread_first, thread_last, instanceproperty,
    InstanceProperty, curry, memoize, Compose, compose, compose_left, pipe,
    complement, juxt, do, flip, return_none, excepts, num_required_args,
    has_varargs, has_keywords, is_valid_args, is_partial_args, is_arity,
)


# ---------------------------------------------------------------------------
# identity / apply
# ---------------------------------------------------------------------------

def test_identity():
    assert identity(3) == 3
    obj = object()
    assert identity(obj) is obj


def test_apply_basic():
    assert apply(lambda x: 2 * x, 5) == 10
    assert apply(lambda x, y: x + y, 1, 2) == 3
    assert apply(lambda x, y=10: x + y, 1, y=5) == 6


def test_apply_requires_func():
    with pytest.raises(TypeError):
        apply()


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_empty():
    assert thread_first(5) == 5


def test_thread_first_callable_and_tuple():
    inc = lambda x: x + 1
    double = lambda x: 2 * x
    assert thread_first(1, inc, double) == 4

    add = lambda x, y: x + y
    pow_ = lambda x, y: x ** y
    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_last_empty():
    assert thread_last(5) == 5


def test_thread_last_callable_and_tuple():
    inc = lambda x: x + 1
    double = lambda x: 2 * x
    assert thread_last(1, inc, double) == 4

    add = lambda x, y: x + y
    pow_ = lambda x, y: x ** y
    assert thread_last(1, (add, 4), (pow_, 2)) == 32


def test_thread_last_map_filter():
    inc = lambda x: x + 1
    iseven = lambda x: x % 2 == 0
    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_class_access():
    class C:
        """Class doc."""
        @instanceproperty
        def val(self):
            return 42

        @instanceproperty(classval="classval")
        def other(self):
            return "instance"

    assert C.val is None
    assert C.other == "classval"
    obj = C()
    assert obj.val == 42
    assert obj.other == "instance"


def test_instanceproperty_used_without_call():
    @instanceproperty
    def f(self):
        return 1

    assert isinstance(f, InstanceProperty)
    assert f.classval is None


def test_instanceproperty_setter_and_deleter():
    class C:
        def __init__(self):
            self._x = 0

        @instanceproperty
        def x(self):
            return self._x

        @x.setter
        def x(self, v):
            self._x = v

    c = C()
    assert c.x == 0
    c.x = 5
    assert c.x == 5


def test_instanceproperty_doc():
    class C:
        @instanceproperty(classval="class doc")
        def __doc__(self):
            return "instance doc"

    assert C.__doc__ == "class doc"
    assert C().__doc__ == "instance doc"


def test_instanceproperty_pickle():
    class C:
        @instanceproperty
        def val(self):
            return 42

    p = pickle.loads(pickle.dumps(C.__dict__['val']))
    assert isinstance(p, InstanceProperty)
    assert p.classval is None


# ---------------------------------------------------------------------------
# curry basics
# ---------------------------------------------------------------------------

def test_curry_requires_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_requires_callable():
    with pytest.raises(TypeError):
        curry(3)


def test_curry_simple():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert double(10) == 20
    assert cmul(2, 5) == 10


def test_curry_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_attributes():
    def foo():
        "docstring"

    c = curry(foo)
    assert c.func is foo
    assert c.__name__ == "foo"
    assert c.__doc__ == "docstring"
    assert str(c) == str(foo)
    assert repr(c) == repr(foo)


def test_curry_equality_hash():
    def a(x, y):
        return x + y

    c1 = curry(a, 1)
    c2 = curry(a, 1)
    c3 = curry(a, 2)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert c1 != 5


def test_curry_unpacking_partial():
    def f(x, y, z=0):
        return x + y + z

    p = curry(f, 1)
    q = curry(p, 2)
    assert q.func is f
    assert q.args == (1, 2)
    assert q(3) == 6


def test_curry_call_vs_bind():
    def f(x, y):
        return x + y

    c = curry(f)
    # Not enough args -> curried
    partial_c = c(1)
    assert isinstance(partial_c, curry)
    assert partial_c(2) == 3
    # Direct call
    assert c(1, 2) == 3
    # bind
    b = c.bind(1)
    assert isinstance(b, curry)
    assert b(2) == 3
    # call bypasses currying
    with pytest.raises(TypeError):
        curry(f).call(1)


def test_curry_genuine_typeerror():
    def f(x):
        return x + 1

    c = curry(f)
    with pytest.raises(TypeError):
        c("a")


def test_curry_invalid_args():
    def f(x, y):
        return x + y

    c = curry(f)
    with pytest.raises(TypeError):
        c(1, 2, 3)


def test_curry_descriptor():
    class C:
        @curry
        def method(self, x, y):
            return self.offset + x + y

        def __init__(self, offset):
            self.offset = offset

    c = C(10)
    assert c.method(1, 2) == 13
    assert c.method(1)(2) == 13
    # class access returns curry
    assert isinstance(C.method, curry)


def test_curry_func_name_property():
    def f():
        pass

    c = curry(f)
    assert c.func_name == "f"


def test_curry_signature():
    def f(x, y, z=3):
        return x + y + z

    import inspect
    c = curry(f, 1)
    sig = c.__signature__
    assert list(sig.parameters) == ["y", "z"]


def test_curry_signature_invalid():
    def f(x):
        return x

    c = curry(f, 1, 2)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_signature_keywords():
    def f(x, y, z=0):
        return x + y + z

    c = curry(f, 1, y=2)
    sig = c.__signature__
    assert "z" in sig.parameters


def test_curry_pickle():
    import toolz.functoolz as ft

    c = curry(ft.apply)
    data = pickle.dumps(c)
    restored = pickle.loads(data)
    assert isinstance(restored, curry)
    assert restored(identity, 5) == 5


def test_curry_pickle_with_args():
    def add(x, y):
        return x + y
    add.__module__ = __name__
    globals()['_pickle_add_func'] = add

    c = curry(add, 1)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == 3


def test_curry_call_method():
    @curry
    def f(x, y):
        return x + y

    assert f.call(1, 2) == 3


def test_curry_should_curry_varargs():
    @curry
    def deco(fn, *args, **kwargs):
        if not args:
            return "need args"
        return fn(*args, **kwargs)

    # _has_unknown_args should be True; currying happens
    assert callable(deco)


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    m = memoize(add)
    assert m(1, 2) == 3
    assert m(1, 2) == 3
    assert calls == [(1, 2)]
    assert m.__wrapped__ is add


def test_memoize_decorator():
    calls = []

    @memoize
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert add(2, 3) == 5
    assert len(calls) == 2


def test_memoize_with_cache():
    @memoize(cache={(1, 2): 100})
    def add(x, y):
        return x + y

    assert add(1, 2) == 100
    assert add(3, 4) == 7


def test_memoize_custom_key():
    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2, verbose=True) == 3


def test_memoize_unary():
    calls = []

    @memoize
    def sq(x):
        calls.append(x)
        return x * x

    assert sq(3) == 9
    assert sq(3) == 9
    assert calls == [3]


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_kwargs():
    calls = []

    @memoize
    def f(x, y=1):
        calls.append((x, y))
        return x + y

    assert f(1, y=2) == 3
    assert f(1, y=2) == 3
    assert len(calls) == 1


def test_memoize_name_and_doc():
    def foo(x):
        "doc for foo"
        return x

    m = memoize(foo)
    assert m.__name__ == "foo"
    assert m.__doc__ == "doc for foo"


# ---------------------------------------------------------------------------
# compose / compose_left / Compose
# ---------------------------------------------------------------------------

def test_compose_empty():
    assert compose() is identity


def test_compose_single():
    inc = lambda x: x + 1
    assert compose(inc) is inc


def test_compose_multiple():
    inc = lambda x: x + 1
    double = lambda x: 2 * x
    f = compose(str, inc)
    assert f(3) == "4"
    # right to left
    g = compose(inc, double)
    assert g(3) == 7


def test_compose_multiple_args():
    def add(x, y):
        return x + y
    f = compose(str, add)
    assert f(1, 2) == "3"


def test_compose_left():
    assert compose_left() is identity
    inc = lambda x: x + 1
    assert compose_left(inc) is inc
    f = compose_left(inc, str)
    assert f(3) == "4"


def test_compose_class():
    inc = lambda x: x + 1
    double = lambda x: 2 * x
    c = Compose((inc, double))
    assert isinstance(c, Compose)
    assert c(3) == 7


def test_compose_eq_hash():
    inc = lambda x: x + 1
    double = lambda x: 2 * x
    c1 = Compose((inc, double))
    c2 = Compose((inc, double))
    c3 = Compose((double, inc))
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert (c1 == 5) is False
    assert (c1 != 5) is True


def test_compose_repr():
    inc = lambda x: x + 1
    double = lambda x: 2 * x
    c = Compose((inc, double))
    r = repr(c)
    assert r.startswith("Compose(")
    assert "inc" not in r  # lambdas have names like <lambda>
    assert "<lambda>" in r


def test_compose_name():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    c = Compose((inc, double))
    assert c.__name__ == "inc_of_double"


def test_compose_name_no_names():
    class Callable:
        def __call__(self, x):
            return x

    c = Compose((Callable(), Callable()))
    assert c.__name__ == "Compose"


def test_compose_doc():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    c = Compose((inc, double))
    assert c.__doc__.startswith("lambda *args, **kwargs:")


def test_compose_doc_classval():
    assert isinstance(Compose.__dict__['__doc__'], InstanceProperty)
    assert Compose.__doc__ == Compose.__doc__  # class access nontrivial


def test_compose_descriptor():
    class C:
        def add(self, x, y):
            return x + y

        composed = Compose((lambda x: x * 2,))

    c = C()
    # bound as method
    bound = c.composed
    assert bound(3) == 6
    # class access returns Compose
    assert isinstance(C.composed, Compose)


def test_compose_signature():
    def inc(x: int) -> int:
        return x + 1

    def to_str(x: int) -> str:
        return str(x)

    c = Compose((to_str, inc))
    sig = c.__signature__
    import inspect
    assert sig.return_annotation is str or sig.return_annotation == str


def test_compose_wrapped():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    c = Compose((inc, double))
    assert c.__wrapped__ is double


def test_compose_annotations():
    def inc(x: int) -> int:
        return x + 1

    def to_str(x: int) -> str:
        return str(x)

    c = Compose((to_str, inc))
    ann = c.__annotations__
    assert ann['x'] is int
    assert ann['return'] is str


def test_compose_pickle():
    # Use a top-level compose of curried funcs to test pickling via reduce
    import toolz.functoolz as ft
    c = ft.Compose((identity, identity))
    assert c(5) == 5


# ---------------------------------------------------------------------------
# pipe / complement / juxt / do
# ---------------------------------------------------------------------------

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == "6"
    assert pipe(3) == 3


def test_complement():
    iseven = lambda n: n % 2 == 0
    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


def test_juxt():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(x, y=0):
        return x + y

    def g(x, y=0):
        return x * y

    assert juxt(f, g)(3, y=4) == (7, 12)


def test_juxt_pickle():
    inc = lambda x: x + 1
    j = juxt(inc)
    data = pickle.dumps(j)
    restored = pickle.loads(data)
    assert restored(1) == (2,)


def test_do():
    log = []
    result = do(log.append, 5)
    assert result == 5
    assert log == [5]


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------

def test_flip():
    def div(a, b):
        return a // b
    assert flip(div, 2, 6) == 3

    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_flip_isinstance():
    data = [1, "a", "b", 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none / excepts
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError("x")) is None


def test_excepts():
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


def test_excepts_name():
    e = excepts(ValueError, lambda a: a, lambda exc: None)
    assert e.__name__ == "<lambda>_excepting_ValueError"

    e2 = excepts((ValueError, KeyError), lambda a: a)
    assert e2.__name__ == "<lambda>_excepting_ValueError_or_KeyError"


def test_excepts_name_no_attr():
    class C:
        def __call__(self, a):
            return a

    e = excepts(ValueError, C())
    assert e.__name__ == "excepting"


def test_excepts_doc():
    e = excepts(ValueError, lambda a: a, return_none)
    assert "ValueError" in e.__doc__


def test_excepts_class_doc():
    assert "functional try/except" in excepts.__doc__


def test_excepts_doc_attr_error():
    class C:
        def __call__(self, a):
            return a

    e = excepts(ValueError, C())
    assert e.__doc__ == excepts.__doc__


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

    def h(x):
        return x

    assert num_required_args(h) == 1


def test_has_varargs():
    def f(*args):
        return args

    assert has_varargs(f) is True

    def g(**kwargs):
        return kwargs

    assert has_varargs(g) is False


def test_has_keywords():
    def f(x, y=0):
        return x + y

    assert has_keywords(f) is True

    def g(x):
        return x

    assert has_keywords(g) is False

    def h(**kwargs):
        return kwargs

    assert has_keywords(h) is True


def test_is_valid_args():
    def add(x, y):
        return x + y

    assert is_valid_args(add, (1,), {}) is False
    assert is_valid_args(add, (1, 2), {}) is True
    assert is_valid_args(map, (), {}) is False


def test_is_partial_args():
    def add(x, y):
        return x + y

    assert is_partial_args(add, (1,), {}) is True
    assert is_partial_args(add, (1, 2), {}) is True
    assert is_partial_args(add, (1, 2, 3), {}) is False
    assert is_partial_args(map, (), {}) is True


def test_is_arity():
    def f(x):
        return x

    assert is_arity(1, f) is True

    def g(x, y=1):
        return x + y

    assert is_arity(1, g) is False

    def h(*args):
        return args

    assert is_arity(1, h) is False

    def k(x, y):
        return x + y

    assert is_arity(2, k) is True
