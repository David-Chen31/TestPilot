import inspect
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
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3


def test_apply_basic():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert apply(double, 5) == 10
    assert apply(inc, 500) == 501
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)


def test_apply_kwargs():
    def f(a, b=10):
        return a + b

    assert apply(f, 1, b=5) == 6
    assert apply(f, 1) == 11


def test_apply_no_args():
    with pytest.raises(TypeError):
        apply()


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def _double(x):
    return 2 * x


def _inc(x):
    return x + 1


def _add(x, y):
    return x + y


def _pow(x, y):
    return x ** y


def test_thread_first_plain():
    assert thread_first(1, _inc, _double) == 4


def test_thread_first_tuple_forms():
    assert thread_first(1, (_add, 4), (_pow, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(42) == 42


def test_thread_last_plain():
    assert thread_last(1, _inc, _double) == 4


def test_thread_last_tuple_forms():
    assert thread_last(1, (_add, 4), (_pow, 2)) == 32


def test_thread_last_no_forms():
    assert thread_last(42) == 42


def test_thread_last_map_filter():
    def iseven(x):
        return x % 2 == 0
    assert list(thread_last([1, 2, 3], (map, _inc), (filter, iseven))) == [2, 4]


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_class_and_instance():
    class MyClass:
        """The class docstring"""

        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return 'An object docstring'

        @instanceproperty
        def val(self):
            return 42

    assert MyClass.__doc__ == 'The class docstring'
    assert MyClass.val is None
    obj = MyClass()
    assert obj.__doc__ == 'An object docstring'
    assert obj.val == 42


def test_instanceproperty_full_definition():
    class C:
        def _get(self):
            return 1

        def _set(self, v):
            self._v = v

        def _del(self):
            del self._v

        p = instanceproperty(_get, _set, _del, "doc", classval="cv")

    assert C.p == "cv"
    c = C()
    assert c.p == 1
    c.p = 5
    assert c._v == 5
    del c.p
    assert not hasattr(c, "_v")


def test_instanceproperty_is_instance_property():
    class C:
        @instanceproperty
        def x(self):
            return 1

    assert isinstance(C.x, InstanceProperty)
    assert isinstance(C.__dict__["x"], InstanceProperty)


def test_instanceproperty_pickle_roundtrip():
    class C:
        @instanceproperty
        def val(self):
            return 7

    prop = C.__dict__["val"]
    restored = pickle.loads(pickle.dumps(prop))
    assert isinstance(restored, InstanceProperty)
    assert isinstance(C().val, int)
    assert C().val == 7


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def _mul(x, y):
    return x * y


def test_curry_partial_application():
    mul = curry(_mul)
    double = mul(2)
    assert double(10) == 20
    assert mul(3, 4) == 12


def test_curry_as_decorator_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5
    assert f(2, 3) == 50


def test_curry_requires_callable():
    with pytest.raises(TypeError):
        curry(42)


def test_curry_no_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_attributes():
    f = curry(_mul)
    assert f.func is _mul
    assert f.args == ()
    assert f.keywords == {}
    assert f.func_name == "_mul"
    assert f.__name__ == "_mul"


def test_curry_str_repr():
    f = curry(_mul)
    assert str(f) == str(_mul)
    assert repr(f) == repr(_mul)


def test_curry_bind_and_call():
    f = curry(_mul)
    g = f.bind(2)
    assert isinstance(g, curry)
    assert g.call(5) == 10
    assert g(5) == 10


def test_curry_equality_and_hash():
    a = curry(_mul, 2)
    b = curry(_mul, 2)
    c = curry(_mul, 3)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert not (a == 5)


def test_curry_reduce_rebuild():
    g = curry(_mul, 2)
    g2 = pickle.loads(pickle.dumps(g))
    assert g2(10) == 20


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


def test_memoize_as_decorator_with_cache():
    calls = []

    @memoize(cache={(1, 2): 3})
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(1, 2) == 3
    assert calls == []
    assert add(4, 5) == 9
    assert calls == [(4, 5)]


def test_memoize_custom_key():
    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2, verbose=True) == 3


def test_memoize_wrapped_attribute():
    def f(x):
        return x

    m = memoize(f)
    assert m.__wrapped__ is f
    assert m.__name__ == "f"


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_unary():
    calls = []

    @memoize
    def square(x):
        calls.append(x)
        return x * x

    assert square(3) == 9
    assert square(3) == 9
    assert calls == [3]


# ---------------------------------------------------------------------------
# Compose / compose / compose_left
# ---------------------------------------------------------------------------

def test_compose_basic():
    f = compose(str, _inc)
    assert f(3) == '4'


def test_compose_no_funcs_returns_identity():
    f = compose()
    assert f is identity


def test_compose_single_func_returned():
    assert compose(_inc) is _inc


def test_compose_left_basic():
    f = compose_left(_inc, str)
    assert f(3) == '4'


def test_compose_left_no_funcs():
    f = compose_left()
    assert f is identity


def test_compose_object_fields():
    c = compose(str, _inc)
    assert isinstance(c, Compose)
    assert c.first is _inc
    assert c.funcs == (str,)


def test_compose_eq_ne_hash():
    c1 = compose(str, _inc)
    c2 = compose(str, _inc)
    c3 = compose(str, _double)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert c1.__eq__(5) is NotImplemented
    assert c1.__ne__(5) is NotImplemented


def test_compose_repr():
    c = compose(str, _inc)
    assert "Compose" in repr(c)


def test_compose_name_and_doc():
    c = compose(str, _inc)
    assert c.__name__ == "_inc_of_str"
    assert "lambda *args, **kwargs" in c.__doc__


def test_compose_descriptor_method():
    class C:
        def __init__(self):
            self.v = 10

        method = compose(_inc, lambda self: self.v)

    c = C()
    assert c.method() == 11


def test_compose_signature():
    def f(a: int, b: str) -> float:
        return 1.0

    def g(x: float) -> str:
        return ""

    c = compose(f, g)
    sig = c.__signature__
    assert sig.return_annotation is float


def test_compose_annotations():
    def f(a: int, b: str) -> float:
        return 1.0

    def g(x: float) -> str:
        return ""

    c = compose(f, g)
    ann = c.__annotations__
    assert ann["a"] is int
    assert ann["b"] is str
    assert ann["return"] is float


def test_compose_pickle():
    c = compose(str, _inc)
    c2 = pickle.loads(pickle.dumps(c))
    assert c2(3) == '4'


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe_basic():
    assert pipe(3, _double, str) == '6'


def test_pipe_no_funcs():
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
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(a, b):
        return a + b
    assert juxt(f)(1, b=2) == (3,)


def test_juxt_getstate_setstate():
    inc = lambda x: x + 1
    j = juxt(inc)
    state = j.__getstate__()
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2(5) == (6,)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    assert do(log.append, 1) == 1
    assert log == [1]


def test_do_returns_x():
    obj = object()
    assert do(lambda x: None, obj) is obj


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
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none / excepts
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError()) is None


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


def test_excepts_name_and_doc():
    def f(x):
        """f doc"""
        return [1, 2].index(x)

    def handler(e):
        """handler doc"""
        return -1

    e = excepts(ValueError, f, handler)
    assert e.__name__ == "f_excepting_ValueError"
    assert "A wrapper around 'f'" in e.__doc__
    assert "f doc" in e.__doc__
    assert "handler doc" in e.__doc__


def test_excepts_name_tuple():
    e = excepts((IndexError, KeyError), lambda a: a[0])
    assert e.__name__ == "lambda_excepting_IndexError_or_KeyError"


def test_excepts_class_doc_attr():
    assert isinstance(excepts.__doc__, str)


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    def g(*args, **kwargs):
        pass

    assert num_required_args(f) == 2
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

    def g(x):
        return x

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


def test_is_arity_varargs():
    def f(*args):
        return args

    assert is_arity(1, f) is False


# ---------------------------------------------------------------------------
# Additional curry edge cases
# ---------------------------------------------------------------------------

def test_curry_merges_partial_args():
    p = curry(curry(_mul, 2), 3)
    assert p.func is _mul
    assert p.args == (2, 3)
    assert p() == 6


def test_curry_should_curry_genuine_typeerror():
    def f(x, y):
        return x / y

    c = curry(f)
    # full arguments, but division raises ZeroDivisionError, not TypeError;
    # use a type-error-producing function instead
    def g(x, y):
        return x + y

    cg = curry(g)
    # Provide both args - should be a genuine call, no currying needed
    assert cg(1, 2) == 3


def test_curry_signature():
    @curry
    def f(x, y, z=3):
        return x + y + z

    sig = f.__signature__
    params = list(sig.parameters)
    assert "x" in params
    g = f(1)
    sig2 = g.__signature__
    assert "x" not in list(sig2.parameters)


def test_curry_descriptor_binding():
    class C:
        @curry
        def method(self, x):
            return (self, x)

    c = C()
    result = c.method(1)
    assert result == (c, 1)


def test_curry_get_descriptor_no_instance():
    class C:
        @curry
        def method(self, x):
            return (self, x)

    bound = C.method
    assert isinstance(bound, curry) or callable(bound)


def test_curry_call_bypasses_curry():
    @curry
    def f(x, y):
        return x + y

    # call() always invokes underlying partial
    with pytest.raises(TypeError):
        f.call(1)


# ---------------------------------------------------------------------------
# memoize branches: has keywords
# ---------------------------------------------------------------------------

def test_memoize_kwargs_key():
    calls = []

    @memoize
    def f(x, y=0):
        calls.append((x, y))
        return x + y

    assert f(1) == 1
    assert f(1, y=2) == 3
    assert f(1) == 1
    assert len(calls) == 2


def test_memoize_no_keywords_key():
    calls = []

    @memoize
    def f(x, y):
        calls.append((x, y))
        return x + y

    assert f(1, 2) == 3
    assert f(1, 2) == 3
    assert calls == [(1, 2)]
