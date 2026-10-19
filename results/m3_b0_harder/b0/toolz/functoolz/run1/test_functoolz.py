import inspect
import pickle
from functools import partial

import pytest

from toolz.functoolz import (
    identity,
    apply,
    thread_first,
    thread_last,
    memoize,
    compose,
    compose_left,
    pipe,
    complement,
    juxt,
    do,
    curry,
    flip,
    excepts,
    instanceproperty,
    InstanceProperty,
    Compose,
    return_none,
    num_required_args,
    has_varargs,
    has_keywords,
    is_valid_args,
    is_partial_args,
    is_arity,
)
from toolz.utils import no_default


def inc(x):
    return x + 1


def double(x):
    return 2 * x


def add(x, y):
    return x + y


def mul(x, y):
    return x * y


def div(a, b):
    return a // b


def iseven(x):
    return x % 2 == 0


def add_self(self, x):
    return self.value + x


def may_raise_value(x):
    return [1, 2].index(x)


def pickle_add(x, y):
    return x + y


def annotated_first(x: int) -> str:
    return str(x)


def annotated_second(y: str) -> float:
    return float(y)


@curry
def curried_add(x, y):
    return x + y


def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3
    assert identity(None) is None


def test_apply():
    with pytest.raises(TypeError, match="func argument is required"):
        apply()

    assert apply(add, 1, 2) == 3
    assert apply(add, 1, y=2) == 3
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20,
        501,
        16000,
    )


def test_thread_first_and_thread_last():
    assert thread_first(1, inc, double) == 4
    assert thread_first(1, (add, 4), (pow, 2)) == 25
    assert thread_first(5) == 5

    assert thread_last(1, inc, double) == 4
    assert thread_last(1, (add, 4), (pow, 2)) == 32
    assert thread_last(5) == 5

    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


def test_memoize_basic_and_metadata():
    calls = []

    def f(x):
        """double a value"""
        calls.append(x)
        return double(x)

    mf = memoize(f)

    assert mf(3) == 6
    assert mf(3) == 6
    assert calls == [3]
    assert mf.__name__ == "f"
    assert mf.__doc__ == "double a value"
    assert mf.__wrapped__ is f


def test_memoize_cache_and_custom_key():
    calls = []

    @memoize(cache={(1, 2): 3})
    def add_cached(x, y):
        calls.append((x, y))
        return x + y

    assert add_cached(1, 2) == 3
    assert calls == []
    assert add_cached(2, 3) == 5
    assert calls == [(2, 3)]

    calls2 = []

    @memoize(key=lambda args, kwargs: args)
    def add_ignoring_kwargs(x, y, verbose=False):
        calls2.append((x, y))
        return x + y

    assert add_ignoring_kwargs(1, 2, verbose=True) == 3
    assert add_ignoring_kwargs(1, 2, verbose=False) == 3
    assert calls2 == [(1, 2)]


def test_memoize_cache_is_updated():
    cache = {}
    calls = []

    def f(x):
        calls.append(x)
        return x + 1

    mf = memoize(f, cache=cache)

    assert mf(5) == 6
    assert mf(5) == 6
    assert calls == [5]
    assert cache == {5: 6}


def test_memoize_unhashable_arguments():
    mf = memoize(identity)

    with pytest.raises(
        TypeError, match="Arguments to memoized function must be hashable"
    ):
        mf([1, 2])


def test_compose_and_compose_left():
    assert compose() is identity
    assert compose(inc) is inc
    assert compose_left() is identity
    assert compose_left(inc) is inc

    assert compose(inc, double)(3) == 7
    assert compose_left(inc, double)(3) == 8
    assert compose(str, inc)(3) == "4"
    assert compose_left(inc, str)(3) == "4"
    assert compose(str, add)(2, 3) == "5"


def test_compose_class_behavior():
    c = Compose((inc, double))

    assert c.first is double
    assert c.funcs == (inc,)
    assert c(5) == 11
    assert c.__wrapped__ is double
    assert c.__name__ == "inc_of_double"
    assert c.__doc__ == "lambda *args, **kwargs: inc(double(*args, **kwargs))"
    assert repr(c) == "Compose" + repr((inc, double))

    assert c == Compose((inc, double))
    assert c != Compose((double, inc))
    assert c.__eq__(object()) is NotImplemented
    assert c.__ne__(object()) is NotImplemented
    assert (c == object()) is False
    assert hash(c) == hash(Compose((inc, double)))

    first, funcs = c.__getstate__()
    assert first is double
    assert funcs == (inc,)

    d = Compose((str, add))
    d.__setstate__((first, funcs))
    assert d.first is double
    assert d.funcs == (inc,)
    assert d(5) == 11

    assert Compose((str, add))(2, 3) == "5"


def test_compose_fallback_names():
    class NoName:
        def __call__(self, x):
            return x

    c = Compose((inc, NoName()))

    assert c.__doc__ == "A composition of functions"
    assert c.__name__ == "Compose"


def test_compose_signature_and_annotations():
    c = compose(annotated_first, annotated_second)

    sig = inspect.signature(c)
    assert list(sig.parameters) == ["y"]
    assert sig.parameters["y"].annotation is str
    assert sig.return_annotation is str

    assert c.__annotations__ == {"y": str, "return": str}


def test_compose_as_descriptor():
    class C:
        value = 10
        method = compose(inc, add_self)

    obj = C()

    assert obj.method(5) == 16
    assert C.method(obj, 5) == 16


def test_instanceproperty():
    class C:
        @instanceproperty(classval="class-value")
        def value(self):
            return 42

    assert C.value == "class-value"
    assert C().value == 42

    p = instanceproperty(classval="cv")
    assert isinstance(p, partial)

    class D:
        @p
        def val(self):
            return 7

    assert D.val == "cv"
    assert D().val == 7


def test_instanceproperty_reduce():
    def getter(obj):
        return 11

    ip = InstanceProperty(getter, classval="cls")
    restore, state = ip.__reduce__()

    assert restore is InstanceProperty
    assert state == (getter, None, None, None, "cls")

    ip2 = restore(*state)

    class E:
        value = ip2

    assert E.value == "cls"
    assert E().value == 11


def test_pipe_complement_do():
    assert pipe(3, double, str) == "6"
    assert pipe(3) == 3

    assert complement(iseven)(2) is False
    assert complement(iseven)(3) is True

    log = []
    assert do(log.append, 5) == 5
    assert log == [5]


def test_juxt():
    j = juxt(inc, double)

    assert j(3) == (4, 6)
    assert j.funcs == (inc, double)
    assert juxt([inc, double])(3) == (4, 6)
    assert juxt((inc, double))(3) == (4, 6)
    assert juxt()(3) == ()
    assert juxt(lambda x, y: x + y, lambda x, y: x * y)(2, 3) == (5, 6)

    assert j.__getstate__() == (inc, double)

    j2 = juxt(inc)
    j2.__setstate__((double, inc))
    assert j2.funcs == (double, inc)
    assert j2(2) == (4, 3)


def test_flip():
    assert flip(div, 2, 6) == 3

    div_by_two = flip(div, 2)
    assert isinstance(div_by_two, curry)
    assert div_by_two(6) == 3

    assert flip(pow, 2, 3) == 9

    data = [1, "a", 2, 1.5, object()]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2]


def test_excepts_basic():
    e = excepts(ValueError, may_raise_value, lambda err: -1)

    assert e(1) == 0
    assert e(3) == -1

    e2 = excepts((IndexError, KeyError), lambda a: a[0])

    assert e2([]) is None
    assert e2([1]) == 1
    assert e2({}) is None
    assert e2({0: 1}) == 1

    with pytest.raises(ZeroDivisionError):
        excepts(ValueError, lambda: 1 / 0)()


def test_excepts_handler_receives_exception():
    seen = []

    def handler(exc):
        seen.append(exc)
        return "handled"

    e = excepts(ValueError, lambda x: int(x), handler)

    assert e("not-an-int") == "handled"
    assert len(seen) == 1
    assert isinstance(seen[0], ValueError)


def test_excepts_name_and_doc():
    e = excepts(ValueError, may_raise_value)

    assert e.__name__ == "may_raise_value_excepting_ValueError"

    doc = e.__doc__
    assert doc.startswith(
        "A wrapper around 'may_raise_value' that will except:\nValueError\n"
    )
    assert "and handle any exceptions with 'return_none'." in doc

    e2 = excepts((IndexError, KeyError), may_raise_value)
    assert e2.__name__ == "may_raise_value_excepting_IndexError_or_KeyError"
    assert "(IndexError, KeyError)" in e2.__doc__

    class CallableWithoutName:
        def __call__(self, x):
            return x

    e3 = excepts(ValueError, CallableWithoutName())
    assert e3.__name__ == "excepting"
    assert e3.__doc__ == excepts.__doc__


def test_return_none():
    err = ValueError("x")
    assert return_none(err) is None


def test_curry_init_errors():
    with pytest.raises(TypeError, match=r"__init__\(\) takes at least 2 arguments"):
        curry()

    with pytest.raises(TypeError, match="Input must be callable"):
        curry(42)


def test_curry_basic_and_metadata():
    def multiply(x, y):
        """multiply numbers"""
        return x * y

    cm = curry(multiply)

    assert cm(2, 3) == 6
    assert cm.func is multiply
    assert cm.args == ()
    assert cm.keywords == {}
    assert cm.func_name == "multiply"
    assert cm.__name__ == "multiply"
    assert cm.__doc__ == "multiply numbers"
    assert cm.__module__ == multiply.__module__
    assert cm.__qualname__ == multiply.__qualname__


def test_curry_partial_application_and_errors():
    c = curry(add)
    double_add = c(2)

    assert isinstance(double_add, curry)
    assert double_add.args == (2,)
    assert double_add(3) == 5

    with pytest.raises(TypeError):
        c(1, 2, 3)


def test_curry_keyword_merging():
    def f(x, y, a=10):
        return a * (x + y)

    cf = curry(f)
    add1 = cf(a=1)

    assert isinstance(add1, curry)
    assert add1.keywords == {"a": 1}
    assert add1(2, 3) == 5

    def kw(a, b=0):
        return a + b

    c1 = curry(kw, a=1)
    c2 = curry(c1, b=2)

    assert c2.keywords == {"a": 1, "b": 2}
    assert c2() == 3

    p = partial(add, 2)
    cp = curry(p, 3)

    assert cp.func is add
    assert cp.args == (2, 3)
    assert cp() == 5


def test_curry_bind_and_call():
    c = curry(add)

    bound = c.bind(2)

    assert isinstance(bound, curry)
    assert bound.args == (2,)
    assert bound(3) == 5

    assert c.call(2, 3) == 5

    with pytest.raises(TypeError):
        c.call(2)


def test_curry_equality_hash_string():
    c = curry(add)

    assert str(c) == str(add)
    assert repr(c) == repr(add)
    assert c == curry(add)
    assert hash(c) == hash(curry(add))
    assert c != curry(add, 1)
    assert c != 10


def test_curry_signature():
    c = curry(add)
    sig = inspect.signature(c)

    assert list(sig.parameters) == ["x", "y"]
    assert sig.parameters["x"].default is no_default
    assert sig.parameters["y"].default is no_default

    c2 = curry(add, 1)
    sig2 = inspect.signature(c2)

    assert list(sig2.parameters) == ["y"]
    assert sig2.parameters["y"].default is no_default

    bad = curry(add, 1, 2, 3)
    with pytest.raises(TypeError, match="curry object has incorrect arguments"):
        bad.__signature__


def test_curry_descriptor():
    class C:
        value = 10
        method = curry(add_self)

    assert C.method is C.__dict__["method"]

    obj = C()

    assert isinstance(obj.method, curry)
    assert obj.method.func is add_self
    assert obj.method.args == (obj,)
    assert obj.method(5) == 15
    assert C.method(obj, 5) == 15


def test_curry_pickle():
    c = curry(pickle_add, 1)
    restored = pickle.loads(pickle.dumps(c))

    assert isinstance(restored, curry)
    assert restored.func is pickle_add
    assert restored.args == (1,)
    assert restored(2) == 3

    part = curried_add(1)
    restored_part = pickle.loads(pickle.dumps(part))

    assert isinstance(restored_part, curry)
    assert restored_part.func is curried_add.func
    assert restored_part.args == (1,)
    assert restored_part(2) == 3

    whole = pickle.loads(pickle.dumps(curried_add))
    assert whole is curried_add


def test_curry_genuine_type_error_and_varargs():
    def boom(x):
        raise TypeError("boom")

    with pytest.raises(TypeError, match="boom"):
        curry(boom)(1)

    def var(*args):
        if len(args) < 2:
            raise TypeError("need two")
        return args

    cv = curry(var)
    partial_cv = cv(1)

    assert isinstance(partial_cv, curry)
    assert partial_cv.args == (1,)
    assert partial_cv(2) == (1, 2)


def test_introspection_predicates():
    def f(x, y, z=3):
        return x + y + z

    def g(*args, **kwargs):
        pass

    def h(x, /, y=1):
        pass

    def keyword_only(*, x):
        pass

    assert num_required_args(f) == 2
    assert num_required_args(g) == 0
    assert num_required_args(h) == 1

    assert has_varargs(g) is True
    assert has_varargs(f) is False

    assert has_keywords(f) is True
    assert has_keywords(lambda x: x) is False
    assert has_keywords(keyword_only) is True
    assert has_keywords(g) is True


def test_introspection_validity_and_arity():
    def add2(x, y):
        return x + y

    assert is_valid_args(add2, (1,), {}) is False
    assert is_valid_args(add2, (1, 2), {}) is True
    assert is_valid_args(add2, (1, 2, 3), {}) is False
    assert is_valid_args(add2, (), {"x": 1, "y": 2}) is True

    assert is_partial_args(add2, (1,), {}) is True
    assert is_partial_args(add2, (1, 2), {}) is True
    assert is_partial_args(add2, (1, 2, 3), {}) is False
    assert is_partial_args(add2, (), {"x": 1}) is True

    assert is_arity(1, lambda x: x) is True
    assert is_arity(2, add2) is True
    assert is_arity(1, lambda x, y=1: x + y) is False
    assert is_arity(1, lambda *args: args) is False
    assert is_arity(0, lambda: 1) is True


def test_introspection_builtin_registry():
    assert is_valid_args(map, (), {}) is False
    assert is_partial_args(map, (), {}) is True
