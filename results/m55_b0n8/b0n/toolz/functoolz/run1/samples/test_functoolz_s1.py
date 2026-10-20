import pickle
import sys
import inspect
from functools import partial

import pytest

from toolz.functoolz import (
    identity,
    apply,
    thread_first,
    thread_last,
    instanceproperty,
    InstanceProperty,
    curry,
    memoize,
    Compose,
    compose,
    compose_left,
    pipe,
    complement,
    juxt,
    do,
    flip,
    return_none,
    excepts,
    num_required_args,
    has_varargs,
    has_keywords,
    is_valid_args,
    is_partial_args,
    is_arity,
)


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def test_identity():
    assert identity(3) == 3
    obj = object()
    assert identity(obj) is obj
    assert identity(None) is None


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def test_apply_basic():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert apply(double, 5) == 10
    assert apply(inc, 500) == 501


def test_apply_kwargs():
    def f(a, b=0):
        return a + b

    assert apply(f, 1, b=2) == 3


def test_apply_no_func():
    with pytest.raises(TypeError):
        apply()


def test_apply_map():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    result = tuple(map(apply, [double, inc, double], [10, 500, 8000]))
    assert result == (20, 501, 16000)


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


def test_thread_first_basic():
    assert thread_first(1, _inc, _double) == 4


def test_thread_first_tuples():
    assert thread_first(1, (_add, 4), (_pow, 2)) == 25


def test_thread_first_empty():
    assert thread_first(42) == 42


def test_thread_last_basic():
    assert thread_last(1, _inc, _double) == 4


def test_thread_last_tuples():
    assert thread_last(1, (_add, 4), (_pow, 2)) == 32


def test_thread_last_map_filter():
    def iseven(x):
        return x % 2 == 0

    result = list(thread_last([1, 2, 3], (map, _inc), (filter, iseven)))
    assert result == [2, 4]


def test_thread_last_empty():
    assert thread_last(7) == 7


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_class_and_instance():
    class MyClass:
        """The class docstring"""

        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return "An object docstring"

        @instanceproperty
        def val(self):
            return 42

    assert MyClass.__doc__ == "The class docstring"
    assert MyClass.val is None
    obj = MyClass()
    assert obj.__doc__ == "An object docstring"
    assert obj.val == 42


def test_instanceproperty_is_property():
    assert issubclass(InstanceProperty, property)


def test_instanceproperty_setter():
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
    c.x = 5
    assert c.x == 5


def test_instanceproperty_partial_form():
    # Calling with no fget returns a partial.
    ip_factory = instanceproperty(classval="CLASSVAL")
    assert isinstance(ip_factory, partial)

    def getter(self):
        return "instance"

    prop = ip_factory(getter)
    assert isinstance(prop, InstanceProperty)
    assert prop.classval == "CLASSVAL"


def test_instanceproperty_reduce():
    def getter(self):
        return 1

    prop = InstanceProperty(fget=getter, classval=99)
    restored = pickle.loads(pickle.dumps(prop))
    assert isinstance(restored, InstanceProperty)
    assert restored.classval == 99
    assert restored.fget is getter


# ---------------------------------------------------------------------------
# num_required_args / has_varargs / has_keywords /
# is_valid_args / is_partial_args / is_arity
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2

    def g(*args, **kwargs):
        pass

    assert num_required_args(g) == 0


def test_num_required_args_positional_only():
    def f(x, /, y):
        return x + y

    assert num_required_args(f) == 2


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

    def g(x, y):
        return x + y

    assert has_keywords(g) is False


def test_has_keywords_kwonly():
    def f(x, *, y):
        return x + y

    assert has_keywords(f) is True


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


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert double(10) == 20


def test_curry_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_requires_callable():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_no_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_attributes():
    def foo(x, y):
        "docstring"
        return x + y

    c = curry(foo)
    assert c.func is foo
    assert c.__name__ == "foo"
    assert c.__doc__ == "docstring"
    assert c.args == ()
    assert c.keywords == {}
    assert c.func_name == "foo"


def test_curry_str_repr():
    def foo(x):
        return x

    c = curry(foo)
    assert str(c) == str(foo)
    assert repr(c) == repr(foo)


def test_curry_call_and_bind():
    def add(a, b, c):
        return a + b + c

    c = curry(add, 1)
    assert c.call(2, 3) == 6
    bound = c.bind(2)
    assert bound(3) == 6


def test_curry_eq_hash():
    def add(a, b):
        return a + b

    c1 = curry(add, 1)
    c2 = curry(add, 1)
    c3 = curry(add, 2)
    assert c1 == c2
    assert not (c1 == c3)
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert c1 != 5


def test_curry_type_error_propagates():
    def add(a, b):
        return a + b

    c = curry(add)
    with pytest.raises(TypeError):
        c(1, 2, 3)


def test_curry_unpack_partial():
    def add(a, b, c):
        return a + b + c

    p = partial(add, 1, 2)
    c = curry(p, 3)
    assert c() == 6


def test_curry_unpack_curry():
    def add(a, b, c):
        return a + b + c

    c1 = curry(add, 1)
    c2 = curry(c1, 2)
    assert c2(3) == 6
    assert c2.func is add


def test_curry_descriptor():
    class A:
        def __init__(self, factor):
            self.factor = factor

        @curry
        def mul(self, x):
            return self.factor * x

    a = A(3)
    assert a.mul(4) == 12


def test_curry_signature():
    def f(a, b, c=3):
        return a + b + c

    c = curry(f, 1)
    sig = c.__signature__
    assert list(sig.parameters) == ["b", "c"]


def test_curry_signature_bad_args():
    def f(a, b):
        return a + b

    c = curry(f, 1, 2, 3)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_signature_varargs():
    def f(*args):
        return args

    c = curry(f, 1)
    sig = c.__signature__
    assert list(sig.parameters) == ["args"]


def test_curry_signature_keywords():
    def f(a, b, c):
        return a + b + c

    c = curry(f, a=1)
    sig = c.__signature__
    params = sig.parameters
    assert "a" not in params
    assert params["b"].kind == inspect.Parameter.KEYWORD_ONLY


def test_curry_should_curry_with_varargs():
    def f(*args, **kwargs):
        return args, kwargs

    c = curry(f, 1)
    # Calling with a TypeError-raising inner further should still curry.
    result = c(2)
    assert result == ((1, 2), {})


def test_curry_pickle():
    # A module-level curried function is picklable.
    c = curry(_add, 1)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == 3


def test_curry_pickle_decorated():
    # flip is a module-level curry decorator.
    f = flip(_div, 2)
    restored = pickle.loads(pickle.dumps(f))
    assert restored(6) == 3


def _div(a, b):
    return a // b


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
    assert m.__name__ == "add"
    assert m.__wrapped__ is add


def test_memoize_cache_kwarg():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        return x + y

    assert add(1, 2) == 3
    assert add(2, 3) == 5


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y))
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_unary():
    calls = []

    @memoize
    def twice(x):
        calls.append(x)
        return 2 * x

    assert twice(3) == 6
    assert twice(3) == 6
    assert calls == [3]


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_kwargs_distinct():
    calls = []

    @memoize
    def f(x, verbose=False):
        calls.append((x, verbose))
        return x

    assert f(1, verbose=True) == 1
    assert f(1, verbose=False) == 1
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Compose / compose / compose_left
# ---------------------------------------------------------------------------

def test_compose_returns_identity():
    assert compose() is identity


def test_compose_single():
    assert compose(_inc) is _inc


def test_compose_basic():
    result = compose(str, _inc)(3)
    assert result == "4"


def test_compose_left_basic():
    result = compose_left(_inc, str)(3)
    assert result == "4"


def test_compose_empty_left():
    assert compose_left() is identity


def test_compose_obj_basic():
    c = Compose((str, _inc))
    assert isinstance(c, Compose)
    assert c(3) == "4"


def test_compose_doc():
    c = compose(str, _inc)
    assert "lambda *args, **kwargs" in c.__doc__
    assert "str(" in c.__doc__


def test_compose_name():
    c = compose(str, _inc)
    assert c.__name__ == "_of_".join(
        f.__name__ for f in reversed((c.first,) + c.funcs)
    )


def test_compose_repr():
    c = compose(str, _inc)
    r = repr(c)
    assert r.startswith("Compose")
    assert "str" in r
    assert "_inc" in r


def test_compose_eq_hash():
    c1 = compose(str, _inc)
    c2 = compose(str, _inc)
    c3 = compose(str, _double)
    assert c1 == c2
    assert c1 != c3
    assert c1 != 5
    assert not (c1 != c2)
    # NotImplemented propagated -> __ne__ returns NotImplemented, Python falls back
    assert (c1 != 5) is True
    assert hash(c1) == hash(c2)


def test_compose_signature():
    def f(a: int, b: int) -> int:
        return a + b

    def g(x: int) -> str:
        return str(x)

    c = compose(g, f)
    sig = c.__signature__
    assert list(sig.parameters) == ["a", "b"]
    assert sig.return_annotation is str


def test_compose_wrapped():
    def f(x):
        return x

    c = compose(str, f)
    assert c.__wrapped__ is f


def test_compose_pickle():
    c = compose(str, _inc)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(3) == "4"


def test_compose_descriptor():
    class A:
        def __init__(self, factor):
            self.factor = factor

        def method(self, x):
            return self.factor * x

        # Wrap a standalone function that is applied after the method.
        op = compose(_inc, method)

    a = A(2)
    # a.op(3) -> _inc(method(a, 3)) = _inc(6) = 7
    assert a.op(3) == 7


def test_compose_annotations():
    def f(a: int, b: str) -> float:
        return 1.0

    def g(x: float) -> bool:
        return True

    c = compose(g, f)
    ann = c.__annotations__
    assert ann["a"] is int
    assert ann["b"] is str
    assert ann["return"] is bool


def test_compose_annotations_no_return():
    def f(a: int):
        return a

    def g(x):
        return x

    c = compose(g, f)
    ann = c.__annotations__
    assert ann["a"] is int
    assert "return" not in ann


def test_compose_annotations_builtin():
    c = compose(len)
    # len has no annotations.
    assert c.__annotations__ == {}


def test_compose_annotations_nested_compose():
    def f(a: int) -> int:
        return a

    def g(x: int) -> str:
        return str(x)

    def h(y: str) -> bool:
        return True

    inner = compose(g, f)
    outer = compose(h, inner)
    ann = outer.__annotations__
    assert ann["a"] is int
    assert ann["return"] is bool


def test_compose_annotations_class_access():
    c = compose(str, _inc)
    # Accessing __annotations__ on the class returns empty dict-ish descriptor
    # Accessing on instance returns real dict.
    assert isinstance(c.__annotations__, dict)


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe_basic():
    assert pipe(3, _double, str) == "6"


def test_pipe_empty():
    assert pipe(5) == 5


def test_pipe_chain():
    assert pipe(1, _inc, _double, _inc) == 5


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
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_iterable():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(x, y=0):
        return x + y

    def g(x, y=0):
        return x - y

    assert juxt(f, g)(10, y=3) == (13, 7)


def test_juxt_slots():
    j = juxt(_inc, _double)
    assert j.funcs == (_inc, _double)
    with pytest.raises(AttributeError):
        j.other = 1


def test_juxt_pickle():
    j = juxt(_inc, _double)
    restored = pickle.loads(pickle.dumps(j))
    assert restored(5) == (6, 10)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    result = do(log.append, 42)
    assert result == 42
    assert log == [42]


def test_do_compose():
    log = []
    inc = compose(_inc, do(log.append))
    assert inc(1) == 2
    assert inc(11) == 12
    assert log == [1, 11]


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------

def test_flip_basic():
    assert flip(_div, 2, 6) == 3


def test_flip_curried():
    div_by_two = flip(_div, 2)
    assert div_by_two(4) == 2


def test_flip_isinstance():
    data = [1, "a", "b", 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError("x")) is None


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


def test_excepts_name():
    def f(x):
        "f doc"
        return x

    e = excepts(ValueError, f)
    assert e.__name__ == "f_excepting_ValueError"
    assert "f doc" in e.__doc__


def test_excepts_name_tuple():
    def f(x):
        return x

    e = excepts((IndexError, KeyError), f)
    assert e.__name__ == "f_excepting_IndexError_or_KeyError"


def test_excepts_name_no_name():
    obj = excepts(ValueError, lambda x: x)
    assert obj.__name__.startswith("<lambda>")


def test_excepts_class_doc():
    # Class-level __doc__ returns the class docstring.
    assert "A wrapper around" in excepts.__doc__


def test_excepts_doc_no_attr():
    obj = excepts(ValueError, lambda x: x)
    # lambdas have __name__; handler return_none has __name__ too.
    doc = obj.__doc__
    assert "A wrapper around" in doc


# ---------------------------------------------------------------------------
# Additional checks: signature-related branches
# ---------------------------------------------------------------------------

def test_is_valid_args_sigspec():
    def add(x, y):
        return x + y

    sig = inspect.signature(add)
    assert is_valid_args(add, (1,), {}, sig) is False
    assert is_valid_args(add, (1, 2), {}, sig) is True


def test_is_partial_args_sigspec():
    def add(x, y):
        return x + y

    sig = inspect.signature(add)
    assert is_partial_args(add, (1,), {}, sig) is True
    assert is_partial_args(add, (1, 2, 3), {}, sig) is False


def test_num_required_args_sigspec():
    def f(x, y, z=3):
        return x + y + z

    sig = inspect.signature(f)
    assert num_required_args(f, sig) == 2


def test_has_varargs_sigspec():
    def f(*args):
        return args

    sig = inspect.signature(f)
    assert has_varargs(f, sig) is True


def test_has_keywords_sigspec():
    def f(x, y=0):
        return x + y

    sig = inspect.signature(f)
    assert has_keywords(f, sig) is True


def test_is_arity_sigspec():
    def f(x):
        return x

    sig = inspect.signature(f)
    assert is_arity(1, f, sig) is True


def test_curry_call_valid_but_typeerror_in_body():
    def f(x, y):
        raise TypeError("genuine")

    c = curry(f)
    with pytest.raises(TypeError, match="genuine"):
        c(1, 2)


def test_curry_should_curry_false_then_raise():
    def add(a, b):
        return a + b

    c = curry(add, 1)
    with pytest.raises(TypeError):
        c(2, 3, 4)


def test_curry_call_keyword_then_curry():
    def f(a, b, c):
        return a + b + c

    c = curry(f, a=1)
    assert c(b=2)(c=3) == 6


def test_compose_single_func_identity():
    assert compose_left(_inc) is _inc


def test_compose_name_no_name():
    class Callable:
        def __call__(self, x):
            return x

    c = compose(Callable(), str)
    r = repr(c)
    assert "Callable" in r


def test_juxt_empty_call():
    j = juxt()
    assert j() == ()
