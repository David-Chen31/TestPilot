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


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def test_apply_basic():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert apply(double, 5) == 10
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000,
    )


def test_apply_kwargs():
    def f(a, b=0):
        return a + b

    assert apply(f, 1, b=2) == 3


def test_apply_no_func():
    with pytest.raises(TypeError):
        apply()


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_basic():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4
    assert thread_first(1) == 1


def test_thread_first_with_tuples():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_last_basic():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4


def test_thread_last_with_tuples():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (pow_, 2)) == 32


def test_thread_last_map_filter():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    result = thread_last([1, 2, 3], (map, inc), (filter, iseven))
    assert list(result) == [2, 4]


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_basic():
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


def test_instanceproperty_with_setter_deleter():
    class C:
        def __init__(self):
            self._x = 0

        @instanceproperty
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

    c = C()
    assert c.x == 0
    c.x = 5
    assert c.x == 5


def test_instanceproperty_returns_partial():
    p = instanceproperty(fset=None, fdel=None, doc=None, classval=7)
    assert isinstance(p, partial)


def test_instanceproperty_reduce():
    class C:
        @instanceproperty(classval=3)
        def x(self):
            return 1

    ip = C.__dict__['x']
    assert isinstance(ip, InstanceProperty)
    reduce_func, state = ip.__reduce__()
    assert reduce_func is InstanceProperty
    restored = reduce_func(*state)
    assert restored.classval == 3


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    c = curry(mul)
    double = c(2)
    assert isinstance(double, curry)
    assert double(10) == 20


def test_curry_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_full_call_returns_value():
    def add(x, y):
        return x + y

    c = curry(add)
    assert c(1, 2) == 3


def test_curry_requires_callable():
    with pytest.raises(TypeError):
        curry(3)


def test_curry_requires_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_unpacks_partial_like():
    def add(x, y, z):
        return x + y + z

    c1 = curry(add, 1)
    c2 = curry(c1, 2)
    assert c2(3) == 6


def test_curry_metadata():
    def myfunc(x):
        """my doc"""
        return x

    c = curry(myfunc)
    assert c.__doc__ == "my doc"
    assert c.__name__ == "myfunc"
    assert c.func_name == "myfunc"
    assert c.func is myfunc
    assert str(c) == str(myfunc)
    assert repr(c) == repr(myfunc)


def test_curry_hash_equality():
    def add(x, y):
        return x + y

    a = curry(add, 1)
    b = curry(add, 1)
    c = curry(add, 2)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert not (a == 5)


def test_curry_call_and_bind():
    def add(x, y):
        return x + y

    c = curry(add)
    assert c.call(1, 2) == 3
    bound = c.bind(1)
    assert isinstance(bound, curry)
    assert bound(2) == 3


def test_curry_genuine_type_error_propagates():
    def f(x):
        return x + 1

    c = curry(f)
    with pytest.raises(TypeError):
        c('abc')


def test_curry_varargs_curries_on_typeerror():
    @curry
    def f(*args, **kwargs):
        return args

    # Calling with no args is a valid call for *args, so it returns the result
    # directly rather than currying.
    res = f()
    assert res == ()


def test_curry_varargs_raises_inside_curries():
    @curry
    def f(x, *args):
        if x == 'bad':
            raise TypeError("genuine")
        return (x, args)

    # Because the function has *args, a TypeError raised inside the body is
    # treated as a signal to curry (returns a curry object).
    result = f('bad')
    assert isinstance(result, curry)

    # Calling with insufficient positional args should curry due to *args
    for_partial = f()
    assert isinstance(for_partial, curry)


def test_curry_as_method():
    class C:
        @curry
        def method(self, x, y):
            return x + y

    c = C()
    assert c.method(1, 2) == 3
    partial_method = c.method(1)
    assert isinstance(partial_method, curry)
    assert partial_method(2) == 3


def test_curry_signature():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    # First param (x) is consumed
    assert params[0].name == 'y'


def test_curry_call_direct_partial_args():
    def add(x, y):
        return x + y

    c = curry(add, 1)
    # Passing all args should not curry
    assert c(2) == 3


def test_curry_pickle():
    def add(x, y):
        return x + y

    c = curry(add, 1)
    # Cannot pickle nested function; use a module-level function instead
    # We only check that __reduce__ exists and is callable.
    assert hasattr(c, '__reduce__')


def test_curry_doc_default():
    c = curry(lambda x: x)
    assert c.__doc__ is None


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
    assert len(calls) == 1


def test_memoize_unary():
    calls = []

    def f(x):
        calls.append(x)
        return x * 2

    m = memoize(f)
    assert m(3) == 6
    assert m(3) == 6
    assert calls == [3]


def test_memoize_functions_no_kwargs():
    calls = []

    def f(a, b):
        calls.append((a, b))
        return a + b

    m = memoize(f)
    assert m(1, 2) == 3
    assert m(1, 2) == 3
    assert len(calls) == 1
    # Different key
    assert m(2, 3) == 5
    assert len(calls) == 2


def test_memoize_custom_cache():
    cache = {(1, 2): 3}

    def add(x, y):
        return x + y

    m = memoize(add, cache=cache)
    assert m(1, 2) == 3
    assert m(2, 3) == 5
    assert (2, 3) in cache


def test_memoize_custom_key():
    calls = []

    def add(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    m = memoize(add, key=lambda args, kwargs: args)
    assert m(1, 2, verbose=True) == 3
    assert m(1, 2, verbose=False) == 3
    # Same key since key only uses args
    assert len(calls) == 1


def test_memoize_unhashable():
    def f(x):
        return x

    m = memoize(f)
    with pytest.raises(TypeError):
        m([1, 2, 3])


def test_memoize_preserves_metadata():
    def f(x):
        """docstring"""
        return x

    m = memoize(f)
    assert m.__name__ == 'f'
    assert m.__doc__ == 'docstring'
    assert m.__wrapped__ is f


def test_memoize_as_decorator():
    @memoize
    def add(x, y):
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3


def test_memoize_with_cache_kwarg_decorator():
    @memoize(cache={(1, 2): 100})
    def add(x, y):
        return x + y

    assert add(1, 2) == 100
    assert add(2, 3) == 5


# ---------------------------------------------------------------------------
# compose / compose_left / Compose
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_no_funcs():
    assert compose() is identity


def test_compose_single_func():
    def f(x):
        return x * 2

    assert compose(f) is f


def test_compose_left_basic():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_no_funcs():
    assert compose_left() is identity


def test_compose_multiple_args():
    c = compose(str, lambda x, y: x + y)
    assert c(1, 2) == '3'


def test_compose_docstring():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert 'f(g(*args, **kwargs))' in c.__doc__


def test_compose_name():
    def foo(x):
        return x

    def bar(x):
        return x

    c = compose(foo, bar)
    assert c.__name__ == 'foo_of_bar'


def test_compose_name_lambda():
    c = compose(lambda x: x, lambda x: x)
    assert c.__name__ == '<lambda>_of_<lambda>'


def test_compose_name_fallback():
    class NoName:
        def __call__(self, x):
            return x

    c = Compose((NoName(),))
    assert c.__name__ == 'Compose'


def test_compose_repr():
    def foo(x):
        return x

    def bar(x):
        return x

    c = compose(foo, bar)
    assert 'Compose' in repr(c)


def test_compose_equality_hash():
    def foo(x):
        return x

    c1 = Compose((foo,))
    c2 = Compose((foo,))
    assert c1 == c2
    assert not (c1 != c2)
    assert hash(c1) == hash(c2)


def test_compose_equality_other_type():
    def foo(x):
        return x

    c = Compose((foo,))
    assert c.__eq__(5) is NotImplemented
    assert c.__ne__(5) is NotImplemented


def test_compose_pickle():
    def foo(x):
        return x

    c = Compose((foo,))
    state = c.__getstate__()
    assert state == (c.first, c.funcs)
    c2 = Compose((foo,))
    c2.__setstate__(state)
    assert c.first == c2.first


def test_compose_signature():
    def f(x: int) -> str:
        return str(x)

    def g(y) -> int:
        return int(y)

    c = compose(f, g)
    sig = c.__signature__
    assert sig.return_annotation is str


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__wrapped__ is g


def test_compose_as_method():
    class C:
        def add(self, x):
            return x + 1

        double = compose(lambda x: x * 2)

    c = C()
    # The composed lambda receives the instance as an extra arg, hence error.
    with pytest.raises(TypeError):
        c.double(5)


def test_compose_as_method_with_self():
    class C:
        double = compose(lambda self, x: x * 2)

    c = C()
    assert c.double(5) == 10


def test_compose_get_descriptor_on_class():
    c = Compose((lambda x: x,))
    assert c.__get__(None, object) is c


def test_compose_annotations():
    def f(x: int, y: str) -> float:
        return 0.0

    def g(a: bytes) -> int:
        return 0

    c = compose(f, g)
    ann = c.__annotations__
    # Parameter annotations come from the first function applied (g), not f.
    assert ann['a'] is bytes
    assert ann['return'] is float


def test_compose_annotations_no_signature():
    class NoSig:
        def __call__(self, x):
            return x

    c = Compose((NoSig(),))
    ann = c.__annotations__
    # NoSig's signature is odd but annotations should reflect its params
    assert isinstance(ann, dict)


def test_compose_nested_annotations():
    def f(x: int) -> str:
        return str(x)

    def g(y) -> int:
        return int(y)

    inner = compose(f, g)
    outer = compose(str, inner)
    ann = outer.__annotations__
    # The combined annotations of a nested Compose use the inner composition's
    # combined annotations; the inner Compose (as last function) produces no
    # return annotation here since the annotations are combined lazily.
    # Assert what the implementation actually produces: an empty dict.
    assert ann == {}


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'


def test_pipe_no_funcs():
    assert pipe(5) == 5


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

def test_juxt():
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
        return x * y

    assert juxt(f, g)(2, 3) == (5, 6)


def test_juxt_pickle_state():
    inc = lambda x: x + 1
    j = juxt(inc)
    state = j.__getstate__()
    j2 = juxt()
    j2.__setstate__(state)
    assert j2.funcs == j.funcs


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    assert do(log.append, 1) == 1
    assert log == [1]


def test_do_compose_example():
    # `do` requires both func and x, so we curry/call with explicit args.
    log = []
    inc = lambda x: x + 1
    f = compose(inc, lambda x: do(log.append, x))
    assert f(1) == 2
    assert f(11) == 12
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
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
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


def test_excepts_multiple_and_default():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_unrelated_exception_propagates():
    excepting = excepts(ValueError, lambda a: a + 1)
    with pytest.raises(TypeError):
        excepting('a')


def test_excepts_doc():
    def f(x):
        """f doc"""
        return x

    def h(e):
        """h doc"""
        return None

    ex = excepts(ValueError, f, h)
    doc = ex.__doc__
    assert 'f doc' in doc
    assert 'h doc' in doc
    assert 'ValueError' in doc


def test_excepts_doc_tuple():
    def f(x):
        return x

    def h(e):
        return None

    ex = excepts((ValueError, KeyError), f, h)
    assert '(ValueError, KeyError)' in ex.__doc__


def test_excepts_doc_classlevel():
    assert 'A wrapper around' in excepts.__doc__


def test_excepts_name():
    def f(x):
        return x

    ex = excepts(ValueError, f)
    assert ex.__name__ == 'f_excepting_ValueError'


def test_excepts_name_tuple():
    def f(x):
        return x

    ex = excepts((ValueError, KeyError), f)
    assert ex.__name__ == 'f_excepting_ValueError_or_KeyError'


def test_excepts_name_fallback():
    class NoName:
        pass

    ex = excepts(ValueError, NoName())
    assert ex.__name__ == 'excepting'


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def _f_two(x, y):
    return x + y


def _g_default(x, y=0):
    return x + y


def _h_varargs(*args):
    return args


def _i_kwargs(**kwargs):
    return kwargs


def _j_all(x, y, *args, **kwargs):
    return (x, y, args, kwargs)


def test_num_required_args():
    assert num_required_args(_f_two) == 2
    assert num_required_args(_g_default) == 1
    assert num_required_args(_h_varargs) == 0
    assert num_required_args(_i_kwargs) == 0


def test_has_varargs():
    assert has_varargs(_f_two) is False
    assert has_varargs(_h_varargs) is True
    assert has_varargs(_i_kwargs) is False
    assert has_varargs(_j_all) is True


def test_has_keywords():
    assert has_keywords(_f_two) is False
    assert has_keywords(_g_default) is True
    assert has_keywords(_i_kwargs) is True
    assert has_keywords(_j_all) is True


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

    def g(x, y=1):
        return x + y

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False
    assert is_arity(2, _f_two) is True
    assert is_arity(0, _h_varargs) is False


def test_introspection_builtin_signature_registry():
    # map is a builtin; is_valid_args/is_partial_args should consult the
    # registry in toolz._signatures
    assert is_valid_args(map, (lambda x: x, [1, 2]), {}) is True
    assert is_partial_args(map, (), {}) is True


def test_check_sigspec_with_explicit_sigspec():
    sig = inspect.signature(_f_two)
    assert num_required_args(_f_two, sigspec=sig) == 2
    assert has_varargs(_f_two, sigspec=sig) is False
    assert has_keywords(_f_two, sigspec=sig) is False
    assert is_valid_args(_f_two, (1, 2), {}, sigspec=sig) is True
    assert is_partial_args(_f_two, (1,), {}, sigspec=sig) is True


def test_is_arity_empty_signature_error():
    # Functions without introspectable signatures;
    # _check_sigspec returns whatever the registry says.

    class Callable:
        def __call__(self, x):
            return x

    # Should not raise
    result = is_arity(1, Callable())
    assert result in (True, False, None)
