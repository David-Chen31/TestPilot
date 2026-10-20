import inspect
import pickle
import sys
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
# Helpers
# ---------------------------------------------------------------------------

def double(x):
    return 2 * x


def inc(x):
    return x + 1


def add(x, y):
    return x + y


def mul(x, y):
    return x * y


# ---------------------------------------------------------------------------
# identity / apply
# ---------------------------------------------------------------------------

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3


def test_apply():
    assert apply(double, 5) == 10
    assert apply(add, 1, 2) == 3
    assert apply(add, 1, y=2) == 3
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)


def test_apply_no_args_raises():
    with pytest.raises(TypeError):
        apply()


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_plain_funcs():
    assert thread_first(1, inc, double) == 4
    assert thread_first(1) == 1  # no forms -> identity


def test_thread_first_tuple_forms():
    def power(x, y):
        return x ** y
    assert thread_first(1, (add, 4), (power, 2)) == 25


def test_thread_last_plain_funcs():
    assert thread_last(1, inc, double) == 4
    assert thread_last(1) == 1


def test_thread_last_tuple_forms():
    def power(x, y):
        return x ** y
    # pow(2, add(4, 1)) = pow(2, 5) = 32
    assert thread_last(1, (add, 4), (power, 2)) == 32


def test_thread_last_with_map_filter():
    def iseven(x):
        return x % 2 == 0
    result = list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
    assert result == [2, 4]


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


def test_instanceproperty_setter_deleter():
    class C:
        def __init__(self):
            self._x = 0

        def _get(self):
            return self._x

        def _set(self, v):
            self._x = v

        def _del(self):
            self._x = -1

        x = instanceproperty(_get, _set, _del, classval='cls')

    assert C.x == 'cls'
    c = C()
    c.x = 5
    assert c.x == 5
    del c.x
    assert c.x == -1


def test_instanceproperty_returns_partial_when_no_fget():
    p = instanceproperty(fset=None, fdel=None, doc='d', classval=1)
    assert isinstance(p, partial)
    assert p.args == (instanceproperty,)
    assert p.keywords['classval'] == 1


def test_instanceproperty_is_property_subclass():
    prop = instanceproperty(lambda self: 1)
    assert isinstance(prop, InstanceProperty)
    assert isinstance(prop, property)


def test_instanceproperty_pickle_roundtrip():
    class C:
        """doc"""
        val = instanceproperty(lambda self: 1, classval='cv')

    p = C.__dict__['val']
    restored = pickle.loads(pickle.dumps(p))
    assert restored.classval == 'cv'
    assert restored.fget is p.fget


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic_partial_application():
    cmul = curry(mul)
    doubled = cmul(2)
    assert doubled(10) == 20


def test_curry_as_decorator():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    g = f(a=1)
    assert g(2, 3) == 5


def test_curry_no_args_raises():
    with pytest.raises(TypeError):
        curry()


def test_curry_non_callable_raises():
    with pytest.raises(TypeError):
        curry(42)


def test_curry_merges_partial_like():
    c = curry(mul, 2)
    c2 = curry(c, 3)
    assert c2() == 6
    assert c2.func is mul
    assert c2.args == (2, 3)


def test_curry_preserves_name_doc_module():
    def myfunc(x):
        """my doc"""
        return x

    c = curry(myfunc)
    assert c.__name__ == 'myfunc'
    assert c.__doc__ == 'my doc'
    assert c.func_name == 'myfunc'
    assert c.__module__ == myfunc.__module__


def test_curry_str_repr():
    c = curry(mul)
    assert str(c) == str(mul)
    assert repr(c) == repr(mul)


def test_curry_args_keywords():
    c = curry(mul, 2, y=3)
    assert c.args == (2,)
    assert c.keywords == {'y': 3}


def test_curry_equality_and_hash():
    a = curry(mul, 2)
    b = curry(mul, 2)
    c = curry(mul, 3)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert a != mul  # non-curry


def test_curry_ne_other_type():
    assert curry(mul) != object()


def test_curry_call_completes():
    c = curry(add)
    assert c(1, 2) == 3


def test_curry_genuine_typeerror_raised():
    @curry
    def f(x):
        return x + 1

    with pytest.raises(TypeError):
        f(1, 2, 3, 4)  # too many args


def test_curry_bind_and_call():
    c = curry(add)
    bound = c.bind(1)
    assert isinstance(bound, curry)
    assert bound.args == (1,)
    assert bound.call(2) == 3
    assert bound(2) == 3


def test_curry_signature():
    c = curry(mul, 2)
    sig = c.__signature__
    params = list(sig.parameters)
    assert params == ['y']


def test_curry_signature_incorrect_args():
    c = curry(mul, 2, 3, 4)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_descriptor():
    class C:
        @curry
        def method(self, x):
            return self.value + x

        def __init__(self, value):
            self.value = value

    # accessing on class returns the curry itself
    assert isinstance(C.method, curry)
    inst = C(10)
    bound = inst.method
    assert isinstance(bound, curry)
    assert bound(5) == 15


def test_curry_pickle():
    @curry
    def mod_level(x, y):
        return x + y

    # module-level curried functions can be pickled
    c = curry(add, 1)
    restored = pickle.loads(pickle.dumps(c))
    assert restored.args == (1,)
    assert restored(2) == 3


def test_curry_varargs_curries_on_typeerror():
    @curry
    def f(*args, **kwargs):
        if len(args) < 2:
            raise TypeError('need more')
        return args

    # Should curry rather than raise since there are *args
    result = f(1)
    assert isinstance(result, curry)
    assert result(2) == (1, 2)


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]


def test_memoize_with_initial_cache():
    @memoize(cache={(1, 2): 3})
    def add2(x, y):
        return x + y

    assert add2(1, 2) == 3
    assert add2(5, 5) == 10


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def f(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    assert f(1, 2) == 3
    # different kwargs but same args should hit cache
    assert f(1, 2, verbose=True) == 3
    assert len(calls) == 1


def test_memoize_unhashable_raises():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_preserves_doc_and_wrapped():
    @memoize
    def f(x):
        """docstring"""
        return x

    assert f.__doc__ == 'docstring'
    assert f.__wrapped__ is not None
    assert f.__name__ == 'f'


def test_memoize_variadic():
    calls = []

    @memoize
    def f(*args):
        calls.append(args)
        return sum(args)

    assert f(1, 2) == 3
    assert f(1, 2) == 3
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Compose / compose / compose_left
# ---------------------------------------------------------------------------

def test_compose_empty():
    assert compose() is identity


def test_compose_single():
    assert compose(inc) is inc


def test_compose_order():
    assert compose(str, inc)(3) == '4'


def test_compose_left():
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_empty():
    assert compose_left() is identity


def test_compose_call_with_kwargs():
    def f(a, b):
        return a + b
    assert compose(double, f)(1, b=2) == 6


def test_compose_name_doc_repr():
    c = compose(str, inc)
    assert c.__name__ == 'str_of_inc'
    assert 'lambda *args, **kwargs' in c.__doc__
    assert repr(c).startswith('Compose')


def test_compose_doc_without_names():
    c = compose(str, [].append)
    # builtin methods might not have __name__ -> fallback docstring
    assert c.__doc__ is not None


def test_compose_equality():
    c1 = compose(str, inc)
    c2 = compose(str, inc)
    c3 = compose(str, double)
    assert c1 == c2
    assert c1 != c3
    assert (c1 == 42) is False


def test_compose_hash():
    c1 = compose(str, inc)
    c2 = compose(str, inc)
    assert hash(c1) == hash(c2)


def test_compose_pickle():
    c = compose(str, inc)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(3) == '4'
    assert restored == c


def test_compose_signature():
    def f(x: int) -> int:
        return x

    def g(x: int) -> str:
        return str(x)

    c = compose(g, f)
    sig = c.__signature__
    assert sig.return_annotation is str


def test_compose_wrapped():
    c = compose(str, inc)
    assert c.__wrapped__ is inc


def test_compose_annotations():
    def f(x: int, y: int) -> int:
        return x + y

    def g(x: int) -> str:
        return str(x)

    c = compose(g, f)
    ann = c.__annotations__
    assert ann['x'] is int
    assert ann['y'] is int
    assert ann['return'] is str


def test_compose_annotations_no_signature():
    def f(x):
        return x

    c = compose(str, f)
    # f has no annotations -> empty (but str may contribute)
    ann = c.__annotations__
    assert isinstance(ann, dict)


def test_compose_as_method():
    class C:
        def __init__(self, v):
            self.v = v

        f = compose(inc, double)

    c = C(10)
    assert c.f() == 21


def test_compose_descriptor_class_access():
    c = compose(inc, double)
    # On class access within own __get__ - just check __get__(None) works
    sentinel = object()
    assert c.__get__(None, sentinel) is c


def test_compose_char():
    # compositional path through Compose._combined_annotations
    c = compose(compose(str, inc), double)
    assert c.__annotations__ is not None


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe():
    assert pipe(3, double) == 6
    assert pipe(3, double, inc) == 7
    assert pipe(3) == 3


def test_pipe_str():
    assert pipe(3, double, str) == '6'


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------

def test_complement():
    def iseven(n):
        return n % 2 == 0
    isodd = complement(iseven)
    assert isodd(2) is False
    assert isodd(3) is True


# ---------------------------------------------------------------------------
# juxt
# ---------------------------------------------------------------------------

def test_juxt_call():
    j = juxt(inc, double)
    assert j(10) == (11, 20)


def test_juxt_list():
    j = juxt([inc, double])
    assert j(10) == (11, 20)


def test_juxt_kwargs():
    def f(a, b):
        return a - b
    def g(a, b):
        return a + b
    j = juxt(f, g)
    assert j(5, b=2) == (3, 7)


def test_juxt_pickle():
    j = juxt(inc, double)
    restored = pickle.loads(pickle.dumps(j))
    assert restored(5) == (6, 10)


def test_juxt_slots():
    j = juxt(inc, double)
    assert j.funcs == (inc, double)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    result = do(log.append, 1)
    assert result == 1
    assert log == [1]


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


def test_flip_builtin_isinstance():
    data = [1, 'a', 'b', 2, 1.5, 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(Exception()) is None


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
    e = excepts(ValueError, int)
    assert e.__name__ == 'int_excepting_ValueError'


def test_excepts_name_tuple():
    e = excepts((ValueError, KeyError), int)
    assert e.__name__ == 'int_excepting_ValueError_or_KeyError'


def test_excepts_name_no_func_name():
    class Callable:
        def __call__(self, x):
            return x
    e = excepts(ValueError, Callable())
    assert e.__name__ == 'excepting'


def test_excepts_doc_class():
    # Accessed on the class is class documentation
    assert excepts.__doc__ is not None


def test_excepts_instance_doc():
    e = excepts(ValueError, lambda a: a, lambda exc: -1)
    doc = e.__doc__
    assert 'ValueError' in doc


# ---------------------------------------------------------------------------
# introspection: num_required_args, has_varargs, has_keywords
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2


def test_num_required_args_varargs():
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
    assert has_keywords(f) is True


def test_has_keywords_var_keyword():
    def f(**kwargs):
        return kwargs
    assert has_keywords(f) is True


def test_has_keywords_none():
    def f(x, y):
        return x
    assert has_keywords(f) is False


# ---------------------------------------------------------------------------
# is_valid_args / is_partial_args / is_arity
# ---------------------------------------------------------------------------

def test_is_valid_args():
    def add2(x, y):
        return x + y

    assert is_valid_args(add2, (1,), {}) is False
    assert is_valid_args(add2, (1, 2), {}) is True
    assert is_valid_args(map, (), {}) is False


def test_is_partial_args():
    def add2(x, y):
        return x + y

    assert is_partial_args(add2, (1,), {}) is True
    assert is_partial_args(add2, (1, 2), {}) is True
    assert is_partial_args(add2, (1, 2, 3), {}) is False
    assert is_partial_args(map, (), {}) is True


def test_is_arity():
    def f(x):
        return x

    def g(x, y=1):
        return x + y

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False
    assert is_arity(2, g) is False  # y has a default so 1 required is wrong? check


def test_is_arity_varargs_false():
    def f(*args):
        return args
    assert is_arity(1, f) is False


def test_is_arity_keywords_false():
    def f(x, **kwargs):
        return x
    assert is_arity(1, f) is False


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------

def test_curry_signature_with_kwonly():
    @curry
    def f(a, b, *, c=1):
        return a + b + c

    c = f(1)
    sig = c.__signature__
    assert list(sig.parameters) == ['b', 'c']


def test_curry_restore_pickle_func_obj():
    c = curry(add, 1, 2)
    restored = pickle.loads(pickle.dumps(c))
    assert restored() == 3


def test_compose_annotations_compose_in_compose():
    def f(x: int, y: int) -> int:
        return x + y
    def g(x: int) -> str:
        return str(x)
    inner = compose(g, f)
    outer = compose(inner, double)
    ann = outer.__annotations__
    assert ann['x'] is int
    assert ann['return'] is str
