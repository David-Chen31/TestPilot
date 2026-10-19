import pickle
import sys
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


# ---------------------------------------------------------------- helpers

def add(x, y):
    return x + y


def double(x):
    return 2 * x


def inc(x):
    return x + 1


def mul(x, y):
    return x * y


def default_f(x, y, a=10):
    return a * (x + y)


def varargs_f(*args):
    return args


def kwargs_f(**kwargs):
    return kwargs


# ---------------------------------------------------------------- identity

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3
    assert identity(None) is None


# ---------------------------------------------------------------- apply

def test_apply_basic():
    assert apply(double, 5) == 10
    assert apply(add, 1, 2) == 3


def test_apply_kwargs():
    assert apply(default_f, 1, 2, a=3) == 9


def test_apply_map_usage():
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)


def test_apply_no_args_raises():
    with pytest.raises(TypeError):
        apply()


# ---------------------------------------------------------------- threading

def test_thread_first():
    assert thread_first(1, inc, double) == 4
    assert thread_first(1, (add, 4), (double,)) == 10


def test_thread_first_multiple_args():
    def pow_(x, y):
        return x ** y
    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_first_empty():
    assert thread_first(5) == 5


def test_thread_last():
    assert thread_last(1, inc, double) == 4
    assert thread_last(1, (add, 4)) == 5


def test_thread_last_multiple_args():
    def pow_(x, y):
        return x ** y
    assert thread_last(1, (add, 4), (pow_, 2)) == 32


def test_thread_last_map_filter():
    def iseven(x):
        return x % 2 == 0
    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


def test_thread_last_empty():
    assert thread_last(5) == 5


# ---------------------------------------------------------------- instanceproperty

def test_instanceproperty_classval():
    class MyClass:
        '''The class docstring'''

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


def test_instanceproperty_no_fget_returns_partial():
    from functools import partial
    result = instanceproperty()
    assert isinstance(result, partial)


def test_instance_property_is_property():
    class C:
        @instanceproperty
        def x(self):
            return 1
    assert isinstance(C.x, InstanceProperty)
    assert isinstance(C.x, property)


def test_instance_property_setter():
    class C:
        def __init__(self):
            self._v = 0

        @instanceproperty
        def v(self):
            return self._v

        @v.setter
        def v(self, val):
            self._v = val

    c = C()
    assert c.v == 0
    c.v = 5
    assert c.v == 5


# ---------------------------------------------------------------- curry

def test_curry_simple():
    curried = curry(mul)
    double_ = curried(2)
    assert double_(10) == 20
    assert mul(2, 10) == 20


def test_curry_kwargs():
    f = curry(default_f)
    add_ = f(a=1)
    assert add_(2, 3) == 5


def test_curry_decorator():
    @curry
    def f(x, y):
        return x + y

    assert f(1)(2) == 3
    assert f(1, 2) == 3


def test_curry_no_args_raises():
    with pytest.raises(TypeError):
        curry()


def test_curry_not_callable_raises():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_func_and_args():
    c = curry(add, 1)
    assert c.func is add
    assert c.args == (1,)
    assert c.keywords == {}


def test_curry_equality_and_hash():
    c1 = curry(add, 1)
    c2 = curry(add, 1)
    assert c1 == c2
    assert hash(c1) == hash(c2)
    assert c1 != curry(add, 2)
    assert c1 != curry(mul, 1)


def test_curry_str_repr():
    c = curry(add, 1)
    assert str(c) == str(add)
    assert repr(c) == repr(add)


def test_curry_get_descriptor():
    class C:
        @curry
        def method(self, x):
            return x + 1

    c = C()
    assert C.method is C.method  # class access returns curry
    bound = C.method
    assert isinstance(bound, curry)


def test_curry_call_explicit():
    c = curry(add, 1)
    assert c.call(2) == 3


def test_curry_bind():
    c = curry(add)
    bound = c.bind(1)
    assert isinstance(bound, curry)
    assert bound(2) == 3


def test_curry_genuine_type_error_propagates():
    @curry
    def f(x):
        raise TypeError("boom")

    with pytest.raises(TypeError, match="boom"):
        f(1)


def test_curry_varargs_typeerror_curries():
    @curry
    def f(*args, **kwargs):
        # raises TypeError but has *args so it should curry
        raise TypeError("inner")

    # has varargs -> returns a curry object rather than raising
    result = f(1)
    assert isinstance(result, curry)


def test_curry_too_many_args_raises():
    @curry
    def f(x):
        return x

    with pytest.raises(TypeError):
        f(1, 2)


def test_curry_signature():
    import inspect
    c = curry(add, 1)
    sig = c.__signature__
    assert 'y' in sig.parameters
    assert 'x' not in sig.parameters


def test_curry_signature_invalid_args():
    import inspect
    c = curry(add, 1, 2, 3)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_unpack_partial():
    from functools import partial
    p = partial(add, 1)
    c = curry(p, 2)
    assert c.func is add
    assert c.args == (1, 2)
    assert c() == 3


def test_curry_pickle_roundtrip():
    c = curry(mul, 2)
    data = pickle.dumps(c)
    restored = pickle.loads(data)
    assert restored(10) == 20


def test_curry_func_name():
    c = curry(add)
    assert c.func_name == 'add'


# ---------------------------------------------------------------- memoize

def test_memoize_counting():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(1) == 2
    assert f(1) == 2
    assert f(2) == 4
    assert calls == [1, 2]


def test_memoize_with_cache():
    @memoize(cache={(1, 2): 3})
    def add_(x, y):
        return x + y

    assert add_(1, 2) == 3


def test_memoize_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def f(x, y, verbose=False):
        calls.append((x, y))
        return x + y

    assert f(1, 2, verbose=True) == 3
    assert f(1, 2, verbose=False) == 3
    assert calls == [(1, 2)]


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_preserves_name_and_doc():
    def f(x):
        """docstring"""
        return x

    m = memoize(f)
    assert m.__name__ == 'f'
    assert m.__doc__ == 'docstring'
    assert m.__wrapped__ is f


# ---------------------------------------------------------------- Compose / compose

def test_compose_basic():
    assert compose(str, inc)(3) == '4'


def test_compose_no_args():
    assert compose() is identity


def test_compose_single():
    assert compose(inc) is inc


def test_compose_left():
    assert compose_left(inc, str)(3) == '4'


def test_compose_multi_args():
    c = compose(double, add)
    assert c(1, 2) == 6


def test_compose_object():
    c = compose(str, inc)
    assert isinstance(c, Compose)
    assert c(3) == '4'


def test_compose_doc():
    c = compose(str, inc)
    assert 'lambda' in c.__doc__
    assert 'str(inc(*args, **kwargs))' in c.__doc__


def test_compose_doc_no_names():
    import functools
    f = functools.partial(int)
    c = compose(f, inc)
    assert c.__doc__ == 'A composition of functions'


def test_compose_name():
    c = compose(str, inc)
    assert c.__name__ == 'str_of_inc'


def test_compose_repr():
    c = compose(str, inc)
    assert repr(c).startswith('Compose(')


def test_compose_eq_hash():
    c1 = compose(str, inc)
    c2 = compose(str, inc)
    assert c1 == c2
    assert hash(c1) == hash(c2)
    assert (c1 == 5) is False
    assert c1 != 5


def test_compose_descriptor():
    class C:
        f = compose(str, inc)

    c = C()
    assert c.f(1) == '2'
    bound = c.f
    from types import MethodType
    assert isinstance(bound, MethodType)


def test_compose_wrapped():
    c = compose(str, inc)
    assert c.__wrapped__ is inc


def test_compose_signature():
    import inspect
    def f(x: int, y: int) -> int:
        return x + y

    def g(z: int) -> str:
        return str(z)

    c = compose(g, f)
    sig = c.__signature__
    assert sig.return_annotation is str


def test_compose_annotations():
    def f(x: int, y: int) -> int:
        return x + y

    def g(z: int) -> str:
        return str(z)

    c = compose(g, f)
    ann = c.__annotations__
    assert ann['x'] is int
    assert ann['y'] is int
    assert ann['return'] is str


def test_compose_pickle():
    c = compose(str, inc)
    data = pickle.dumps(c)
    restored = pickle.loads(data)
    assert restored(3) == '4'


def test_compose_class_access_doc():
    class C:
        f = compose(str, inc)
    assert C.f.__doc__ is not None


# ---------------------------------------------------------------- pipe

def test_pipe():
    assert pipe(3, double, str) == '6'


def test_pipe_empty():
    assert pipe(3) == 3


# ---------------------------------------------------------------- complement

def test_complement():
    def iseven(n):
        return n % 2 == 0

    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


# ---------------------------------------------------------------- juxt

def test_juxt():
    j = juxt(inc, double)
    assert j(10) == (11, 20)


def test_juxt_list():
    j = juxt([inc, double])
    assert j(10) == (11, 20)


def test_juxt_kwargs():
    def f(x, y):
        return x + y

    def g(x, y):
        return x * y

    j = juxt(f, g)
    assert j(2, 3) == (5, 6)


def test_juxt_pickle():
    j = juxt(inc, double)
    data = pickle.dumps(j)
    restored = pickle.loads(data)
    assert restored(5) == (6, 10)


# ---------------------------------------------------------------- do

def test_do():
    log = []
    result = do(log.append, 5)
    assert result == 5
    assert log == [5]


def test_do_compose():
    log = []
    f = compose(inc, do(log.append))
    assert f(1) == 2
    assert f(11) == 12
    assert log == [1, 11]


# ---------------------------------------------------------------- flip

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


# ---------------------------------------------------------------- return_none

def test_return_none():
    assert return_none(ValueError()) is None
    assert return_none(None) is None


# ---------------------------------------------------------------- excepts

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
    excepting = excepts(ValueError, lambda a: a)
    assert excepting.__name__ == '<lambda>_excepting_ValueError'


def test_excepts_tuple_name():
    excepting = excepts((IndexError, KeyError), lambda a: a)
    assert excepting.__name__ == '<lambda>_excepting_IndexError_or_KeyError'


def test_excepts_doc():
    excepting = excepts(ValueError, list.index, lambda e: -1)
    assert 'A wrapper around' in excepting.__doc__
    assert 'ValueError' in excepting.__doc__
    assert '<lambda>' in excepting.__doc__


def test_excepts_class_access_doc():
    assert excepts.__doc__ is not None


def test_excepts_name_error():
    class BadFunc:
        pass

    excepting = excepts(ValueError, BadFunc())
    assert excepting.__name__ == 'excepting'


# ---------------------------------------------------------------- introspection

def test_num_required_args():
    assert num_required_args(add) == 2
    assert num_required_args(lambda x, y, z=3: x) == 2
    assert num_required_args(varargs_f) == 0


def test_has_varargs():
    assert has_varargs(varargs_f) is True
    assert has_varargs(kwargs_f) is False
    assert has_varargs(add) is False


def test_has_keywords():
    assert has_keywords(default_f) is True
    assert has_keywords(kwargs_f) is True
    assert has_keywords(add) is False


def test_is_valid_args():
    assert is_valid_args(add, (1,), {}) is False
    assert is_valid_args(add, (1, 2), {}) is True
    assert is_valid_args(map, (), {}) is False


def test_is_partial_args():
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


def test_is_arity_varargs():
    def f(*args):
        return args
    assert is_arity(1, f) is False
