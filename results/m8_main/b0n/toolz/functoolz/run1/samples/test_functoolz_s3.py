import inspect
import pickle
import sys
import pytest

from toolz.functoolz import (
    identity, apply, thread_first, thread_last, instanceproperty,
    InstanceProperty, curry, memoize, Compose, compose, compose_left, pipe,
    complement, juxt, do, flip, return_none, excepts,
    num_required_args, has_varargs, has_keywords, is_valid_args,
    is_partial_args, is_arity,
)


# helpers

def inc(x):
    return x + 1


def double(x):
    return 2 * x


def add(x, y):
    return x + y


def div(a, b):
    return a // b


# identity

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3


# apply

def test_apply_basic():
    assert apply(double, 5) == 10


def test_apply_multiple_args():
    assert apply(add, 1, 2) == 3


def test_apply_kwargs():
    def f(a, b=10):
        return a + b
    assert apply(f, 1, b=5) == 6


def test_apply_no_func():
    with pytest.raises(TypeError, match='func argument is required'):
        apply()


# thread_first

def test_thread_first_functions():
    assert thread_first(1, inc, double) == 4


def test_thread_first_tuples():
    assert thread_first(1, (add, 4), (lambda x, y: x ** y, 2)) == 25


def test_thread_first_empty():
    assert thread_first(5) == 5


def test_thread_first_mixed():
    def pow_(x, y):
        return x ** y
    assert thread_first(2, inc, (pow_, 3)) == 27


# thread_last

def test_thread_last_functions():
    assert thread_last(1, inc, double) == 4


def test_thread_last_tuples():
    assert thread_last(1, (add, 4), (lambda x, y: x ** y, 2)) == 32


def test_thread_last_empty():
    assert thread_last(5) == 5


def test_thread_last_map_filter():
    def iseven(x):
        return x % 2 == 0
    result = list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
    assert result == [2, 4]


# instanceproperty / InstanceProperty

def test_instanceproperty_classval():
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


def test_instanceproperty_without_classval_returns_none():
    class C:
        @instanceproperty
        def val(self):
            return 1

    assert C.val is None
    assert C().val == 1


def test_instanceproperty_fget_none_returns_partial():
    from functools import partial
    p = instanceproperty(classval=5)
    assert isinstance(p, partial)


def test_instance_property_is_property():
    def fget(self):
        return 1
    ip = InstanceProperty(fget=fget, classval=10)
    assert isinstance(ip, property)
    assert ip.classval == 10


def test_instance_property_setter():
    class C:
        def __init__(self):
            self._v = 1

        @instanceproperty
        def val(self):
            return self._v

        @val.setter
        def val(self, v):
            self._v = v

    c = C()
    c.val = 99
    assert c.val == 99


def test_instance_property_reduce():
    def fget(self):
        return 7
    ip = InstanceProperty(fget=fget, classval=3)
    cls, state = ip.__reduce__()
    assert cls is InstanceProperty
    restored = cls(*state)
    assert restored.classval == 3


# curry

def test_curry_partial_apply():
    mul = curry(lambda x, y: x * y)
    double_f = mul(2)
    assert isinstance(double_f, curry)
    assert double_f(10) == 20


def test_curry_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add_ = f(a=1)
    assert add_(2, 3) == 5


def test_curry_init_no_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_init_not_callable():
    with pytest.raises(TypeError, match='Input must be callable'):
        curry(5)


def test_curry_from_partial():
    from functools import partial
    def f(a, b, c):
        return a + b + c
    p = partial(f, 1)
    c = curry(p, 2)
    assert c(3) == 6


def test_curry_from_curry():
    @curry
    def f(a, b, c):
        return a + b + c
    c1 = f(1)
    c2 = c1(2)
    assert c2(3) == 6


def test_curry_call_and_bind():
    @curry
    def f(a, b):
        return a + b
    partial_f = f(1)
    assert partial_f.call(2) == 3
    bound = partial_f.bind(2)
    assert bound() == 3


def test_curry_attributes():
    def f(a, b):
        """docstring"""
        return a + b
    c = curry(f, 1)
    assert c.func is f
    assert c.args == (1,)
    assert c.keywords == {}
    assert c.__doc__ == 'docstring'
    assert c.__name__ == 'f'
    assert c.func_name == 'f'


def test_curry_str_repr():
    def f(a, b):
        return a + b
    c = curry(f, 1)
    assert str(c) == str(f)
    assert repr(c) == repr(f)


def test_curry_equality_and_hash():
    def f(a, b):
        return a + b
    c1 = curry(f, 1)
    c2 = curry(f, 1)
    c3 = curry(f, 2)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert c1 != 5


def test_curry_keyword_hash():
    def f(a, b=0):
        return a + b
    c = curry(f, b=1)
    assert isinstance(hash(c), int)


def test_curry_signature():
    def f(a, b, c=3):
        return a + b + c
    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    assert params[0].name == 'b'
    assert params[0].default is inspect.Parameter.empty


def test_curry_incorrect_args_signature():
    def f(a, b):
        return a + b
    c = curry(f, 1, 2, 3)
    with pytest.raises(TypeError, match='incorrect arguments'):
        c.__signature__


def test_curry_genuine_type_error():
    def f(a, b):
        raise TypeError("boom")
    c = curry(f)
    with pytest.raises(TypeError, match="boom"):
        c(1, 2)


def test_curry_varargs_curries_on_type_error():
    def f(*args, **kwargs):
        if len(args) < 2:
            raise TypeError("need more")
        return args
    c = curry(f)
    result = c(1)
    assert isinstance(result, curry)


def test_curry_get_descriptor():
    class C:
        def __init__(self, v):
            self.v = v

        @curry
        def method(self, x):
            return self.v + x

    assert isinstance(C.method, curry)
    obj = C(10)
    assert obj.method(x=5) == 15


def test_curry_call_invalid_args():
    def f(a, b):
        return a + b
    c = curry(f)
    with pytest.raises(TypeError):
        c('x')


# memoize

def test_memoize_basic():
    calls = []

    def f(x):
        calls.append(x)
        return x * 2

    g = memoize(f)
    assert g(3) == 6
    assert g(3) == 6
    assert calls == [3]


def test_memoize_as_decorator():
    @memoize
    def f(x, y):
        return x + y
    assert f(1, 2) == 3
    assert f(1, 2) == 3


def test_memoize_initial_cache():
    @memoize(cache={(1, 2): 3})
    def f(x, y):
        return x + y
    assert f(1, 2) == 3


def test_memoize_multiple_args():
    @memoize
    def f(x, y):
        return x * y
    assert f(2, 3) == 6
    assert f(2, 3) == 6


def test_memoize_kwargs():
    calls = []

    @memoize
    def f(x, y, verbose=False):
        calls.append((x, y))
        return x + y

    assert f(1, 2, verbose=True) == 3
    assert f(1, 2, verbose=True) == 3
    assert len(calls) == 1


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def f(x, y, verbose=False):
        calls.append(1)
        return x + y

    assert f(1, 2, verbose=True) == 3
    assert f(1, 2, verbose=False) == 3
    assert len(calls) == 1


def test_memoize_unhashable_raises():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError, match='hashable'):
        f([1, 2, 3])


def test_memoize_attributes():
    def f(x):
        """docs"""
        return x

    g = memoize(f)
    assert g.__name__ == 'f'
    assert g.__doc__ == 'docs'
    assert g.__wrapped__ is f


# Compose

def test_compose_basic():
    assert compose(str, inc)(3) == '4'


def test_compose_no_funcs():
    assert compose() is identity


def test_compose_one_func():
    def f(x):
        return x
    assert compose(f) is f


def test_compose_left_basic():
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_no_funcs():
    assert compose_left() is identity


def test_compose_class_attributes():
    c = Compose([str, inc])
    assert c.first is inc
    assert c.funcs == (str,)


def test_compose_call():
    c = compose(str, inc)
    assert c(3) == '4'


def test_compose_doc():
    c = compose(str, inc)
    assert 'inc' in c.__doc__
    assert 'str' in c.__doc__


def test_compose_class_doc():
    assert Compose.__doc__


def test_compose_doc_no_name():
    c = compose(lambda x: x, inc)
    assert c.__doc__ == 'A composition of functions'


def test_compose_name():
    c = compose(str, inc)
    assert c.__name__ == 'str_of_inc'


def test_compose_name_no_name():
    c = compose(lambda x: x, inc)
    assert c.__name__ == 'Compose'


def test_compose_repr():
    c = compose(str, inc)
    r = repr(c)
    assert r.startswith('Compose')


def test_compose_eq():
    c1 = compose(str, inc)
    c2 = compose(str, inc)
    c3 = compose(str, double)
    assert c1 == c2
    assert c1 != c3
    assert (c1 == 5) is False
    assert (c1 != 5) is True


def test_compose_NotImplemented():
    c = compose(str, inc)
    assert c.__eq__(5) is NotImplemented
    assert c.__ne__(5) is NotImplemented


def test_compose_hash():
    c1 = compose(str, inc)
    c2 = compose(str, inc)
    assert hash(c1) == hash(c2)


def test_compose_descriptor():
    class C:
        method = compose(inc, double)

    obj = C()
    assert C.method(3) == 7
    assert obj.method(3) == 7


def test_compose_getstate_setstate():
    c = compose(str, inc)
    state = c.__getstate__()
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2(3) == '4'


def test_compose_signature():
    def f(x: int) -> int:
        return x
    def g(y: str) -> str:
        return y
    c = compose(g, f)
    sig = c.__signature__
    assert sig.return_annotation is str


def test_compose_signature_general():
    c = compose(str, inc)
    sig = c.__signature__
    # parameter from first applied (inc), return from last (str)
    assert sig.parameters['x'].annotation is inspect.Parameter.empty


def test_compose_annotations():
    def f(x: int) -> int:
        return x
    def g(y: float) -> str:
        return str(y)
    c = compose(g, f)
    ann = c._combined_annotations()
    assert ann == {'y': float, 'return': str}


def test_compose_wrapped():
    def f(x):
        return x
    c = compose(str, f)
    assert c.__wrapped__ is f


def test_compose_pickle():
    c = compose(str, inc)
    # module-level functions required for pickling; use a built-in composition
    import toolz.functoolz as m
    c2 = compose(str, m.identity)
    data = pickle.dumps(c2)
    restored = pickle.loads(data)
    assert restored(3) == '3'


# pipe

def test_pipe():
    assert pipe(3, double, str) == '6'


def test_pipe_no_funcs():
    assert pipe(3) == 3


# complement

def test_complement():
    def iseven(n):
        return n % 2 == 0
    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


# juxt

def test_juxt_basic():
    j = juxt(inc, double)
    assert j(10) == (11, 20)


def test_juxt_list():
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(x, y=0):
        return x + y
    def g(x, y=0):
        return x - y
    j = juxt(f, g)
    assert j(5, y=2) == (7, 3)


def test_juxt_getstate_setstate():
    j = juxt(inc, double)
    state = j.__getstate__()
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2(1) == (2, 2)


# do

def test_do():
    log = []
    result = do(log.append, 1)
    assert result == 1
    assert log == [1]


def test_do_compose():
    log = []
    log_inc = compose(inc, do(log.append))
    assert log_inc(1) == 2
    assert log_inc(11) == 12
    assert log == [1, 11]


# flip

def test_flip():
    assert flip(div, 2, 6) == 3


def test_flip_curried():
    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_flip_isinstance():
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# return_none

def test_return_none():
    assert return_none(ValueError("x")) is None


# excepts

def test_excepts_basic():
    excepting = excepts(ValueError, lambda a: [1, 2].index(a), lambda _: -1)
    assert excepting(1) == 0
    assert excepting(3) == -1


def test_excepts_default_handler():
    excepting = excepts(IndexError, lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1


def test_excepts_multiple():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_attributes():
    e = excepts(ValueError, lambda a: int(a), return_none)
    assert e.exc is ValueError
    assert e.handler is return_none


def test_excepts_name():
    e = excepts(ValueError, lambda a: a)
    assert e.__name__ == '<lambda>_excepting_ValueError'


def test_excepts_name_tuple():
    e = excepts((IndexError, KeyError), lambda a: a)
    assert e.__name__ == '<lambda>_excepting_IndexError_or_KeyError'


def test_excepts_name_fallback():
    e = excepts(ValueError, 5)
    assert e.__name__ == 'excepting'


def test_excepts_doc():
    e = excepts(ValueError, lambda a: a)
    doc = e.__doc__
    assert 'ValueError' in doc


def test_excepts_no_raise():
    e = excepts(ValueError, lambda a: a + 1)
    assert e(1) == 2


# num_required_args

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z
    assert num_required_args(f) == 2


def test_num_required_args_varargs():
    def g(*args, **kwargs):
        pass
    assert num_required_args(g) == 0


def test_num_required_args_positional_only():
    def f(a, b, /, c):
        return a + b + c
    assert num_required_args(f) == 3


# has_varargs

def test_has_varargs_true():
    def f(*args):
        return args
    assert has_varargs(f) is True


def test_has_varargs_false():
    def g(**kwargs):
        return kwargs
    assert has_varargs(g) is False


# has_keywords

def test_has_keywords_true():
    def f(x, y=0):
        return x + y
    assert has_keywords(f) is True


def test_has_keywords_var_kwargs():
    def f(x, **kwargs):
        return x
    assert has_keywords(f) is True


def test_has_keywords_false():
    def f(x):
        return x
    assert has_keywords(f) is False


# is_valid_args

def test_is_valid_args():
    def f(x, y):
        return x + y
    assert is_valid_args(f, (1,), {}) is False
    assert is_valid_args(f, (1, 2), {}) is True


def test_is_valid_args_map():
    assert is_valid_args(map, (), {}) is False


# is_partial_args

def test_is_partial_args():
    def f(x, y):
        return x + y
    assert is_partial_args(f, (1,), {}) is True
    assert is_partial_args(f, (1, 2), {}) is True
    assert is_partial_args(f, (1, 2, 3), {}) is False


# is_arity

def test_is_arity():
    def f(x):
        return x
    assert is_arity(1, f) is True

    def g(x, y=1):
        return x + y
    assert is_arity(1, g) is False


def test_is_arity_varargs():
    def f(*args):
        return args
    assert is_arity(1, f) is False


def test_is_arity_two():
    def f(x, y):
        return x + y
    assert is_arity(2, f) is True
    assert is_arity(3, f) is False
