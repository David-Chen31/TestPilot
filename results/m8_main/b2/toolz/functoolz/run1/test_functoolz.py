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

    assert apply(double, 5) == 10


def test_apply_kwargs():
    def f(a, b=0):
        return a + b

    assert apply(f, 1, b=2) == 3


def test_apply_no_args_raises():
    with pytest.raises(TypeError, match='func argument is required'):
        apply()


def test_apply_in_map():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_callables():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4


def test_thread_first_tuples():
    def add(x, y):
        return x + y

    def power(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (power, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(5) == 5


def test_thread_last_callables():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4


def test_thread_last_tuples():
    def add(x, y):
        return x + y

    def power(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (power, 2)) == 32


def test_thread_last_no_forms():
    assert thread_last(5) == 5


def test_thread_last_map_filter():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    result = list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
    assert result == [2, 4]


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_basic():
    class MyClass(object):
        """The class docstring"""

        @instanceproperty(classval='classval')
        def val(self):
            return 42

    assert MyClass.val == 'classval'
    obj = MyClass()
    assert obj.val == 42


def test_instanceproperty_class_doc():
    class MyClass:
        """The class docstring"""

        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return 'An object docstring'

    assert MyClass.__doc__ == 'The class docstring'
    assert MyClass().__doc__ == 'An object docstring'


def test_instanceproperty_default_classval_none():
    class MyClass:
        @instanceproperty
        def val(self):
            return 7

    assert MyClass.val is None
    assert MyClass().val == 7


def test_instanceproperty_setter():
    class MyClass:
        def __init__(self):
            self._x = 0

        @instanceproperty
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

    obj = MyClass()
    assert obj.x == 0
    obj.x = 5
    assert obj.x == 5


def test_instance_property_reduce():
    class MyClass:
        @instanceproperty(classval='cv')
        def val(self):
            return 1

    inst = MyClass.__dict__['val']
    assert isinstance(inst, InstanceProperty)
    restored_cls, state = inst.__reduce__()
    assert restored_cls is InstanceProperty
    restored = restored_cls(*state)
    assert restored.classval == 'cv'


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert isinstance(double, curry)
    assert double(10) == 20


def test_curry_as_decorator_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_no_args_raises():
    with pytest.raises(TypeError):
        curry()


def test_curry_not_callable_raises():
    with pytest.raises(TypeError, match='Input must be callable'):
        curry(1)


def test_curry_attributes():
    def mul(x, y):
        """multiplication"""
        return x * y

    cmul = curry(mul, 2)
    assert cmul.func is mul
    assert cmul.args == (2,)
    assert cmul.keywords == {}
    assert cmul.__doc__ == 'multiplication'
    assert cmul.__name__ == 'mul'
    assert cmul.func_name == 'mul'
    assert str(cmul) == str(mul)
    assert repr(cmul) == repr(mul)


def test_curry_equality_and_hash():
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

    a = curry(add, 1)
    assert a.call(2) == 3
    bound = a.bind(2)
    assert isinstance(bound, curry)
    assert bound() == 3


def test_curry_merges_partial_like():
    def add(x, y, z):
        return x + y + z

    p = curry(add, 1)
    p2 = curry(p, 2)
    assert p2.func is add
    assert p2.args == (1, 2)
    assert p2(3) == 6


def test_curry_typeerror_propagates():
    def add(x, y):
        return x + y

    c = curry(add)
    # Too many args: no way to make valid by adding more -> propagate
    with pytest.raises(TypeError):
        c(1, 2, 3)


def test_curry_raises_inside_function():
    def f(x):
        raise TypeError('boom')

    c = curry(f)
    with pytest.raises(TypeError, match='boom'):
        c(1)


def test_curry_varargs_curries_anyway():
    @curry
    def f(*args, **kwargs):
        if len(args) < 2:
            raise TypeError('need more')
        return args

    assert f(1)(2) == (1, 2)


def test_curry_signature():
    import inspect

    def add(x, y=10, z=30):
        return x + y + z

    c = curry(add, 1)
    sig = inspect.signature(c)
    assert list(sig.parameters) == ['y', 'z']


def test_curry_signature_incorrect_args():
    def f(x):
        return x

    c = curry(f, 1)
    import inspect
    with pytest.raises(TypeError, match='curry object has incorrect arguments'):
        inspect.signature(c)


def test_curry_signature_keywords_bound():
    import inspect

    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1, z=10)
    sig = inspect.signature(c)
    params = sig.parameters
    assert 'z' not in params
    assert params['y'].default is not inspect.Parameter.empty


def test_curry_get_descriptor():
    class C:
        @curry
        def method(self, x):
            return x + 1

    inst = C()
    assert inst.method(1) == 2


def test_curry_pickle_basic():
    # module-level function for pickling
    c = curry(_pickle_add, 1)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == 3


def _pickle_add(x, y):
    return x + y


def test_curry_pickle_decorated():
    restored = pickle.loads(pickle.dumps(_decorated_curry))
    assert restored is _decorated_curry


@curry
def _decorated_curry(x, y):
    return x + y


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic():
    calls = []

    @memoize
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert len(calls) == 1
    assert add(2, 3) == 5
    assert len(calls) == 2


def test_memoize_with_initial_cache():
    calls = []

    @memoize(cache={(1, 2): 100})
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(1, 2) == 100
    assert calls == []


def test_memoize_key_function():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3
    assert len(calls) == 1


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError, match='hashable'):
        f([1, 2, 3])


def test_memoize_metadata():
    def add(x, y):
        """add two"""
        return x + y

    m = memoize(add)
    assert m.__name__ == 'add'
    assert m.__doc__ == 'add two'
    assert m.__wrapped__ is add


def test_memoize_unary():
    calls = []

    @memoize
    def inc(x):
        calls.append(x)
        return x + 1

    assert inc(1) == 2
    assert inc(1) == 2
    assert calls == [1]


# ---------------------------------------------------------------------------
# compose / compose_left / Compose
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_no_funcs_identity():
    assert compose() is identity


def test_compose_single_func():
    def f(x):
        return x

    assert compose(f) is f


def test_compose_multiple_args_to_first():
    def add(a, b):
        return a + b

    def double(x):
        return 2 * x

    assert compose(double, add)(1, 2) == 6


def test_compose_left_basic():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_no_funcs():
    assert compose_left() is identity


def test_compose_left_single():
    def f(x):
        return x

    assert compose_left(f) is f


def test_compose_is_compose_instance():
    inc = lambda i: i + 1
    c = compose(str, inc)
    assert isinstance(c, Compose)


def test_compose_equality():
    inc = lambda i: i + 1
    assert compose(str, inc) == compose(str, inc)
    assert compose(str, inc) != compose(inc, str)


def test_compose_neq_not_compose():
    inc = lambda i: i + 1
    c = compose(str, inc)
    assert (c == 5) is False
    assert (c != 5) is True


def test_compose_hash():
    inc = lambda i: i + 1
    assert hash(compose(str, inc)) == hash(compose(str, inc))


def test_compose_repr():
    inc = lambda i: i + 1
    c = compose(str, inc)
    assert repr(c).startswith('Compose(')
    assert 'str' in repr(c)


def test_compose_name():
    def foo(x):
        return x

    def bar(x):
        return x

    c = compose(foo, bar)
    assert c.__name__ == 'foo_of_bar'


def test_compose_name_unavailable():
    class Callable:
        def __call__(self, x):
            return x

    c = compose(Callable(), Callable())
    assert c.__name__ == 'Compose'


def test_compose_doc():
    def foo(x):
        return x

    def bar(x):
        return x

    c = compose(foo, bar)
    assert 'foo' in c.__doc__
    assert 'bar' in c.__doc__


def test_compose_doc_classval():
    def foo(x):
        return x

    c = compose(foo, foo)
    # class-level access returns the doc attribute
    assert Compose.__dict__['__doc__'].classval == Compose.__doc__


def test_compose_doc_no_names():
    class Callable:
        def __call__(self, x):
            return x

    c = compose(Callable(), Callable())
    assert c.__doc__ == 'A composition of functions'


def test_compose_get_descriptor():
    class C:
        @compose(lambda x: x + 1, lambda x: x * 2)
        def method(self):
            pass

    c = C()
    assert c.method() == 3


def test_compose_get_descriptor_class_access():
    inc = lambda i: i + 1
    c = compose(str, inc)

    class C:
        method = c

    assert C.method is c


def test_compose_signature():
    import inspect

    def f(x: int) -> int:
        return x

    def g(y: str) -> str:
        return y

    c = compose(f, g)
    sig = inspect.signature(c)
    assert sig.return_annotation is int


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__wrapped__ is g


def test_compose_pickle():
    c = compose(_pickle_add, _pickle_add)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(1) == 4


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'


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

def test_juxt_multiple():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_list():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(a, b):
        return a + b

    def g(a, b):
        return a * b

    assert juxt(f, g)(a=2, b=3) == (5, 6)


def test_juxt_pickle():
    inc = lambda x: x + 1
    j = juxt(_pickle_add, inc)
    restored = pickle.loads(pickle.dumps(j))
    assert restored(1) == (2, 2)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []

    def inc(x):
        return x + 1

    f = compose(inc, do(log.append))
    assert f(1) == 2
    assert f(11) == 12
    assert log == [1, 11]


def test_do_returns_x():
    log = []
    assert do(log.append, 5) == 5
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
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError('x')) is None
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


def test_excepts_handler_receives_exception():
    seen = []

    def handler(exc):
        seen.append(exc)
        return 'handled'

    def f(x):
        raise ValueError('boom')

    e = excepts(ValueError, f, handler)
    assert e(1) == 'handled'
    assert len(seen) == 1
    assert isinstance(seen[0], ValueError)


def test_excepts_name():
    def my_func(x):
        return x

    e = excepts(ValueError, my_func)
    assert e.__name__ == 'my_func_excepting_ValueError'


def test_excepts_name_tuple():
    def my_func(x):
        return x

    e = excepts((ValueError, KeyError), my_func)
    assert e.__name__ == 'my_func_excepting_ValueError_or_KeyError'


def test_excepts_name_no_attr():
    e = excepts(ValueError, 5)  # 5 has no __name__
    assert e.__name__ == 'excepting'


def test_excepts_doc():
    def my_func(x):
        """my doc"""
        return x

    e = excepts(ValueError, my_func)
    assert 'my_func' in e.__doc__
    assert 'ValueError' in e.__doc__


def test_excepts_doc_tuple():
    def my_func(x):
        """my doc"""
        return x

    e = excepts((ValueError, KeyError), my_func)
    assert '(ValueError, KeyError)' in e.__doc__


def test_excepts_doc_no_attr():
    e = excepts(ValueError, 5)
    assert e.__doc__ == excepts.__doc__


def test_excepts_class_doc():
    assert isinstance(excepts.__dict__['__doc__'], InstanceProperty)
    assert excepts.__doc__ == excepts.__dict__['__doc__'].classval


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2

    def g(*args, **kwargs):
        pass

    assert num_required_args(g) == 0


def test_num_required_args_positional_only():
    def f(x, y, /, z=1):
        return x

    assert num_required_args(f) == 2


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

    def g(x, *, kwonly):
        return x

    def h(x, **kwargs):
        return x

    assert has_keywords(f) is True
    assert has_keywords(g) is True
    assert has_keywords(h) is True

    def basic(x):
        return x

    assert has_keywords(basic) is False


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


def test_is_arity_two():
    def f(x, y):
        return x

    assert is_arity(2, f) is True
    assert is_arity(1, f) is False


def test_is_arity_varargs():
    def f(*args):
        return args

    assert is_arity(1, f) is False


# ---------------------------------------------------------------------------
# misc: instanceproperty returns partial when fget None
# ---------------------------------------------------------------------------

def test_instanceproperty_returns_partial():
    from functools import partial

    p = instanceproperty()
    assert isinstance(p, partial)
    # applying to fget yields InstanceProperty
    def f(self):
        return 1

    result = p(f)
    assert isinstance(result, InstanceProperty)
