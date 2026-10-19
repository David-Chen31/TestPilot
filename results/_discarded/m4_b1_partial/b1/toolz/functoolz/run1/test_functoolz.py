import pickle
import inspect
import sys
import functools

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
# Module-level helpers (picklable for pickle tests)
# ---------------------------------------------------------------------------

def inc(x):
    return x + 1


def double(x):
    return 2 * x


def add(x, y):
    return x + y


def power(x, y):
    return x ** y


def _pickle_add(x, y):
    return x + y


_pickle_add = curry(_pickle_add)


class _pickle_property_host:
    """Module-level host so its fget can be pickled by qualified name."""
    def __init__(self):
        self._x = 10

    @instanceproperty(classval='cv')
    def x(self):
        return self._x


# ---------------------------------------------------------------------------
# identity / apply
# ---------------------------------------------------------------------------

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3
    assert identity(None) is None


def test_apply_basic():
    assert apply(double, 5) == 10


def test_apply_kwargs():
    def f(x, y=0):
        return x + y

    assert apply(f, 1, y=10) == 11


def test_apply_multiple_args():
    def add3(a, b, c):
        return a + b + c

    assert apply(add3, 1, 2, 3) == 6


def test_apply_no_func():
    with pytest.raises(TypeError):
        apply()


def test_apply_with_map():
    result = tuple(map(apply, [double, inc, double], [10, 500, 8000]))
    assert result == (20, 501, 16000)


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_simple():
    assert thread_first(1, inc, double) == 4


def test_thread_first_forms():
    assert thread_first(1, (add, 4), (power, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(5) == 5


def test_thread_last_simple():
    assert thread_last(1, inc, double) == 4


def test_thread_last_forms():
    assert thread_last(1, (add, 4), (power, 2)) == 32


def test_thread_last_no_forms():
    assert thread_last(7) == 7


def test_thread_last_with_map_filter():
    def iseven(x):
        return x % 2 == 0

    result = list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
    assert result == [2, 4]


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_classval_and_instance_value():
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


def test_instanceproperty_partial_form():
    spec = instanceproperty(classval='classvalue')
    assert callable(spec)

    class C:
        @spec
        def value(self):
            return 'instance'

    assert C.value == 'classvalue'
    assert C().value == 'instance'


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
    assert c.x == 0
    c.x = 5
    assert c.x == 5


def test_instance_property_class_is_property():
    class C:
        @instanceproperty
        def v(self):
            return 1

    assert isinstance(C.__dict__['v'], InstanceProperty)
    assert isinstance(C.__dict__['v'], property)


def test_instanceproperty_pickle_roundtrip():
    # Pickling an InstanceProperty directly fails because its fget is a
    # function that is shadowed by the property in the class namespace (the
    # class attribute ``x`` is the InstanceProperty, not the function), so
    # pickle cannot find the function by qualified name.
    prop = _pickle_property_host.__dict__['x']
    with pytest.raises(pickle.PicklingError):
        pickle.dumps(prop)

    # Verify the __reduce__ shape describes the descriptor correctly without
    # requiring it to be pickled; the reduce tuple is (cls, state).
    reduce_fn, state = prop.__reduce__()
    assert reduce_fn is InstanceProperty
    assert state == (prop.fget, prop.fset, prop.fdel, prop.__doc__,
                     prop.classval)
    # And that reconstructing from that state works fine.
    restored = InstanceProperty(*state)
    assert restored.classval == 'cv'
    assert restored.fget is prop.fget
    assert restored.__get__(_pickle_property_host()) == 10
    assert restored.__get__(None, _pickle_property_host) == 'cv'


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    mul = curry(mul)
    double_ = mul(2)
    assert double_(10) == 20


def test_curry_decorator_with_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add_ = f(a=1)
    assert add_(2, 3) == 5


def test_curry_requires_callable():
    with pytest.raises(TypeError):
        curry(1)


def test_curry_requires_arg():
    with pytest.raises(TypeError):
        curry()


def test_curry_unpacks_existing_curry():
    @curry
    def add3(a, b, c):
        return a + b + c

    partial_add = add3(1)(2)
    assert partial_add(3) == 6
    combined = partial_add.bind(10)
    assert combined() == 13


def test_curry_call_and_bind():
    c = curry(add, 1)
    assert c.call(2) == 3
    c2 = curry(power, 2)
    assert c2.call(3) == 8
    # bind prepends args to the already-bound args: curry(power, 2) has args
    # (2,), and bind(2) yields curry(power, 2, 2), so call needs no more args.
    bound = c2.bind(2)
    assert bound() == 4
    assert bound.call() == 4
    assert bound.args == (2, 2)


def test_curry_attributes():
    def f(x):
        """docstring"""
        return x

    c = curry(f)
    assert c.func is f
    assert c.func_name == 'f'
    assert c.__name__ == 'f'
    assert c.__doc__ == 'docstring'
    assert str(c) == str(f)
    assert repr(c) == repr(f)


def test_curry_default_name_for_callable_object():
    class Callable:
        def __call__(self, x):
            return x

    c = curry(Callable())
    assert c.__name__ == '<curry>'


def test_curry_equality_and_hash():
    def add_(x, y):
        return x + y

    a = curry(add_, 1)
    b = curry(add_, 1)
    c = curry(add_, 2)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert a != object()


def test_curry_genuine_typeerror_propagates():
    def f(x):
        return x + 1

    c = curry(f)
    with pytest.raises(TypeError):
        c('not a number')


def test_curry_varargs_decorator_pattern():
    @curry
    def f(a, b):
        return a + b

    g = f(1)
    assert g(2) == 3


def test_curry_too_many_args_raises():
    @curry
    def f(a, b):
        return a + b

    with pytest.raises(TypeError):
        f(1, 2, 3)


def test_curry_signature():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1)
    sig = c.__signature__
    assert 'x' not in sig.parameters
    assert sig.parameters['y'].default is not inspect.Parameter.empty
    assert sig.parameters['z'].default == 3


def test_curry_signature_incorrect_args():
    def f(x, y):
        return x + y

    c = curry(f, 1, 2, 3)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_keyword_signature():
    def f(a, b, c=1):
        return a + b + c

    c = curry(f, b=2)
    sig = c.__signature__
    assert sig.parameters['b'].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters['b'].default == 2


def test_curry_method_descriptor():
    class C:
        @curry
        def add(self, x, y):
            return x + y

    obj = C()
    # Accessing via class returns the curry object itself.
    assert isinstance(C.add, curry)
    # Accessing via instance returns a curry bound to the instance.
    assert obj.add(1, 2) == 3


def test_curry_pickle_roundtrip():
    c = _pickle_add(1)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == 3


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic():
    calls = []

    @memoize
    def add_(x, y):
        calls.append((x, y))
        return x + y

    assert add_(1, 2) == 3
    assert add_(1, 2) == 3
    assert len(calls) == 1


def test_memoize_initial_cache():
    @memoize(cache={(1, 2): 3})
    def add_(x, y):
        raise AssertionError("should not be called")

    assert add_(1, 2) == 3


def test_memoize_unary():
    calls = []

    @memoize
    def square(x):
        calls.append(x)
        return x * x

    assert square(3) == 9
    assert square(3) == 9
    assert calls == [3]


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add_(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    assert add_(1, 2) == 3
    assert add_(1, 2, verbose=True) == 3
    assert len(calls) == 1


def test_memoize_unhashable_raises():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_metadata():
    @memoize
    def f(x):
        """mydoc"""
        return x

    assert f.__name__ == 'f'
    assert f.__doc__ == 'mydoc'
    assert f.__wrapped__ is not None


# ---------------------------------------------------------------------------
# compose / compose_left / Compose
# ---------------------------------------------------------------------------

def test_compose_basic():
    assert compose(str, inc)(3) == '4'
    assert compose(inc, double)(3) == 7


def test_compose_no_args():
    assert compose() is identity


def test_compose_single():
    def f(x):
        return x

    assert compose(f) is f


def test_compose_left_basic():
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_no_args():
    assert compose_left() is identity


def test_compose_kwargs_to_first():
    def f(a, b=1):
        return a + b

    assert compose(str, f)(2, b=3) == '5'


def test_compose_class_and_attributes():
    c = compose(str, inc)
    assert isinstance(c, Compose)
    assert c.first is inc
    assert c.funcs == (str,)
    assert c.__name__ == 'str_of_inc'
    assert c.__wrapped__ is inc


def test_compose_doc():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert 'lambda *args, **kwargs:' in c.__doc__
    assert 'f(g(*args, **kwargs))' in c.__doc__


def test_compose_doc_no_names():
    class Callable:
        def __call__(self, x):
            return x

    c = compose(Callable(), Callable())
    assert c.__doc__ == 'A composition of functions'
    assert c.__name__ == 'Compose'


def test_compose_repr():
    c = compose(str, inc)
    assert repr(c).startswith('Compose(')
    assert repr(c).endswith(')')


def test_compose_eq_and_hash():
    c1 = compose(str, inc)
    c2 = compose(str, inc)
    c3 = compose(str, double)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)


def test_compose_not_implemented_eq():
    c = compose(str, inc)
    assert c.__eq__(object()) is NotImplemented
    assert c.__ne__(object()) is NotImplemented


def test_compose_signature():
    def f(x: int) -> str:
        return str(x)

    def g(y: int) -> int:
        return y + 1

    c = compose(f, g)
    sig = c.__signature__
    assert sig.return_annotation is str


def test_compose_descriptor_binding():
    def method(self, x):
        return x + 1

    class C:
        value = compose(str, method)

    obj = C()
    # Compose binds itself as a MethodType when accessed via instance,
    # passing the instance as the first arg to `first`.
    result = obj.value(1)
    assert result == '2'


def test_compose_pickle_roundtrip():
    c = compose(str, inc)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(1) == '2'


def test_compose_annotations():
    def f(x: int) -> str:
        return str(x)

    def g(y: int) -> int:
        return y + 1

    c = compose(f, g)
    ann = c.__annotations__
    assert ann == {'y': int, 'return': str}


def test_compose_nested_annotations():
    def f(x: int) -> str:
        return str(x)

    def g(y: int) -> int:
        return y + 1

    def h(z: int) -> int:
        return z * 2

    inner = compose(f, g)
    outer = compose(inner, h)
    ann = outer.__annotations__
    assert ann == {'z': int, 'return': str}


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe_basic():
    assert pipe(3, double, str) == '6'


def test_pipe_no_funcs():
    assert pipe(5) == 5


def test_pipe_multiple():
    assert pipe(1, inc, inc, inc) == 4


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
    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_from_list():
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(x, y=0):
        return x + y

    def g(x, y=0):
        return x - y

    assert juxt(f, g)(5, y=2) == (7, 3)


def test_juxt_stores_funcs_tuple():
    j = juxt(inc, double)
    assert j.funcs == (inc, double)


def test_juxt_pickle():
    j = juxt(inc, double)
    restored = pickle.loads(pickle.dumps(j))
    assert restored(10) == (11, 20)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do_returns_x_and_side_effect():
    log = []
    result = do(log.append, 5)
    assert result == 5
    assert log == [5]


def test_do_with_compose():
    log = []
    f = compose(inc, lambda x: do(log.append, x))
    assert f(1) == 2
    assert f(11) == 12
    assert log == [1, 11]


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------

def test_flip_basic():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3


def test_flip_curried():
    def div(a, b):
        return a // b

    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_flip_isinstance_use():
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none / excepts
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError('x')) is None


def test_excepts_basic():
    excepting = excepts(
        ValueError,
        lambda a: [1, 2].index(a),
        lambda _: -1,
    )
    assert excepting(1) == 0
    assert excepting(3) == -1


def test_excepts_default_handler():
    excepting = excepts(ValueError, lambda a: int(a))
    assert excepting('5') == 5
    assert excepting('bad') is None


def test_excepts_multiple_exceptions():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_non_matching_exception_propagates():
    excepting = excepts(ValueError, lambda a: a + 1)
    with pytest.raises(TypeError):
        excepting('x')


def test_excepts_doc():
    excepting = excepts(ValueError, lambda a: a, lambda e: None)
    doc = excepting.__doc__
    assert 'ValueError' in doc
    assert '<lambda>' in doc


def test_excepts_name():
    excepting = excepts(ValueError, lambda a: a)
    assert excepting.__name__ == '<lambda>_excepting_ValueError'


def test_excepts_name_multiple():
    excepting = excepts((IndexError, KeyError), lambda a: a)
    assert excepting.__name__ == '<lambda>_excepting_IndexError_or_KeyError'


def test_excepts_name_attribute_error():
    class Callable:
        def __call__(self, x):
            return x

    excepting = excepts(ValueError, Callable())
    assert excepting.__name__ == 'excepting'


def test_excepts_classval_doc():
    assert excepts.__doc__ is not None


# ---------------------------------------------------------------------------
# introspection helpers
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2

    def g(*args, **kwargs):
        return args

    assert num_required_args(g) == 0


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


def test_is_valid_args():
    def add_(x, y):
        return x + y

    assert is_valid_args(add_, (1,), {}) is False
    assert is_valid_args(add_, (1, 2), {}) is True


def test_is_partial_args():
    def add_(x, y):
        return x + y

    assert is_partial_args(add_, (1,), {}) is True
    assert is_partial_args(add_, (1, 2), {}) is True
    assert is_partial_args(add_, (1, 2, 3), {}) is False


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


def test_is_valid_args_with_signature():
    def add_(x, y):
        return x + y

    sig = inspect.signature(add_)
    assert is_valid_args(add_, (1, 2), {}, sig) is True
    assert is_valid_args(add_, (1,), {}, sig) is False


def test_is_partial_args_with_signature():
    def add_(x, y):
        return x + y

    sig = inspect.signature(add_)
    assert is_partial_args(add_, (1,), {}, sig) is True
    assert is_partial_args(add_, (1, 2, 3), {}, sig) is False
