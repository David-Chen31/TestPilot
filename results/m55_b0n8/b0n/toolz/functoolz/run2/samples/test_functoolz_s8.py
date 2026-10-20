import pickle
import sys
import types
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
# identity / apply
# ---------------------------------------------------------------------------

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3
    assert identity(None) is None


def test_apply_basic():
    assert apply(lambda x: 2 * x, 5) == 10
    assert apply(lambda x, y: x + y, 1, 2) == 3
    assert apply(lambda x, y=0: x + y, 1, y=5) == 6


def test_apply_with_map():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)


def test_apply_requires_func():
    with pytest.raises(TypeError):
        apply()


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4
    assert thread_first(1) == 1

    def add(x, y):
        return x + y

    def power(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (power, 2)) == 25


def test_thread_last():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4
    assert thread_last(1) == 1

    def add(x, y):
        return x + y

    def power(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (power, 2)) == 32


def test_thread_last_iterables():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


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


def test_instanceproperty_setter():
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


def test_instanceproperty_is_property_subclass():
    class C:
        @instanceproperty
        def val(self):
            return 1

    assert isinstance(C.__dict__['val'], InstanceProperty)
    assert isinstance(C.__dict__['val'], property)


def test_instanceproperty_pickle():
    class C:
        @instanceproperty
        def val(self):
            return 7

    prop = C.__dict__['val']
    restored = pickle.loads(pickle.dumps(prop))
    assert isinstance(restored, InstanceProperty)
    assert restored.classval is None
    assert restored.fget is prop.fget


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_partial_application():
    def mul(x, y):
        return x * y

    mul_c = curry(mul)
    double = mul_c(2)
    assert double(10) == 20
    assert mul_c(2, 3) == 6


def test_curry_as_decorator_with_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5
    assert f(2, 3) == 50


def test_curry_requires_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_requires_callable():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_unpacks_partial():
    def add(x, y, z):
        return x + y + z

    p = partial(add, 1)
    c = curry(p, 2)
    assert c(3) == 6


def test_curry_unpacks_keywords_from_partial():
    def f(x, y, a=0, b=0):
        return x + y + a + b

    p = partial(f, a=10)
    c = curry(p, b=100)
    assert c(1, 2) == 113


def test_curry_attributes():
    def myfunc(x, y):
        """docstring"""
        return x + y

    c = curry(myfunc, 1)
    assert c.func is myfunc
    assert c.args == (1,)
    assert c.keywords == {}
    assert c.__doc__ == 'docstring'
    assert c.__name__ == 'myfunc'
    assert c.__qualname__ == myfunc.__qualname__
    assert c.__module__ == myfunc.__module__
    assert c.func_name == 'myfunc'


def test_curry_str_repr():
    def myfunc(x):
        return x

    c = curry(myfunc)
    assert str(c) == str(myfunc)
    assert repr(c) == repr(myfunc)
    assert str(c) == str(myfunc)


def test_curry_equality_hash():
    def myfunc(x, y):
        return x + y

    a = curry(myfunc, 1)
    b = curry(myfunc, 1)
    c = curry(myfunc, 2)
    assert a == b
    assert not (a == c)
    assert a != c
    assert hash(a) == hash(b)
    assert not (a == 5)


def test_curry_call_and_bind():
    def add(x, y, z):
        return x + y + z

    c = curry(add, 1)
    assert c.call(2, 3) == 6
    b = c.bind(2)
    assert isinstance(b, curry)
    assert b(3) == 6


def test_curry_genuine_typeerror_reraises():
    def f(x):
        return x + 'a'  # will fail for int

    with pytest.raises(TypeError):
        curry(f)(1)


def test_curry_signature():
    import inspect

    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1)
    sig = c.__signature__
    assert list(sig.parameters) == ['y', 'z']


def test_curry_signature_incorrect_arguments():
    def f(x):
        return x

    c = curry(f, 1, 2)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_keyword_only_in_signature():
    import inspect

    def f(x, y=1):
        return x + y

    c = curry(f, x=1)
    sig = c.__signature__
    assert 'y' in sig.parameters


def test_curry_descriptor():
    class C:
        def __init__(self, n):
            self.n = n

        @curry
        def add(self, a, b):
            return self.n + a + b

    c = C(10)
    assert c.add(1, 2) == 13
    assert isinstance(C.__dict__['add'], curry)


def test_curry_pickle():
    # Use a module-level function
    c = curry(_module_add, 1)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == 3
    assert restored.func is _module_add
    assert restored.args == (1,)


def _module_add(x, y):
    return x + y


def test_curry_pickle_keywords():
    c = curry(_module_addkw, x=1)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(y=2) == 3


def _module_addkw(x=0, y=0):
    return x + y


def test_curry_pickle_decorated():
    # memoize is a module-level curry-decorated function
    assert isinstance(memoize, curry)
    restored = pickle.loads(pickle.dumps(memoize))
    assert restored is memoize


def test_curry_with_varargs_still_curries():
    def f(*args, **kwargs):
        return (args, kwargs)

    c = curry(f)
    partial_call = c(1)
    assert isinstance(partial_call, curry)
    assert partial_call(2) == ((1, 2), {})


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
    assert m.__name__ == 'add'


def test_memoize_as_decorator():
    @memoize
    def add(x, y):
        return x + y

    assert add(1, 2) == 3


def test_memoize_cache_argument():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        return x + y

    assert add(1, 2) == 3


def test_memoize_key_argument():
    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2, verbose=True) == 3


def test_memoize_unary():
    calls = []

    def square(x):
        calls.append(x)
        return x * x

    m = memoize(square)
    assert m(3) == 9
    assert m(3) == 9
    assert calls == [3]


def test_memoize_unhashable():
    def f(x):
        return x

    m = memoize(f)
    with pytest.raises(TypeError):
        m([1, 2, 3])


def test_memoize_docstring_preserved():
    def f(x):
        """my doc"""
        return x

    m = memoize(f)
    assert m.__doc__ == 'my doc'


def test_memoize_no_name():
    class Callable:
        def __call__(self, x):
            return x

    m = memoize(Callable())
    assert m(1) == 1


# ---------------------------------------------------------------------------
# compose / compose_left / pipe
# ---------------------------------------------------------------------------

def test_compose_no_args_identity():
    assert compose() is identity


def test_compose_single_func():
    def f(x):
        return x + 1

    assert compose(f) is f


def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'
    assert compose(inc, inc, inc)(0) == 3


def test_compose_left():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'
    assert compose_left() is identity


def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'
    assert pipe(3) == 3


def test_compose_equality_hash():
    f = lambda x: x
    g = lambda x: x
    c1 = compose(f, g)
    c2 = compose(f, g)
    assert c1 == c2
    assert hash(c1) == hash(c2)
    assert not (c1 == 5)
    assert (c1 != 5) is True


def test_compose_neq_notimplemented_other_way():
    f = lambda x: x
    c = compose(f, f)
    # Comparing against something that doesn't know how to compare
    assert c != object()


def test_compose_doc():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__doc__.startswith('lambda *args, **kwargs: ')
    assert 'f(g(' in c.__doc__


def test_compose_doc_no_name():
    class Callable:
        def __call__(self, x):
            return x

    c = compose(Callable(), Callable())
    assert c.__doc__ == 'A composition of functions'


def test_compose_name():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__name__ == 'f_of_g'


def test_compose_name_no_name():
    class Callable:
        def __call__(self, x):
            return x

    c = compose(Callable(), Callable())
    assert c.__name__ == 'Compose'


def test_compose_repr():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert repr(c).startswith('Compose(')


def test_compose_pickle():
    c = compose(_module_inc, _module_double)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(3) == _module_inc(_module_double(3))


def _module_inc(x):
    return x + 1


def _module_double(x):
    return 2 * x


def test_compose_signature():
    def f(x: int, y: int) -> str:
        return str(x + y)

    def g(a: int, b: int) -> int:
        return a + b

    c = compose(f, g)
    sig = c.__signature__
    assert sig.return_annotation is str
    assert list(sig.parameters) == ['a', 'b']


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__wrapped__ is g


def test_compose_descriptor():
    class C:
        def __init__(self, base):
            self.base = base

        def transform(self, x):
            return x + self.base

        method = compose(lambda x: x * 2, transform)

    c = C(3)
    assert c.method(2) == (2 + 3) * 2


def test_compose_annotations():
    def f(x: int) -> str:
        return str(x)

    def g(a: int) -> int:
        return a

    c = compose(f, g)
    ann = c.__annotations__
    assert ann['a'] is int
    assert ann['return'] is str


def test_compose_annotations_nested():
    def f(x: int) -> str:
        return str(x)

    def g(a: int) -> int:
        return a

    def h(a: int) -> int:
        return a

    c = compose(f, compose(g, h))
    ann = c.__annotations__
    assert ann['a'] is int
    assert ann['return'] is str


def test_compose_annotations_class_access():
    def f(x):
        return x

    c = compose(f, f)
    # accessed via class returns descriptor
    desc = Compose.__dict__['__annotations__']
    assert desc.__get__(None, Compose) is desc


@pytest.mark.skipif(sys.version_info < (3, 14),
                    reason="annotationlib is 3.14+")
def test_compose_annotate_forwardref():
    import annotationlib

    def f(x):
        return x

    c = compose(f, f)
    ann = c.__annotate__(annotationlib.Format.FORWARDREF)
    assert isinstance(ann, dict)


def test_compose_getstate_setstate():
    f = lambda x: x
    c = compose(f, f)
    state = c.__getstate__()
    first, funcs = state
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2.first == first
    assert c2.funcs == funcs


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


def test_juxt_empty():
    assert juxt()() == ()


def test_juxt_kwargs():
    def f(x, y=1):
        return x + y

    def g(x, y=1):
        return x * y

    assert juxt(f, g)(2, y=3) == (5, 6)


def test_juxt_pickle():
    f = _module_inc
    g = _module_double
    j = juxt(f, g)
    restored = pickle.loads(pickle.dumps(j))
    assert restored(3) == (4, 6)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    assert do(log.append, 5) == 5
    assert log == [5]


def test_do_compose():
    log = []
    inc = lambda x: x + 1
    f = compose(inc, do(log.append))
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
    assert return_none(ValueError()) is None
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


def test_excepts_doc():
    def f(x):
        """f doc"""
        return x

    def handler(e):
        """handler doc"""
        return None

    e = excepts(ValueError, f, handler)
    doc = e.__doc__
    assert "'f'" in doc
    assert 'ValueError' in doc
    assert "'handler'" in doc
    assert 'f doc' in doc
    assert 'handler doc' in doc
    # class level access
    assert excepts.__doc__ is not None


def test_excepts_doc_tuple():
    def f(x):
        return x

    e = excepts((ValueError, KeyError), f)
    assert '(ValueError, KeyError)' in e.__doc__


def test_excepts_doc_attribute_error():
    class NoName:
        def __call__(self, x):
            return x

    e = excepts(ValueError, NoName(), NoName())
    assert e.__doc__ == excepts.__doc__


def test_excepts_name():
    def f(x):
        return x

    e = excepts(ValueError, f)
    assert e.__name__ == 'f_excepting_ValueError'


def test_excepts_name_tuple():
    def f(x):
        return x

    e = excepts((ValueError, KeyError), f)
    assert e.__name__ == 'f_excepting_ValueError_or_KeyError'


def test_excepts_name_attribute_error():
    class NoName:
        def __call__(self, x):
            return x

    e = excepts(ValueError, NoName())
    assert e.__name__ == 'excepting'


# ---------------------------------------------------------------------------
# introspection helpers
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2

    def g(*args, **kwargs):
        return None

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


def test_is_arity_varargs():
    def f(*args):
        return args

    assert is_arity(1, f) is False


def test_is_arity_keywords():
    def f(x, **kwargs):
        return x

    assert is_arity(1, f) is False


def test_sigspec_passed_through():
    import inspect

    def f(x, y):
        return x + y

    sig = inspect.signature(f)
    assert num_required_args(f, sig) == 2
    assert has_varargs(f, sig) is False
    assert has_keywords(f, sig) is False
    assert is_valid_args(f, (1, 2), {}, sig) is True
    assert is_partial_args(f, (1,), {}, sig) is True
    assert is_arity(2, f, sig) is True


def test_builtin_has_no_signature_path():
    # These builtins use the fallback in toolz._signatures
    assert is_partial_args(map, (), {}) is True
    assert is_valid_args(map, (), {}) is False
