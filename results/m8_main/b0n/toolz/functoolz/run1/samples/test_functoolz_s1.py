import pickle
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
# identity / apply
# ---------------------------------------------------------------------------

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3


def test_apply_basic():
    assert apply(lambda x: x * 2, 5) == 10


def test_apply_kwargs():
    def f(a, b, c=0):
        return a + b + c
    assert apply(f, 1, 2, c=3) == 6


def test_apply_no_func():
    with pytest.raises(TypeError):
        apply()


def test_apply_map():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000
    )


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4


def test_thread_first_tuples():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(5) == 5


def test_thread_last_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4


def test_thread_last_tuples():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (pow_, 2)) == 32


def test_thread_last_no_forms():
    assert thread_last(5) == 5


def test_thread_last_map_filter():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_class_and_instance():
    class MyClass(object):
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


def test_instanceproperty_is_instance_property():
    class C:
        @instanceproperty
        def x(self):
            return 1

    assert isinstance(C.__dict__['x'], InstanceProperty)
    assert isinstance(C.__dict__['x'], property)


def test_instanceproperty_with_arguments():
    dec = instanceproperty(classval='classval')
    assert isinstance(dec, partial)

    class C:
        @dec
        def x(self):
            return 'instance'

    assert C.x == 'classval'
    assert C().x == 'instance'


def test_instanceproperty_reduce():
    class C:
        @instanceproperty(classval=5)
        def x(self):
            return 10

    prop = C.__dict__['x']
    cls, state = prop.__reduce__()
    assert cls is InstanceProperty
    assert state == (prop.fget, prop.fset, prop.fdel, prop.__doc__, 5)


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert double(10) == 20


def test_curry_decorator_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_no_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_non_callable():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_attributes():
    def foo(x, y):
        """dostring"""
        return x + y

    c = curry(foo)
    assert c.func is foo
    assert c.args == ()
    assert c.keywords == {}
    assert c.__name__ == 'foo'
    assert c.__doc__ == 'dostring'
    assert str(c) == str(foo)
    assert repr(c) == repr(foo)
    assert c.func_name == 'foo'


def test_curry_unpacks_partial_style():
    def add3(a, b, c):
        return a + b + c

    p = partial(add3, 1, 2)
    c = curry(p)
    assert c.func is add3
    assert c.args == (1, 2)
    assert c() == 6


def test_curry_unpacks_curry():
    def add3(a, b, c):
        return a + b + c

    c1 = curry(add3, 1)
    c2 = curry(c1, 2)
    assert c2.func is add3
    assert c2.args == (1, 2)
    assert c2(3) == 6


def test_curry_eq_hash():
    def f(x, y):
        return x + y

    c1 = curry(f, 1)
    c2 = curry(f, 1)
    c3 = curry(f, 2)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert c1 != "not a curry"


def test_curry_call_and_bind():
    def f(x, y, z):
        return x + y + z

    c = curry(f, 1)
    assert c.call(2, 3) == 6
    bound = c.bind(2)
    assert isinstance(bound, curry)
    assert bound(3) == 6


def test_curry_genuine_type_error_raises():
    @curry
    def f(x):
        return x + 1

    with pytest.raises(TypeError):
        f("a", "b")


def test_curry_varargs_returns_curry_on_typeerror():
    def varargs(*args, **kwargs):
        if len(args) < 2:
            raise TypeError("need 2")
        return args, kwargs

    c = curry(varargs, 1)
    # may return curry rather than raising because *args
    result = c(2, 3)
    assert result == ((1, 2, 3), {})


def test_curry_signature():
    def f(a, b, c=3):
        return a + b + c

    c = curry(f, 1)
    sig = c.__signature__
    names = list(sig.parameters)
    # 'a' consumed by positional args
    assert 'a' not in names
    assert 'b' in names
    assert 'c' in names


def test_curry_signature_incorrect_args():
    def f(a):
        return a

    # Force a bad partial through bind (bypass normal validation)
    c = curry(f)
    c2 = curry.__new__(curry)
    c2._partial = partial(f, 1, 2)
    c2._sigspec = None
    c2._has_unknown_args = None
    with pytest.raises(TypeError):
        c2.__signature__


def test_curry_get_descriptor():
    class C:
        @curry
        def method(self, x):
            return self.value + x

        value = 10

    c = C()
    # Accessing via class returns the curry itself
    assert isinstance(C.method, curry)
    # Accessing via instance binds
    assert C.method.__get__(c, C)(5) == 15


def test_curry_pickle_roundtrip():
    def add(x, y):
        return x + y

    # make it picklable via qualname lookup on module
    add.__qualname__ = 'add'
    globals()['add'] = add
    try:
        c = curry(add, 1)
        restored = pickle.loads(pickle.dumps(c))
        assert restored(2) == 3
    finally:
        del globals()['add']


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
    assert calls == [(1, 2)]


def test_memoize_with_cache():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        return x + y

    assert add(1, 2) == 3
    assert add(1, 1) == 2


def test_memoize_unary():
    calls = []

    @memoize
    def sq(x):
        calls.append(x)
        return x * x

    assert sq(3) == 9
    assert sq(3) == 9
    assert calls == [3]


def test_memoize_key_func():
    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_attributes():
    @memoize
    def foo(x):
        """foo doc"""
        return x

    assert foo.__name__ == 'foo'
    assert foo.__doc__ == 'foo doc'
    assert foo.__wrapped__ is not None


# ---------------------------------------------------------------------------
# compose / compose_left / Compose
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_empty():
    assert compose() is identity


def test_compose_single():
    f = lambda x: x + 1
    assert compose(f) is f


def test_compose_left():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_empty():
    assert compose_left() is identity


def test_compose_class():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    c = compose(double, inc)
    assert isinstance(c, Compose)
    assert c(3) == 8
    assert c.first is inc
    assert c.funcs == (double,)


def test_compose_equality():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c == compose(f, g)
    assert c != compose(g, f)
    assert (c == "other") is False


def test_compose_hash():
    def f(x):
        return x

    def g(x):
        return x

    assert hash(compose(f, g)) == hash(compose(f, g))


def test_compose_name_and_doc():
    def foo(x):
        return x

    def bar(x):
        return x

    c = compose(foo, bar)
    assert c.__name__ == 'foo_of_bar'
    assert 'bar' in c.__doc__
    assert 'foo' in c.__doc__


def test_compose_name_no_name():
    c = compose(len, lambda x: x)
    # lambda has __name__ == '<lambda>', len does not
    assert c.__name__ == 'len_of_<lambda>'


def test_compose_doc_no_name():
    class NoName:
        def __call__(self, x):
            return x

    c = compose(NoName(), lambda x: x)
    assert c.__doc__ == 'A composition of functions'


def test_compose_repr():
    def foo(x):
        return x

    def bar(x):
        return x

    c = compose(foo, bar)
    assert repr(c).startswith('Compose(')


def test_compose_signature():
    def f(x: int, y: str) -> str:
        return y * x

    def g(a: int, b: int) -> int:
        return a + b

    c = compose(f, g)
    sig = c.__signature__
    # parameters from g (first applied), return from f (last)
    assert list(sig.parameters) == ['a', 'b']
    assert sig.return_annotation == str


def test_compose_annotations():
    def f(x: int) -> str:
        return str(x)

    def g(a: float, b: float) -> int:
        return int(a + b)

    c = compose(f, g)
    annotations = c.__annotations__
    assert annotations['a'] == float
    assert annotations['b'] == float
    assert annotations['return'] == str


def test_compose_annotations_classval():
    # On the class, __annotations__ is the descriptor itself (empty dict)
    assert isinstance(Compose.__annotations__, dict)


def test_compose_get_descriptor():
    def f(x):
        return x + 1

    def g(self, x):
        return self.base + x

    c = compose(f, g)

    class C:
        base = 10
        method = c

    obj = C()
    # g(obj, x) then f(...)
    assert C.method is c
    assert obj.method(5) == 16


def test_compose_pickle():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    c = compose(f, g)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(3) == 7


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__wrapped__ is g


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'


def test_pipe_empty():
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

def test_juxt_basic():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_list():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_single_callable():
    inc = lambda x: x + 1
    j = juxt(inc)
    assert j(1) == (2,)


def test_juxt_kwargs():
    f = lambda x, y=0: x + y
    g = lambda x, y=0: x - y
    assert juxt(f, g)(5, y=2) == (7, 3)


def test_juxt_pickle():
    inc = len
    double = len
    j = juxt(inc, double)
    restored = pickle.loads(pickle.dumps(j))
    assert restored([1, 2, 3]) == (3, 3)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    result = do(log.append, 1)
    assert result == 1
    assert log == [1]


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
# return_none / excepts
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError("x")) is None


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


def test_excepts_attributes():
    def foo(a):
        return a

    def handler(e):
        return e

    e = excepts(ValueError, foo, handler)
    assert e.exc is ValueError
    assert e.func is foo
    assert e.handler is handler
    assert e.__name__ == 'foo_excepting_ValueError'


def test_excepts_name_tuple():
    def foo(a):
        return a

    e = excepts((IndexError, KeyError), foo)
    assert e.__name__ == 'foo_excepting_IndexError_or_KeyError'


def test_excepts_name_no_name():
    class NoName:
        def __call__(self, a):
            return a

    e = excepts(ValueError, NoName())
    assert e.__name__ == 'excepting'


def test_excepts_doc():
    def foo(a):
        """foo doc"""
        return a

    def handler(e):
        """handler doc"""
        return e

    e = excepts(ValueError, foo, handler)
    assert "A wrapper around 'foo'" in e.__doc__
    assert 'foo doc' in e.__doc__
    assert 'handler doc' in e.__doc__


def test_excepts_doc_classval():
    assert 'A wrapper around' in excepts.__doc__


def test_excepts_doc_no_name():
    class NoName:
        def __call__(self, a):
            return a

    e = excepts(ValueError, NoName())
    assert 'TypeError' in e.__doc__ or 'wrapper' in e.__doc__.lower()


# ---------------------------------------------------------------------------
# introspection helpers
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2

    def g(*args, **kwargs):
        pass

    assert num_required_args(g) == 0


def test_num_required_args_positional_only():
    def f(x, y, z=3, /, w=4):
        return x + y + z + w

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

    def g(x, *args):
        return x

    assert has_keywords(f) is True
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

    def g(x, y=1):
        return x + y

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False


def test_is_arity_varargs():
    def f(*args):
        return args

    assert is_arity(1, f) is False


# ---------------------------------------------------------------------------
# misc: InstanceProperty through curry func/args/keywords
# ---------------------------------------------------------------------------

def test_curry_func_is_instance_property():
    def f(x):
        return x

    c = curry(f)
    # accessing class attribute returns InstanceProperty descriptor
    assert isinstance(curry.__dict__['func'], InstanceProperty)
    assert c.func is f


def test_compose_empty_no_first():
    # Compose with one func: funcs is empty tuple
    c = Compose((lambda x: x,))
    assert c(5) == 5
    assert c.funcs == ()


def test_curry_eq_with_non_curry():
    def f(x):
        return x

    c = curry(f)
    assert (c == 5) is False
