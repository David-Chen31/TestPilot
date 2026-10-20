import pickle

import pytest

from toolz.functoolz import (
    identity, apply, thread_first, thread_last, instanceproperty,
    InstanceProperty, curry, memoize, Compose, compose, compose_left, pipe,
    complement, juxt, do, flip, return_none, excepts, num_required_args,
    has_varargs, has_keywords, is_valid_args, is_partial_args, is_arity,
)


def _mod_add(x, y):
    return x + y


# ---------------- identity / apply ----------------

def test_identity():
    assert identity(3) == 3
    obj = object()
    assert identity(obj) is obj


def test_apply_basic():
    assert apply(lambda x: 2 * x, 5) == 10
    assert apply(pow, 2, 3) == 8
    assert apply(pow, 2, 3, 5) == 32


def test_apply_kwargs():
    def f(a, b=10):
        return a + b
    assert apply(f, 1, b=2) == 3


def test_apply_requires_func():
    with pytest.raises(TypeError):
        apply()


# ---------------- thread_first / thread_last ----------------

def test_thread_first_callable():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1
    assert thread_first(1, inc, double) == 4


def test_thread_first_tuple():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y
    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_first_empty():
    assert thread_first(5) == 5


def test_thread_last_callable():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1
    assert thread_last(1, inc, double) == 4


def test_thread_last_tuple():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y
    assert thread_last(1, (add, 4), (pow_, 2)) == 32


def test_thread_last_empty():
    assert thread_last(5) == 5


# ---------------- instanceproperty / InstanceProperty ----------------

def test_instanceproperty_classval():
    class MyClass:
        @instanceproperty(classval='class-level')
        def val(self):
            return 42

    assert MyClass.val == 'class-level'
    assert MyClass().val == 42


def test_instanceproperty_no_classval():
    class MyClass:
        @instanceproperty
        def val(self):
            return 7

    assert MyClass.val is None
    assert MyClass().val == 7


def test_instanceproperty_partial_form():
    factory = instanceproperty(classval='cls')

    def fget(self):
        return 'inst'
    prop = factory(fget)
    assert isinstance(prop, InstanceProperty)
    assert prop.classval == 'cls'


def test_instanceproperty_setter():
    class MyClass:
        def __init__(self):
            self._v = 0

        @instanceproperty
        def v(self):
            return self._v

        @v.setter
        def v(self, value):
            self._v = value

    obj = MyClass()
    obj.v = 5
    assert obj.v == 5


def test_instanceproperty_reduce():
    def fget(self):
        return 1
    prop = InstanceProperty(fget=fget, classval='x')
    cls, state = prop.__reduce__()
    assert cls is InstanceProperty
    assert state == (fget, None, None, prop.__doc__, 'x')


# ---------------- curry ----------------

def test_curry_basic():
    def mul(x, y):
        return x * y
    mul_c = curry(mul)
    double = mul_c(2)
    assert double(10) == 20


def test_curry_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)
    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_requires_arg():
    with pytest.raises(TypeError):
        curry()


def test_curry_requires_callable():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_attributes():
    def myfunc(x, y=0):
        """myfunc doc"""
        return x + y
    myfunc.__module__ = 'mymod'
    c = curry(myfunc)
    assert c.func is myfunc
    assert c.__doc__ == 'myfunc doc'
    assert c.__name__ == 'myfunc'
    assert c.__module__ == 'mymod'
    assert c.func_name == 'myfunc'
    assert c.args == ()
    assert c.keywords == {}


def test_curry_str_repr():
    def myfunc(x):
        return x
    c = curry(myfunc)
    assert str(c) == str(myfunc)
    assert repr(c) == repr(myfunc)


def test_curry_eq():
    def f(x, y):
        return x + y
    a = curry(f, 1)
    b = curry(f, 1)
    c = curry(f, 2)
    assert a == b
    assert not (a == c)
    assert a != c
    assert (a == 5) is False


def test_curry_hash():
    def f(x):
        return x
    a = curry(f, 1)
    b = curry(f, 1)
    assert hash(a) == hash(b)


def test_curry_call_raises_typeerror():
    def f(x):
        return x
    c = curry(f)
    with pytest.raises(TypeError):
        c(1, 2)


def test_curry_bind():
    c = curry(_mod_add)
    bound = c.bind(1)
    assert isinstance(bound, curry)
    assert bound(2) == 3


def test_curry_call_method():
    c = curry(_mod_add, 1)
    assert c.call(2) == 3


def test_curry_descriptor():
    class MyClass:
        @curry
        def method(self, x, y):
            return x + y
    obj = MyClass()
    assert MyClass.method is MyClass.__dict__['method']
    bound = obj.method
    assert bound(1, 2) == 3


def test_curry_reduce_roundtrip():
    c = curry(_mod_add, 1)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == 3


def test_curry_reduce_roundtrip_keywords():
    c = curry(_mod_add, y=5)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == 7


def test_curry_signature():
    def add(x, y):
        return x + y
    c = curry(add, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    assert len(params) == 1
    assert params[0].name == 'y'


def test_curry_signature_invalid():
    def add(x, y):
        return x + y
    c = curry(add, 1, 2, 3)
    with pytest.raises(TypeError):
        _ = c.__signature__


def test_curry_from_partial_like_unpacking():
    import functools

    def add(x, y, z):
        return x + y + z
    p = functools.partial(add, 1)
    c = curry(p, 2)
    assert c(3) == 6


# ---------------- memoize ----------------

def test_memoize_basic():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y
    m = memoize(add)
    assert m(1, 2) == 3
    assert m(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_unary():
    calls = []

    def sq(x):
        calls.append(x)
        return x * x
    m = memoize(sq)
    assert m(3) == 9
    assert m(3) == 9
    assert calls == [3]


def test_memoize_initial_cache():
    @memoize(cache={(1, 2): 100})
    def add(x, y):
        return x + y
    assert add(1, 2) == 100
    assert add(2, 3) == 5


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y))
        return x + y
    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3
    assert calls == [(1, 2)]


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x
    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_attributes():
    def add(x, y):
        """add doc"""
        return x + y
    m = memoize(add)
    assert m.__name__ == 'add'
    assert m.__doc__ == 'add doc'
    assert m.__wrapped__ is add


# ---------------- compose / Compose / compose_left ----------------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_left_basic():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'


def test_compose_no_args():
    assert compose() is identity


def test_compose_single_arg():
    f = lambda x: x
    assert compose(f) is f


def test_compose_class():
    c = compose(str, int)
    assert isinstance(c, Compose)
    assert c('5') == '5'


def test_compose_name():
    def foo(x):
        return x

    def bar(x):
        return x
    c = compose(foo, bar)
    assert c.__name__ == 'foo_of_bar'


def test_compose_name_fallback():
    class Callable:
        def __call__(self, x):
            return x
    c = compose(Callable())
    assert isinstance(c, Callable)
    c2 = compose(Callable(), Callable())
    assert c2.__name__ == 'Compose'


def test_compose_repr():
    def foo(x):
        return x

    def bar(x):
        return x
    c = compose(foo, bar)
    assert 'Compose' in repr(c)


def test_compose_eq():
    def foo(x):
        return x

    def bar(x):
        return x
    assert compose(foo, bar) == compose(foo, bar)
    assert not (compose(foo, bar) == compose(bar, foo))
    assert compose(foo, bar).__ne__(5) is NotImplemented


def test_compose_hash():
    def foo(x):
        return x

    def bar(x):
        return x
    assert hash(compose(foo, bar)) == hash(compose(foo, bar))


def test_compose_signature():
    def f(a: int, b: int) -> int:
        return a + b

    def g(x) -> str:
        return str(x)
    c = compose(g, f)
    sig = c.__signature__
    assert list(sig.parameters) == ['a', 'b']
    assert sig.return_annotation is str


def test_compose_descriptor():
    class MyClass:
        method = compose(lambda self, x: x + 1)

    obj = MyClass()
    assert obj.method(1) == 2


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x
    c = compose(f, g)
    assert c.__wrapped__ is g


def test_compose_doc():
    def f(x):
        return x

    def g(x):
        return x
    c = compose(f, g)
    assert isinstance(c.__doc__, str)
    assert 'f' in c.__doc__
    assert 'g' in c.__doc__


def test_compose_getstate_setstate():
    def f(x):
        return x

    def g(x):
        return x
    c = compose(f, g)
    state = c.__getstate__()
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2(3) == c(3)


def test_compose_annotations():
    def f(a: int, b: int) -> int:
        return a + b

    def g(x: int) -> str:
        return str(x)
    c = compose(g, f)
    ann = c.__annotations__
    assert ann.get('a') is int
    assert ann.get('b') is int
    assert ann.get('return') is str


def test_compose_combined_annotations_single():
    def f(a: int) -> int:
        return a
    c = compose(f)
    # compose(single) returns the function itself, not a Compose
    assert c is f


# ---------------- pipe ----------------

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'
    assert pipe(5) == 5


# ---------------- complement ----------------

def test_complement():
    def iseven(n):
        return n % 2 == 0
    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


# ---------------- juxt ----------------

def test_juxt_basic():
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


def test_juxt_state():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    j = juxt(inc, double)
    assert j.__getstate__() == (inc, double)
    j2 = juxt.__new__(juxt)
    j2.__setstate__((inc, double))
    assert j2(3) == (4, 6)


# ---------------- do ----------------

def test_do():
    log = []
    result = do(log.append, 5)
    assert result == 5
    assert log == [5]


# ---------------- flip ----------------

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


# ---------------- return_none ----------------

def test_return_none():
    assert return_none(ValueError('x')) is None


# ---------------- excepts ----------------

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


def test_excepts_uncaught_propagates():
    excepting = excepts(ValueError, lambda a: a + 1)
    with pytest.raises(TypeError):
        excepting('x')


def test_excepts_name():
    e = excepts(ValueError, lambda a: a)
    assert e.__name__ == '<lambda>_excepting_ValueError'


def test_excepts_name_tuple():
    e = excepts((ValueError, KeyError), lambda a: a)
    assert e.__name__ == '<lambda>_excepting_ValueError_or_KeyError'


def test_excepts_name_fallback():
    e = excepts(ValueError, 5)
    assert e.__name__ == 'excepting'


def test_excepts_doc_classlevel():
    assert isinstance(excepts.__doc__, str)
    assert 'wrapper' in excepts.__doc__


def test_excepts_doc_instance():
    def myfunc(a):
        """myfunc docs"""
        return a

    def handler(e):
        """handler docs"""
        return None
    e = excepts(ValueError, myfunc, handler)
    assert 'myfunc' in e.__doc__
    assert 'handler' in e.__doc__
    assert 'ValueError' in e.__doc__


def test_excepts_attributes():
    e = excepts(ValueError, lambda a: a, return_none)
    assert e.exc is ValueError
    assert e.handler is return_none


# ---------------- introspection helpers ----------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z
    assert num_required_args(f) == 2

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

    def g(x):
        return x
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

    def h(*args):
        return args
    assert is_arity(1, h) is False
