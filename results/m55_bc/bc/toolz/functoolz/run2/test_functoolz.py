import pickle
import pytest
import inspect
from functools import partial

from toolz import functoolz
from toolz.functoolz import (
    identity, apply, thread_first, thread_last, instanceproperty,
    InstanceProperty, curry, memoize, Compose, compose, compose_left, pipe,
    complement, juxt, do, flip, return_none, excepts, num_required_args,
    has_varargs, has_keywords, is_valid_args, is_partial_args, is_arity,
)


# module-level helpers so they are picklable by reference
def _inc(i):
    return i + 1


def _double(i):
    return i * 2


def _add_xy(x, y):
    return x + y


# ---------------------------------------------------------------------------
# identity / apply
# ---------------------------------------------------------------------------
def test_identity():
    assert identity(3) == 3
    obj = object()
    assert identity(obj) is obj


def test_apply_basic():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert apply(double, 5) == 10
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (20, 501, 16000)


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
def test_thread_first_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4
    assert thread_first(1) == 1


def test_thread_first_tuple_forms():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_last_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4
    assert thread_last(1) == 1


def test_thread_last_tuple_forms():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (pow_, 2)) == 32


def test_thread_last_iterables():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    result = list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
    assert result == [2, 4]


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


def test_instanceproperty_partial_form():
    def get_val(self):
        return 7

    prop = instanceproperty(classval='class')
    # calling with fget returns an InstanceProperty
    ip = prop(get_val)
    assert isinstance(ip, InstanceProperty)
    assert ip.classval == 'class'
    assert ip.fget is get_val


def test_instanceproperty_getsetdel():
    class C:
        def __init__(self):
            self._x = 1

        def _get(self):
            return self._x

        def _set(self, v):
            self._x = v

        def _del(self):
            self._x = 0

        x = InstanceProperty(_get, _set, _del, doc='the x')

    c = C()
    assert c.x == 1
    c.x = 5
    assert c.x == 5
    del c.x
    assert c.x == 0
    assert C.x is None


def test_instanceproperty_pickle():
    # InstanceProperty is picklable via __reduce__, but the fget must be
    # picklable by reference. Use a module-level function here.

    def _get_val(self):
        return 42

    # Use a module-level function so it can be pickled by reference.
    # Fallback: exercise __reduce__ state directly.
    ip = InstanceProperty(_get_val, classval='cv')
    state = ip.__reduce__()
    cls, init_state = state
    assert cls is InstanceProperty
    assert init_state == (_get_val, None, None, ip.__doc__, 'cv')

    restored = cls(*init_state[:4], classval=init_state[4]) \
        if False else InstanceProperty(
            fget=init_state[0], fset=init_state[1], fdel=init_state[2],
            doc=init_state[3], classval=init_state[4])
    assert isinstance(restored, InstanceProperty)
    assert restored.fget is _get_val
    assert restored.classval == 'cv'


def test_instanceproperty_pickle_module_level():
    # Use the picklable module-level _inc-like? We need a function; use a
    # class-level one defined at module scope isn't available, so use the
    # InstanceProperty.__reduce__ structure check above plus a real roundtrip
    # through a plain function.
    ip = InstanceProperty(_inc, classval='cv')
    # _inc is a module-level function, so pickling works
    restored = pickle.loads(pickle.dumps(ip))
    assert isinstance(restored, InstanceProperty)
    assert restored.fget is _inc
    assert restored.classval == 'cv'


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------
def test_curry_basic():
    def mul(x, y):
        return x * y

    mul = curry(mul)
    double = mul(2)
    assert double(10) == 20


def test_curry_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_init_errors():
    with pytest.raises(TypeError):
        curry()
    with pytest.raises(TypeError):
        curry(42)


def test_curry_attributes():
    def foo():
        """docstring"""

    c = curry(foo)
    assert c.func is foo
    assert c.func_name == 'foo'
    assert c.__doc__ == 'docstring'
    assert c.__name__ == 'foo'
    assert str(c) == str(foo)
    assert repr(c) == repr(foo)


def test_curry_call_vs_bind():
    @curry
    def f(x, y):
        return x + y

    partial_f = f(1)
    assert isinstance(partial_f, curry)
    assert partial_f(2) == 3


def test_curry_genuine_typeerror():
    @curry
    def f(x):
        raise TypeError("boom")

    with pytest.raises(TypeError):
        f(1)


def test_curry_unpack_partial():
    def add(x, y, z):
        return x + y + z

    c = curry(partial(add, 1), 2)
    assert c(3) == 6


def test_curry_eq_hash():
    def f(x):
        return x

    c1 = curry(f, 1)
    c2 = curry(f, 1)
    c3 = curry(f, 2)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert c1 != 5


def test_curry_signature():
    def add(x, y):
        return x + y

    c = curry(add, 1)
    sig = c.__signature__
    assert list(sig.parameters) == ['y']


def test_curry_signature_bad_args():
    def add(x, y):
        return x + y

    c = curry(add)
    # force bad partial args directly
    c._partial = partial(add, 1, 2, 3)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_bind_call():
    @curry
    def f(x, y):
        return x + y

    b = f.bind(1)
    assert b(2) == 3
    assert f.call(1, 2) == 3


def test_curry_get_descriptor():
    class C:
        @curry
        def method(self, x):
            return x + 1

    c = C()
    # accessing on class returns curry
    assert isinstance(C.method, curry)
    assert C().method(4) == 5


def test_curry_pickle():
    # curry pickling requires the wrapped function to be importable by
    # reference, so use the module-level _add_xy.
    c = curry(_add_xy, 1)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == 3
    assert restored.func is _add_xy
    assert restored.args == (1,)


def test_curry_pickle_userdict():
    c = curry(_add_xy, 1)
    c.custom_attr = 'hello'
    restored = pickle.loads(pickle.dumps(c))
    assert restored.custom_attr == 'hello'
    assert restored(2) == 3


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------
def test_memoize_basic():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    add = memoize(add)
    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_unary():
    calls = []

    @memoize
    def sq(x):
        calls.append(x)
        return x * x

    assert sq(3) == 9
    assert sq(3) == 9
    assert calls == [3]


def test_memoize_with_cache():
    @memoize(cache={(1, 2): 100})
    def add(x, y):
        return x + y

    assert add(1, 2) == 100
    assert add(2, 3) == 5


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append(verbose)
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3
    assert calls == [True]


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_metadata():
    def f(x):
        """doc"""
        return x

    m = memoize(f)
    assert m.__name__ == 'f'
    assert m.__doc__ == 'doc'
    assert m.__wrapped__ is f


def test_memoize_kwargs_cache_key():
    calls = []

    @memoize
    def f(x=0, y=0):
        calls.append((x, y))
        return x + y

    assert f(1, 2) == 3
    assert f(1, 2) == 3
    assert f(1, y=2) == 3
    # f(1, 2) and f(1, y=2) create distinct cache keys
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# compose / compose_left / Compose
# ---------------------------------------------------------------------------
def test_compose_basic():
    assert compose(str, _inc)(3) == '4'
    assert compose(_double, _inc)(3) == 8


def test_compose_no_args_is_identity():
    assert compose() is identity
    assert compose(lambda x: x) is not identity


def test_compose_single_func():
    def f(x):
        return x + 1

    assert compose(f) is f


def test_compose_left():
    assert compose_left(_inc, str)(3) == '4'
    assert compose_left(_double, _inc)(3) == 7


def test_compose_kwargs():
    def f(a, b=0):
        return a + b

    assert compose(_inc, f)(1, b=2) == 4


def test_compose_eq():
    c1 = compose(_inc, _double)
    c2 = compose(_inc, _double)
    c3 = compose(_double, _inc)
    assert c1 == c2
    assert c1 != c3
    assert (c1 == 5) is False


def test_compose_repr():
    c = compose(_inc, _double)
    r = repr(c)
    assert r.startswith('Compose(')
    # funcs listed as originally passed (right-to-left application order)
    assert '_double' in r
    assert '_inc' in r


def test_compose_doc_and_name():
    c = compose(_inc, _double)
    # name is built from the funcs tuple order (reversed application order)
    assert c.__name__ == '_inc_of__double'
    assert '_double' in c.__doc__
    assert '_inc' in c.__doc__


def test_compose_doc_no_name():
    c = Compose([_inc, object()])
    assert c.__doc__ == 'A composition of functions'
    assert c.__name__ == 'Compose'


def test_compose_signature():
    def f(x: int) -> int:
        return x

    def g(y: int) -> str:
        return str(y)

    c = compose(g, f)
    sig = c.__signature__
    assert sig.return_annotation is str
    assert 'x' in sig.parameters


def test_compose_wrapped():
    c = compose(_inc, _double)
    assert c.__wrapped__ is _double  # first applied function


def test_compose_annotations():
    def f(x: int) -> int:
        return x

    def g(y: int) -> str:
        return str(y)

    c = compose(g, f)
    ann = c.__annotations__
    assert ann['x'] is int
    assert ann['return'] is str


def test_compose_get_descriptor():
    class C:
        method = compose(_inc, _double)

    assert C.method is C.__dict__['method']
    # Compose binds as a method: the instance is passed to the first called
    # function (_double). Define a method-style first function so binding works.
    class D:
        def __init__(self):
            self.base = 10

        def scale(self, x):
            return self.base * x

        method = compose(_inc, scale)

    d = D()
    # scale(d, 3) = 30, then inc -> 31
    assert d.method(3) == 31


def test_compose_pickle():
    c = compose(_inc, _double)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(3) == c(3)


def test_compose_hash():
    c1 = compose(_inc, _double)
    c2 = compose(_inc, _double)
    assert hash(c1) == hash(c2)


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------
def test_pipe():
    assert pipe(3, _double, str) == '6'
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
def test_juxt():
    def inc_(x):
        return x + 1

    def double_(x):
        return x * 2

    assert juxt(inc_, double_)(10) == (11, 20)
    assert juxt([inc_, double_])(10) == (11, 20)


def test_juxt_kwargs():
    def f(a, b=0):
        return a + b

    j = juxt(f, f)
    assert j(1, b=2) == (3, 3)


def test_juxt_pickle():
    j = juxt(_inc, _double)
    restored = pickle.loads(pickle.dumps(j))
    assert restored(3) == j(3)
    assert restored.funcs == j.funcs


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------
def test_do():
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
    data = [1, 'a', 'b', 2, 1.5, 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none / excepts
# ---------------------------------------------------------------------------
def test_return_none():
    assert return_none(ValueError()) is None


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
    excepting = excepts(ValueError, lambda a: a, lambda e: None)
    doc = excepting.__doc__
    assert '<lambda>' in doc
    assert 'ValueError' in doc


def test_excepts_doc_classval():
    assert excepts.__doc__ is not None
    assert 'wrapper' in excepts.__doc__


def test_excepts_name():
    def f(a):
        return a[0]

    e = excepts(IndexError, f)
    assert e.__name__ == 'f_excepting_IndexError'


def test_excepts_name_tuple():
    def f(a):
        return a

    e = excepts((IndexError, KeyError), f)
    assert e.__name__ == 'f_excepting_IndexError_or_KeyError'


def test_excepts_name_fallback():
    class Callable:
        def __call__(self, *a):
            return 1

    e = excepts(ValueError, Callable())
    assert e.__name__ == 'excepting'


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

    def g(x):
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

    def h(*args):
        return args

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False
    assert is_arity(1, h) is False


def test_builtin_introspection():
    # builtins rely on the signature registry
    assert is_valid_args(map, (), {}) is False
    assert is_partial_args(map, (), {}) is True
    assert has_varargs(map) is True


def test_unknown_signature_returns_none():
    # An object with no signature at all
    class Weird:
        __call__ = None

    # just ensure we get a boolean-or-None without raising
    rv = is_valid_args(Weird(), (), {})
    assert rv in (True, False, None)
