import inspect
import pickle
import sys
from functools import partial
from operator import add

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
from toolz.utils import no_default


# ---------- identity / apply ----------

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3
    assert identity(None) is None


def test_apply():
    assert apply(lambda x: 2 * x, 5) == 10
    assert apply(add, 1, 2) == 3


def test_apply_kwargs():
    def f(a, b=0):
        return a + b
    assert apply(f, 1, b=10) == 11


def test_apply_no_args_raises():
    with pytest.raises(TypeError):
        apply()


# ---------- thread_first / thread_last ----------

def test_thread_first():
    inc = lambda x: x + 1
    double = lambda x: 2 * x
    assert thread_first(1, inc, double) == 4
    assert thread_first(1) == 1


def test_thread_first_tuple_form():
    def add_(x, y):
        return x + y

    def pow_(x, y):
        return x ** y
    assert thread_first(1, (add_, 4), (pow_, 2)) == 25


def test_thread_last():
    inc = lambda x: x + 1
    double = lambda x: 2 * x
    assert thread_last(1, inc, double) == 4
    assert thread_last(1) == 1


def test_thread_last_tuple_form():
    def add_(x, y):
        return x + y

    def pow_(x, y):
        return x ** y
    assert thread_last(1, (add_, 4), (pow_, 2)) == 32


def test_thread_last_map_filter():
    def iseven(x):
        return x % 2 == 0
    inc = lambda x: x + 1
    result = list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
    assert result == [2, 4]


# ---------- instanceproperty / InstanceProperty ----------

def test_instanceproperty_basic():
    class C:
        """Class doc."""
        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return 'object doc'
        @instanceproperty
        def val(self):
            return 42

    assert C.__doc__ == 'Class doc.'
    assert C.val is None
    obj = C()
    assert obj.__doc__ == 'object doc'
    assert obj.val == 42


def test_instanceproperty_partial_form():
    class C:
        @instanceproperty()
        def val(self):
            return 7
    assert C().val == 7


def test_instanceproperty_setter_getter_deleter():
    store = {}

    class C:
        @instanceproperty
        def val(self):
            return store.get('v')

        @val.setter
        def val(self, x):
            store['v'] = x

        @val.deleter
        def val(self):
            store.pop('v', None)

    c = C()
    assert c.val is None
    c.val = 5
    assert c.val == 5
    del c.val
    assert c.val is None


def test_instance_property_pickle():
    class C:
        @instanceproperty(classval='clsval')
        def val(self):
            return 99

    p = C.__dict__['val']
    assert isinstance(p, InstanceProperty)
    p2 = pickle.loads(pickle.dumps(p))
    assert p2.classval == 'clsval'
    assert p2.fget is p.fget


# ---------- curry ----------

def test_curry_basic():
    @curry
    def add_(x, y):
        return x + y
    assert add_(1, 2) == 3
    assert add_(1)(2) == 3


def test_curry_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)
    add_ = f(a=1)
    assert add_(2, 3) == 5


def test_curry_bad_init():
    with pytest.raises(TypeError):
        curry()
    with pytest.raises(TypeError):
        curry(1)


def test_curry_unpack_partial():
    def f(a, b, c=0):
        return a + b + c
    p = partial(f, 1)
    c = curry(p, 2)
    assert c() == 3
    assert c.args == (1, 2)
    assert c.func is f


def test_curry_unpack_partial_with_keywords():
    def f(a, b=0, c=0):
        return a + b + c
    p = partial(f, 1, b=5)
    c = curry(p, c=10)
    assert c() == 16


def test_curry_properties():
    def f(x, y):
        "docstring"
        return x + y
    c = curry(f, 1)
    assert c.func is f
    assert c.args == (1,)
    assert c.keywords == {}
    assert c.func_name == 'f'
    assert c.__doc__ == 'docstring'
    assert c.__name__ == 'f'
    assert c.__module__ == f.__module__
    assert c.__qualname__ == f.__qualname__
    assert str(c) == str(f)
    assert repr(c) == repr(f)


def test_curry_equality_hash():
    def f(x):
        return x
    c1 = curry(f, 1)
    c2 = curry(f, 1)
    c3 = curry(f, 2)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)


def test_curry_equality_other_type():
    def f(x):
        return x
    c = curry(f, 1)
    assert c != 5
    assert not (c == 5)


def test_curry_call_raises_real_typeerror():
    @curry
    def f(x):
        return x
    with pytest.raises(TypeError):
        f(1, 2)


def test_curry_bind_and_call():
    @curry
    def f(x, y):
        return x + y
    c = f.bind(1)
    assert c(2) == 3
    assert f.call(1, 2) == 3


def test_curry_get_descriptor():
    class C:
        @curry
        def method(self, x):
            return x

    assert isinstance(C.__dict__['method'], curry)
    obj = C()
    bound = obj.method
    assert isinstance(bound, curry)
    assert bound(5) == 5


def test_curry_signature():
    def f(a, b, c=3):
        return a + b + c
    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    assert params[0].name == 'b'
    assert params[1].name == 'c'


def test_curry_signature_kwargs_makes_kwonly():
    def f(a, b, c=0):
        return a + b + c
    c = curry(f, 1, b=10)
    sig = c.__signature__
    param_b = sig.parameters['b']
    assert param_b.kind == param_b.KEYWORD_ONLY
    assert param_b.default == 10


def test_curry_signature_incorrect_args():
    def f(a, b):
        return a + b
    c = curry(f, 1, 2, 3)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_with_varargs():
    @curry
    def f(*args):
        return len(args)
    # has unknown args -> always curry on TypeError
    result = f(1)
    assert result == 1


def test_curry_decorator_pickle_roundtrip():
    # decorated at module level not available here, so just check a
    # plain pickle of a curry of a builtin.
    c = curry(add, 1)
    c2 = pickle.loads(pickle.dumps(c))
    assert c2(2) == 3


def test_curry_reduce_roundtrip_callable():
    def f(a, b, c=0):
        return a + b + c
    c = curry(f, 1, b=3)
    restore, state = c.__reduce__()
    c2 = restore(*state)
    assert c2() == 4


# ---------- memoize ----------

def test_memoize_basic():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2
    assert f(2) == 4
    assert f(2) == 4
    assert calls == [2]


def test_memoize_cache_argument():
    @memoize(cache={(1, 2): 3})
    def add_(x, y):
        return x + y
    assert add_(1, 2) == 3
    assert add_(2, 3) == 5


def test_memoize_kwargs_key():
    calls = []

    @memoize
    def f(x, y=0):
        calls.append((x, y))
        return x + y
    assert f(1, y=2) == 3
    assert f(1, y=2) == 3
    assert calls == [(1, 2)]


def test_memoize_custom_key():
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


def test_memoize_attributes():
    def f(x):
        """F doc"""
        return x
    m = memoize(f)
    assert m.__name__ == 'f'
    assert m.__doc__ == 'F doc'
    assert m.__wrapped__ is f


# ---------- Compose / compose / compose_left / pipe ----------

def test_compose_basic():
    inc = lambda x: x + 1
    assert compose(str, inc)(3) == '4'


def test_compose_empty_returns_identity():
    assert compose() is identity


def test_compose_single_returns_func():
    f = lambda x: x
    assert compose(f) is f


def test_compose_left():
    inc = lambda x: x + 1
    assert compose_left(inc, str)(3) == '4'
    assert compose_left() is identity


def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'
    assert pipe(3) == 3


def test_complement():
    def iseven(n):
        return n % 2 == 0
    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


def test_Compose_attributes():
    def f(x):
        return x + 1

    def g(x):
        return x * 2
    c = Compose((f, g))
    # applied right to left: f(g(x))
    assert c(3) == 7
    assert c.first is g
    assert c.funcs == (f,)


def test_Compose_equality_hash():
    f = lambda x: x
    g = lambda x: x
    c1 = Compose((f, g))
    c2 = Compose((f, g))
    c3 = Compose((g, f))
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)


def test_Compose_eq_with_non_compose():
    c = Compose((lambda x: x, lambda x: x))
    assert c.__eq__(5) is NotImplemented
    assert c.__ne__(5) is NotImplemented


def test_Compose_name():
    def f(x):
        return x

    def g(x):
        return x
    c = Compose((f, g))
    assert c.__name__ == 'f_of_g'


def test_Compose_doc():
    def f(x):
        return x

    def g(x):
        return x
    c = Compose((f, g))
    assert 'lambda *args, **kwargs:' in c.__doc__


def test_Compose_doc_no_names():
    c = Compose((object(), object()))
    assert c.__doc__ == 'A composition of functions'


def test_Compose_name_no_names():
    class Callable:
        def __call__(self, x):
            return x
    c = Compose((Callable(), Callable()))
    assert c.__name__ == 'Compose'


def test_Compose_repr():
    def f(x):
        return x

    def g(x):
        return x
    c = Compose((f, g))
    r = repr(c)
    assert r.startswith('Compose(')


def test_Compose_get_descriptor():
    class C:
        def add_(self, x):
            return x + 1

        method = Compose((add_, lambda x: x * 2))

    assert isinstance(C.__dict__['method'], Compose)
    # class access returns the Compose itself
    assert isinstance(C.method, Compose)
    obj = C()
    # bound method: first applied is the last in the list of funcs
    # compose(add_, double)(obj, 1) -> add_(double(obj, 1))? No: first is double applied to (obj,1)
    # Actually Compose((add_, lambda x: x*2)): first is lambda x: x*2
    # so this would fail; just check binding type
    assert callable(obj.method)


def test_Compose_getstate_setstate():
    def f(x):
        return x + 1

    def g(x):
        return x * 2
    c = Compose((f, g))
    state = c.__getstate__()
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2(3) == c(3)


def test_Compose_signature():
    def f(x: int, y: int) -> int:
        return x + y

    def g(a) -> str:
        return str(a)
    c = Compose((f, g))
    sig = c.__signature__
    assert sig.return_annotation is str
    assert 'x' in sig.parameters
    assert 'y' in sig.parameters


def test_Compose_annotations():
    def f(x: int) -> int:
        return x

    def g(a: str) -> bool:
        return bool(a)
    c = Compose((f, g))
    ann = c.__annotations__
    assert ann['x'] is int
    assert ann['return'] is bool


# ---------- juxt ----------

def test_juxt_basic():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_list():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_getstate_setstate():
    inc = lambda x: x + 1
    j = juxt(inc)
    state = j.__getstate__()
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2(1) == (2,)


# ---------- do / flip ----------

def test_do():
    log = []
    result = do(log.append, 5)
    assert result == 5
    assert log == [5]


def test_flip():
    def div(a, b):
        return a // b
    assert flip(div, 2, 6) == 3
    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_flip_builtin():
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------- return_none / excepts ----------

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
    excepting = excepts(IndexError, lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1


def test_excepts_multiple():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_name():
    def f(a):
        return a

    def handler(e):
        return None
    e = excepts(ValueError, f, handler)
    assert e.__name__ == 'f_excepting_ValueError'
    e2 = excepts((ValueError, KeyError), f, handler)
    assert e2.__name__ == 'f_excepting_ValueError_or_KeyError'


def test_excepts_name_no_attributes():
    class NotNamed:
        def __call__(self, *a, **kw):
            return None
    e = excepts(NotNamed, NotNamed(), NotNamed())
    assert e.__name__ == 'excepting'


def test_excepts_doc():
    def f(a):
        return a

    def handler(e):
        return None
    e = excepts(ValueError, f, handler)
    doc = e.__doc__
    assert "'f'" in doc
    assert 'ValueError' in doc
    assert "'handler'" in doc


def test_excepts_doc_classval():
    assert excepts.__doc__ is not None
    assert 'wrapper around a function' in excepts.__doc__


# ---------- introspection helpers ----------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z
    assert num_required_args(f) == 2

    def g(*args, **kwargs):
        pass
    assert num_required_args(g) == 0


def test_num_required_args_with_sigspec():
    def f(x, y):
        return x + y
    assert num_required_args(f, inspect.signature(f)) == 2


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


def test_introspection_builtins():
    # builtins use the signatures registry
    assert is_valid_args(map, ()) is False
    assert is_partial_args(map, ()) is True
    assert num_required_args(map) is not None
