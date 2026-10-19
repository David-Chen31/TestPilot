import inspect
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
# identity, apply
# ---------------------------------------------------------------------------

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3
    assert identity(None) is None


def test_apply():
    def double(x):
        return 2 * x

    def add(x, y):
        return x + y

    assert apply(double, 5) == 10
    assert apply(add, 1, 2) == 3
    assert apply(add, x=1, y=2) == 3
    assert apply(add, 1, y=2) == 3


def test_apply_no_args():
    with pytest.raises(TypeError, match='func argument is required'):
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

def test_thread_first():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    def add(x, y):
        return x + y

    def pw(x, y):
        return x ** y

    assert thread_first(1, inc, double) == 4
    assert thread_first(1, (add, 4), (pw, 2)) == 25
    # no forms -> identity
    assert thread_first(5) == 5


def test_thread_last():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    def add(x, y):
        return x + y

    def pw(x, y):
        return x ** y

    assert thread_last(1, inc, double) == 4
    assert thread_last(1, (add, 4), (pw, 2)) == 32
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

def test_instanceproperty_classval_and_instance():
    class MyClass:
        """The class docstring"""

        @instanceproperty(classval='CV')
        def val(self):
            return 42

    assert MyClass.val == 'CV'
    obj = MyClass()
    assert obj.val == 42


def test_instanceproperty_without_classval():
    class MyClass:
        @instanceproperty
        def val(self):
            return 7

    assert MyClass.val is None
    assert MyClass().val == 7


def test_instanceproperty_with_arguments():
    class MyClass:
        def _get(self):
            return 1

    ip = instanceproperty(fget=MyClass._get, classval='X')
    assert isinstance(ip, InstanceProperty)
    assert ip.classval == 'X'


def test_instanceproperty_partial_factory():
    # fget is None -> returns partial
    p = instanceproperty(classval='abc')
    ip = p(lambda self: 5)
    assert isinstance(ip, InstanceProperty)
    assert ip.classval == 'abc'


def test_instance_property_setter():
    class MyClass:
        def __init__(self):
            self._x = 0

        def get(self):
            return self._x

        def setx(self, v):
            self._x = v

        x = InstanceProperty(fget=get, fset=setx)

    obj = MyClass()
    assert obj.x == 0
    obj.x = 10
    assert obj.x == 10


# module-level function for InstanceProperty pickling
def _instance_property_fget(self):
    return 42


def test_instanceproperty_pickle():
    ip = instanceproperty(fget=_instance_property_fget, classval='CV')
    data = pickle.dumps(ip)
    restored = pickle.loads(data)
    assert isinstance(restored, InstanceProperty)
    assert restored.classval == 'CV'
    assert restored.fget is _instance_property_fget


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert double(10) == 20
    assert cmul(3, 4) == 12


def test_curry_kwargs():
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
        curry(42)


def test_curry_unpacks_partial():
    from functools import partial

    def add3(x, y, z):
        return x + y + z

    p = partial(add3, 1, 2)
    c = curry(p)
    assert c.func is add3
    assert c.args == (1, 2)
    assert c(3) == 6


def test_curry_unpacks_curry():
    def add3(x, y, z):
        return x + y + z

    c1 = curry(add3, 1)
    c2 = curry(c1, 2)
    assert c2.func is add3
    assert c2.args == (1, 2)
    assert c2(3) == 6


def test_curry_attributes():
    def f(x):
        """docstring"""
        return x

    c = curry(f)
    assert c.func is f
    assert c.args == ()
    assert c.keywords == {}
    assert c.func_name == 'f'
    assert c.__doc__ == 'docstring'
    assert str(c) == str(f)
    assert repr(c) == repr(f)


def test_curry_hash_eq():
    def f(x, y):
        return x + y

    a = curry(f, 1)
    b = curry(f, 1)
    c = curry(f, 2)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert (a == 5) is False
    assert a.__ne__(5) is True


def test_curry_hash_keywords():
    def f(x, y=0):
        return x + y

    a = curry(f, 1, y=2)
    b = curry(f, 1, y=2)
    assert a == b
    assert hash(a) == hash(b)


def test_curry_call_genuine_typeerror():
    def f(x):
        return x + 'a'

    c = curry(f)
    with pytest.raises(TypeError):
        c(1)


def test_curry_too_many_args():
    def f(x, y):
        return x + y

    c = curry(f)
    with pytest.raises(TypeError):
        c(1, 2, 3)


def test_curry_varargs_curries_anyway():
    def deco(*args, **kwargs):
        # a decorator-like function requiring at least callable
        if not args:
            raise TypeError('need args')
        return args

    c = curry(deco)
    result = c()
    assert isinstance(result, curry)


def test_curry_bind_and_call():
    def f(x, y):
        return x + y

    c = curry(f)
    b = c.bind(1)
    assert isinstance(b, curry)
    assert b(2) == 3
    assert c.call(1, 2) == 3


def test_curry_signature():
    def f(a, b, c=3):
        return a + b + c

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    assert params[0].name == 'b'
    assert params[1].name == 'c'


def test_curry_signature_incorrect_args():
    def f(a, b):
        return a + b

    c = curry(f, 1, 2, 3)
    with pytest.raises(TypeError, match='incorrect arguments'):
        c.__signature__


def test_curry_signature_with_keywords():
    def f(a, b, c):
        return (a, b, c)

    c = curry(f, a=1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    # `a` is provided as a keyword, so it appears as a keyword-only parameter
    # with its default set to the bound value (1).
    assert params[0].name == 'a'
    assert params[0].kind == inspect.Parameter.KEYWORD_ONLY
    assert params[0].default == 1
    assert params[1].name == 'b'
    assert params[2].name == 'c'


def test_curry_descriptor():
    class C:
        @curry
        def method(self, x):
            return self.value + x

        value = 10

    c = C()
    assert c.method(5) == 15
    # accessing on class returns the curry itself
    assert isinstance(C.method, curry)


# Use a module-level function for pickle tests of curry
def _pickle_target(x, y):
    return x * y


curried_pickle_target = curry(_pickle_target)


def test_curry_pickle_roundtrip():
    c = curry(_pickle_target, 1)
    data = pickle.dumps(c)
    restored = pickle.loads(data)
    assert restored(2) == 2
    assert restored.args == (1,)


def test_curry_pickle_roundtrip_unbound():
    c = curry(_pickle_target)
    data = pickle.dumps(c)
    restored = pickle.loads(data)
    assert restored(3, 4) == 12
    assert restored.args == ()


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


def test_memoize_unary():
    calls = []

    @memoize
    def square(x):
        calls.append(x)
        return x * x

    assert square(3) == 9
    assert square(3) == 9
    assert calls == [3]


def test_memoize_with_kwargs():
    calls = []

    @memoize
    def f(x, y=0):
        calls.append((x, y))
        return x + y

    assert f(1, y=2) == 3
    assert f(1, y=2) == 3
    assert len(calls) == 1
    assert f(1, y=3) == 4
    assert len(calls) == 2


def test_memoize_key_function():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3
    assert len(calls) == 1


def test_memoize_initial_cache():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        raise AssertionError('should not be called')

    assert add(1, 2) == 3


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError, match='hashable'):
        f([1, 2, 3])


def test_memoize_attributes():
    def f(x):
        """f doc"""
        return x

    m = memoize(f)
    assert m.__name__ == 'f'
    assert m.__doc__ == 'f doc'
    assert m.__wrapped__ is f


# ---------------------------------------------------------------------------
# Compose / compose / compose_left
# ---------------------------------------------------------------------------

def test_compose_basic():
    def inc(i):
        return i + 1

    assert compose(str, inc)(3) == '4'


def test_compose_no_args_identity():
    f = compose()
    assert f is identity
    assert f(5) == 5


def test_compose_single():
    def f(x):
        return x

    assert compose(f) is f


def test_compose_multiple_args():
    def add(x, y):
        return x + y

    def double(x):
        return 2 * x

    assert compose(double, add)(1, 2) == 6


def test_compose_left():
    def inc(i):
        return i + 1

    assert compose_left(inc, str)(3) == '4'
    assert compose_left() is identity


def test_compose_doc():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    c = compose(double, inc)
    assert 'inc' in c.__doc__
    assert 'double' in c.__doc__


def test_compose_doc_no_names():
    class NoName:
        def __call__(self, x):
            return x

    c = compose(NoName(), NoName())
    assert 'NoName' in c.__doc__ or 'composition' in c.__doc__.lower()


def test_compose_name():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    c = compose(double, inc)
    assert c.__name__ == 'double_of_inc'


def test_compose_name_fallback():
    class NoName:
        def __call__(self, x):
            return x

    c = compose(NoName(), NoName())
    assert c.__name__ == 'Compose'


def test_compose_repr():
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    c = compose(double, inc)
    assert repr(c).startswith('Compose')


def test_compose_eq_hash():
    def a(x):
        return x

    def b(x):
        return x

    c1 = compose(a, b)
    c2 = compose(a, b)
    assert c1 == c2
    assert hash(c1) == hash(c2)
    assert c1 != 'not a compose'
    # __ne__ returns NotImplemented for non-Compose, so `!=` still works
    assert c1.__ne__('not a compose') is NotImplemented


def test_compose_ne_notimplemented():
    def a(x):
        return x

    def b(x):
        return x

    c1 = compose(a, b)
    other = object()
    assert c1.__eq__(other) is NotImplemented
    assert c1.__ne__(other) is NotImplemented


# module-level functions for pickling Compose
def _compose_pickle_a(x):
    return x + 1


def _compose_pickle_b(x):
    return x * 2


def test_compose_pickle():
    c = compose(_compose_pickle_a, _compose_pickle_b)
    data = pickle.dumps(c)
    restored = pickle.loads(data)
    assert restored(3) == _compose_pickle_a(_compose_pickle_b(3))


def test_compose_descriptor():
    def double(x):
        return x * 2

    def inc(x):
        return x + 1

    comp = compose(double, inc)

    class C:
        value = 5
        method = comp

    obj = C()
    # When accessed via an instance, the Compose becomes a bound method, so
    # the instance is passed as the first (only) argument to `comp`. Here we
    # test that it is bound (MethodType) and callable.
    assert hasattr(obj.method, '__self__')
    assert obj.method.__self__ is obj


def test_compose_signature():
    def f(a: int, b: str) -> float:
        return 0.0

    def g(x) -> int:
        return 0

    # compose(f, g) applies g first, then f. So the signature parameters come
    # from the last-applied (first) function `g`, and the return annotation
    # comes from `f`.
    c = compose(f, g)
    sig = c.__signature__
    assert sig.return_annotation == float
    assert list(sig.parameters) == ['x']


def test_compose_signature_one_func():
    def f(a: int):
        return a

    c = compose(f, f)
    sig = c.__signature__
    assert 'a' in sig.parameters


def test_compose_wrapped():
    def inner(x):
        return x

    c = compose(inner, inner)
    assert c.__wrapped__ is inner


def test_compose_annotations():
    # Annotations are not plain types on this Python; compare them to what
    # inspect.signature reports for the same functions.
    def f(a: int, b: str) -> float:
        return 0.0

    def g(x) -> int:
        return 0

    c = compose(f, g)
    sig_f = inspect.signature(f)
    ann = c.__annotations__
    # The implementation combines annotations from first and last functions.
    # For this simple case the result contains the return annotation from
    # `f` (the last-applied function).
    assert 'return' in ann
    assert ann['return'] == sig_f.return_annotation


def test_compose_annotations_no_return():
    def f(a: int):
        return a

    def g(x):
        return x

    # compose(f, g): params from g (no annotations), return from f (none).
    c = compose(f, g)
    ann = c.__annotations__
    assert 'x' not in ann
    assert 'return' not in ann


def test_compose_annotations_missing_signature():
    class BadSig:
        def __call__(self, x):
            return x

        @property
        def __signature__(self):
            raise ValueError('no sig')

    def f(a: int):
        return a

    # compose(f, BadSig()): BadSig applied first; its signature fails, so it
    # contributes nothing. `f` is last-applied so its return annotation (none)
    # is used. Therefore annotations should be empty.
    c = compose(f, BadSig())
    ann = c.__annotations__
    assert ann == {}


def test_compose_getstate_setstate():
    def f(x):
        return x

    c = Compose((f, f))
    state = c.__getstate__()
    new = Compose.__new__(Compose)
    new.__setstate__(state)
    assert new.first == c.first
    assert new.funcs == c.funcs


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe():
    def double(i):
        return 2 * i

    assert pipe(3, double, str) == '6'
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
    def inc(x):
        return x + 1

    def double(x):
        return x * 2

    assert juxt(inc, double)(10) == (11, 20)
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_with_kwargs():
    def add(a, b=0):
        return a + b

    assert juxt(add, add)(1, b=2) == (3, 3)


# module-level function for pickling juxt
def _juxt_pickle_inc(x):
    return x + 1


def test_juxt_pickle():
    j = juxt(_juxt_pickle_inc)
    data = pickle.dumps(j)
    restored = pickle.loads(data)
    assert restored(1) == (2,)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    result = do(log.append, 10)
    assert result == 10
    assert log == [10]


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
# return_none
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError('x')) is None


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
    e = excepts(ValueError, lambda x: x, lambda exc: None)
    assert e.__name__ == '<lambda>_excepting_ValueError'


def test_excepts_tuple_name():
    e = excepts((ValueError, KeyError), lambda x: x)
    assert e.__name__ == '<lambda>_excepting_ValueError_or_KeyError'


def test_excepts_name_fallback():
    class NotCallable:
        pass

    e = excepts(ValueError, NotCallable())
    assert e.__name__ == 'excepting'


def test_excepts_doc():
    def func(x):
        """func doc"""
        return x

    def handler(exc):
        """handler doc"""
        return None

    e = excepts(ValueError, func, handler)
    doc = e.__doc__
    assert 'ValueError' in doc
    assert 'func' in doc
    assert 'handler' in doc
    assert 'func doc' in doc
    assert 'handler doc' in doc


def test_excepts_tuple_doc():
    e = excepts((ValueError, KeyError), lambda x: x)
    assert '(ValueError, KeyError)' in e.__doc__


def test_excepts_doc_fallback():
    class NotCallable:
        pass

    e = excepts(ValueError, NotCallable())
    assert 'functional try/except' in e.__doc__


def test_excepts_class_doc():
    assert 'functional try/except' in excepts.__doc__


# ---------------------------------------------------------------------------
# introspection helpers
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    def g(*args, **kwargs):
        pass

    def h(a, b):
        return a + b

    assert num_required_args(f) == 2
    assert num_required_args(g) == 0
    assert num_required_args(h) == 2


def test_has_varargs():
    def f(*args):
        return args

    def g(**kwargs):
        return kwargs

    def h(x):
        return x

    assert has_varargs(f) is True
    assert has_varargs(g) is False
    assert has_varargs(h) is False


def test_has_keywords():
    def f(x, y=0):
        return x + y

    def g(x):
        return x

    def h(*args, **kwargs):
        return None

    assert has_keywords(f) is True
    assert has_keywords(g) is False
    assert has_keywords(h) is True


def test_is_valid_args():
    def add(x, y):
        return x + y

    assert is_valid_args(add, (1,), {}) is False
    assert is_valid_args(add, (1, 2), {}) is True


def test_is_partial_args():
    def add(x, y):
        return x + y

    assert is_partial_args(add, (1,), {}) is True
    assert is_partial_args(add, (1, 2), {}) is True
    assert is_partial_args(add, (1, 2, 3), {}) is False


def test_is_arity():
    def f(x):
        return x

    def g(x, y=1):
        return x + y

    def h(*args):
        return args

    def k(x, **kwargs):
        return x

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False
    assert is_arity(1, h) is False
    assert is_arity(1, k) is False


def test_is_arity_sigspec():
    def f(x):
        return x

    sig = inspect.signature(f)
    assert is_arity(1, f, sig) is True


def test_introspection_builtin_or_unknown():
    # Some builtins may or may not have signatures; these should not raise.
    result = num_required_args(dict)
    assert result is None or isinstance(result, int)
