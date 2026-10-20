import functools
import inspect
import pytest

from funcy.decorators import (
    decorator,
    make_decorator,
    Call,
    has_single_arg,
    has_1pos_and_kwonly,
    get_argnames,
    arggetter,
    update_wrapper,
    wraps,
)


# ---------------------------------------------------------------------------
# has_single_arg
# ---------------------------------------------------------------------------

def test_has_single_arg_true():
    def f(call):
        pass
    assert has_single_arg(f) is True


def test_has_single_arg_with_default():
    def f(call=None):
        pass
    assert has_single_arg(f) is True


def test_has_single_arg_false_two_args():
    def f(a, b):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_false_no_args():
    def f():
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_starargs():
    def f(*args):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_kwargs():
    def f(**kwargs):
        pass
    assert has_single_arg(f) is False


# ---------------------------------------------------------------------------
# has_1pos_and_kwonly
# ---------------------------------------------------------------------------

def test_has_1pos_and_kwonly_simple():
    def f(call):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_with_kwonly():
    def f(call, *, a, b=1):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_posonly():
    def f(call, /, *, a):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_varargs_rejects():
    def f(call, *args, a):
        pass
    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_two_positional():
    def f(a, b):
        pass
    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_no_positional():
    def f(*, a):
        pass
    assert has_1pos_and_kwonly(f) is False


# ---------------------------------------------------------------------------
# get_argnames
# ---------------------------------------------------------------------------

def test_get_argnames_basic():
    def f(a, b, c):
        pass
    assert get_argnames(f) == ('a', 'b', 'c')


def test_get_argnames_kwonly():
    def f(a, *, b, c):
        pass
    assert get_argnames(f) == ('a', 'b', 'c')


def test_get_argnames_varargs_varkw():
    def f(a, *args, **kwargs):
        pass
    assert get_argnames(f) == ('a', 'args', 'kwargs')


def test_get_argnames_includes_defaults():
    def f(a, b=2):
        pass
    assert get_argnames(f) == ('a', 'b')


def test_get_argnames_empty():
    def f():
        pass
    assert get_argnames(f) == ()


def test_get_argnames_uses_original():
    def original(a, b):
        pass

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        pass
    wrapper.__original__ = original

    assert get_argnames(wrapper) == ('a', 'b')


# ---------------------------------------------------------------------------
# arggetter
# ---------------------------------------------------------------------------

def test_arggetter_positional():
    def f(a, b):
        pass
    get = arggetter(f)
    assert get('a', (1, 2), {}) == 1
    assert get('b', (1, 2), {}) == 2


def test_arggetter_keyword():
    def f(a, b):
        pass
    get = arggetter(f)
    assert get('a', (), {'a': 10, 'b': 20}) == 10
    assert get('b', (), {'a': 10, 'b': 20}) == 20


def test_arggetter_default():
    def f(a, b=42):
        pass
    get = arggetter(f)
    assert get('b', (1,), {}) == 42


def test_arggetter_kwonly_default():
    def f(a, *, b=99):
        pass
    get = arggetter(f)
    assert get('b', (1,), {}) == 99


def test_arggetter_varargs():
    def f(a, *args):
        pass
    get = arggetter(f)
    assert get('args', (1, 2, 3), {}) == (2, 3)


def test_arggetter_varkw():
    def f(a, **kwargs):
        pass
    get = arggetter(f)
    res = get('kwargs', (1,), {'x': 5, 'y': 6})
    assert res == {'x': 5, 'y': 6}


def test_arggetter_varkw_excludes_named():
    def f(a, b, **kwargs):
        pass
    get = arggetter(f)
    res = get('kwargs', (1,), {'b': 2, 'x': 3})
    assert res == {'x': 3}


def test_arggetter_unknown_arg_raises():
    def f(a):
        pass
    get = arggetter(f)
    with pytest.raises(TypeError, match="doesn't have argument named"):
        get('zzz', (), {})


def test_arggetter_missing_required_raises():
    def f(a, b):
        pass
    get = arggetter(f)
    with pytest.raises(TypeError, match="missing required argument"):
        get('b', (), {})


def test_arggetter_cache_returns_same():
    def f(a):
        pass
    g1 = arggetter(f)
    g2 = arggetter(f)
    assert g1 is g2


def test_arggetter_posonly():
    def f(a, b, /, c):
        pass
    get = arggetter(f)
    assert get('a', (1, 2, 3), {}) == 1
    assert get('b', (1, 2, 3), {}) == 2
    assert get('c', (1, 2, 3), {}) == 3


def test_arggetter_method_skips_self():
    class C:
        def method(self, x):
            pass
    get = arggetter(C.method)
    # call as unbound with self prepended (via Call)
    assert get('x', (C(), 7), {}) == 7


# ---------------------------------------------------------------------------
# Call
# ---------------------------------------------------------------------------

def test_call_basic_call():
    def f(a, b):
        return a + b
    c = Call(f, (1,), {'b': 2})
    assert c() == 3


def test_call_override_args():
    def f(a, b, c=0):
        return (a, b, c)
    c = Call(f, (1,), {'b': 2})
    assert c(10) == (10, 2, 0)
    assert c(c=5) == (1, 2, 5)


def test_call_getattr_arg():
    def f(a, b):
        return None
    c = Call(f, (1,), {'b': 2})
    assert c.a == 1
    assert c.b == 2


def test_call_getattr_caches():
    def f(a):
        return None
    c = Call(f, (5,), {})
    assert c.a == 5
    # cached in instance dict
    assert 'a' in c.__dict__
    assert c.__dict__['a'] == 5


def test_call_getattr_unknown():
    def f(a):
        return None
    c = Call(f, (1,), {})
    with pytest.raises(AttributeError):
        c.nonexistent


def test_call_str():
    def myfunc(a, b):
        return None
    c = Call(myfunc, (1,), {'b': 2})
    s = str(c)
    assert s == "myfunc(1, b=2)"


def test_call_repr():
    def myfunc(a):
        return None
    c = Call(myfunc, (1,), {})
    assert repr(c) == "<Call myfunc(1)>"


def test_call_str_qualname_fallback():
    class Obj:
        def __call__(self, a):
            return None
    obj = Obj()
    c = Call(obj, (1,), {})
    s = str(c)
    assert '1' in s
    assert s.startswith('<') or s.startswith('Obj') or '1' in s


# ---------------------------------------------------------------------------
# make_decorator
# ---------------------------------------------------------------------------

def test_make_decorator_simple():
    def deco(call):
        return call() * 2

    dec = make_decorator(deco)

    @dec
    def f(x):
        return x + 1

    assert f(3) == 8


def test_make_decorator_with_dargs():
    def deco(call, n):
        return call() + n

    dec = make_decorator(deco, (10,))

    @dec
    def f(x):
        return x

    assert f(5) == 15


def test_make_decorator_with_dkwargs():
    def deco(call, n=1):
        return call() * n

    dec = make_decorator(deco, (), {'n': 3})

    @dec
    def f(x):
        return x

    assert f(4) == 12


def test_make_decorator_stores_introspection():
    def deco(call):
        return call()

    dec = make_decorator(deco, (1,), {'x': 2})
    assert dec._func is deco
    assert dec._args == (1,)
    assert dec._kwargs == {'x': 2}


def test_make_decorator_preserves_metadata():
    def deco(call):
        return call()

    dec = make_decorator(deco)

    @dec
    def myfunction(a, b):
        """docstring here"""
        return a + b

    assert myfunction.__name__ == 'myfunction'
    assert myfunction.__doc__ == 'docstring here'


def test_make_decorator_call_accesses_args():
    def deco(call):
        return call.a * call.b

    dec = make_decorator(deco)

    @dec
    def f(a, b):
        return None

    assert f(3, 4) == 12


# ---------------------------------------------------------------------------
# decorator
# ---------------------------------------------------------------------------

def test_decorator_single_arg():
    @decorator
    def deco(call):
        return call() + 1

    @deco
    def f(x):
        return x

    assert f(10) == 11


def test_decorator_single_arg_uses_args():
    @decorator
    def deco(call):
        return call()

    @deco
    def f(a, b, c=3):
        return a + b + c

    assert f(1, 2) == 6
    assert f(1, 2, c=10) == 13


def test_decorator_with_args_fab():
    @decorator
    def deco(call, n):
        return call() * n

    @deco(n=5)
    def f(x):
        return x

    assert f(2) == 10


def test_decorator_with_args_direct():
    @decorator
    def deco(call, n):
        return call() + n

    @deco(10)
    def f(x):
        return x

    assert f(1) == 11


def test_decorator_with_args_kwonly():
    @decorator
    def deco(call, *, mult=2):
        return call() * mult

    @deco()
    def f(x):
        return x

    @deco(mult=3)
    def g(x):
        return x

    assert f(4) == 8
    assert g(4) == 12


def test_decorator_with_varargs():
    @decorator
    def deco(call, *multipliers):
        result = call()
        for m in multipliers:
            result *= m
        return result

    @deco(2, 3)
    def f(x):
        return x

    assert f(5) == 30


def test_decorator_with_varkwargs():
    @decorator
    def deco(call, **opts):
        return call() + opts.get('offset', 0)

    @deco(offset=100)
    def f(x):
        return x

    assert f(1) == 101


def test_decorator_preserves_func_name():
    @decorator
    def deco(call):
        return call()

    @deco
    def myfunc(a):
        """some doc"""
        return a

    assert myfunc.__name__ == 'myfunc'
    assert myfunc.__doc__ == 'some doc'


# ---------------------------------------------------------------------------
# update_wrapper and wraps
# ---------------------------------------------------------------------------

def test_update_wrapper_basic():
    def wrapped(a, b):
        """wrapped doc"""
        pass

    def wrapper(*args, **kwargs):
        pass

    result = update_wrapper(wrapper, wrapped)
    assert result is wrapper
    assert wrapper.__name__ == 'wrapped'
    assert wrapper.__doc__ == 'wrapped doc'
    assert wrapper.__wrapped__ is wrapped


def test_update_wrapper_sets_original():
    def wrapped(a):
        pass

    def wrapper(*args):
        pass

    update_wrapper(wrapper, wrapped)
    assert wrapper.__original__ is wrapped


def test_update_wrapper_original_chain():
    def base(a):
        pass

    def mid(*args):
        pass
    update_wrapper(mid, base)

    def outer(*args):
        pass
    update_wrapper(outer, mid)

    # __original__ should point all the way back to base
    assert outer.__original__ is base


def test_wraps_basic():
    def wrapped(a, b):
        """doc"""
        pass

    def wrapper(*args, **kwargs):
        pass

    result = wraps(wrapped)(wrapper)
    assert result is wrapper
    assert wrapper.__name__ == 'wrapped'
    assert wrapper.__original__ is wrapped


def test_wraps_returns_partial():
    def wrapped(a):
        pass

    p = wraps(wrapped)
    assert isinstance(p, functools.partial)

    def wrapper(*args):
        pass
    result = p(wrapper)
    assert result is wrapper
    assert wrapper.__name__ == 'wrapped'


def test_wraps_custom_assigned():
    def wrapped(a):
        pass
    wrapped.custom_attr = 'value'

    def wrapper(*args):
        pass

    wraps(wrapped, assigned=('custom_attr',))(wrapper)
    assert wrapper.custom_attr == 'value'
