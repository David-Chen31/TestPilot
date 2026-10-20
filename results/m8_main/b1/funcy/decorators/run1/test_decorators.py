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
# has_single_arg / has_1pos_and_kwonly
# ---------------------------------------------------------------------------

def test_has_single_arg_true():
    def f(call):
        pass
    assert has_single_arg(f) is True


def test_has_single_arg_with_default():
    def f(call=1):
        pass
    assert has_single_arg(f) is True


def test_has_single_arg_false_multiple():
    def f(a, b):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_false_varargs():
    def f(*args):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_false_varkw():
    def f(**kwargs):
        pass
    assert has_single_arg(f) is False


def test_has_1pos_and_kwonly_true():
    def f(a, *, b=1):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_true_single_pos():
    def f(a):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_false_two_pos():
    def f(a, b):
        pass
    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_false_varargs():
    def f(a, *args, b=1):
        pass
    assert has_1pos_and_kwonly(f) is False


# ---------------------------------------------------------------------------
# get_argnames
# ---------------------------------------------------------------------------

def test_get_argnames_simple():
    def f(a, b, c):
        pass
    assert get_argnames(f) == ('a', 'b', 'c')


def test_get_argnames_defaults_and_kwonly():
    def f(a, b=1, *, c=2):
        pass
    assert get_argnames(f) == ('a', 'b', 'c')


def test_get_argnames_varargs_varkw():
    def f(a, *args, **kwargs):
        pass
    assert get_argnames(f) == ('a', 'args', 'kwargs')


def test_get_argnames_uses_original_attr():
    def original(a, b):
        pass

    def wrapper():
        pass
    wrapper.__original__ = original
    assert get_argnames(wrapper) == ('a', 'b')


# ---------------------------------------------------------------------------
# wraps / update_wrapper
# ---------------------------------------------------------------------------

def test_wraps_copies_metadata():
    def orig(a, b=1):
        "docstring"
        pass

    @wraps(orig)
    def wrapper(*args, **kwargs):
        pass

    assert wrapper.__name__ == 'orig'
    assert wrapper.__doc__ == 'docstring'
    assert wrapper.__wrapped__ is orig
    assert hasattr(wrapper, '__original__')
    assert wrapper.__original__ is orig


def test_wraps_propagates_original():
    def inner(a):
        pass

    def middle(b):
        pass
    middle.__original__ = inner

    @wraps(middle)
    def wrapper(*args, **kwargs):
        pass

    assert wrapper.__original__ is inner


def test_update_wrapper_returns_wrapper():
    def orig(a):
        return a

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    result = update_wrapper(wrapper, orig)
    assert result is wrapper
    assert wrapper.__name__ == 'orig'
    assert wrapper.__original__ is orig


# ---------------------------------------------------------------------------
# Call
# ---------------------------------------------------------------------------

def test_call_basic():
    def f(a, b):
        return a + b

    call = Call(f, (1,), {'b': 2})
    assert call() == 3


def test_call_no_args_uses_saved():
    def f(a, b=0):
        return (a, b)

    call = Call(f, (1,), {'b': 5})
    assert call() == (1, 5)


def test_call_extends_args_and_kwargs():
    def f(a, b, c, d=0):
        return (a, b, c, d)

    call = Call(f, (1,), {'b': 2, 'c': 3})
    assert call() == (1, 2, 3, 0)


def test_call_extends_extra_positional():
    def f(*args, **kwargs):
        return (args, kwargs)

    call = Call(f, (1,), {'x': 2})
    assert call(10, y=3) == ((1, 10), {'x': 2, 'y': 3})


def test_call_attributes():
    def f(a, b=2):
        pass

    call = Call(f, (1,), {})
    assert call.a == 1
    assert call.b == 2
    assert call._func is f
    assert call._args == (1,)


def test_call_getattr_missing():
    def f(a):
        pass

    call = Call(f, (1,), {})
    with pytest.raises(AttributeError):
        call.zzz


def test_call_kwarg_overrides_positional():
    def f(a, b):
        return (a, b)

    call = Call(f, (1, 2), {})
    assert call.b == 2


def test_call_varargs():
    def f(a, *args):
        return (a, args)

    call = Call(f, (1, 2, 3), {})
    assert call.args == (2, 3)


def test_call_varkw():
    def f(a, **kwargs):
        return (a, kwargs)

    call = Call(f, (1,), {'x': 2, 'y': 3})
    assert call.kwargs == {'x': 2, 'y': 3}


def test_call_str_and_repr():
    def myfunc(a, b=2):
        pass

    call = Call(myfunc, (1,), {'b': 5})
    s = str(call)
    # str uses __qualname__ which includes the test-local scope
    assert myfunc.__qualname__ in s
    assert s.endswith('(1, b=5)')
    assert repr(call) == '<Call %s>' % s


def test_call_str_no_qualname():
    class NoQualName:
        def __call__(self, a):
            pass

    call = Call(NoQualName(), (1,), {})
    # Has a __call__ method; str shouldn't crash
    assert isinstance(str(call), str)


def test_call_missing_required_arg():
    def f(a, b):
        pass

    call = Call(f, (1,), {})
    # Call.__getattr__ converts TypeError into AttributeError
    with pytest.raises(AttributeError, match="missing required argument"):
        call.b


# ---------------------------------------------------------------------------
# arggetter
# ---------------------------------------------------------------------------

def test_arggetter_positional():
    def f(a, b, c):
        pass
    get = arggetter(f)
    assert get('a', (1, 2, 3), {}) == 1
    assert get('b', (1, 2, 3), {}) == 2
    assert get('c', (1, 2, 3), {}) == 3


def test_arggetter_from_kwargs():
    def f(a, b, c):
        pass
    get = arggetter(f)
    assert get('b', (1,), {'b': 20}) == 20
    assert get('c', (), {'c': 30}) == 30


def test_arggetter_defaults():
    def f(a, b=10, *, c=20):
        pass
    get = arggetter(f)
    assert get('b', (1,), {}) == 10
    assert get('c', (1,), {}) == 20


def test_arggetter_missing_required():
    def f(a, b):
        pass
    get = arggetter(f)
    with pytest.raises(TypeError, match="missing required argument"):
        get('b', (1,), {})


def test_arggetter_unknown_name():
    def f(a):
        pass
    get = arggetter(f)
    with pytest.raises(TypeError, match="doesn't have argument named"):
        get('zzz', (1,), {})


def test_arggetter_varargs():
    def f(a, *args):
        pass
    get = arggetter(f)
    assert get('args', (1, 2, 3), {}) == (2, 3)


def test_arggetter_varkw():
    def f(a, **kwargs):
        pass
    get = arggetter(f)
    assert get('kwargs', (1,), {'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_arggetter_kwonly_not_positional():
    def f(a, *, b):
        pass
    get = arggetter(f)
    assert get('b', (1,), {'b': 5}) == 5
    with pytest.raises(TypeError):
        get('b', (1, 5), {})


def test_arggetter_bound_method():
    class C:
        def m(self, a):
            pass

    obj = C()
    get = arggetter(obj.m)
    assert get('a', (5,), {}) == 5


def test_arggetter_caching():
    def f(a):
        pass
    g1 = arggetter(f)
    g2 = arggetter(f)
    assert g1 is g2


# ---------------------------------------------------------------------------
# make_decorator
# ---------------------------------------------------------------------------

def test_make_decorator_basic():
    def deco(call, extra=0):
        return call() + extra

    dec = make_decorator(deco, (), {'extra': 100})

    @dec
    def add(a, b):
        return a + b

    assert add(1, 2) == 103


def test_make_decorator_introspection_attrs():
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
    def myfunc():
        "my doc"
        return 1

    assert myfunc.__name__ == 'myfunc'
    assert myfunc.__doc__ == 'my doc'


# ---------------------------------------------------------------------------
# decorator (flat style: single arg)
# ---------------------------------------------------------------------------

def test_decorator_flat_single_arg():
    @decorator
    def double(call):
        return call() * 2

    @double
    def add(a, b):
        return a + b

    assert add(1, 2) == 6
    assert add.__name__ == 'add'


def test_decorator_flat_arguments_access():
    seen = {}

    @decorator
    def spy(call):
        seen['a'] = call.a
        seen['b'] = call.b
        return call()

    @spy
    def f(a, b):
        return a + b

    assert f(3, 4) == 7
    assert seen == {'a': 3, 'b': 4}


# ---------------------------------------------------------------------------
# decorator (fab style: one positional + kwonly)
# ---------------------------------------------------------------------------

def test_decorator_fab_with_args():
    @decorator
    def times(call, factor=1):
        return call() * factor

    @times(factor=3)
    def f(x):
        return x

    assert f(5) == 15


def test_decorator_fab_no_args_returns_decorator_fab():
    # For the fab style, calling with a single positional arg and no kwargs
    # goes through decorator_fab(_func=<arg>) -> returns the decorated wrapper.
    @decorator
    def times(call, factor=2):
        return call() * factor

    @times
    def f(x):
        return x

    # @times <-> times(f) which is the wrapped function (not called), so
    # calling f(5) uses default factor=2 -> 10.
    assert f(5) == 10


# ---------------------------------------------------------------------------
# decorator (multi-arg style: decorator fab with dargs)
# ---------------------------------------------------------------------------

def test_decorator_multiarg_fab():
    @decorator
    def add_const(call, a, b):
        return call() + a + b

    dec = add_const(10, 20)

    @dec
    def f():
        return 1

    assert f() == 31


def test_decorator_multiarg_kwargs():
    @decorator
    def adder(call, x=0, y=0):
        return call() + x + y

    dec = adder(x=5, y=7)

    @dec
    def f():
        return 1

    assert f() == 13


# ---------------------------------------------------------------------------
# integration: decorator wrapping a function that uses Call args
# ---------------------------------------------------------------------------

def test_decorator_with_varargs_varkw():
    @decorator
    def passthrough(call):
        return call()

    @passthrough
    def f(*args, **kwargs):
        return (args, kwargs)

    assert f(1, 2, x=3) == ((1, 2), {'x': 3})


def test_decorator_method_wrapping():
    @decorator
    def double(call):
        return call() * 2

    class C:
        @double
        def val(self, a):
            return a

    assert C().val(4) == 8
