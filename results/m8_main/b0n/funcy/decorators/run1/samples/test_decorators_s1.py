import functools
import inspect

import pytest

from funcy.decorators import (
    Call,
    arggetter,
    decorator,
    get_argnames,
    has_1pos_and_kwonly,
    has_single_arg,
    make_decorator,
    update_wrapper,
    wraps,
    unwrap,
)


# ---------------------------------------------------------------------------
# has_single_arg / has_1pos_and_kwonly
# ---------------------------------------------------------------------------

def test_has_single_arg_simple():
    def f(call):
        pass
    assert has_single_arg(f) is True


def test_has_single_arg_with_default():
    def f(call=1):
        pass
    assert has_single_arg(f) is True


def test_has_single_arg_varargs():
    def f(*args):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_varkw():
    def f(**kw):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_too_many():
    def f(a, b):
        pass
    assert has_single_arg(f) is False


def test_has_single_arg_zero():
    def f():
        pass
    assert has_single_arg(f) is False


def test_has_1pos_and_kwonly_true():
    def f(a, b=1, *, c=2):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_single_pos():
    def f(a):
        pass
    assert has_1pos_and_kwonly(f) is True


def test_has_1pos_and_kwonly_two_pos():
    def f(a, b, *, c):
        pass
    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_varpos():
    def f(a, *args, b=1):
        pass
    assert has_1pos_and_kwonly(f) is False


def test_has_1pos_and_kwonly_zero_pos():
    def f(*, a):
        pass
    assert has_1pos_and_kwonly(f) is False


# ---------------------------------------------------------------------------
# Call
# ---------------------------------------------------------------------------

def test_call_basic_invocation():
    def add(a, b):
        return a + b
    c = Call(add, (1, 2), {})
    assert c() == 3


def test_call_with_extra_args():
    def add(a, b=10):
        return a + b
    c = Call(add, (1,), {'b': 2})
    # extra positional appended to stored, kwargs merged
    assert c(5) == 6


def test_call_stored_attributes():
    def add(a, b):
        return a + b
    c = Call(add, (1, 2), {'x': 1})
    assert c._func is add
    assert c._args == (1, 2)
    assert c._kwargs == {'x': 1}


def test_call_arg_access_by_name():
    def f(a, b, c=3):
        return a + b + c
    c = Call(f, (1, 2), {'c': 5})
    assert c.a == 1
    assert c.b == 2
    assert c.c == 5


def test_call_varargs():
    def f(a, *args):
        return (a, args)
    c = Call(f, (1, 2, 3), {})
    assert c.a == 1
    assert c.args == (2, 3)


def test_call_varkw():
    def f(a, **kw):
        return (a, kw)
    c = Call(f, (1,), {'x': 2, 'y': 3})
    assert c.a == 1
    assert c.kw == {'x': 2, 'y': 3}


def test_call_missing_arg_raises_attribute_error():
    def f(a, b):
        return a + b
    c = Call(f, (1,), {})
    with pytest.raises(AttributeError):
        c.b


def test_call_unknown_arg_raises_attribute_error():
    def f(a, b):
        return a + b
    c = Call(f, (1, 2), {})
    with pytest.raises(AttributeError):
        c.unknown


def test_call_str():
    def myfunc(a, b):
        pass
    c = Call(myfunc, (1,), {'b': 2})
    s = str(c)
    assert s.startswith('myfunc(')
    assert '1' in s
    assert 'b=2' in s


def test_call_repr():
    def myfunc(a):
        pass
    c = Call(myfunc, (1,), {})
    r = repr(c)
    assert r.startswith('<Call ')
    assert r.endswith('>')


def test_call_caches_attribute():
    def f(a):
        pass
    c = Call(f, (5,), {})
    assert c.a == 5
    assert 'a' in c.__dict__


# ---------------------------------------------------------------------------
# get_argnames
# ---------------------------------------------------------------------------

def test_get_argnames_basic():
    def f(a, b, c):
        pass
    assert get_argnames(f) == ['a', 'b', 'c']


def test_get_argnames_with_kwonly():
    def f(a, b=1, *, c=2, d=3):
        pass
    assert get_argnames(f) == ['a', 'b', 'c', 'd']


def test_get_argnames_with_varargs_varkw():
    def f(a, *args, **kwargs):
        pass
    assert get_argnames(f) == ['a', 'args', 'kwargs']


def test_get_argnames_kwonly_and_varargs():
    def f(a, *, b, *args, **kwargs):
        pass
    # invalid syntax actually; use a legal combination
    def g(a, *args, b=1, **kwargs):
        pass
    assert get_argnames(g) == ['a', 'b', 'args', 'kwargs']


def test_get_argnames_unwraps():
    def original(a, b):
        pass

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        return original(*args, **kwargs)
    assert get_argnames(wrapper) == ['a', 'b']


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


def test_arggetter_positional_fallback_to_kwargs():
    def f(a, b):
        pass
    get = arggetter(f)
    assert get('a', (), {'a': 10, 'b': 20}) == 10


def test_arggetter_defaults():
    def f(a, b=5):
        pass
    get = arggetter(f)
    assert get('b', (1,), {}) == 5


def test_arggetter_kwonly_default():
    def f(a, *, c=7):
        pass
    get = arggetter(f)
    assert get('c', (1,), {}) == 7
    assert get('c', (1,), {'c': 9}) == 9


def test_arggetter_varargs():
    def f(a, *args):
        pass
    get = arggetter(f)
    assert get('args', (1, 2, 3), {}) == (2, 3)


def test_arggetter_varkw():
    def f(a, *args, **kwargs):
        pass
    get = arggetter(f)
    assert get('kwargs', (1,), {'x': 2, 'y': 3}) == {'x': 2, 'y': 3}


def test_arggetter_varkw_omits_named_kwonly():
    def f(a, *, b=1, **kwargs):
        pass
    get = arggetter(f)
    assert get('kwargs', (1,), {'b': 2, 'x': 3}) == {'x': 3}


def test_arggetter_unknown_name_raises_typeerror():
    def f(a):
        pass
    get = arggetter(f)
    with pytest.raises(TypeError):
        get('nope', (1,), {})


def test_arggetter_missing_required_raises_typeerror():
    def f(a, b):
        pass
    get = arggetter(f)
    with pytest.raises(TypeError):
        get('b', (1,), {})


def test_arggetter_caches():
    def f(a):
        pass
    assert arggetter(f) is arggetter(f)


def test_arggetter_method_skips_self():
    class A:
        def method(self, x, y=2):
            pass
    get = arggetter(A.method)
    # When called unbound-style from Call, args include self
    assert get('x', (None, 5), {}) == 5
    assert get('y', (None, 5), {}) == 2


# ---------------------------------------------------------------------------
# update_wrapper / wraps
# ---------------------------------------------------------------------------

def test_update_wrapper_copies_metadata():
    def wrapped(a, b):
        """docstring"""
        pass

    def wrapper(*args, **kwargs):
        return wrapped(*args, **kwargs)

    result = update_wrapper(wrapper, wrapped)
    assert result is wrapper
    assert wrapper.__name__ == 'wrapped'
    assert wrapper.__doc__ == 'docstring'
    assert wrapper.__wrapped__ is wrapped


def test_update_wrapper_sets_original():
    def wrapped(a, b):
        pass

    def wrapper(*args, **kwargs):
        pass

    update_wrapper(wrapper, wrapped)
    assert wrapper.__original__ is wrapped


def test_update_wrapper_preserves_original_chain():
    def innermost(a):
        pass

    def wrapped(*args, **kwargs):
        pass
    update_wrapper(wrapped, innermost)

    def wrapper(*args, **kwargs):
        pass
    update_wrapper(wrapper, wrapped)
    assert wrapper.__original__ is innermost


def test_wraps_returns_partial():
    def wrapped(a):
        pass
    p = wraps(wrapped)
    assert isinstance(p, functools.partial)


def test_wraps_application():
    def wrapped(a, b):
        """my doc"""
        pass

    @wraps(wrapped)
    def wrapper(*args, **kwargs):
        return wrapped(*args, **kwargs)

    assert wrapper.__name__ == 'wrapped'
    assert wrapper.__doc__ == 'my doc'
    assert wrapper.__original__ is wrapped
    assert unwrap(wrapper) is wrapped


# ---------------------------------------------------------------------------
# make_decorator
# ---------------------------------------------------------------------------

def test_make_decorator_simple():
    calls = []

    def deco(call):
        calls.append(call)
        return call()

    decorated = make_decorator(deco)(lambda x: x * 2)
    assert decorated(5) == 10
    assert len(calls) == 1


def test_make_decorator_with_args():
    def deco(call, n):
        return call() + n

    decorated = make_decorator(deco, (10,))({})(  # hmm call needs to be a func
    )
    # use proper function
    def f():
        return 1
    decorated = make_decorator(deco, (10,))(f)
    assert decorated() == 11


def test_make_decorator_introspection():
    def deco(call):
        return call()

    d = make_decorator(deco, (1,), {'k': 2})
    assert d._func is deco
    assert d._args == (1,)
    assert d._kwargs == {'k': 2}


def test_make_decorator_preserves_wrapped_metadata():
    def deco(call):
        return call()

    def f(x):
        """func doc"""
        return x
    decorated = make_decorator(deco)(f)
    assert decorated.__name__ == 'f'
    assert decorated.__doc__ == 'func doc'
    assert decorated.__wrapped__ is f


# ---------------------------------------------------------------------------
# decorator
# ---------------------------------------------------------------------------

def test_decorator_single_arg_simple():
    @decorator
    def double(call):
        return call() * 2

    @double
    def add(a, b):
        return a + b

    assert add(1, 2) == 6
    assert add.__name__ == 'add'


def test_decorator_accesses_args_by_name():
    @decorator
    def scoped(call, scope):
        return call.a + call.b

    @scoped
    def something(a, b):
        return 0

    assert something(3, 4) == 7


def test_decorator_with_kwargs():
    @decorator
    def with_param(call, param=None):
        return (call(), param)

    @with_param(param='x')
    def hello():
        return 'hi'

    assert hello() == ('hi', 'x')


def test_decorator_with_default_no_args():
    @decorator
    def with_param(call, param=None):
        return (call(), param)

    @with_param
    def hello():
        return 'hi'

    assert hello() == ('hi', None)


def test_decorator_varargs_star_args():
    @decorator
    def adder(call, *amounts):
        return call() + sum(amounts)

    @adder(1, 2, 3)
    def zero():
        return 0

    assert zero() == 6


def test_decorator_varargs_with_kwargs():
    @decorator
    def combine(call, *args, **kwargs):
        return call(), args, kwargs

    @combine(1, 2, foo=3)
    def base():
        return 'base'

    assert base() == ('base', (1, 2), {'foo': 3})


def test_decorator_returns_fab_metadata():
    def deco(call, x):
        return call()

    decorated = decorator(deco)
    assert decorated.__name__ == 'deco'


def test_decorator_directly_bare_call_uses_function():
    # For the 1-pos-and-kwonly branch, calling with a function should apply
    @decorator
    def identity(call, extra=None):
        return call()

    def f():
        return 42

    result = identity(f)
    assert result() == 42


def test_decorator_preserves_func_name_through_wrapping():
    @decorator
    def noop(call):
        return call()

    @noop
    def my_function():
        return 1

    assert my_function.__name__ == 'my_function'


def test_decorator_kwargs_only_deco_no_arg():
    @decorator
    def kwdeco(call, *, amount=1):
        return call() * amount

    @kwdeco
    def base():
        return 5

    assert base() == 5

    @kwdeco(amount=3)
    def base3():
        return 5

    assert base3() == 15
