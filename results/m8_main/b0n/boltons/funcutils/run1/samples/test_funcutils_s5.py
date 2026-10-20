# -*- coding: utf-8 -*-
"""Tests for boltons.funcutils."""

import functools
import inspect
import sys
import threading

import pytest

from boltons.funcutils import (
    CachedInstancePartial,
    ExistingArgument,
    FunctionBuilder,
    InstancePartial,
    MissingArgument,
    NO_DEFAULT,
    copy_function,
    dir_dict,
    format_exp_repr,
    format_invocation,
    format_nonexp_repr,
    get_module_callables,
    inspect_formatargspec,
    mro_items,
    noop,
    once,
    partial,
    partial_ordering,
    update_wrapper,
    wraps,
)


# ---------------------------------------------------------------------------
# inspect_formatargspec
# ---------------------------------------------------------------------------

def test_inspect_formatargspec_basic():
    s = inspect_formatargspec(['a', 'b', 'c'], defaults=(2, 3))
    assert s == '(a, b=2, c=3)'


def test_inspect_formatargspec_no_defaults():
    assert inspect_formatargspec(['a', 'b']) == '(a, b)'


def test_inspect_formatargspec_varargs_and_varkw():
    s = inspect_formatargspec(['a'], varargs='args', varkw='kwargs')
    assert s == '(a, *args, **kwargs)'


def test_inspect_formatargspec_kwonly():
    s = inspect_formatargspec(['a'], kwonlyargs=['b', 'c'],
                              kwonlydefaults={'c': 5})
    assert s == '(a, *, b, c=5)'


def test_inspect_formatargspec_kwonly_with_varargs():
    s = inspect_formatargspec(['a'], varargs='args', kwonlyargs=['b'])
    assert s == '(a, *args, b)'


def test_inspect_formatargspec_annotations():
    s = inspect_formatargspec(['a', 'b'], annotations={'a': int, 'return': str})
    assert s == "(a: int, b) -> str"


def test_inspect_formatargspec_custom_formatters():
    s = inspect_formatargspec(
        ['a'], defaults=(1,),
        formatarg=lambda n: n.upper(),
        formatvalue=lambda v: '=<%r>' % v,
    )
    assert s == '(A=<1>)'


# ---------------------------------------------------------------------------
# get_module_callables
# ---------------------------------------------------------------------------

def test_get_module_callables_on_self_module():
    import boltons.funcutils as mod
    types, funcs = get_module_callables(mod)
    assert 'FunctionBuilder' in types
    assert 'InstancePartial' in types
    assert 'noop' in funcs
    assert 'update_wrapper' in funcs
    # imported items with a different __module__ should be filtered out
    assert 'functools' not in types


def test_get_module_callables_by_string():
    types, funcs = get_module_callables('boltons.funcutils')
    assert 'FunctionBuilder' in types
    assert 'once' in funcs


def test_get_module_callables_ignore():
    import boltons.funcutils as mod
    types, funcs = get_module_callables(
        mod, ignore=lambda name: name.startswith('_') or name == 'noop')
    assert 'noop' not in funcs
    assert 'FunctionBuilder' in types


# ---------------------------------------------------------------------------
# mro_items
# ---------------------------------------------------------------------------

def test_mro_items_basic():
    class A(object):
        x = 1
    class B(A):
        y = 2
    items = dict(mro_items(B))
    assert items['x'] == 1
    assert items['y'] == 2


def test_mro_items_returns_iterator():
    it = mro_items(int)
    assert iter(it) is it
    # consume to check it produces key/value pairs
    key, value = next(it)
    assert isinstance(key, str)


# ---------------------------------------------------------------------------
# dir_dict
# ---------------------------------------------------------------------------

def test_dir_dict_basic():
    class Foo(object):
        def __init__(self):
            self.a = 1
        def m(self):
            return 2
    d = dir_dict(Foo())
    assert d['a'] == 1
    assert callable(d['m'])


def test_dir_dict_skips_raising_attribute():
    class Bad(object):
        @property
        def boom(self):
            raise RuntimeError('nope')
        @property
        def ok(self):
            return 5
    d = dir_dict(Bad())
    assert 'boom' not in d
    assert d['ok'] == 5


def test_dir_dict_raise_exc():
    class Bad(object):
        @property
        def boom(self):
            raise RuntimeError('nope')
    with pytest.raises(RuntimeError):
        dir_dict(Bad(), raise_exc=True)


# ---------------------------------------------------------------------------
# copy_function
# ---------------------------------------------------------------------------

def test_copy_function_self_reference():
    func = lambda: func
    func_copy = copy_function(func)
    assert func_copy() is func
    assert func_copy is not func


def test_copy_function_dict_not_copied():
    def orig():
        return 1
    orig.marker = 'x'
    c = copy_function(orig, copy_dict=False)
    assert not hasattr(c, 'marker')
    c2 = copy_function(orig)
    assert c2.marker == 'x'


def test_copy_function_defaults():
    def orig(a, b=3):
        return a + b
    c = copy_function(orig)
    assert c(1) == 4
    assert c.__defaults__ == (3,)


# ---------------------------------------------------------------------------
# partial_ordering
# ---------------------------------------------------------------------------

def test_partial_ordering():
    @partial_ordering
    class MySet(set):
        def __le__(self, other):
            return self.issubset(other)
        def __ge__(self, other):
            return self.issuperset(other)

    a = MySet([1, 2, 3])
    b = MySet([1, 2])
    c = MySet([1, 2, 4])
    assert (b < a) is True
    assert (b > a) is False
    assert (b < c) is True
    assert (a < c) is False
    assert (c > a) is False
    assert (a == a) is True


def test_partial_ordering_does_not_override_existing():
    @partial_ordering
    class Foo(object):
        def __le__(self, other):
            return True
        def __ge__(self, other):
            return True
        def __lt__(self, other):
            return 'custom'
    assert Foo() < Foo() == 'custom'


# ---------------------------------------------------------------------------
# InstancePartial / CachedInstancePartial
# ---------------------------------------------------------------------------

def test_instance_partial_as_method():
    def add(self, x, y):
        return self.base + x + y

    class Obj(object):
        base = 10
        add = InstancePartial(add, 1)

    o = Obj()
    assert o.add(2) == 13


def test_partial_alias_is_cached_instance_partial():
    assert partial is CachedInstancePartial


def test_cached_instance_partial_caches_method():
    def mul(self, x):
        return self.base * x

    class Obj(object):
        base = 3
        mul = CachedInstancePartial(mul, 2)

    o = Obj()
    first = o.mul
    second = o.mul
    assert first == second
    assert first(1) == 6
    # stored on instance __dict__
    assert 'mul' in o.__dict__


def test_cached_instance_partial_class_access():
    def f(self, x):
        return x

    class Obj(object):
        f = CachedInstancePartial(f, 1)

    obj = Obj()
    # accessing through type should provide a bound-ish callable
    bound = Obj.f.__get__(obj, Obj)
    assert bound() == 1


def test_instance_partial_partialmethod_attr():
    def f(a, b):
        return a + b
    ip = InstancePartial(f, 1)
    pm = ip._partialmethod
    assert isinstance(pm, functools.partialmethod)
    pm2 = ip.__partialmethod__
    assert isinstance(pm2, functools.partialmethod)


# ---------------------------------------------------------------------------
# format_invocation
# ---------------------------------------------------------------------------

def test_format_invocation_basic():
    assert format_invocation('func', args=(1, 2), kwargs={'c': 3}) == 'func(1, 2, c=3)'


def test_format_invocation_args_only():
    assert format_invocation('a_func', args=(1,)) == 'a_func(1)'


def test_format_invocation_kwargs_list():
    assert format_invocation('kw_func', kwargs=[('a', 1), ('b', 2)]) == 'kw_func(a=1, b=2)'


def test_format_invocation_sorts_dict_kwargs():
    assert format_invocation('f', kwargs={'b': 2, 'a': 1}) == 'f(a=1, b=2)'


def test_format_invocation_no_args():
    assert format_invocation('f') == 'f()'


def test_format_invocation_custom_repr():
    assert format_invocation('f', args=('x',), repr=str) == "f(x)"


def test_format_invocation_unexpected_kwarg():
    with pytest.raises(TypeError):
        format_invocation('f', bad=1)


# ---------------------------------------------------------------------------
# format_exp_repr / format_nonexp_repr
# ---------------------------------------------------------------------------

class _Flag(object):
    def __init__(self, length, width, depth=None):
        self.length = length
        self.width = width
        self.depth = depth


def test_format_exp_repr_positional_and_optional():
    flag = _Flag(5, 10)
    assert format_exp_repr(flag, ['length', 'width'], [], ['depth']) == 'Flag(5, 10)'


def test_format_exp_repr_required_and_optional():
    flag2 = _Flag(5, 15, 2)
    assert format_exp_repr(flag2, ['length'], ['width', 'depth']) == 'Flag(5, width=15, depth=2)'


def test_format_exp_repr_dedup_and_opt_key():
    flag = _Flag(5, 10)
    out = format_exp_repr(flag, ['length'], ['width', 'width'],
                          opt_key=lambda v: v is None)
    assert out == 'Flag(5, width=10)'


def test_format_exp_repr_custom_opt_key():
    flag = _Flag(5, 10)
    # opt_key returns True for values that should be *omitted*
    out = format_exp_repr(flag, ['length'], [], ['width'],
                          opt_key=lambda v: v == 10)
    assert out == 'Flag(5)'


def test_format_nonexp_repr_basic():
    flag = _Flag(5, 10)
    assert format_nonexp_repr(flag, ['length', 'width'], ['depth']) == \
        '<Flag length=5 width=10>'


def test_format_nonexp_repr_fallback_id():
    flag = _Flag(5, 10)
    out = format_nonexp_repr(flag)
    assert out == '<Flag id=%s>' % id(flag)


def test_format_nonexp_repr_optional_omitted():
    flag = _Flag(1, 2, None)
    out = format_nonexp_repr(flag, ['length'], ['depth'])
    assert out == '<Flag length=1>'


def test_format_nonexp_repr_with_value_present():
    flag = _Flag(1, 2, 3)
    out = format_nonexp_repr(flag, ['length'], ['depth'])
    assert out == '<Flag length=1 depth=3>'


# ---------------------------------------------------------------------------
# wraps / update_wrapper
# ---------------------------------------------------------------------------

def test_update_wrapper_copies_metadata_and_signature():
    def func(a, b=1, *args, c=2, **kwargs):
        "docstring"
        return a + b

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapped = update_wrapper(wrapper, func)
    assert wrapped.__name__ == 'func'
    assert wrapped.__doc__ == 'docstring'
    sig = inspect.signature(wrapped)
    assert list(sig.parameters) == ['a', 'b', 'args', 'c', 'kwargs']
    assert wrapped(1) == 2
    assert wrapped(1, 2, c=3) == 3
    assert wrapped.__wrapped__ is func


def test_update_wrapper_injected():
    def func(a, b):
        return a + b

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapped = update_wrapper(wrapper, func, injected='a')
    sig = inspect.signature(wrapped)
    assert list(sig.parameters) == ['b']


def test_update_wrapper_injected_missing_raises():
    def func(a):
        return a

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    with pytest.raises(MissingArgument):
        update_wrapper(wrapper, func, injected='zzz')


def test_update_wrapper_injected_to_varkw():
    def func(a, **kwargs):
        return a

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapped = update_wrapper(wrapper, func, injected='zzz')
    assert wrapped(1) == 1


def test_update_wrapper_expected():
    def func(a):
        return a

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapped = update_wrapper(wrapper, func, expected=[('extra', 5)])
    sig = inspect.signature(wrapped)
    assert 'extra' in sig.parameters
    assert sig.parameters['extra'].default == 5


def test_update_wrapper_expected_string():
    def func(a):
        return a

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapped = update_wrapper(wrapper, func, expected='newarg')
    sig = inspect.signature(wrapped)
    assert 'newarg' in sig.parameters
    assert sig.parameters['newarg'].default is NO_DEFAULT


def test_update_wrapper_hide_wrapped():
    def func():
        return 1

    def wrapper(*args, **kwargs):
        return func()

    wrapped = update_wrapper(wrapper, func, hide_wrapped=True)
    assert not hasattr(wrapped, '__wrapped__')


def test_update_wrapper_unexpected_kwarg():
    def func():
        pass

    def wrapper():
        pass

    with pytest.raises(TypeError):
        update_wrapper(wrapper, func, bogus=True)


def test_update_wrapper_rejects_classmethod():
    class C(object):
        @classmethod
        def cm(cls):
            pass

    def wrapper(*args, **kwargs):
        pass

    with pytest.raises(TypeError):
        update_wrapper(wrapper, C.__dict__['cm'])


def test_wraps_decorator():
    def print_return(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    @print_return
    def example():
        "docstring"
        return 'value'

    assert example() == 'value'
    assert example.__name__ == 'example'
    assert example.__doc__ == 'docstring'


def test_wraps_with_injected():
    def deco(func):
        @wraps(func, injected='hidden')
        def wrapper(*args, **kwargs):
            return func(1, *args, **kwargs)
        return wrapper

    @deco
    def target(hidden, visible):
        return hidden + visible

    sig = inspect.signature(target)
    assert list(sig.parameters) == ['visible']
    assert target(2) == 3


# ---------------------------------------------------------------------------
# FunctionBuilder
# ---------------------------------------------------------------------------

def test_function_builder_simple():
    fb = FunctionBuilder('return_five', doc='returns the integer 5',
                         body='return 5')
    f = fb.get_func()
    assert f() == 5
    assert f.__name__ == 'return_five'
    assert f.__doc__ == 'returns the integer 5'
    assert f.__module__ is None
    assert '__source__' in f.__dict__


def test_function_builder_varkw():
    fb = FunctionBuilder('return_five', body='return 5')
    fb.varkw = 'kw'
    f_kw = fb.get_func()
    assert f_kw(ignored_arg='ignored_val') == 5


def test_function_builder_args_defaults():
    fb = FunctionBuilder('add', args=['a', 'b'], defaults=(3,),
                         body='return a + b')
    f = fb.get_func()
    assert f(1) == 4
    assert f(1, 2) == 3


def test_function_builder_no_add_source_no_dict():
    fb = FunctionBuilder('f', body='return 1')
    fb.dict['marker'] = 'x'
    f = fb.get_func(add_source=False, with_dict=False)
    assert '__source__' not in f.__dict__
    assert 'marker' not in f.__dict__


def test_function_builder_with_dict():
    fb = FunctionBuilder('f', body='return 1')
    fb.dict['marker'] = 'x'
    f = fb.get_func()
    assert f.marker == 'x'


def test_function_builder_async():
    fb = FunctionBuilder('af', body='return 42', is_async=True)
    f = fb.get_func()
    assert f() == 42  # sanity: coroutine functions don't run here without await


def test_function_builder_from_func_roundtrip():
    def orig(a, b=2, *args, c=3, **kwargs):
        "the doc"
        return (a, b, args, c, kwargs)

    fb = FunctionBuilder.from_func(orig)
    assert fb.name == 'orig'
    assert fb.doc == 'the doc'
    assert fb.args == ['a', 'b']
    assert fb.defaults == (2,)
    assert fb.varargs == 'args'
    assert fb.kwonlyargs == ['c']
    assert fb.kwonlydefaults == {'c': 3}
    assert fb.varkw == 'kwargs'

    f = fb.get_func()
    assert f(1) == (1, 2, (), 3, {})


def test_function_builder_from_func_non_callable():
    with pytest.raises(TypeError):
        FunctionBuilder.from_func(42)


def test_function_builder_posonly():
    def orig(a, b, /, c):
        return a + b + c

    fb = FunctionBuilder.from_func(orig)
    assert fb.posonlyargs == ['a', 'b']
    inv = fb.get_invocation_str()
    # positional-only must be forwarded positionally
    assert inv.startswith('a, b')


def test_function_builder_sig_str_annotations():
    fb = FunctionBuilder('f', args=['a'], annotations={'a': int, 'return': str},
                         body='return a')
    s = fb.get_sig_str()
    assert 'int' in s
    assert '-> str' in s
    s2 = fb.get_sig_str(with_annotations=False)
    assert 'int' not in s2


def test_function_builder_get_defaults_dict():
    fb = FunctionBuilder('f', args=['a', 'b', 'c'], defaults=(2, 3))
    assert fb.get_defaults_dict() == {'b': 2, 'c': 3}


def test_function_builder_get_defaults_dict_kwonly():
    fb = FunctionBuilder('f', args=['a'], kwonlyargs=['k'],
                         kwonlydefaults={'k': 9})
    assert fb.get_defaults_dict() == {'k': 9}


def test_function_builder_get_arg_names():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,), kwonlyargs=['c'])
    assert fb.get_arg_names() == ('a', 'b', 'c')
    assert fb.get_arg_names(only_required=True) == ('a', 'c')


def test_function_builder_add_arg():
    fb = FunctionBuilder('f', body='return a')
    fb.add_arg('a')
    assert fb.args == ['a']
    fb.add_arg('b', 5)
    assert fb.args == ['a', 'b']
    assert fb.defaults == (5,)


def test_function_builder_add_kwonly_arg():
    fb = FunctionBuilder('f')
    fb.add_arg('k', 7, kwonly=True)
    assert fb.kwonlyargs == ['k']
    assert fb.kwonlydefaults == {'k': 7}


def test_function_builder_add_existing_arg():
    fb = FunctionBuilder('f', args=['a'])
    with pytest.raises(ExistingArgument):
        fb.add_arg('a')


def test_function_builder_add_existing_kwonly_arg():
    fb = FunctionBuilder('f', kwonlyargs=['k'])
    with pytest.raises(ExistingArgument):
        fb.add_arg('k', kwonly=True)


def test_function_builder_remove_arg():
    fb = FunctionBuilder('f', args=['a', 'b', 'c'], defaults=(2, 3))
    fb.remove_arg('b')
    assert fb.args == ['a', 'c']
    assert fb.defaults == (3,)


def test_function_builder_remove_kwonly_arg():
    fb = FunctionBuilder('f', kwonlyargs=['k'], kwonlydefaults={'k': 1})
    fb.remove_arg('k')
    assert fb.kwonlyargs == []
    assert fb.kwonlydefaults == {}


def test_function_builder_remove_missing_arg():
    fb = FunctionBuilder('f', args=['a'])
    with pytest.raises(MissingArgument) as exc_info:
        fb.remove_arg('zzz')
    assert exc_info.value.arg_name == 'zzz'


def test_function_builder_unexpected_kwarg():
    with pytest.raises(TypeError):
        FunctionBuilder('f', nope=1)


def test_function_builder_get_invocation_str():
    fb = FunctionBuilder('f', args=['a', 'b', 'c'], defaults=(2, 3),
                         kwonlyargs=['k'], varkw='kw')
    inv = fb.get_invocation_str()
    assert inv.startswith('a, ')
    assert 'b=b' in inv
    assert 'c=c' in inv
    assert 'k=k' in inv
    assert '**kw' in inv


def test_function_builder_get_invocation_str_with_varargs():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,), varargs='args')
    inv = fb.get_invocation_str()
    # with varargs, regular args stay positional
    assert inv == 'a, b, *args'


def test_function_builder_lambda_name_sanitized():
    fb = FunctionBuilder('lambda', body='return 1')
    # name is used as identifier; just ensure it compiles
    f = fb.get_func()
    assert f() == 1


def test_function_builder_body_indent():
    fb = FunctionBuilder('f', body='x = 1\nreturn x', indent=2)
    f = fb.get_func()
    assert f() == 1


# ---------------------------------------------------------------------------
# noop
# ---------------------------------------------------------------------------

def test_noop():
    assert noop() is None
    assert noop(1, 2, a=3) is None


# ---------------------------------------------------------------------------
# once
# ---------------------------------------------------------------------------

def test_once_caches_result():
    calls = []

    @once
    def expensive():
        calls.append(1)
        return 'initialized'

    assert expensive() == 'initialized'
    assert expensive() == 'initialized'
    assert len(calls) == 1


def test_once_thread_safe():
    call_count = []
    started = threading.Event()
    release = threading.Event()

    @once
    def slow():
        call_count.append(1)
        started.set()
        release.wait(timeout=2)
        return 42

    results = []

    def worker():
        results.append(slow())

    t1 = threading.Thread(target=worker)
    t1.start()
    started.wait(timeout=2)
    t2 = threading.Thread(target=worker)
    t2.start()
    release.set()
    t1.join(timeout=2)
    t2.join(timeout=2)

    assert results == [42, 42]
    assert len(call_count) == 1


def test_once_preserves_metadata():
    @once
    def my_setup():
        "does setup"
        return 1

    assert my_setup.__name__ == 'my_setup'
    assert my_setup.__doc__ == 'does setup'


# ---------------------------------------------------------------------------
# Exceptions hierarchy
# ---------------------------------------------------------------------------

def test_exception_hierarchy():
    assert issubclass(MissingArgument, ValueError)
    assert issubclass(ExistingArgument, ValueError)
