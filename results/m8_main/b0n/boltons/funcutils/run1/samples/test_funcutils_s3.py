import sys
import functools
import inspect
import threading
import time
import types

import pytest

from boltons.funcutils import (
    inspect_formatargspec,
    get_module_callables,
    mro_items,
    dir_dict,
    copy_function,
    partial_ordering,
    InstancePartial,
    CachedInstancePartial,
    format_invocation,
    format_exp_repr,
    format_nonexp_repr,
    wraps,
    update_wrapper,
    FunctionBuilder,
    MissingArgument,
    ExistingArgument,
    noop,
    once,
    NO_DEFAULT,
)


# ---------------------------------------------------------------------------
# inspect_formatargspec
# ---------------------------------------------------------------------------

def test_inspect_formatargspec_simple():
    assert inspect_formatargspec(['a', 'b']) == '(a, b)'


def test_inspect_formatargspec_defaults():
    assert inspect_formatargspec(['a', 'b'], defaults=(2,)) == '(a, b=2)'


def test_inspect_formatargspec_varargs_varkw():
    assert inspect_formatargspec(['a'], varargs='args', varkw='kwargs') == (
        '(a, *args, **kwargs)')


def test_inspect_formatargspec_kwonly_without_varargs():
    assert inspect_formatargspec([], kwonlyargs=['x']) == '(*, x)'


def test_inspect_formatargspec_kwonly_with_defaults():
    res = inspect_formatargspec([], kwonlyargs=['x'], kwonlydefaults={'x': 5})
    assert res == '(*, x=5)'


def test_inspect_formatargspec_annotations():
    res = inspect_formatargspec(['a'], annotations={'a': int, 'return': str})
    assert res == '(a: int) -> str'


def test_inspect_formatargspec_custom_formatters():
    res = inspect_formatargspec(
        ['a'],
        varargs='args',
        formatarg=lambda n: n.upper(),
        formatvarargs=lambda n: 'VARARGS_' + n,
    )
    assert res == '(A, VARARGS_args)'


# ---------------------------------------------------------------------------
# get_module_callables
# ---------------------------------------------------------------------------

def test_get_module_callables_with_module_object():
    import math
    typs, funcs = get_module_callables(math)
    assert 'sqrt' in funcs
    assert funcs['sqrt'] is math.sqrt


def test_get_module_callables_with_string():
    typs, funcs = get_module_callables('math')
    assert 'sqrt' in funcs


def test_get_module_callables_ignore():
    import math
    typs, funcs = get_module_callables(math, ignore=lambda n: n.startswith('sqrt'))
    assert 'sqrt' not in funcs


def test_get_module_callables_types_only_own():
    import json
    typs, funcs = get_module_callables(json)
    # JSONDecoder is defined in json.decoder not json, so not necessarily here.
    # But classes like 'JSONDecoder' may be imported. We just check dict types.
    assert isinstance(typs, dict)
    assert isinstance(funcs, dict)


# ---------------------------------------------------------------------------
# mro_items
# ---------------------------------------------------------------------------

def test_mro_items_int():
    names = {k for k, v in mro_items(int)
             if not k.startswith('__') and 'bytes' not in k and not callable(v)}
    assert names == {'denominator', 'imag', 'numerator', 'real'}


def test_mro_items_custom():
    class A:
        x = 1

    class B(A):
        y = 2

    items = dict(mro_items(B))
    assert items['x'] == 1
    assert items['y'] == 2


# ---------------------------------------------------------------------------
# dir_dict
# ---------------------------------------------------------------------------

def test_dir_dict_basic():
    class C:
        a = 1

        def m(self):
            return 2

    d = dir_dict(C())
    assert d['a'] == 1
    assert d['m']() == 2


def test_dir_dict_raise_exc():
    class Bad:
        @property
        def prop(self):
            raise RuntimeError('boom')

    d = dir_dict(Bad())
    assert 'prop' not in d

    with pytest.raises(RuntimeError):
        dir_dict(Bad(), raise_exc=True)


# ---------------------------------------------------------------------------
# copy_function
# ---------------------------------------------------------------------------

def test_copy_function_self_reference():
    func = lambda: func
    assert func() is func
    func_copy = copy_function(func)
    assert func_copy() is func
    assert func_copy is not func


def test_copy_function_attributes():
    def f():
        return 1
    f.custom = 'x'
    copied = copy_function(f)
    assert copied.custom == 'x'
    copied2 = copy_function(f, copy_dict=False)
    assert not hasattr(copied2, 'custom')


def test_copy_function_defaults_and_kwdefaults():
    def f(a, b=2, *, c=3):
        return a + b + c
    copied = copy_function(f)
    assert f(1) == copied(1)
    assert copied.__defaults__ == f.__defaults__
    assert copied.__kwdefaults__ == f.__kwdefaults__


# ---------------------------------------------------------------------------
# partial_ordering
# ---------------------------------------------------------------------------

def test_partial_ordering_set():
    @partial_ordering
    class MySet(set):
        def __le__(self, other):
            return self.issubset(other)

        def __ge__(self, other):
            return self.issuperset(other)

    a = MySet([1, 2, 3])
    b = MySet([1, 2])
    c = MySet([1, 2, 4])
    assert b < a
    assert not (b > a)
    assert b < c
    assert not (a < c)
    assert not (c > a)
    assert a == MySet([1, 2, 3])


def test_partial_ordering_does_not_override():
    def custom_lt(self, other):
        return 'custom'

    @partial_ordering
    class X:
        __lt__ = custom_lt

        def __le__(self, other):
            return True

        def __ge__(self, other):
            return True

    assert X() < X() == 'custom'


# ---------------------------------------------------------------------------
# InstancePartial / CachedInstancePartial
# ---------------------------------------------------------------------------

def test_instance_partial_basic():
    def add(a, b):
        return a + b

    class C:
        method = InstancePartial(add, 10)

    c = C()
    assert c.method(5) == 15


def test_instance_partial_partialmethod_property():
    def add(a, b):
        return a + b
    ip = InstancePartial(add, 1)
    pm = ip._partialmethod
    assert isinstance(pm, functools.partialmethod)
    assert ip.__partialmethod__.func is add


def test_cached_instance_partial():
    def add(self, b):
        return b + 1

    class C:
        method = CachedInstancePartial(add, 'unused')

    c = C()
    # accessing twice returns same bound method (cached)
    m1 = c.method
    m2 = c.method
    assert m1 is m2


def test_cached_instance_partial_class_access():
    def add(self, b):
        return b + 1

    class C:
        method = CachedInstancePartial(add)

    # class access returns bound MethodType of the partial
    bound = C.method
    assert isinstance(bound, types.MethodType)


def test_partial_as_wraps():
    def func(a, b=1):
        """doc"""
        return a + b

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    assert wrapper(1) == 2
    assert wrapper.__name__ == 'func'
    assert wrapper.__doc__ == 'doc'


# ---------------------------------------------------------------------------
# format_invocation
# ---------------------------------------------------------------------------

def test_format_invocation_basic():
    assert format_invocation('func', args=(1, 2), kwargs={'c': 3}) == 'func(1, 2, c=3)'


def test_format_invocation_no_kwargs():
    assert format_invocation('a_func', args=(1,)) == 'a_func(1)'


def test_format_invocation_kwargs_list():
    assert format_invocation('kw_func', kwargs=[('a', 1), ('b', 2)]) == 'kw_func(a=1, b=2)'


def test_format_invocation_empty():
    assert format_invocation('f') == 'f()'


def test_format_invocation_custom_repr():
    assert format_invocation('f', args=('x',), repr=str) == 'f(x)'


def test_format_invocation_bad_kwarg():
    with pytest.raises(TypeError):
        format_invocation('f', bad=True)


# ---------------------------------------------------------------------------
# format_exp_repr
# ---------------------------------------------------------------------------

class _Flag:
    def __init__(self, length, width, depth=None):
        self.length = length
        self.width = width
        self.depth = depth


def test_format_exp_repr_positional():
    flag = _Flag(5, 10)
    assert format_exp_repr(flag, ['length', 'width'], [], ['depth']) == 'Flag(5, 10)'


def test_format_exp_repr_kwargs():
    flag2 = _Flag(5, 15, 2)
    assert format_exp_repr(flag2, ['length'], ['width', 'depth']) == 'Flag(5, width=15, depth=2)'


def test_format_exp_repr_opt_key_custom():
    flag = _Flag(5, 10, 3)
    res = format_exp_repr(flag, [], [], ['depth'], opt_key=lambda v: v is None)
    assert res == 'Flag(depth=3)'


# ---------------------------------------------------------------------------
# format_nonexp_repr
# ---------------------------------------------------------------------------

def test_format_nonexp_repr_basic():
    flag = _Flag(5, 10)
    assert format_nonexp_repr(flag, ['length', 'width'], ['depth']) == '<Flag length=5 width=10>'


def test_format_nonexp_repr_id_fallback():
    flag = _Flag(5, 10)
    res = format_nonexp_repr(flag)
    assert res.startswith('<Flag id=')
    assert res.endswith('>')


# ---------------------------------------------------------------------------
# wraps / update_wrapper
# ---------------------------------------------------------------------------

def test_wraps_injected():
    def func(a, injected_arg, b):
        return a + b

    @wraps(func, injected=['injected_arg'])
    def wrapper(a, b):
        return func(a, 0, b)

    assert wrapper(1, 2) == 3
    sig = inspect.signature(wrapper)
    assert 'injected_arg' not in sig.parameters


def test_update_wrapper_returns_copy():
    def func(a):
        return a

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    new = update_wrapper(wrapper, func)
    assert new is not wrapper
    assert new(3) == 3
    assert new.__name__ == 'func'


def test_update_wrapper_expected():
    def func(a):
        return a

    def wrapper(a, extra=None):
        return func(a)

    new = update_wrapper(wrapper, func, expected='extra')
    assert 'extra' in inspect.signature(new).parameters


def test_update_wrapper_expected_pairs():
    def func(a):
        return a

    def wrapper(a, extra=None):
        return func(a)

    new = update_wrapper(wrapper, func, expected=[('extra', 99)])
    sig = inspect.signature(new)
    assert sig.parameters['extra'].default == 99


def test_update_wrapper_bad_expected():
    def func(a):
        return a

    with pytest.raises(ValueError):
        update_wrapper(func, func, expected=5)


def test_update_wrapper_expected_bad_name():
    def func(a):
        return a

    with pytest.raises(ValueError):
        update_wrapper(func, func, expected=[(1, 2)])


def test_update_wrapper_classmethod_not_supported():
    class C:
        @classmethod
        def m(cls):
            pass

    def wrapper(*a, **k):
        return None

    with pytest.raises(TypeError):
        update_wrapper(wrapper, C.__dict__['m'])


def test_update_wrapper_injected_missing_raises():
    def func(a):
        return a

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    with pytest.raises(MissingArgument):
        update_wrapper(wrapper, func, injected=['nonexistent'])


def test_update_wrapper_injected_to_varkw():
    def func(a, **kwargs):
        return a

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    new = update_wrapper(wrapper, func, injected=['nonexistent'])
    assert new(1) == 1


def test_update_wrapper_unexpected_kwarg():
    def func(a):
        return a

    def wrapper(*a, **k):
        return None

    with pytest.raises(TypeError):
        update_wrapper(wrapper, func, bogus=True)


def test_update_wrapper_hide_wrapped():
    def func(a):
        return a

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    new = update_wrapper(wrapper, func, hide_wrapped=True)
    assert not hasattr(new, '__wrapped__')

    new2 = update_wrapper(wrapper, func)
    assert new2.__wrapped__ is func


def test_update_wrapper_async():
    async def func(a):
        return a

    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    new = update_wrapper(wrapper, func)
    assert inspect.iscoroutinefunction(new)


# ---------------------------------------------------------------------------
# FunctionBuilder
# ---------------------------------------------------------------------------

def test_function_builder_basic():
    fb = FunctionBuilder('return_five', doc='returns the integer 5',
                         body='return 5')
    f = fb.get_func()
    assert f() == 5
    assert f.__doc__ == 'returns the integer 5'


def test_function_builder_varkw():
    fb = FunctionBuilder('return_five', body='return 5')
    fb.varkw = 'kw'
    f_kw = fb.get_func()
    assert f_kw(ignored_arg='x') == 5


def test_function_builder_from_func():
    def add(a, b=2):
        """Add."""
        return a + b

    fb = FunctionBuilder.from_func(add)
    assert fb.name == 'add'
    assert fb.args == ['a', 'b']
    assert fb.defaults == (2,)
    assert fb.doc == 'Add.'


def test_function_builder_from_func_not_callable():
    with pytest.raises(TypeError):
        FunctionBuilder.from_func(42)


def test_function_builder_unexpected_kwarg():
    with pytest.raises(TypeError):
        FunctionBuilder('x', bad_kw=True)


def test_function_builder_add_arg():
    fb = FunctionBuilder('f', body='return a + b')
    fb.add_arg('a')
    fb.add_arg('b', 5)
    f = fb.get_func()
    assert f(1) == 6
    assert f(1, 2) == 3


def test_function_builder_add_arg_existing():
    fb = FunctionBuilder('f')
    fb.add_arg('a')
    with pytest.raises(ExistingArgument):
        fb.add_arg('a')


def test_function_builder_add_kwonly_arg():
    fb = FunctionBuilder('f', body='return x')
    fb.add_arg('x', 7, kwonly=True)
    f = fb.get_func()
    assert f() == 7
    assert f(x=3) == 3


def test_function_builder_add_kwonly_existing():
    fb = FunctionBuilder('f')
    fb.add_arg('x', kwonly=True)
    with pytest.raises(ExistingArgument):
        fb.add_arg('x', kwonly=True)


def test_function_builder_remove_arg():
    fb = FunctionBuilder('f', body='return b')
    fb.add_arg('a')
    fb.add_arg('b')
    fb.remove_arg('a')
    assert fb.args == ['b']
    assert fb.get_func()() is not None  # it's a function returning b (unbound)


def test_function_builder_remove_arg_with_defaults():
    def f(a, b=2, c=3):
        return a + b + c

    fb = FunctionBuilder.from_func(f)
    fb.remove_arg('b')
    assert fb.args == ['a', 'c']
    assert fb.defaults == (3,)


def test_function_builder_remove_arg_missing():
    fb = FunctionBuilder('f')
    with pytest.raises(MissingArgument) as exc_info:
        fb.remove_arg('nonexistent')
    assert exc_info.value.arg_name == 'nonexistent'


def test_function_builder_remove_kwonly_arg():
    fb = FunctionBuilder('f', body='return 1')
    fb.add_arg('x', 5, kwonly=True)
    fb.remove_arg('x')
    assert fb.kwonlyargs == []
    assert 'x' not in fb.kwonlydefaults


def test_function_builder_get_sig_str():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,))
    assert fb.get_sig_str() == '(a, b=2)'


def test_function_builder_get_sig_str_no_annotations():
    fb = FunctionBuilder('f', args=['a'], annotations={'a': int})
    assert fb.get_sig_str(with_annotations=False) == '(a)'


def test_function_builder_get_invocation_str():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,))
    assert fb.get_invocation_str() == 'a, b=b'


def test_function_builder_get_invocation_str_varargs():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,), varargs='args',
                         varkw='kwargs')
    assert fb.get_invocation_str() == 'a, b, *args, **kwargs'


def test_function_builder_get_invocation_str_kwonly():
    fb = FunctionBuilder('f', kwonlyargs=['x'])
    assert fb.get_invocation_str() == 'x=x'


def test_function_builder_get_defaults_dict():
    fb = FunctionBuilder('f', args=['a', 'b', 'c'], defaults=(2, 3))
    assert fb.get_defaults_dict() == {'b': 2, 'c': 3}


def test_function_builder_get_defaults_dict_kwonly():
    fb = FunctionBuilder('f', args=['a'], defaults=(1,))
    fb.add_arg('k', 9, kwonly=True)
    assert fb.get_defaults_dict() == {'a': 1, 'k': 9}


def test_function_builder_get_arg_names():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,))
    fb.add_arg('k', 5, kwonly=True)
    assert fb.get_arg_names() == ('a', 'b', 'k')
    assert fb.get_arg_names(only_required=True) == ('a',)


def test_function_builder_get_func_with_dict():
    fb = FunctionBuilder('f', body='return 1', dict={'custom': 'val'})
    f = fb.get_func()
    assert f.custom == 'val'


def test_function_builder_get_func_without_dict():
    fb = FunctionBuilder('f', body='return 1', dict={'custom': 'val'})
    f = fb.get_func(with_dict=False)
    assert not hasattr(f, 'custom')


def test_function_builder_get_func_source():
    fb = FunctionBuilder('f', body='return 1')
    f = fb.get_func()
    assert hasattr(f, '__source__')
    f2 = fb.get_func(add_source=False)
    assert not hasattr(f2, '__source__')


def test_function_builder_async():
    fb = FunctionBuilder('f', body='return 1', is_async=True)
    f = fb.get_func()
    assert inspect.iscoroutinefunction(f)


def test_function_builder_defaults_and_kwdefaults_set():
    fb = FunctionBuilder('f', args=['a'], defaults=(5,),
                         kwonlyargs=['k'], kwonlydefaults={'k': 9},
                         body='return (a, k)')
    f = fb.get_func()
    assert f.__defaults__ == (5,)
    assert f.__kwdefaults__ == {'k': 9}
    assert f() == (5, 9)


def test_function_builder_module():
    fb = FunctionBuilder('f', body='return 1', module='mymod')
    f = fb.get_func()
    assert f.__module__ == 'mymod'


def test_function_builder_posonly():
    def f(a, b, /, c):
        return a + b + c
    fb = FunctionBuilder.from_func(f)
    assert fb.posonlyargs == ['a', 'b']
    # posonly args never keyword-forwarded
    inv = fb.get_invocation_str()
    assert 'a' in inv.split(', ')
    assert 'b' in inv.split(', ')


# ---------------------------------------------------------------------------
# noop
# ---------------------------------------------------------------------------

def test_noop():
    assert noop() is None
    assert noop(1, 2, a=3) is None


# ---------------------------------------------------------------------------
# once
# ---------------------------------------------------------------------------

def test_once_basic():
    calls = []

    @once
    def setup():
        calls.append(1)
        return 'result'

    assert setup() == 'result'
    assert setup() == 'result'
    assert calls == [1]


def test_once_thread_safe():
    calls = []

    @once
    def setup():
        time.sleep(0.02)
        calls.append(1)
        return 'value'

    results = []
    threads = []

    def worker():
        results.append(setup())

    for _ in range(5):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert results == ['value'] * 5
    assert calls == [1]
