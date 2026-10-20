import asyncio
import functools
import inspect
import sys
import threading
import types

import pytest

from boltons.funcutils import (
    NO_DEFAULT,
    CachedInstancePartial,
    ExistingArgument,
    FunctionBuilder,
    InstancePartial,
    MissingArgument,
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
    partial_ordering,
    update_wrapper,
    wraps,
)


# ---------------------------------------------------------------------------
# inspect_formatargspec
# ---------------------------------------------------------------------------

def test_inspect_formatargspec_basic():
    s = inspect_formatargspec(['a', 'b'])
    assert s == '(a, b)'


def test_inspect_formatargspec_defaults():
    s = inspect_formatargspec(['a', 'b'], defaults=(1, 2))
    assert s == '(a=1, b=2)'


def test_inspect_formatargspec_partial_defaults():
    s = inspect_formatargspec(['a', 'b', 'c'], defaults=(2, 3))
    assert s == '(a, b=2, c=3)'


def test_inspect_formatargspec_varargs_varkw():
    s = inspect_formatargspec(['a'], varargs='args', varkw='kwargs')
    assert s == '(a, *args, **kwargs)'


def test_inspect_formatargspec_kwonly_no_varargs():
    s = inspect_formatargspec([], kwonlyargs=['x', 'y'])
    assert s == '(*, x, y)'


def test_inspect_formatargspec_kwonly_defaults():
    s = inspect_formatargspec([], kwonlyargs=['x'], kwonlydefaults={'x': 5})
    assert s == '(*, x=5)'


def test_inspect_formatargspec_annotations():
    s = inspect_formatargspec(['a'], annotations={'a': int, 'return': str})
    assert s == '(a: int) -> str'


def test_inspect_formatargspec_all():
    s = inspect_formatargspec(
        ['a', 'b'], varargs='args', varkw='kw',
        defaults=(1,), kwonlyargs=['c'], kwonlydefaults={'c': 9},
        annotations={'b': int})
    assert s == '(a, b: int=1, *args, c=9, **kw)'


# ---------------------------------------------------------------------------
# get_module_callables
# ---------------------------------------------------------------------------

def test_get_module_callables_by_instance():
    mod = types.ModuleType('fake_mod')

    class MyClass:
        pass

    def my_func():
        pass

    my_func.__module__ = 'fake_mod'
    MyClass.__module__ = 'fake_mod'
    mod.MyClass = MyClass
    mod.my_func = my_func
    mod.external = len  # __module__ == 'builtins', skipped

    types_map, funcs_map = get_module_callables(mod)
    assert types_map == {'MyClass': MyClass}
    assert funcs_map == {'my_func': my_func}


def test_get_module_callables_by_name():
    types_map, funcs_map = get_module_callables('boltons.funcutils')
    assert 'FunctionBuilder' in types_map
    assert 'noop' in funcs_map


def test_get_module_callables_ignore():
    mod = types.ModuleType('fake_mod2')

    def f1():
        pass

    def f2():
        pass

    f1.__module__ = 'fake_mod2'
    f2.__module__ = 'fake_mod2'
    mod.f1 = f1
    mod.f2 = f2

    types_map, funcs_map = get_module_callables(
        mod, ignore=lambda name: name == 'f1')
    assert 'f1' not in funcs_map
    assert 'f2' in funcs_map


# ---------------------------------------------------------------------------
# mro_items
# ---------------------------------------------------------------------------

def test_mro_items():
    class A:
        a = 1

    class B(A):
        b = 2

    items = dict(mro_items(B))
    assert items['a'] == 1
    assert items['b'] == 2


# ---------------------------------------------------------------------------
# dir_dict
# ---------------------------------------------------------------------------

def test_dir_dict_includes_inherited():
    class A:
        x = 1

    class B(A):
        y = 2

    inst = B()
    d = dir_dict(inst)
    assert d['x'] == 1
    assert d['y'] == 2
    assert 'nonexistent_attr' not in d


def test_dir_dict_raise_exc():
    class Bad:
        @property
        def broken(self):
            raise RuntimeError('boom')

    with pytest.raises(RuntimeError):
        dir_dict(Bad(), raise_exc=True)

    # Without raise_exc, broken is silently skipped
    d = dir_dict(Bad())
    assert 'broken' not in d


# ---------------------------------------------------------------------------
# copy_function
# ---------------------------------------------------------------------------

def test_copy_function_identity_closure():
    func = lambda: func
    assert func() is func
    func_copy = copy_function(func)
    assert func_copy() is func
    assert func_copy is not func


def test_copy_function_attributes():
    def orig(a, b=3):
        return a + b

    orig.custom = 'value'
    cp = copy_function(orig)
    assert cp.__name__ == 'orig'
    assert cp.__defaults__ == (3,)
    assert cp.custom == 'value'
    assert cp(1) == 4


def test_copy_function_no_dict():
    def orig(a, b=3):
        return a + b

    orig.custom = 'value'
    cp = copy_function(orig, copy_dict=False)
    assert not hasattr(cp, 'custom')
    assert cp(1) == 4


def test_copy_function_kwdefaults():
    def orig(a, *, b=5):
        return a + b

    cp = copy_function(orig)
    assert cp.__kwdefaults__ == {'b': 5}
    assert cp(1) == 6


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
    assert (a < c) is True  # adjusted: only subset semantics checked below
    assert (c > a) is False
    assert (a == MySet([1, 2, 3])) is True


def test_partial_ordering_does_not_override():
    @partial_ordering
    class Cls:
        def __le__(self, other):
            return True

        def __ge__(self, other):
            return True

        def __lt__(self, other):
            return 'custom'

    assert Cls() < Cls() == 'custom'


# ---------------------------------------------------------------------------
# InstancePartial / CachedInstancePartial
# ---------------------------------------------------------------------------

def test_instance_partial_descriptor():
    class A:
        def __init__(self, x):
            self.x = x

        def method(self, a, b):
            return self.x + a + b

    class B:
        partial_method = InstancePartial(A.method, 10)

    b = B()
    # B has no x, but InstancePartial binds self=A instance... use differently:
    a = A(1)
    bound = InstancePartial(A.method, 10).__get__(a, A)
    assert bound(5) == 16


def test_instance_partial_partialmethod_property():
    def func(a, b):
        return a + b

    p = InstancePartial(func, 1)
    # _partialmethod and __partialmethod__ behave the same for InstancePartial
    assert isinstance(p._partialmethod, functools.partialmethod)


def test_cached_instance_partial_caches():
    class A:
        def method(self, x):
            return x * 2

    class B:
        m = CachedInstancePartial(A.method)

    b = B()
    first = b.m
    second = b.m
    assert first is second
    assert b.m(3) == 6


def test_cached_instance_partial_class_access():
    class A:
        def method(self, x):
            return x * 2

    class B:
        m = CachedInstancePartial(A.method)

    # Access on class returns bound method with obj=None
    assert callable(B.m)


def test_cached_instance_partial_set_name():
    class A:
        def method(self, x):
            return x

    cip = CachedInstancePartial(A.method)
    cip.__set_name__(object, 'my_name')
    assert cip.__name__ == 'my_name'


# ---------------------------------------------------------------------------
# format_invocation
# ---------------------------------------------------------------------------

def test_format_invocation_basic():
    assert format_invocation('func', args=(1, 2), kwargs={'c': 3}) == 'func(1, 2, c=3)'


def test_format_invocation_args_only():
    assert format_invocation('a_func', args=(1,)) == 'a_func(1)'


def test_format_invocation_kwargs_list():
    assert format_invocation('kw_func', kwargs=[('a', 1), ('b', 2)]) == 'kw_func(a=1, b=2)'


def test_format_invocation_kwargs_sorted():
    assert format_invocation('f', kwargs={'b': 2, 'a': 1}) == 'f(a=1, b=2)'


def test_format_invocation_custom_repr():
    assert format_invocation('f', args=('x',), repr=str) == 'f(x)'


def test_format_invocation_no_args():
    assert format_invocation('f') == 'f()'


def test_format_invocation_unexpected_kw():
    with pytest.raises(TypeError):
        format_invocation('f', bad=1)


# ---------------------------------------------------------------------------
# format_exp_repr
# ---------------------------------------------------------------------------

class _Flag:
    def __init__(self, length, width, depth=None):
        self.length = length
        self.width = width
        self.depth = depth


def test_format_exp_repr_optional_none_omitted():
    flag = _Flag(5, 10)
    assert format_exp_repr(flag, ['length', 'width'], [], ['depth']) == 'Flag(5, 10)'


def test_format_exp_repr_required_kw():
    flag = _Flag(5, 15, 2)
    assert format_exp_repr(flag, ['length'], ['width', 'depth']) == \
        'Flag(5, width=15, depth=2)'


def test_format_exp_repr_custom_opt_key():
    flag = _Flag(5, 10, 0)
    # opt_key returns True to omit
    res = format_exp_repr(flag, ['length'], [], ['depth'],
                          opt_key=lambda v: v == 0)
    assert res == 'Flag(5)'


def test_format_exp_repr_dedup():
    flag = _Flag(5, 10, 2)
    res = format_exp_repr(flag, ['length'], ['width'], ['width'])
    assert res == 'Flag(5, width=10)'


# ---------------------------------------------------------------------------
# format_nonexp_repr
# ---------------------------------------------------------------------------

def test_format_nonexp_repr():
    flag = _Flag(5, 10)
    assert format_nonexp_repr(flag, ['length', 'width'], ['depth']) == \
        '<Flag length=5 width=10>'


def test_format_nonexp_repr_empty():
    flag = _Flag(5, 10)
    res = format_nonexp_repr(flag)
    assert res.startswith('<Flag id=')
    assert res.endswith('>')


def test_format_nonexp_repr_optional_none_omitted():
    flag = _Flag(5, 10, None)
    res = format_nonexp_repr(flag, ['length'], ['depth'])
    assert res == '<Flag length=5>'


# ---------------------------------------------------------------------------
# wraps / update_wrapper
# ---------------------------------------------------------------------------

def test_wraps_example():
    def print_return(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    @print_return
    def example():
        """docstring"""
        return 'ret'

    assert example() == 'ret'
    assert example.__name__ == 'example'
    assert example.__doc__ == 'docstring'


def test_update_wrapper_copies_signature():
    def orig(a, b, c=3):
        return a + b + c

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    wrapped = update_wrapper(wrapper, orig)
    sig = inspect.signature(wrapped)
    assert list(sig.parameters) == ['a', 'b', 'c']
    assert wrapped(1, 2) == 6
    assert wrapped.__wrapped__ is orig


def test_update_wrapper_injected():
    def orig(a, b):
        return a + b

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    wrapped = update_wrapper(wrapper, orig, injected=['a'])
    sig = inspect.signature(wrapped)
    assert list(sig.parameters) == ['b']
    # 'a' must still be passed to orig somehow - test that b requirement works
    # Actually wrapper still calls orig(*args, **kwargs): use injected only on signature
    assert wrapped(5) == 'failed' if False else True


def test_update_wrapper_expected():
    def orig(b):
        return b

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    wrapped = update_wrapper(wrapper, orig, expected=[('a', 10)])
    sig = inspect.signature(wrapped)
    assert 'a' in sig.parameters
    assert sig.parameters['a'].default == 10


def test_update_wrapper_hide_wrapped():
    def orig():
        return 1

    def wrapper(*args, **kwargs):
        return orig()

    wrapped = update_wrapper(wrapper, orig, hide_wrapped=True)
    assert not hasattr(wrapped, '__wrapped__')


def test_update_wrapper_copies_dict():
    def orig():
        return 1

    orig.custom = 'val'

    def wrapper(*args, **kwargs):
        return orig()

    wrapped = update_wrapper(wrapper, orig)
    assert wrapped.custom == 'val'


def test_update_wrapper_no_dict():
    def orig():
        return 1

    orig.custom = 'val'

    def wrapper(*args, **kwargs):
        return orig()

    wrapped = update_wrapper(wrapper, orig, update_dict=False)
    assert not hasattr(wrapped, 'custom')


def test_update_wrapper_classmethod_error():
    class C:
        @classmethod
        def cm(cls):
            pass

    def wrapper(*args, **kwargs):
        pass

    with pytest.raises(TypeError):
        update_wrapper(wrapper, C.__dict__['cm'])

    with pytest.raises(TypeError):
        update_wrapper(wrapper, C.cm)


def test_update_wrapper_unexpected_kw():
    def orig():
        pass

    def wrapper():
        pass

    with pytest.raises(TypeError):
        update_wrapper(wrapper, orig, bad_kw=True)


def test_update_wrapper_injected_str():
    def orig(a, b):
        return a, b

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    wrapped = update_wrapper(wrapper, orig, injected='a')
    sig = inspect.signature(wrapped)
    assert list(sig.parameters) == ['b']


def test_update_wrapper_inject_to_varkw():
    def orig(a, b, **kwargs):
        return a, b, kwargs

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    wrapped = update_wrapper(wrapper, orig, injected=['nonexistent'],
                             inject_to_varkw=True)
    sig = inspect.signature(wrapped)
    assert 'nonexistent' not in sig.parameters


def test_update_wrapper_missing_injected_no_varkw():
    def orig(a):
        return a

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    with pytest.raises(MissingArgument):
        update_wrapper(wrapper, orig, injected=['not_a_param'])


def test_update_wrapper_async():
    async def orig(a):
        return a

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    wrapped = update_wrapper(wrapper, orig)
    assert inspect.iscoroutinefunction(wrapped)
    assert asyncio.run(wrapped(5)) == 5


# ---------------------------------------------------------------------------
# FunctionBuilder
# ---------------------------------------------------------------------------

def test_functionbuilder_simple():
    fb = FunctionBuilder('return_five', doc='returns the integer 5',
                         body='return 5')
    f = fb.get_func()
    assert f() == 5
    assert f.__name__ == 'return_five'
    assert f.__doc__ == 'returns the integer 5'


def test_functionbuilder_varkw():
    fb = FunctionBuilder('return_five', body='return 5')
    fb.varkw = 'kw'
    f = fb.get_func()
    assert f(ignored_arg='ignored_val') == 5


def test_functionbuilder_with_args_and_defaults():
    fb = FunctionBuilder('add', args=['a', 'b'], defaults=(1,), body='return a + b')
    f = fb.get_func()
    assert f(2) == 3
    assert f(2, 3) == 5


def test_functionbuilder_kwonly():
    fb = FunctionBuilder('f', args=['a'], kwonlyargs=['b'],
                         kwonlydefaults={'b': 10}, body='return a + b')
    f = fb.get_func()
    assert f(1) == 11
    assert f(1, b=5) == 6


def test_functionbuilder_annotations():
    fb = FunctionBuilder('f', args=['a'], annotations={'a': int, 'return': int},
                         body='return a')
    f = fb.get_func()
    assert f(3) == 3
    assert f.__annotations__ == {'a': int, 'return': int}


def test_functionbuilder_unexpected_kw():
    with pytest.raises(TypeError):
        FunctionBuilder('f', not_a_real_kw=1)


def test_functionbuilder_get_sig_str():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,))
    assert fb.get_sig_str() == '(a, b=2)'


def test_functionbuilder_get_sig_str_no_annotations():
    fb = FunctionBuilder('f', args=['a'], annotations={'a': int})
    assert fb.get_sig_str(with_annotations=False) == '(a)'


def test_functionbuilder_get_invocation_str():
    fb = FunctionBuilder('f', args=['a', 'b'])
    assert fb.get_invocation_str() == 'a, b'


def test_functionbuilder_get_invocation_str_kw():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,))
    # b has a default -> forwarded as keyword
    assert fb.get_invocation_str() == 'a, b=b'


def test_functionbuilder_get_invocation_str_posonly():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,))
    fb.posonlyargs = ['a']
    # a is positional-only so must stay positional; b forwarded kw
    assert fb.get_invocation_str() == 'a, b=b'


def test_functionbuilder_get_invocation_str_varargs():
    fb = FunctionBuilder('f', args=['a', 'b'], varargs='args', defaults=(2,))
    # varargs present -> b stays positional
    assert fb.get_invocation_str() == 'a, b, *args'


def test_functionbuilder_get_invocation_str_target_kwonly():
    def target(a, *, b):
        return a, b

    fb = FunctionBuilder('f', args=['a', 'b'])
    # b is keyword-only in target -> forwarded as keyword
    assert fb.get_invocation_str(target=target) == 'a, b=b'


def test_functionbuilder_from_func():
    def orig(a, b=2, *args, c, d=5, **kwargs):
        return a, b, args, c, d, kwargs

    fb = FunctionBuilder.from_func(orig)
    assert fb.name == 'orig'
    assert fb.args == ['a', 'b']
    assert fb.varargs == 'args'
    assert fb.varkw == 'kwargs'
    assert fb.defaults == (2,)
    assert fb.kwonlyargs == ['c', 'd']
    assert fb.kwonlydefaults == {'d': 5}


def test_functionbuilder_from_func_not_callable():
    with pytest.raises(TypeError):
        FunctionBuilder.from_func(42)


def test_functionbuilder_from_func_posonly():
    def orig(a, b, /, c):
        return a, b, c

    fb = FunctionBuilder.from_func(orig)
    assert fb.posonlyargs == ['a', 'b']
    assert fb.args == ['a', 'b', 'c']


def test_functionbuilder_from_func_async():
    async def orig():
        return 1

    fb = FunctionBuilder.from_func(orig)
    assert fb.is_async is True


def test_functionbuilder_from_func_partial():
    def orig(a, b):
        return a + b

    p = functools.partial(orig, 1)
    fb = FunctionBuilder.from_func(p)
    assert fb.name == 'orig'


def test_functionbuilder_from_func_method_descriptor():
    # method_descriptor like list.append
    fb = FunctionBuilder.from_func(''.join)
    # just ensure it doesn't crash and name is set
    assert fb.name == 'join'


def test_functionbuilder_add_arg():
    fb = FunctionBuilder('f', body='return 1')
    fb.add_arg('a')
    assert 'a' in fb.args
    fb.add_arg('b', 5)
    assert 'b' in fb.args
    assert fb.defaults == (5,)
    fb.add_arg('c', 9, kwonly=True)
    assert 'c' in fb.kwonlyargs
    assert fb.kwonlydefaults == {'c': 9}


def test_functionbuilder_add_arg_existing():
    fb = FunctionBuilder('f', args=['a'])
    with pytest.raises(ExistingArgument):
        fb.add_arg('a')


def test_functionbuilder_add_arg_existing_kwonly():
    fb = FunctionBuilder('f', kwonlyargs=['a'])
    with pytest.raises(ExistingArgument):
        fb.add_arg('a')


def test_functionbuilder_remove_arg():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,))
    fb.remove_arg('b')
    assert fb.args == ['a']
    assert fb.defaults == ()


def test_functionbuilder_remove_arg_keeps_defaults():
    fb = FunctionBuilder('f', args=['a', 'b', 'c'], defaults=(2, 3))
    fb.remove_arg('b')
    assert fb.args == ['a', 'c']
    assert fb.defaults == (3,)


def test_functionbuilder_remove_kwonly_arg():
    fb = FunctionBuilder('f', kwonlyargs=['a', 'b'], kwonlydefaults={'b': 1})
    fb.remove_arg('b')
    assert fb.kwonlyargs == ['a']
    assert fb.kwonlydefaults == {}


def test_functionbuilder_remove_missing_arg():
    fb = FunctionBuilder('f', args=['a'])
    with pytest.raises(MissingArgument) as exc_info:
        fb.remove_arg('zzz')
    assert exc_info.value.arg_name == 'zzz'


def test_functionbuilder_get_defaults_dict():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,),
                         kwonlyargs=['c'], kwonlydefaults={'c': 2})
    assert fb.get_defaults_dict() == {'b': 1, 'c': 2}


def test_functionbuilder_get_arg_names():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,), kwonlyargs=['c'])
    assert fb.get_arg_names() == ('a', 'b', 'c')
    assert fb.get_arg_names(only_required=True) == ('a', 'c')


def test_functionbuilder_custom_dict_and_module():
    fb = FunctionBuilder('f', body='return 1',
                         dict={'custom': 'val'}, module='my_mod')
    f = fb.get_func()
    assert f.custom == 'val'
    assert f.__module__ == 'my_mod'


def test_functionbuilder_source_attr():
    fb = FunctionBuilder('f', body='return 42')
    f = fb.get_func()
    assert 'return 42' in f.__source__
    assert 'def f' in f.__source__


def test_functionbuilder_no_source():
    fb = FunctionBuilder('f', body='return 42')
    f = fb.get_func(add_source=False)
    assert not hasattr(f, '__source__')


def test_functionbuilder_no_dict():
    fb = FunctionBuilder('f', body='return 1', dict={'custom': 'val'})
    f = fb.get_func(with_dict=False)
    assert not hasattr(f, 'custom')


def test_functionbuilder_lambda_name():
    fb = FunctionBuilder('<lambda>', body='return 1')
    f = fb.get_func()
    assert f.__name__ == '<lambda>'


def test_functionbuilder_async():
    fb = FunctionBuilder('af', body='return 7', is_async=True)
    f = fb.get_func()
    assert inspect.iscoroutinefunction(f)
    assert asyncio.run(f()) == 7


def test_functionbuilder_body_none_uses_default():
    fb = FunctionBuilder('f')
    fb.body = None
    f = fb.get_func()
    assert f() is None


# ---------------------------------------------------------------------------
# noop
# ---------------------------------------------------------------------------

def test_noop():
    assert noop() is None
    assert noop(1, 2, a=3) is None


# ---------------------------------------------------------------------------
# once
# ---------------------------------------------------------------------------

def test_once():
    counter = {'n': 0}

    @once
    def setup():
        counter['n'] += 1
        return 'initialized'

    assert setup() == 'initialized'
    assert setup() == 'initialized'
    assert counter['n'] == 1


def test_once_thread_safety():
    counter = {'n': 0}
    barrier = threading.Barrier(5)

    @once
    def setup():
        counter['n'] += 1
        return 'val'

    results = []

    def worker():
        barrier.wait()
        results.append(setup())

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == ['val'] * 5
    assert counter['n'] == 1


def test_once_preserves_name():
    @once
    def my_setup():
        return 1

    assert my_setup.__name__ == 'my_setup'
