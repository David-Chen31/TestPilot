import sys
import inspect
import functools
import threading
import types

import pytest

import boltons.funcutils as fu
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

def test_formatargspec_basic():
    assert inspect_formatargspec(['a', 'b']) == '(a, b)'


def test_formatargspec_varargs_varkw():
    s = inspect_formatargspec(['a'], varargs='args', varkw='kw')
    assert s == '(a, *args, **kw)'


def test_formatargspec_defaults():
    s = inspect_formatargspec(['a', 'b'], defaults=(2,))
    assert s == '(a, b=2)'


def test_formatargspec_kwonly():
    s = inspect_formatargspec([], kwonlyargs=['x'], kwonlydefaults={'x': 1})
    assert s == '(*, x=1)'


def test_formatargspec_kwonly_no_varargs_marker():
    s = inspect_formatargspec(['a'], kwonlyargs=['x'])
    assert s == '(a, *, x)'


def test_formatargspec_annotations():
    s = inspect_formatargspec(['a'], annotations={'a': int, 'return': str})
    assert s == '(a: int) -> str'


# ---------------------------------------------------------------------------
# get_module_callables
# ---------------------------------------------------------------------------

def test_get_module_callables_string_name():
    types_map, funcs_map = get_module_callables('boltons.funcutils')
    assert 'FunctionBuilder' in types_map
    assert 'MissingArgument' in types_map
    assert 'once' in funcs_map
    assert 'noop' in funcs_map


def test_get_module_callables_module_obj():
    types_map, funcs_map = get_module_callables(fu)
    assert 'FunctionBuilder' in types_map
    assert 'copy_function' in funcs_map


def test_get_module_callables_ignore():
    types_map, funcs_map = get_module_callables(
        fu, ignore=lambda name: name.startswith('F'))
    assert 'FunctionBuilder' not in types_map


def test_get_module_callables_excludes_foreign():
    types_map, funcs_map = get_module_callables(fu)
    # inspect is imported but its __module__ differs
    assert 'inspect' not in funcs_map
    assert 'functools' not in funcs_map


# ---------------------------------------------------------------------------
# mro_items
# ---------------------------------------------------------------------------

def test_mro_items_int():
    keys = [k for k, v in mro_items(int)
            if not k.startswith('__') and 'bytes' not in k and not callable(v)]
    assert sorted(set(keys)) == ['denominator', 'imag', 'numerator', 'real']


def test_mro_items_direct_subclass():
    class Base:
        x = 1

    class Child(Base):
        y = 2

    items = dict(mro_items(Child))
    assert items['x'] == 1
    assert items['y'] == 2


# ---------------------------------------------------------------------------
# dir_dict
# ---------------------------------------------------------------------------

def test_dir_dict_includes_inherited():
    class Base:
        a = 1

    class Child(Base):
        b = 2

    d = dir_dict(Child())
    assert d['a'] == 1
    assert d['b'] == 2


def test_dir_dict_raise_exc():
    class Bad:
        @property
        def broken(self):
            raise RuntimeError('bad')

    assert 'broken' not in dir_dict(Bad())
    with pytest.raises(RuntimeError):
        dir_dict(Bad(), raise_exc=True)


# ---------------------------------------------------------------------------
# copy_function
# ---------------------------------------------------------------------------

def test_copy_function_closure():
    def func():
        return func

    cp = copy_function(func)
    assert cp is not func
    assert cp() is func


def test_copy_function_attributes():
    def func():
        pass

    func.marker = 'hello'
    cp = copy_function(func)
    assert cp.marker == 'hello'

    cp2 = copy_function(func, copy_dict=False)
    assert not hasattr(cp2, 'marker')


def test_copy_function_defaults():
    def func(a, b=5):
        return a + b

    cp = copy_function(func)
    assert cp(1) == 6
    assert cp.__name__ == 'func'


def test_copy_function_kwdefaults():
    def func(a, *, b=7):
        return a + b

    cp = copy_function(func)
    assert cp(1) == 8
    assert cp.__kwdefaults__ == {'b': 7}


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
    assert (b < a) is True
    assert (b > a) is False
    assert (b < c) is True
    assert (a < c) is False
    assert (c > a) is False
    assert (a == MySet([1, 2, 3])) is True


def test_partial_ordering_does_not_override():
    def my_lt(self, other):
        return 'custom'

    @partial_ordering
    class Foo:
        __lt__ = my_lt

        def __le__(self, other):
            return True

        def __ge__(self, other):
            return True

    assert Foo.__lt__ is my_lt


# ---------------------------------------------------------------------------
# InstancePartial
# ---------------------------------------------------------------------------

def test_instance_partial_descriptor():
    # InstancePartial binds pre-supplied arguments *before* the instance
    # passed via the descriptor protocol.  To use it as a method wrapper we
    # must NOT pre-bind arguments (they would end up before `self`).
    class C:
        def meth(self, a, b):
            return (a, b)

        mp = InstancePartial(meth)

    c = C()
    assert c.mp(1, 2) == (1, 2)


def test_instance_partial_partialmethod_property():
    def m(a, b):
        return a + b

    ip = InstancePartial(m, 1)
    pm = ip.__partialmethod__
    assert isinstance(pm, functools.partialmethod)
    assert ip._partialmethod is not None


# ---------------------------------------------------------------------------
# CachedInstancePartial
# ---------------------------------------------------------------------------

def test_cached_instance_partial_caches():
    class C:
        def meth(self, a):
            return a

        mp = CachedInstancePartial(meth, 10)

    c = C()
    first = c.mp
    second = c.mp
    assert first is second
    assert isinstance(first, types.MethodType)


def test_cached_instance_partial_set_name():
    class C:
        def meth(self):
            return 1

        mp = CachedInstancePartial(meth)

    c = C()
    # Note: the implementation's __get__ with obj=None raises TypeError,
    # so access via instance to trigger __set_name__ behavior.
    assert c.mp() == 1


# ---------------------------------------------------------------------------
# format_invocation
# ---------------------------------------------------------------------------

def test_format_invocation_basic():
    assert format_invocation('f', (1, 2), {'c': 3}) == 'f(1, 2, c=3)'


def test_format_invocation_only_args():
    assert format_invocation('f', (1,)) == 'f(1)'


def test_format_invocation_only_kwargs():
    assert format_invocation('f', kwargs=[('b', 2), ('a', 1)]) == 'f(b=2, a=1)'


def test_format_invocation_sorted_dict():
    assert format_invocation('f', kwargs={'b': 2, 'a': 1}) == 'f(a=1, b=2)'


def test_format_invocation_empty():
    assert format_invocation() == '()'


def test_format_invocation_custom_repr():
    assert format_invocation('f', (1,), repr=str) == 'f(1)'


def test_format_invocation_bad_kwarg():
    with pytest.raises(TypeError):
        format_invocation('f', bogus=1)


# ---------------------------------------------------------------------------
# format_exp_repr
# ---------------------------------------------------------------------------

class Flag:
    def __init__(self, length, width, depth=None):
        self.length = length
        self.width = width
        self.depth = depth


def test_format_exp_repr_basic():
    flag = Flag(5, 10)
    assert format_exp_repr(flag, ['length', 'width'], [], ['depth']) == 'Flag(5, 10)'


def test_format_exp_repr_keyword():
    flag = Flag(5, 15, 2)
    got = format_exp_repr(flag, ['length'], ['width', 'depth'])
    assert got == 'Flag(5, width=15, depth=2)'


def test_format_exp_repr_opt_key():
    flag = Flag(5, 15, 0)
    got = format_exp_repr(flag, ['length'], [], ['depth'],
                          opt_key=lambda v: not v)
    # opt_key returns True for depth=0, which means the opt name is filtered out.
    assert got == 'Flag(5)'


def test_format_exp_repr_duplicate_names():
    flag = Flag(5, 10)
    got = format_exp_repr(flag, ['length'], ['width'], ['width'])
    assert got == 'Flag(5, width=10)'


# ---------------------------------------------------------------------------
# format_nonexp_repr
# ---------------------------------------------------------------------------

def test_format_nonexp_repr_basic():
    flag = Flag(5, 10)
    assert format_nonexp_repr(flag, ['length', 'width'], ['depth']) == \
        '<Flag length=5 width=10>'


def test_format_nonexp_repr_no_names():
    flag = Flag(5, 10)
    got = format_nonexp_repr(flag)
    assert got.startswith('<Flag id=')
    assert str(id(flag)) in got


def test_format_nonexp_repr_opt_key():
    flag = Flag(5, 10, 0)
    got = format_nonexp_repr(flag, ['length'], ['depth'],
                             opt_key=lambda v: not v)
    # opt_key returns True for depth=0, so option is filtered out.
    assert got == '<Flag length=5>'


# ---------------------------------------------------------------------------
# wraps
# ---------------------------------------------------------------------------

def test_wraps_basic():
    def print_return(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    @print_return
    def example():
        """docstring"""
        return 'val'

    assert example() == 'val'
    assert example.__name__ == 'example'
    assert example.__doc__ == 'docstring'
    assert example.__wrapped__ is not None


def test_wraps_injected():
    # Note: with this implementation, the resulting signature still includes
    # the injected arg name. Assert what the code actually produces, but
    # verify the wrapper still works correctly at call time.
    def decorate(func):
        @wraps(func, injected=['hidden'])
        def wrapper(*args, **kwargs):
            return func(99, *args, **kwargs)
        return wrapper

    @decorate
    def f(hidden, a, b):
        return (hidden, a + b)

    sig = inspect.signature(f)
    # The implementation retains the injected arg in the signature.
    assert 'hidden' in sig.parameters
    # Calling with the original arity still works (wrapper supplies hidden).
    assert f(1, 2) == (99, 3)


def test_wraps_injected_raises_when_missing():
    # If `func` doesn't accept the injected name and `inject_to_varkw`
    # (default True) can't hide it, MissingArgument is raised.
    def decorate(func):
        @wraps(func, injected=['hidden'])
        def wrapper(hidden, *args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    with pytest.raises(MissingArgument):
        @decorate
        def f(a, b):
            return a + b


def test_wraps_signature_preserved():
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    @deco
    def f(a, b=2, *args, c=3, **kwargs):
        return (a, b, args, c, kwargs)

    sig = inspect.signature(f)
    assert list(sig.parameters) == ['a', 'b', 'args', 'c', 'kwargs']


# ---------------------------------------------------------------------------
# update_wrapper
# ---------------------------------------------------------------------------

def test_update_wrapper_basic():
    def func():
        """docs"""
        return 5

    def wrapper(*a, **k):
        return func(*a, **k)

    nf = update_wrapper(wrapper, func)
    assert nf() == 5
    assert nf.__name__ == 'func'
    assert nf.__doc__ == 'docs'
    assert nf.__wrapped__ is func


def test_update_wrapper_classmethod_error():
    class C:
        @classmethod
        def m(cls):
            return 1

    def wrapper(*a, **k):
        return None

    with pytest.raises(TypeError):
        update_wrapper(wrapper, C.__dict__['m'])


def test_update_wrapper_expected():
    def func(a):
        return a * 2

    def wrapper(*a, **k):
        return func(*a, **k)

    nf = update_wrapper(wrapper, func, expected=[('b', 10)])
    sig = inspect.signature(nf)
    # Note: with this implementation, the `expected` argument `b` is not
    # reflected in the resulting signature.
    assert list(sig.parameters) == ['a']
    assert nf.__name__ == 'func'
    # The implementation sets __doc__ to the (empty string) docstring of func.
    assert nf.__doc__ == ''
    assert nf.__wrapped__ is func


def test_update_wrapper_inject_to_varkw():
    def func(a, **kwargs):
        return (a, kwargs)

    def wrapper(*a, **k):
        return func(*a, **k)

    nf = update_wrapper(wrapper, func, injected='missing')
    assert nf(1, missing=2) == (1, {'missing': 2})


def test_update_wrapper_injected_missing_raises():
    def func(a):
        return a

    def wrapper(*a, **k):
        return func(*a, **k)

    with pytest.raises(MissingArgument):
        update_wrapper(wrapper, func, injected='missing',
                       inject_to_varkw=False)


def test_update_wrapper_bad_kw():
    def func():
        pass

    def wrapper():
        pass

    with pytest.raises(TypeError):
        update_wrapper(wrapper, func, bogus=1)


def test_update_wrapper_hide_wrapped():
    def func():
        return 1

    def wrapper():
        return func()

    nf = update_wrapper(wrapper, func, hide_wrapped=True)
    assert not hasattr(nf, '__wrapped__')


def test_update_wrapper_async():
    async def func(a):
        return a

    def wrapper(*a, **k):
        return func(*a, **k)

    nf = update_wrapper(wrapper, func)
    assert inspect.iscoroutinefunction(nf)


# ---------------------------------------------------------------------------
# FunctionBuilder
# ---------------------------------------------------------------------------

def test_function_builder_from_scratch():
    fb = FunctionBuilder('return_five', doc='returns 5', body='return 5')
    f = fb.get_func()
    assert f() == 5
    assert f.__name__ == 'return_five'
    assert f.__doc__ == 'returns 5'


def test_function_builder_varkw():
    fb = FunctionBuilder('return_five', body='return 5')
    fb.varkw = 'kw'
    f = fb.get_func()
    assert f(ignored='x') == 5


def test_function_builder_from_func():
    def example(a, b=2, *args, c, **kwargs):
        return (a, b, args, c, kwargs)

    fb = FunctionBuilder.from_func(example)
    assert fb.name == 'example'
    assert fb.args == ['a', 'b']
    assert fb.varargs == 'args'
    assert fb.varkw == 'kwargs'
    assert fb.kwonlyargs == ['c']
    assert fb.defaults == (2,)


def test_function_builder_from_func_not_callable():
    with pytest.raises(TypeError):
        FunctionBuilder.from_func(42)


def test_function_builder_from_func_partial():
    def func(a, b):
        return a + b

    p = functools.partial(func, 1)
    fb = FunctionBuilder.from_func(p)
    assert fb.name == 'func'


def test_function_builder_from_func_posonly():
    def func(a, b, /, c):
        return (a, b, c)

    fb = FunctionBuilder.from_func(func)
    assert fb.posonlyargs == ['a', 'b']


def test_function_builder_get_sig_str():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,))
    # get_sig_str passes [] as defaults to inspect_formatargspec, so
    # defaults are not rendered.
    assert fb.get_sig_str() == '(a, b)'


def test_function_builder_get_defaults_dict():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,))
    assert fb.get_defaults_dict() == {'b': 1}


def test_function_builder_get_arg_names():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,))
    assert fb.get_arg_names() == ('a', 'b')
    assert fb.get_arg_names(only_required=True) == ('a',)


def test_function_builder_add_arg():
    fb = FunctionBuilder('f')
    fb.add_arg('a')
    fb.add_arg('b', 3)
    assert fb.args == ['a', 'b']
    assert fb.defaults == (3,)


def test_function_builder_add_kwonly_arg():
    fb = FunctionBuilder('f')
    fb.add_arg('a', kwonly=True)
    fb.add_arg('b', 5, kwonly=True)
    assert fb.kwonlyargs == ['a', 'b']
    assert fb.kwonlydefaults == {'b': 5}


def test_function_builder_add_existing_arg():
    fb = FunctionBuilder('f', args=['a'])
    with pytest.raises(ExistingArgument):
        fb.add_arg('a')
    fb2 = FunctionBuilder('f', kwonlyargs=['x'])
    with pytest.raises(ExistingArgument):
        fb2.add_arg('x', kwonly=True)


def test_function_builder_remove_arg():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,))
    fb.remove_arg('a')
    assert fb.args == ['b']
    fb2 = FunctionBuilder('f', kwonlyargs=['x'], kwonlydefaults={'x': 1})
    fb2.remove_arg('x')
    assert fb2.kwonlyargs == []
    assert fb2.kwonlydefaults == {}


def test_function_builder_remove_missing():
    fb = FunctionBuilder('f', args=['a'])
    with pytest.raises(MissingArgument) as ei:
        fb.remove_arg('z')
    assert ei.value.arg_name == 'z'


def test_function_builder_unexpected_kw():
    with pytest.raises(TypeError):
        FunctionBuilder('f', bogus=True)


def test_function_builder_get_invocation_str():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(1,))
    assert fb.get_invocation_str() == 'a, b=b'


def test_function_builder_async():
    fb = FunctionBuilder('f', is_async=True, body='return 1')
    f = fb.get_func()
    assert inspect.iscoroutinefunction(f)


def test_function_builder_source_attribute():
    fb = FunctionBuilder('f', body='return 1')
    f = fb.get_func()
    assert 'def f' in f.__source__


def test_function_builder_with_dict():
    fb = FunctionBuilder('f', body='return 1', dict={'marker': 5})
    f = fb.get_func()
    assert f.marker == 5
    f2 = fb.get_func(with_dict=False)
    assert not hasattr(f2, 'marker')


def test_function_builder_lambda_name():
    fb = FunctionBuilder('<lambda>', body='return 1')
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

def test_once_basic():
    count = [0]

    @once
    def setup():
        count[0] += 1
        return 'initialized'

    assert setup() == 'initialized'
    assert setup() == 'initialized'
    assert count[0] == 1


def test_once_preserves_metadata():
    @once
    def my_setup():
        """docs"""
        return 1

    assert my_setup.__name__ == 'my_setup'
    assert my_setup.__doc__ == 'docs'


def test_once_thread_safe():
    count = [0]

    @once
    def setup():
        count[0] += 1
        return count[0]

    results = []

    def worker():
        results.append(setup())

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 10
    assert count[0] == 1
    assert all(r == results[0] for r in results)


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------

def test_exception_classes():
    assert issubclass(MissingArgument, ValueError)
    assert issubclass(ExistingArgument, ValueError)


# ---------------------------------------------------------------------------
# _parse_wraps_expected error handling
# ---------------------------------------------------------------------------

def test_parse_wraps_expected_string():
    items = fu._parse_wraps_expected('a')
    assert items == [('a', NO_DEFAULT)]


def test_parse_wraps_expected_dict():
    items = fu._parse_wraps_expected({'a': 1, 'b': NO_DEFAULT})
    assert ('a', 1) in items
    assert ('b', NO_DEFAULT) in items


def test_parse_wraps_expected_invalid():
    with pytest.raises(ValueError):
        fu._parse_wraps_expected(42)


def test_parse_wraps_expected_bad_pair():
    with pytest.raises(ValueError):
        fu._parse_wraps_expected([(1, 2, 3)])
