import sys
import types
import inspect
import functools
import threading
import queue

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

def test_formatargspec_basic():
    assert inspect_formatargspec(['a', 'b']) == '(a, b)'


def test_formatargspec_defaults():
    assert inspect_formatargspec(['a', 'b'], defaults=(1, 2)) == '(a=1, b=2)'
    assert inspect_formatargspec(['a', 'b', 'c'], defaults=(2,)) == '(a, b=2, c)'.replace('c)', 'c)')


def test_formatargspec_varargs_varkw():
    assert inspect_formatargspec(['a'], varargs='args') == '(a, *args)'
    assert inspect_formatargspec(['a'], varkw='kw') == '(a, **kw)'
    assert inspect_formatargspec([], varargs='a', varkw='kw') == '(*a, **kw)'


def test_formatargspec_kwonly_only_bare_star():
    res = inspect_formatargspec([], kwonlyargs=['x', 'y'])
    assert res == '(*, x, y)'


def test_formatargspec_kwonly_with_defaults():
    res = inspect_formatargspec([], kwonlyargs=['x', 'y'],
                                kwonlydefaults={'y': 5})
    assert res == '(*, x, y=5)'


def test_formatargspec_annotations():
    res = inspect_formatargspec(['a'], annotations={'a': int, 'return': str})
    assert res == '(a: int) -> str'


def test_formatargspec_no_return_annotation():
    res = inspect_formatargspec(['a'], annotations={'a': int})
    assert res == '(a: int)'


# ---------------------------------------------------------------------------
# get_module_callables
# ---------------------------------------------------------------------------

def test_get_module_callables_by_name():
    types_map, funcs_map = get_module_callables('json')
    assert 'JSONEncoder' in types_map
    assert 'dumps' in funcs_map


def test_get_module_callables_by_instance():
    mod = types.ModuleType('test_mod_xyz')
    def some_func():
        return 1
    some_func.__module__ = 'test_mod_xyz'
    class SomeClass:
        pass
    SomeClass.__module__ = 'test_mod_xyz'
    mod.some_func = some_func
    mod.SomeClass = SomeClass
    mod.imported = len  # not in module
    sys.modules['test_mod_xyz'] = mod
    try:
        types_map, funcs_map = get_module_callables(mod)
        assert types_map == {'SomeClass': SomeClass}
        assert funcs_map == {'some_func': some_func}
    finally:
        del sys.modules['test_mod_xyz']


def test_get_module_callables_ignore():
    types_map, funcs_map = get_module_callables(
        'json', ignore=lambda name: name.startswith('JSON'))
    assert 'JSONEncoder' not in types_map
    assert 'dumps' in funcs_map


# ---------------------------------------------------------------------------
# mro_items
# ---------------------------------------------------------------------------

def test_mro_items_int():
    items = dict(mro_items(int))
    assert 'numerator' in items
    assert 'denominator' in items
    assert 'real' in items
    assert 'imag' in items


def test_mro_items_custom_hierarchy():
    class A:
        a_val = 1
    class B(A):
        b_val = 2
    keys = {k for k, v in mro_items(B)}
    assert 'a_val' in keys
    assert 'b_val' in keys


# ---------------------------------------------------------------------------
# dir_dict
# ---------------------------------------------------------------------------

def test_dir_dict_includes_inherited():
    class A:
        a_val = 1
    class B(A):
        b_val = 2
    d = dir_dict(B())
    assert d['a_val'] == 1
    assert d['b_val'] == 2


def test_dir_dict_swallows_exceptions_by_default():
    class WithProp:
        @property
        def bad(self):
            raise RuntimeError('boom')
    d = dir_dict(WithProp())
    assert 'bad' not in d


def test_dir_dict_raise_exc():
    class WithProp:
        @property
        def bad(self):
            raise RuntimeError('boom')
    with pytest.raises(RuntimeError):
        dir_dict(WithProp(), raise_exc=True)


# ---------------------------------------------------------------------------
# copy_function
# ---------------------------------------------------------------------------

def test_copy_function_identity_and_attrs():
    def func(a, b=2, *args, **kwargs):
        return a
    func.custom = 'hello'
    cp = copy_function(func)
    assert cp is not func
    assert cp.__name__ == 'func'
    assert cp.__defaults__ == (2,)
    assert cp.custom == 'hello'


def test_copy_function_no_dict():
    def func():
        return 42
    func.custom = 'hello'
    cp = copy_function(func, copy_dict=False)
    assert not hasattr(cp, 'custom')


def test_copy_function_closure_globals():
    sentinel = [1]
    def outer():
        x = sentinel
        def inner():
            return x
        return inner
    inner = outer()
    cp = copy_function(inner)
    assert cp() is sentinel


def test_copy_function_kwdefaults():
    def func(*, a=5):
        return a
    cp = copy_function(func)
    assert cp() == 5
    assert cp.__kwdefaults__ == {'a': 5}


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
    assert (b == MySet([1, 2])) is True


def test_partial_ordering_does_not_override():
    class WithLt:
        def __le__(self, other):
            return True
        def __ge__(self, other):
            return True
        def __lt__(self, other):
            return 'custom'
    original_lt = WithLt.__lt__
    partial_ordering(WithLt)
    assert WithLt.__lt__ is original_lt


# ---------------------------------------------------------------------------
# InstancePartial / CachedInstancePartial
# ---------------------------------------------------------------------------

def test_instance_partial_as_method():
    class C:
        def method(self, a, b):
            return a + b
        partial_method = InstancePartial(method, 10)

    c = C()
    assert c.partial_method(5) == 15


def test_cached_instance_partial():
    class C:
        def method(self, a, b):
            return a + b
        partial_method = CachedInstancePartial(method, 10)

    c = C()
    assert c.partial_method(5) == 15
    # cache hit path
    assert c.partial_method(7) == 17
    assert 'partial_method' in c.__dict__


def test_cached_instance_partial_set_name_and_doc():
    class C:
        def method(self, a):
            """my method doc"""
            return a
        partial_method = CachedInstancePartial(method, 10)

    assert C.partial_method.__name__ == 'partial_method'
    c = C()
    assert c.partial_method() == 10
    assert C.partial_method.__doc__ == 'my method doc'


# ---------------------------------------------------------------------------
# format_invocation
# ---------------------------------------------------------------------------

def test_format_invocation_basic():
    assert format_invocation('func', args=(1, 2), kwargs={'c': 3}) == 'func(1, 2, c=3)'


def test_format_invocation_no_kwargs():
    assert format_invocation('a_func', args=(1,)) == 'a_func(1)'


def test_format_invocation_kwargs_list():
    assert format_invocation('kw_func', kwargs=[('a', 1), ('b', 2)]) == 'kw_func(a=1, b=2)'


def test_format_invocation_sorted_dict_keys():
    assert format_invocation('f', kwargs={'b': 2, 'a': 1}) == 'f(a=1, b=2)'


def test_format_invocation_custom_repr():
    assert format_invocation('f', args=(1,), repr=str) == 'f(1)'


def test_format_invocation_unexpected_kwarg():
    with pytest.raises(TypeError):
        format_invocation('f', unexpected=True)


def test_format_invocation_empty():
    assert format_invocation('f') == 'f()'


# ---------------------------------------------------------------------------
# format_exp_repr
# ---------------------------------------------------------------------------

def test_format_exp_repr_positional():
    class Flag:
        def __init__(self, length, width, depth=None):
            self.length = length
            self.width = width
            self.depth = depth
    flag = Flag(5, 10)
    assert format_exp_repr(flag, ['length', 'width'], [], ['depth']) == 'Flag(5, 10)'


def test_format_exp_repr_req_opt():
    class Flag:
        def __init__(self, length, width, depth=None):
            self.length = length
            self.width = width
            self.depth = depth
    flag = Flag(5, 15, 2)
    res = format_exp_repr(flag, ['length'], ['width', 'depth'])
    assert res == 'Flag(5, width=15, depth=2)'


def test_format_exp_repr_dedup_names():
    class Obj:
        a = 1
        b = 2
    obj = Obj()
    res = format_exp_repr(obj, [], ['a', 'b', 'a'])
    assert res == 'Obj(a=1, b=2)'


def test_format_exp_repr_custom_opt_key():
    class Obj:
        x = 0
        y = 1
    obj = Obj()
    res = format_exp_repr(obj, [], [], ['x', 'y'], opt_key=lambda v: v == 0)
    assert res == 'Obj(y=1)'


# ---------------------------------------------------------------------------
# format_nonexp_repr
# ---------------------------------------------------------------------------

def test_format_nonexp_repr_basic():
    class Flag:
        def __init__(self, length, width, depth=None):
            self.length = length
            self.width = width
            self.depth = depth
    flag = Flag(5, 10)
    assert format_nonexp_repr(flag, ['length', 'width'], ['depth']) == '<Flag length=5 width=10>'


def test_format_nonexp_repr_fallback_id():
    class Flag:
        pass
    flag = Flag()
    res = format_nonexp_repr(flag)
    assert res.startswith('<Flag id=')
    assert res.endswith('>')


def test_format_nonexp_repr_opt_key():
    class Obj:
        x = 0
        y = 1
    obj = Obj()
    res = format_nonexp_repr(obj, [], ['x', 'y'], opt_key=lambda v: v == 0)
    assert res == '<Obj y=1>'


# ---------------------------------------------------------------------------
# wraps / update_wrapper
# ---------------------------------------------------------------------------

def test_wraps_name_doc():
    def print_return(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    @print_return
    def example():
        """docstring"""
        return 'example return value'

    assert example() == 'example return value'
    assert example.__name__ == 'example'
    assert example.__doc__ == 'docstring'


def test_update_wrapper_signature_copy():
    def target(a, b=2):
        return a + b

    def wrapper(*args, **kwargs):
        return target(*args, **kwargs)

    wrapped = update_wrapper(wrapper, target)
    sig = inspect.signature(wrapped)
    assert list(sig.parameters) == ['a', 'b']
    assert wrapped(1) == 3
    assert wrapped(1, 5) == 6


def test_update_wrapper_injected():
    def target(internal, x):
        return internal + x

    def wrapper(*args, **kwargs):
        return target(10, *args, **kwargs)

    wrapped = update_wrapper(wrapper, target, injected=['internal'])
    sig = inspect.signature(wrapped)
    assert list(sig.parameters) == ['x']
    assert wrapped(5) == 15


def test_update_wrapper_injected_string():
    def target(internal, x):
        return internal + x

    def wrapper(*args, **kwargs):
        return target(10, *args, **kwargs)

    wrapped = update_wrapper(wrapper, target, injected='internal')
    assert list(inspect.signature(wrapped).parameters) == ['x']


def test_update_wrapper_expected():
    def target(x):
        return x

    def wrapper(*args, **kwargs):
        return target(*args, **kwargs)

    wrapped = update_wrapper(wrapper, target, expected=[('y', 5)])
    sig = inspect.signature(wrapped)
    assert list(sig.parameters) == ['x', 'y']
    assert sig.parameters['y'].default == 5


def test_update_wrapper_missing_injected_varkw_ignored():
    def target(**kwargs):
        return kwargs

    def wrapper(*args, **kwargs):
        return target(*args, **kwargs)

    # inject_to_varkw default True, should not raise
    wrapped = update_wrapper(wrapper, target, injected=['missing'])
    assert wrapped() == {}


def test_update_wrapper_missing_injected_raises():
    def target(x):
        return x

    def wrapper(*args, **kwargs):
        return target(*args, **kwargs)

    with pytest.raises(MissingArgument):
        update_wrapper(wrapper, target, injected=['missing'])


def test_update_wrapper_hide_wrapped():
    def target():
        return 1

    def wrapper(*args, **kwargs):
        return target()

    wrapped = update_wrapper(wrapper, target, hide_wrapped=True)
    assert not hasattr(wrapped, '__wrapped__')


def test_update_wrapper_classmethod_raises():
    class C:
        @classmethod
        def cm(cls):
            pass
    with pytest.raises(TypeError):
        update_wrapper(lambda: None, C.cm)


def test_update_wrapper_unexpected_kwarg():
    def target():
        pass
    with pytest.raises(TypeError):
        update_wrapper(lambda: None, target, bogus=True)


def test_update_wrapper_async():
    async def target(x):
        return x

    def wrapper(*args, **kwargs):
        return target(*args, **kwargs)

    wrapped = update_wrapper(wrapper, target)
    assert inspect.iscoroutinefunction(wrapped)


# ---------------------------------------------------------------------------
# FunctionBuilder
# ---------------------------------------------------------------------------

def test_function_builder_simple():
    fb = FunctionBuilder('return_five', doc='returns the integer 5',
                         body='return 5')
    f = fb.get_func()
    assert f() == 5
    assert f.__doc__ == 'returns the integer 5'
    assert f.__name__ == 'return_five'


def test_function_builder_varkw():
    fb = FunctionBuilder('return_five', body='return 5')
    fb.varkw = 'kw'
    f = fb.get_func()
    assert f(ignored_arg='ignored_val') == 5


def test_function_builder_from_func():
    def orig(a, b=2, *args, c=3, **kwargs):
        return (a, b, args, c, kwargs)
    fb = FunctionBuilder.from_func(orig)
    assert fb.args == ['a', 'b']
    assert fb.varargs == 'args'
    assert fb.varkw == 'kwargs'
    assert fb.kwonlyargs == ['c']
    f = fb.get_func()
    assert f(1) == (1, 2, (), 3, {})


def test_function_builder_from_func_non_callable():
    with pytest.raises(TypeError):
        FunctionBuilder.from_func(42)


def test_function_builder_unexpected_kwarg():
    with pytest.raises(TypeError):
        FunctionBuilder('f', bogus_arg=1)


def test_function_builder_get_sig_str():
    fb = FunctionBuilder('f', args=['a', 'b'],
                         defaults=(1,),
                         kwonlyargs=['c'],
                         annotations={'a': int})
    assert fb.get_sig_str() == '(a: int, b=1, *, c)'
    assert fb.get_sig_str(with_annotations=False) == '(a, b=1, *, c)'


def test_function_builder_get_defaults_dict():
    fb = FunctionBuilder('f', args=['a', 'b', 'c'],
                         defaults=(2, 3),
                         kwonlyargs=['d'],
                         kwonlydefaults={'d': 4})
    assert fb.get_defaults_dict() == {'b': 2, 'c': 3, 'd': 4}


def test_function_builder_get_arg_names():
    fb = FunctionBuilder('f', args=['a', 'b'],
                         defaults=(1,),
                         kwonlyargs=['c'],
                         kwonlydefaults={'c': 2})
    assert fb.get_arg_names() == ('a', 'b', 'c')
    assert fb.get_arg_names(only_required=True) == ('a',)


def test_function_builder_add_arg():
    fb = FunctionBuilder('f')
    fb.add_arg('a')
    fb.add_arg('b', default=5)
    fb.add_arg('c', kwonly=True)
    fb.add_arg('d', default=7, kwonly=True)
    assert fb.args == ['a', 'b']
    assert fb.defaults == (5,)
    assert fb.kwonlyargs == ['c', 'd']
    assert fb.kwonlydefaults == {'d': 7}


def test_function_builder_add_arg_existing():
    fb = FunctionBuilder('f', args=['a'])
    with pytest.raises(ExistingArgument):
        fb.add_arg('a')


def test_function_builder_add_arg_existing_kwonly():
    fb = FunctionBuilder('f', kwonlyargs=['a'])
    with pytest.raises(ExistingArgument):
        fb.add_arg('a', kwonly=True)


def test_function_builder_remove_arg():
    fb = FunctionBuilder('f', args=['a', 'b', 'c'], defaults=(2, 3))
    fb.remove_arg('b')
    assert fb.args == ['a', 'c']
    assert fb.defaults == (3,)


def test_function_builder_remove_kwonly_arg():
    fb = FunctionBuilder('f', kwonlyargs=['a', 'b'],
                         kwonlydefaults={'a': 1, 'b': 2})
    fb.remove_arg('a')
    assert fb.kwonlyargs == ['b']
    assert fb.kwonlydefaults == {'b': 2}


def test_function_builder_remove_missing_arg():
    fb = FunctionBuilder('f', args=['a'])
    with pytest.raises(MissingArgument) as exc_info:
        fb.remove_arg('nope')
    assert exc_info.value.arg_name == 'nope'


def test_function_builder_custom_dict():
    fb = FunctionBuilder('f', body='return 1', dict={'custom': 'val'})
    f = fb.get_func()
    assert f.custom == 'val'
    f2 = fb.get_func(with_dict=False)
    assert not hasattr(f2, 'custom')


def test_function_builder_add_source():
    fb = FunctionBuilder('f', body='return 1')
    f = fb.get_func()
    assert hasattr(f, '__source__')
    f2 = fb.get_func(add_source=False)
    assert not hasattr(f2, '__source__')


def test_function_builder_async():
    fb = FunctionBuilder('f', body='return 1', is_async=True)
    f = fb.get_func()
    assert inspect.iscoroutinefunction(f)


def test_function_builder_posonly_varargs_forwarding():
    def orig(a, /, b, *args):
        return (a, b, args)
    fb = FunctionBuilder.from_func(orig)
    assert fb.posonlyargs == ['a']
    invocation = fb.get_invocation_str()
    assert invocation.startswith('a')


def test_function_builder_invocation_target_kwonly():
    def target(*, a, b):
        return a + b

    def wrapper(*args, **kwargs):
        return target(*args, **kwargs)

    wrapped = update_wrapper(wrapper, target)
    assert wrapped(a=1, b=2) == 3


def test_function_builder_from_partial():
    def base(a, b=2):
        return a + b
    p = functools.partial(base, 5)
    fb = FunctionBuilder.from_func(p)
    assert fb.name == 'base'


def test_function_builder_lambda_name():
    fb = FunctionBuilder('<lambda>', body='return 3')
    f = fb.get_func()
    assert f() == 3
    assert f.__name__ == '<lambda>'


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

def test_exceptions_subclass_value_error():
    assert issubclass(MissingArgument, ValueError)
    assert issubclass(ExistingArgument, ValueError)


# ---------------------------------------------------------------------------
# noop
# ---------------------------------------------------------------------------

def test_noop():
    assert noop() is None
    assert noop(1, 2, a=3) is None


# ---------------------------------------------------------------------------
# once
# ---------------------------------------------------------------------------

def test_once_single_call():
    calls = [0]

    @once
    def setup():
        calls[0] += 1
        return 'initialized'

    assert setup() == 'initialized'
    assert setup() == 'initialized'
    assert calls[0] == 1


def test_once_preserves_name():
    @once
    def my_setup():
        return 1
    assert my_setup.__name__ == 'my_setup'


def test_once_threadsafe():
    calls = [0]

    @once
    def setup():
        calls[0] += 1
        return 'val'

    results = []
    barrier = threading.Barrier(5)

    def worker():
        barrier.wait()
        results.append(setup())

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == ['val'] * 5
    assert calls[0] == 1
