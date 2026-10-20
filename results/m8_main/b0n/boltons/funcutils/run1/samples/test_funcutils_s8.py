import sys
import inspect
import functools
import threading
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

def test_inspect_formatargspec_basic():
    result = inspect_formatargspec(['a', 'b'])
    assert result == '(a, b)'


def test_inspect_formatargspec_defaults():
    result = inspect_formatargspec(['a', 'b', 'c'], defaults=(2, 3))
    assert result == '(a, b=2, c=3)'


def test_inspect_formatargspec_varargs():
    result = inspect_formatargspec(['a'], varargs='args')
    assert result == '(a, *args)'


def test_inspect_formatargspec_varkw():
    result = inspect_formatargspec(['a'], varkw='kwargs')
    assert result == '(a, **kwargs)'


def test_inspect_formatargspec_kwonly():
    result = inspect_formatargspec(['a'], kwonlyargs=['b', 'c'])
    assert result == '(a, *, b, c)'


def test_inspect_formatargspec_kwonly_defaults():
    result = inspect_formatargspec(
        ['a'], kwonlyargs=['b', 'c'],
        kwonlydefaults={'b': 5})
    assert result == '(a, *, b=5, c)'


def test_inspect_formatargspec_annotations():
    result = inspect_formatargspec(['a'], annotations={'a': int})
    assert result == '(a: int)'


def test_inspect_formatargspec_return_annotation():
    result = inspect_formatargspec(['a'], annotations={'return': int})
    assert result == '(a) -> int'


def test_inspect_formatargspec_combined():
    result = inspect_formatargspec(
        ['a', 'b'], varargs='args', varkw='kw', defaults=(1,),
        kwonlyargs=['k'], kwonlydefaults={'k': 9},
        annotations={'a': int, 'return': str})
    assert result == '(a: int, b=1, *args, k=9, **kw) -> str'


def test_inspect_formatargspec_kwonly_without_varargs():
    # No varargs but kwonlyargs -> bare '*' marker
    result = inspect_formatargspec([], kwonlyargs=['x'])
    assert result == '(*, x)'


# ---------------------------------------------------------------------------
# get_module_callables
# ---------------------------------------------------------------------------

def test_get_module_callables_self_module():
    types_map, funcs_map = get_module_callables(sys.modules[__name__])
    # the module itself defines only functions/classes from this module
    # FunctionBuilder is a class defined in boltons.funcutils, not here
    assert isinstance(types_map, dict)
    assert isinstance(funcs_map, dict)


def test_get_module_callables_by_name():
    import boltons.funcutils as fu
    types_map, funcs_map = get_module_callables('boltons.funcutils')
    assert 'FunctionBuilder' in types_map
    assert 'InstancePartial' in types_map
    assert 'CachedInstancePartial' in types_map
    assert 'copy_function' in funcs_map
    assert 'noop' in funcs_map


def test_get_module_callables_ignore():
    import boltons.funcutils as fu
    types_map, funcs_map = get_module_callables(
        'boltons.funcutils', ignore=lambda name: name.startswith('_'))
    assert not any(k.startswith('_') for k in types_map)
    assert not any(k.startswith('_') for k in funcs_map)
    assert 'noop' in funcs_map


def test_get_module_callables_ignore_hides_known():
    import boltons.funcutils as fu
    types_map, funcs_map = get_module_callables(
        'boltons.funcutils', ignore=lambda name: name == 'noop')
    assert 'noop' not in funcs_map
    assert 'copy_function' in funcs_map


# ---------------------------------------------------------------------------
# mro_items
# ---------------------------------------------------------------------------

def test_mro_items_int():
    collected = list(mro_items(int))
    keys = {k for k, v in collected}
    assert 'numerator' in keys
    assert 'denominator' in keys
    assert 'real' in keys
    assert 'imag' in keys


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
    class Base:
        a = 1

    class Child(Base):
        b = 2

    obj = Child()
    d = dir_dict(obj)
    assert d['a'] == 1
    assert d['b'] == 2


def test_dir_dict_raise_exc():
    class Boom:
        @property
        def bad(self):
            raise RuntimeError("nope")

    obj = Boom()
    # default: swallow errors
    d = dir_dict(obj)
    assert 'bad' not in d

    with pytest.raises(RuntimeError):
        dir_dict(obj, raise_exc=True)


# ---------------------------------------------------------------------------
# copy_function
# ---------------------------------------------------------------------------

def test_copy_function_basic():
    def orig(a, b=5):
        return a + b

    orig.attr = 'hello'
    cp = copy_function(orig)
    assert cp is not orig
    assert cp(1) == 6
    assert cp.__name__ == 'orig'
    assert cp.attr == 'hello'


def test_copy_function_no_dict():
    def orig():
        return 42

    orig.attr = 'x'
    cp = copy_function(orig, copy_dict=False)
    assert not hasattr(cp, 'attr')
    assert cp() == 42


def test_copy_function_closure():
    def outer(x):
        def inner():
            return x
        return inner

    f = outer(10)
    cp = copy_function(f)
    assert cp() == 10
    assert cp() == f()


def test_copy_function_self_reference():
    func = lambda: func
    assert func() is func
    cp = copy_function(func)
    assert cp() is func
    assert cp is not func


def test_copy_function_kwdefaults():
    def orig(a, *, b=3):
        return a + b

    cp = copy_function(orig)
    assert cp(1) == 4
    assert cp(1, b=10) == 11


# ---------------------------------------------------------------------------
# partial_ordering
# ---------------------------------------------------------------------------

@partial_ordering
class MySet(set):
    def __le__(self, other):
        return self.issubset(other)

    def __ge__(self, other):
        return self.issuperset(other)


def test_partial_ordering():
    a = MySet([1, 2, 3])
    b = MySet([1, 2])
    c = MySet([1, 2, 4])
    assert (b < a) is True
    assert (b > a) is False
    assert (b < c) is True
    assert (a < c) is False
    assert (c > a) is False


def test_partial_ordering_eq():
    a = MySet([1, 2])
    b = MySet([1, 2])
    assert (a == b) is True
    assert (a != b) is False


def test_partial_ordering_missing_le_ge():
    class Bare:
        pass

    # No __le__/__ge__, but adding decorator shouldn't error
    @partial_ordering
    class Wrapped(Bare):
        pass

    assert hasattr(Wrapped, '__lt__')
    assert hasattr(Wrapped, '__gt__')
    assert hasattr(Wrapped, '__eq__')


def test_partial_ordering_preserves_existing():
    class Exists:
        def __lt__(self, other):
            return 'custom'

    @partial_ordering
    class Wrapped(Exists):
        def __le__(self, other):
            return True

        def __ge__(self, other):
            return True

    assert Wrapped().__lt__(None) == 'custom'


# ---------------------------------------------------------------------------
# InstancePartial
# ---------------------------------------------------------------------------

def test_instance_partial_basic():
    def add(a, b):
        return a + b

    p = InstancePartial(add, 1)
    assert p(2) == 3
    assert isinstance(p, functools.partial)


def test_instance_partial_as_method():
    class C:
        def method(self, x, y):
            return (self, x, y)

    c = C()
    c.part = InstancePartial(C.method, 5)
    result = c.part(9)
    assert result == (c, 5, 9)


def test_instance_partial_partialmethod():
    def add(a, b):
        return a + b

    p = InstancePartial(add, 1)
    pm = p._partialmethod
    assert isinstance(pm, functools.partialmethod)


# ---------------------------------------------------------------------------
# CachedInstancePartial
# ---------------------------------------------------------------------------

def test_cached_instance_partial_basic():
    def add(a, b):
        return a + b

    p = CachedInstancePartial(add, 1)
    assert p(2) == 3


def test_cached_instance_partial_caches_bound_method():
    class C:
        def method(self, x):
            return x * 2

    c = C()
    c.part = CachedInstancePartial(C.method)
    first = c.part
    second = c.part
    assert first is second
    assert first(5) == 10


def test_cached_instance_partial_name_set_via_set_name():
    class C:
        def method(self, x):
            return x

        part = CachedInstancePartial(method)

    # access through class name -> obj None path
    assert C.__dict__['part'].__name__ == 'part'
    c = C()
    bound = c.part
    assert bound(3) == 3


def test_cached_instance_partial_class_access_returns_method():
    class C:
        def method(self, x):
            return x

        part = CachedInstancePartial(method)

    m = C.part
    assert isinstance(m, type(C.part))  # MethodType
    assert m(c=C(), x=7) == 7


# ---------------------------------------------------------------------------
# format_invocation
# ---------------------------------------------------------------------------

def test_format_invocation_basic():
    assert format_invocation('func', args=(1, 2), kwargs={'c': 3}) == 'func(1, 2, c=3)'


def test_format_invocation_no_args():
    assert format_invocation('f') == 'f()'


def test_format_invocation_only_args():
    assert format_invocation('a_func', args=(1,)) == 'a_func(1)'


def test_format_invocation_kwargs_sorted():
    assert format_invocation('kw_func', kwargs={'b': 2, 'a': 1}) == 'kw_func(a=1, b=2)'


def test_format_invocation_kwargs_list():
    assert format_invocation('kw_func', kwargs=[('a', 1), ('b', 2)]) == 'kw_func(a=1, b=2)'


def test_format_invocation_custom_repr():
    def r(v):
        return '<%s>' % v
    result = format_invocation('f', args=(1,), kwargs={'a': 2}, repr=r)
    assert result == 'f(<1>, a=<2>)'


def test_format_invocation_unexpected_kwarg():
    with pytest.raises(TypeError):
        format_invocation('f', foo=1)


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


def test_format_exp_repr_with_opt_values():
    flag2 = Flag(5, 15, 2)
    assert format_exp_repr(flag2, ['length'], ['width', 'depth']) == 'Flag(5, width=15, depth=2)'


def test_format_exp_repr_custom_opt_key():
    flag = Flag(5, 10, 0)
    # Only include optional when not None -> here depth=0 is not None so it's kept
    result = format_exp_repr(flag, ['length'], ['width', 'depth'], opt_key=lambda v: v is None)
    assert result == 'Flag(5, width=10, depth=0)'


def test_format_exp_repr_dedup_names():
    flag = Flag(5, 10, 1)
    # same name in req_names and opt_names only once
    result = format_exp_repr(flag, ['length'], ['width'], ['width', 'depth'])
    assert result == 'Flag(5, width=10, depth=1)'


# ---------------------------------------------------------------------------
# format_nonexp_repr
# ---------------------------------------------------------------------------

def test_format_nonexp_repr_basic():
    flag = Flag(5, 10)
    assert format_nonexp_repr(flag, ['length', 'width'], ['depth']) == '<Flag length=5 width=10>'


def test_format_nonexp_repr_id_when_no_names():
    flag = Flag(5, 10)
    result = format_nonexp_repr(flag)
    assert result.startswith('<Flag id=')
    assert result.endswith('>')


def test_format_nonexp_repr_id_when_all_opt_none():
    flag = Flag(5, 10)  # depth is None
    result = format_nonexp_repr(flag, ['length', 'width'], ['depth'])
    assert result == '<Flag length=5 width=10>'


def test_format_nonexp_repr_custom_opt_key():
    flag = Flag(5, 10, 0)
    result = format_nonexp_repr(flag, ['length'], ['depth'], opt_key=lambda v: v == 0)
    assert result == '<Flag length=5>'


# ---------------------------------------------------------------------------
# wraps
# ---------------------------------------------------------------------------

def test_wraps_returns_decorator():
    def orig():
        """docstring"""
        return 42

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    dec = wraps(orig)
    wrapped = dec(wrapper)
    assert wrapped() == 42
    assert wrapped.__name__ == 'orig'
    assert wrapped.__doc__ == 'docstring'


def test_wraps_injected():
    def orig(a, b):
        return a + b

    def wrapper(a, **kwargs):
        return orig(a, **kwargs)

    wrapped = wraps(orig, injected='b')(wrapper)
    sig = inspect.signature(wrapped)
    assert 'b' not in sig.parameters
    assert wrapped(3, b=4) == 7


def test_wraps_expected():
    def orig(a):
        return a

    def wrapper(a, b):
        return orig(a)

    wrapped = wraps(orig, expected='b')(wrapper)
    sig = inspect.signature(wrapped)
    assert 'b' in sig.parameters
    assert wrapped(1, 2) == 1


# ---------------------------------------------------------------------------
# update_wrapper
# ---------------------------------------------------------------------------

def test_update_wrapper_basic():
    def orig():
        """docstring"""
        return 'value'

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    w = update_wrapper(wrapper, orig)
    assert w() == 'value'
    assert w.__name__ == 'orig'
    assert w.__doc__ == 'docstring'
    assert w.__wrapped__ is orig


def test_update_wrapper_signature_preserved():
    def orig(a, b, c=3):
        return a + b + c

    def wrapper(*args, **kwargs):
        return orig(*args, **kwargs)

    w = update_wrapper(wrapper, orig)
    sig = inspect.signature(w)
    assert list(sig.parameters) == ['a', 'b', 'c']
    assert w(1, 2) == 6
    assert w(1, 2, c=10) == 13
    assert w(a=1, b=2) == 6


def test_update_wrapper_injected():
    def orig(a, b):
        return a - b

    def wrapper(a, **kwargs):
        return orig(a, **kwargs)

    w = update_wrapper(wrapper, orig, injected=['b'])
    sig = inspect.signature(w)
    assert 'b' not in sig.parameters
    assert w(10, b=3) == 7


def test_update_wrapper_injected_str():
    def orig(a, b):
        return a - b

    def wrapper(a, **kwargs):
        return orig(a, **kwargs)

    w = update_wrapper(wrapper, orig, injected='b')
    assert w(10, b=3) == 7


def test_update_wrapper_injected_into_varkw():
    def orig(a):
        return a

    def wrapper(a, **kwargs):
        return orig(a)

    # removing 'z' which isn't there, but varkw allows it
    w = update_wrapper(wrapper, orig, injected=['z'])
    assert w(5) == 5


def test_update_wrapper_injected_missing_raises():
    def orig(a):
        return a

    def wrapper(a):
        return orig(a)

    with pytest.raises(MissingArgument):
        update_wrapper(wrapper, orig, injected=['z'])


def test_update_wrapper_expected():
    def orig(a):
        return a

    def wrapper(a, b=2):
        return orig(a)

    w = update_wrapper(wrapper, orig, expected=[('b', 2)])
    sig = inspect.signature(w)
    assert 'b' in sig.parameters
    assert w(1) == 1
    assert w(1, b=5) == 1


def test_update_wrapper_hide_wrapped():
    def orig():
        return 1

    def wrapper():
        return orig()

    w = update_wrapper(wrapper, orig, hide_wrapped=True)
    assert not hasattr(w, '__wrapped__')


def test_update_wrapper_unexpected_kwarg():
    def orig():
        return 1

    def wrapper():
        return orig()

    with pytest.raises(TypeError):
        update_wrapper(wrapper, orig, bogus=123)


def test_update_wrapper_classmethod_staticmethod():
    class C:
        @classmethod
        def cm(cls):
            pass

        @staticmethod
        def sm():
            pass

    def wrapper():
        pass

    with pytest.raises(TypeError):
        update_wrapper(wrapper, C.cm)

    with pytest.raises(TypeError):
        update_wrapper(wrapper, C.sm)


def test_update_wrapper_async():
    async def orig(a, b=2):
        return a + b

    async def wrapper(*args, **kwargs):
        return await orig(*args, **kwargs)

    w = update_wrapper(wrapper, orig)
    assert inspect.iscoroutinefunction(w)


def test_update_wrapper_build_from():
    def target(a, b):
        return a + b

    def wrapper(a, b):
        return target(a, b)

    w = update_wrapper(wrapper, target, build_from=wrapper)
    assert w(2, 3) == 5


# ---------------------------------------------------------------------------
# FunctionBuilder
# ---------------------------------------------------------------------------

def test_function_builder_simple():
    fb = FunctionBuilder('return_five', doc='returns five', body='return 5')
    f = fb.get_func()
    assert f() == 5
    assert f.__name__ == 'return_five'
    assert f.__doc__ == 'returns five'


def test_function_builder_args():
    fb = FunctionBuilder('add', args=['a', 'b'], body='return a + b')
    f = fb.get_func()
    assert f(2, 3) == 5


def test_function_builder_defaults():
    fb = FunctionBuilder('add', args=['a', 'b'], defaults=(10,),
                         body='return a + b')
    f = fb.get_func()
    assert f(2) == 12
    assert f(2, 3) == 5


def test_function_builder_varargs_varkw():
    fb = FunctionBuilder('count', varargs='args', varkw='kwargs',
                         body='return (len(args), len(kwargs))')
    f = fb.get_func()
    assert f(1, 2, 3, x=1, y=2) == (3, 2)


def test_function_builder_kwonly():
    fb = FunctionBuilder('f', args=['a'], kwonlyargs=['b'],
                         kwonlydefaults={'b': 10},
                         body='return a + b')
    f = fb.get_func()
    assert f(1) == 11
    assert f(1, b=5) == 6


def test_function_builder_annotations():
    fb = FunctionBuilder('f', args=['a'], annotations={'a': int, 'return': int},
                         body='return a')
    f = fb.get_func()
    assert f.__annotations__ == {'a': int, 'return': int}


def test_function_builder_async():
    fb = FunctionBuilder('f', args=['a'], body='return a + 1', is_async=True)
    f = fb.get_func()
    assert inspect.iscoroutinefunction(f)


def test_function_builder_dict_attrs():
    fb = FunctionBuilder('f', body='return 1', **{'dict': {'attr': 'x'}})
    f = fb.get_func()
    assert f.attr == 'x'


def test_function_builder_dict_attrs_disabled():
    fb = FunctionBuilder('f', body='return 1', **{'dict': {'attr': 'x'}})
    f = fb.get_func(with_dict=False)
    assert not hasattr(f, 'attr')


def test_function_builder_source_attr():
    fb = FunctionBuilder('f', body='return 1')
    f = fb.get_func()
    assert hasattr(f, '__source__')
    assert 'def f' in f.__source__

    f2 = fb.get_func(add_source=False)
    assert not hasattr(f2, '__source__')


def test_function_builder_from_func():
    def orig(a, b=2, *args, c=3, **kwargs):
        return a

    fb = FunctionBuilder.from_func(orig)
    assert fb.name == 'orig'
    assert fb.args == ['a', 'b']
    assert fb.varargs == 'args'
    assert fb.varkw == 'kwargs'
    assert fb.kwonlyargs == ['c']
    assert fb.kwonlydefaults == {'c': 3}
    assert fb.defaults == (2,)


def test_function_builder_from_func_not_callable():
    with pytest.raises(TypeError):
        FunctionBuilder.from_func(42)


def test_function_builder_from_func_async():
    async def orig(a):
        return a

    fb = FunctionBuilder.from_func(orig)
    assert fb.is_async is True


def test_function_builder_get_sig_str():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,),
                         varargs='args', varkw='kw',
                         kwonlyargs=['c'], kwonlydefaults={'c': 3})
    sig = fb.get_sig_str()
    assert sig == '(a, b=2, *args, c=3, **kw)'


def test_function_builder_get_sig_str_no_annotations():
    fb = FunctionBuilder('f', args=['a'], annotations={'a': int})
    assert fb.get_sig_str(with_annotations=True) == '(a: int)'
    assert fb.get_sig_str(with_annotations=False) == '(a)'


def test_function_builder_add_arg():
    fb = FunctionBuilder('f', body='return a')
    fb.add_arg('a', default=5)
    f = fb.get_func()
    assert f() == 5
    assert f(10) == 10


def test_function_builder_add_arg_kwonly():
    fb = FunctionBuilder('f', body='return b')
    fb.add_arg('b', default=7, kwonly=True)
    f = fb.get_func()
    assert f() == 7
    assert f(b=3) == 3


def test_function_builder_add_arg_existing():
    fb = FunctionBuilder('f', args=['a'], body='return a')
    with pytest.raises(ExistingArgument):
        fb.add_arg('a')


def test_function_builder_add_arg_existing_kwonly():
    fb = FunctionBuilder('f', kwonlyargs=['b'], body='return 1')
    with pytest.raises(ExistingArgument):
        fb.add_arg('b', kwonly=True)


def test_function_builder_remove_arg():
    fb = FunctionBuilder('f', args=['a', 'b'], body='return b')
    fb.remove_arg('a')
    f = fb.get_func()
    assert list(inspect.signature(f).parameters) == ['b']
    assert f(9) == 9


def test_function_builder_remove_arg_with_defaults():
    fb = FunctionBuilder('f', args=['a', 'b', 'c'], defaults=(2, 3),
                         body='return (a, b, c)')
    fb.remove_arg('b')
    f = fb.get_func()
    # remaining defaults should map to a and c
    assert f(1) == (1, 2, 3)
    assert f(1, 10) == (1, 10, 3)


def test_function_builder_remove_arg_kwonly():
    fb = FunctionBuilder('f', kwonlyargs=['b'], kwonlydefaults={'b': 1},
                         body='return 1')
    fb.remove_arg('b')
    assert 'b' not in fb.kwonlyargs
    assert 'b' not in fb.kwonlydefaults


def test_function_builder_remove_arg_missing():
    fb = FunctionBuilder('f', args=['a'], body='return a')
    with pytest.raises(MissingArgument) as exc_info:
        fb.remove_arg('z')
    assert exc_info.value.arg_name == 'z'


def test_function_builder_get_defaults_dict():
    fb = FunctionBuilder('f', args=['a', 'b', 'c'], defaults=(2, 3),
                         body='return 1')
    assert fb.get_defaults_dict() == {'b': 2, 'c': 3}


def test_function_builder_get_defaults_dict_kwonly():
    fb = FunctionBuilder('f', args=['a'], defaults=(1,),
                         kwonlyargs=['b'], kwonlydefaults={'b': 5},
                         body='return 1')
    assert fb.get_defaults_dict() == {'a': 1, 'b': 5}


def test_function_builder_get_arg_names():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,),
                         kwonlyargs=['c'], kwonlydefaults={'c': 3},
                         body='return 1')
    assert fb.get_arg_names() == ('a', 'b', 'c')
    assert fb.get_arg_names(only_required=True) == ('a',)


def test_function_builder_unexpected_kwarg():
    with pytest.raises(TypeError):
        FunctionBuilder('f', bogus=1)


def test_function_builder_invocation_str_positional():
    fb = FunctionBuilder('f', args=['a', 'b'], body='return 1')
    assert fb.get_invocation_str() == 'a, b'


def test_function_builder_invocation_str_defaults_as_kwargs():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,), body='return 1')
    assert fb.get_invocation_str() == 'a, b=b'


def test_function_builder_invocation_varargs_positional():
    fb = FunctionBuilder('f', args=['a', 'b'], defaults=(2,),
                         varargs='args', body='return 1')
    assert fb.get_invocation_str() == 'a, b, *args'


def test_function_builder_invocation_kwonly():
    fb = FunctionBuilder('f', args=['a'], kwonlyargs=['b'], body='return 1')
    assert fb.get_invocation_str() == 'a, b=b'


def test_function_builder_invocation_varkw():
    fb = FunctionBuilder('f', args=['a'], varkw='kw', body='return 1')
    assert fb.get_invocation_str() == 'a, **kw'


def test_function_builder_posonly_not_kwarg():
    def orig(a, b, /, c):
        return a + b + c

    fb = FunctionBuilder.from_func(orig)
    assert fb.posonlyargs == ['a', 'b']
    # posonly 'a','b' stay positional, 'c' has no default -> positional
    assert fb.get_invocation_str() == 'a, b, c'


def test_function_builder_target_kwonly_forwarded_as_kw():
    def target(*args, only_kw=None, **kwargs):
        return only_kw

    fb = FunctionBuilder('f', args=['only_kw'], body='return 1')
    assert fb.get_invocation_str(target=target) == 'only_kw=only_kw'


def test_function_builder_from_func_partial():
    def orig(a, b=1):
        return a + b

    p = functools.partial(orig, 5)
    fb = FunctionBuilder.from_func(p)
    assert fb.name == 'orig'


def test_function_builder_compile_filename():
    fb = FunctionBuilder('f', body='return 1', filename='myfile')
    f = fb.get_func()
    assert f.__code__.co_filename.startswith('<myfile-')


# ---------------------------------------------------------------------------
# MissingArgument / ExistingArgument
# ---------------------------------------------------------------------------

def test_missing_argument_is_value_error():
    assert issubclass(MissingArgument, ValueError)
    with pytest.raises(ValueError):
        raise MissingArgument('x')


def test_existing_argument_is_value_error():
    assert issubclass(ExistingArgument, ValueError)
    with pytest.raises(ValueError):
        raise ExistingArgument('x')


# ---------------------------------------------------------------------------
# noop
# ---------------------------------------------------------------------------

def test_noop():
    assert noop() is None
    assert noop(1, 2, 3) is None
    assert noop(a=1, b=2) is None


# ---------------------------------------------------------------------------
# once
# ---------------------------------------------------------------------------

def test_once_caches_result():
    calls = []

    @once
    def setup():
        calls.append(1)
        return 'done'

    assert setup() == 'done'
    assert setup() == 'done'
    assert len(calls) == 1


def test_once_none_result_cached():
    calls = []

    @once
    def setup():
        calls.append(1)
        return None

    assert setup() is None
    assert setup() is None
    assert len(calls) == 1


def test_once_preserves_name_doc():
    @once
    def my_setup():
        """A docstring"""
        return 1

    assert my_setup.__name__ == 'my_setup'
    assert my_setup.__doc__ == 'A docstring'


def test_once_thread_safe():
    calls = []
    barrier = threading.Barrier(5)
    results = []

    @once
    def setup():
        calls.append(1)
        return 'result'

    def worker():
        barrier.wait()
        results.append(setup())

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1
    assert results == ['result'] * 5


# ---------------------------------------------------------------------------
# NO_DEFAULT sentinel
# ---------------------------------------------------------------------------

def test_no_default_sentinel():
    assert NO_DEFAULT is not None
    fb = FunctionBuilder('f', body='return 1')
    fb.add_arg('a')
    assert fb.defaults is None or fb.defaults == ()
    f = fb.get_func()
    # no default -> required
    with pytest.raises(TypeError):
        f()
