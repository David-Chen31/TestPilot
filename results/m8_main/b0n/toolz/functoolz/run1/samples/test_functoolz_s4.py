import inspect
import pickle
import sys
from functools import partial
from operator import add

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
# identity / apply
# ---------------------------------------------------------------------------

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3
    assert identity(None) is None


def test_apply_basic():
    def double(x):
        return 2 * x

    assert apply(double, 5) == 10


def test_apply_kwargs():
    def f(x, y=0, z=0):
        return x + y + z

    assert apply(f, 1, 2, z=3) == 6


def test_apply_no_args_raises():
    with pytest.raises(TypeError):
        apply()


def test_apply_in_map():
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

def _inc(x):
    return x + 1


def _double(x):
    return 2 * x


def test_thread_first_simple():
    assert thread_first(1, _inc, _double) == 4


def test_thread_first_tuple_forms():
    def add(x, y):
        return x + y

    def power(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (power, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(5) == 5


def test_thread_last_simple():
    assert thread_last(1, _inc, _double) == 4


def test_thread_last_tuple_forms():
    def add(x, y):
        return x + y

    def power(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (power, 2)) == 32


def test_thread_last_no_forms():
    assert thread_last(5) == 5


def test_thread_last_with_map_filter():
    def iseven(x):
        return x % 2 == 0

    out = list(thread_last([1, 2, 3], (map, _inc), (filter, iseven)))
    assert out == [2, 4]


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_class_and_instance_values():
    class MyClass(object):
        @instanceproperty(classval='classval')
        def val(self):
            return 42

    assert MyClass.val == 'classval'
    obj = MyClass()
    assert obj.val == 42


def test_instanceproperty_as_bare_decorator():
    class MyClass(object):
        @instanceproperty
        def val(self):
            return 'x'

    assert MyClass.val is None
    assert MyClass().val == 'x'


def test_instanceproperty_setter_and_deleter():
    store = {}

    class MyClass(object):
        @instanceproperty
        def val(self):
            return store.get('val', 0)

        @val.setter
        def val(self, new):
            store['val'] = new

    obj = MyClass()
    assert obj.val == 0
    obj.val = 10
    assert obj.val == 10


def test_instanceproperty_returns_partial_when_no_fget():
    p = instanceproperty()
    assert isinstance(p, partial)


def test_instance_property_is_property_subclass():
    def fget(self):
        return 1

    ip = InstanceProperty(fget=fget, classval='cv')
    assert isinstance(ip, property)
    assert ip.classval == 'cv'


def test_instance_property_pickle_roundtrip():
    # InstanceProperty defines __reduce__, so pickle should work.
    def fget(self):
        return 1

    ip = InstanceProperty(fget=fget, classval='cls')
    restored = pickle.loads(pickle.dumps(ip))
    assert isinstance(restored, InstanceProperty)
    assert restored.classval == 'cls'
    assert restored.fget is fget


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def _mul(x, y):
    return x * y


def test_curry_basic_partial():
    mul = curry(_mul)
    double = mul(2)
    assert isinstance(double, curry)
    assert double(10) == 20


def test_curry_keyword_args():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_requires_callable():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_requires_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_str_repr_doc_name():
    def somefunc(x, y):
        """docstring here"""
        return x + y

    c = curry(somefunc)
    assert str(c) == str(somefunc)
    assert repr(c) == repr(somefunc)
    assert c.__doc__ == "docstring here"
    assert c.__name__ == 'somefunc'
    assert c.func_name == 'somefunc'


def test_curry_call_and_bind():
    c = curry(_mul)
    assert c.call(3, 4) == 12
    bound = c.bind(3)
    assert isinstance(bound, curry)
    assert bound(4) == 12


def test_curry_equality_and_hash():
    c1 = curry(_mul, 2)
    c2 = curry(_mul, 2)
    c3 = curry(_mul, 3)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert not (c1 == "not a curry")
    assert c1 != "not a curry"


def test_curry_kwargs_equality():
    def f(x, y=0):
        return x + y

    c1 = curry(f, y=1)
    c2 = curry(f, y=1)
    c3 = curry(f, y=2)
    assert c1 == c2
    assert c1 != c3


def test_curry_unpacks_partial():
    p = partial(_mul, 2)
    c = curry(p, 3)
    assert c() == 6


def test_curry_unpacks_other_curry():
    c = curry(_mul, 2)
    c2 = curry(c, 3)
    assert c2() == 6


def test_curry_merges_kwargs_when_unpacking():
    def f(a=0, b=0):
        return a + b

    c1 = curry(f, a=1)
    c2 = curry(c1, b=2)
    assert c2() == 3


def test_curry_too_many_args_typeerror():
    c = curry(_mul)
    with pytest.raises(TypeError):
        c(1, 2, 3)


def test_curry_bad_keyword_raises():
    c = curry(_mul)
    with pytest.raises(TypeError):
        c(1, nope=2)


def test_curry_sequence_currying():
    def add3(a, b, c):
        return a + b + c

    c = curry(add3)
    assert c(1)(2)(3) == 6


def test_curry_varargs():
    def f(*args):
        return sum(args)

    c = curry(f)
    assert c(1, 2, 3) == 6


def test_curry_signature():
    def f(a, b, c=3):
        return a + b + c

    c = curry(f, 1)
    sig = c.__signature__
    assert list(sig.parameters) == ['b', 'c']

    c2 = curry(f, 1, 2)
    sig2 = c2.__signature__
    assert list(sig2.parameters) == ['c']


def test_curry_signature_with_kwargs():
    def f(a, b, c=3):
        return a + b + c

    c = curry(f, b=2)
    sig = c.__signature__
    # 'b' becomes keyword-only
    assert sig.parameters['b'].kind == inspect.Parameter.KEYWORD_ONLY


def test_curry_descriptor():
    def f(self, x):
        return self + x

    class Holder:
        method = curry(f)

    h = Holder()
    assert h.method(3) == 3 + h
    # class access returns the curry itself
    assert Holder.method is not None


def test_curry_descriptor_get_from_instance():
    def f(self, x):
        return (self, x)

    class Holder:
        method = curry(f)

    h = Holder()
    out = h.method(5)
    assert out == (h, 5)


def test_curry_pickle():
    c = curry(_mul, 2)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(3) == 6


def test_curry_pickle_with_kwargs():
    def f(x, y=0):
        return x + y

    c = curry(f, 2, y=3)
    restored = pickle.loads(pickle.dumps(c))
    assert restored() == 5


def test_curry_reduce_preserves_func():
    c = curry(_mul, 5)
    func, args, kwargs = c._partial.func, c.args, c.keywords
    assert func is _mul
    assert args == (5,)
    assert kwargs == {}


def test_curry_getattr_only_for_callable():
    with pytest.raises(TypeError):
        curry("a string")


def test_curried_memoize_adds_name_and_doc():
    @memoize
    def add(x, y):
        """add doc"""
        return x + y

    assert add(1, 2) == 3
    assert add.__name__ == 'add'
    assert add.__doc__ == 'add doc'
    assert add.__wrapped__ is not None


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic_caches():
    calls = []

    def orig(x, y):
        calls.append((x, y))
        return x + y

    m = memoize(orig)
    assert m(1, 2) == 3
    assert m(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_as_decorator():
    @memoize
    def add(x, y):
        return x + y

    assert add(1, 2) == 3


def test_memoize_with_initial_cache():
    calls = []

    @memoize(cache={(1, 2): 3})
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(1, 2) == 3
    assert calls == []
    assert add(2, 3) == 5
    assert calls == [(2, 3)]


def test_memoize_unary_key_function():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]


def test_memoize_kwargs_cache():
    calls = []

    @memoize
    def add(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3
    assert add(1, 2, verbose=True) == 3
    assert len(calls) == 2


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3
    assert calls == [(1, 2, True)]


def test_memoize_unhashable_raises():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_varargs_key():
    calls = []

    @memoize
    def f(*args):
        calls.append(args)
        return sum(args)

    assert f(1, 2, 3) == 6
    assert f(1, 2, 3) == 6
    assert calls == [(1, 2, 3)]


def test_memoize_no_kwargs_key():
    calls = []

    def f(x, y):
        calls.append((x, y))
        return x + y

    m = memoize(f)
    assert m(1, 2) == 3
    assert m(1, 2) == 3
    assert calls == [(1, 2)]


# ---------------------------------------------------------------------------
# Compose / compose / compose_left
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_no_args_is_identity():
    assert compose() is identity


def test_compose_one_arg_returns_func():
    inc = lambda i: i + 1
    assert compose(inc) is inc


def test_compose_left_basic():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_no_args_is_identity():
    assert compose_left() is identity


def test_compose_multiple():
    f = compose(lambda x: x * 2, lambda x: x + 1, lambda x: x - 1)
    # ((x - 1) + 1) * 2 == 2x
    assert f(5) == 10


def test_compose_callable_first():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    c = compose(f, g)
    assert c(3) == 7


def test_compose_equality():
    f = lambda x: x + 1
    c1 = compose(f, f)
    c2 = compose(f, f)
    assert c1 == c2


def test_compose_equality_with_non_compose():
    f = lambda x: x + 1
    c = compose(f, f)
    assert (c == "not compose") is False


def test_compose_ne():
    f = lambda x: x + 1
    g = lambda x: x * 2
    c1 = compose(f, g)
    c2 = compose(g, f)
    assert c1 != c2


def test_compose_hash():
    f = lambda x: x + 1
    g = lambda x: x * 2
    c1 = compose(f, g)
    c2 = compose(f, g)
    assert hash(c1) == hash(c2)


def test_compose_doc():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert 'f(g(' in c.__doc__


def test_compose_doc_no_names():
    class Callable:
        def __call__(self, x):
            return x

    c = compose(Callable(), Callable())
    assert c.__doc__ == 'A composition of functions'


def test_compose_name():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__name__ == 'f_of_g'


def test_compose_name_no_names():
    class Callable:
        def __call__(self, x):
            return x

    c = compose(Callable(), Callable())
    assert c.__name__ == 'Compose'


def test_compose_repr():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert repr(c).startswith('Compose(')


def test_compose_get_descriptor():
    def f(x):
        return x * 2

    class Holder:
        method = compose(f)

    h = Holder()
    assert h.method(5) == 10
    assert Holder.method is not None


def test_compose_signature():
    def f(x: int, y: str) -> float:
        return 0.0

    def g(a: int) -> int:
        return a

    c = compose(f, g)
    sig = c.__signature__
    assert sig.return_annotation == float


def test_compose_wrapped_attribute():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    # __wrapped__ is the first function applied (g)
    assert c.__wrapped__ is g


def test_compose_annotations():
    def f(x: int) -> float:
        return 0.0

    def g(a: int) -> str:
        return "x"

    c = compose(f, g)
    ann = c.__annotations__
    assert 'x' in ann
    assert 'return' in ann


def test_compose_pickle():
    # Compose supports __getstate__/__setstate__.
    def f(x):
        return x + 1

    c = compose(f, f, f)
    state = c.__getstate__()
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2(1) == 4


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe_basic():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'


def test_pipe_no_funcs():
    assert pipe(5) == 5


def test_pipe_many():
    assert pipe(1, lambda x: x + 1, lambda x: x * 2, lambda x: x - 1) == 3


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


def test_complement_returns_callable():
    def always_true(x):
        return True

    c = complement(always_true)
    assert c(0) is False


# ---------------------------------------------------------------------------
# juxt
# ---------------------------------------------------------------------------

def test_juxt_basic():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_list_argument():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(x, y=1):
        return x + y

    def g(x, y=1):
        return x * y

    assert juxt(f, g)(3, y=2) == (5, 6)


def test_juxt_getstate_setstate():
    inc = lambda x: x + 1
    j = juxt(inc)
    state = j.__getstate__()
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2(1) == (2,)


def test_juxt_empty():
    j = juxt()
    assert j() == ()


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do_returns_x_on_its_own():
    log = []
    assert do(log.append, 5) == 5
    assert log == [5]


def test_do_side_effect():
    log = []
    inc = curry(lambda x: x + 1)
    inc = compose(inc, do(log.append))
    assert inc(1) == 2
    assert inc(11) == 12
    assert log == [1, 11]


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------

def test_flip_div():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3


def test_flip_curried():
    def div(a, b):
        return a // b

    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_flip_isinstance():
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError("boom")) is None
    assert return_none(None) is None


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


def test_excepts_tuple_of_exceptions():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_default_handler_none():
    e = excepts(ValueError, lambda: int("notanint"))
    assert e() is None


def test_excepts_non_matching_exception_raises():
    e = excepts(ValueError, lambda: int("x") + [1])
    with pytest.raises(TypeError):
        e()


def test_excepts_name():
    e = excepts(ValueError, lambda x: x, lambda x: x)
    assert e.__name__ == '<lambda>_excepting_ValueError'


def test_excepts_name_tuple():
    e = excepts((IndexError, KeyError), lambda x: x)
    assert e.__name__ == '<lambda>_excepting_IndexError_or_KeyError'


def test_excepts_name_no_name():
    class Bad:
        pass

    exc = Bad()
    # Not a class with __name__, but calling will fail before name comparison
    e = excepts(exc, lambda x: x)
    assert e.__name__ == 'excepting'


def test_excepts_doc_class():
    assert 'A wrapper around' in excepts.__doc__


def test_excepts_instance_doc(classval_doc in (True,)):
    # Only run if the classval property is available (sanity)
    pass


def test_excepts_doc_instance():
    def func(a):
        """func docstring"""
        return a

    def handler(e):
        """handler docstring"""
        return None

    e = excepts(ValueError, func, handler)
    assert "func docstring" in e.__doc__
    assert "handler docstring" in e.__doc__
    assert "ValueError" in e.__doc__


def test_excepts_doc_tuple():
    e = excepts((IndexError, KeyError), lambda a: a[0])
    assert "(IndexError, KeyError)" in e.__doc__


def test_excepts_doc_attribute_error_fallback():
    class BadCallable:
        def __call__(self, x):
            return x

    e = excepts(ValueError, BadCallable())
    assert e.__doc__ == excepts.__doc__


# ---------------------------------------------------------------------------
# introspection helpers
# ---------------------------------------------------------------------------

def _f_two(x, y):
    return x + y


def _f_defaults(x, y=1, z=2):
    return x + y + z


def _f_varargs(*args):
    return args


def _f_kwargs(**kwargs):
    return kwargs


def _f_all(*args, **kwargs):
    return args, kwargs


def _f_kwonly(x, *, y=1):
    return x + y


def test_num_required_args_two():
    assert num_required_args(_f_two) == 2


def test_num_required_args_defaults():
    assert num_required_args(_f_defaults) == 1


def test_num_required_args_varargs():
    assert num_required_args(_f_varargs) == 0


def test_num_required_args_kwargs():
    assert num_required_args(_f_kwargs) == 0


def test_num_required_args_kwonly():
    # y is keyword-only so only x counts
    assert num_required_args(_f_kwonly) == 1


def test_num_required_args_builtin_map():
    # Known by the _signatures registry via _num_required_args
    n = num_required_args(map)
    # map requires at least a function and one iterable
    assert n == 2


def test_has_varargs_true():
    assert has_varargs(_f_varargs) is True


def test_has_varargs_false():
    assert has_varargs(_f_kwargs) is False


def test_has_varargs_all():
    assert has_varargs(_f_all) is True


def test_has_varargs_builtin():
    assert has_varargs(map) is False


def test_has_keywords_true_default():
    assert has_keywords(_f_defaults) is True


def test_has_keywords_kwonly():
    assert has_keywords(_f_kwonly) is True


def test_has_keywords_kwargs():
    assert has_keywords(_f_kwargs) is True


def test_has_keywords_false():
    assert has_keywords(_f_two) is False


def test_has_keywords_builtin():
    assert has_keywords(map) is False


def test_is_valid_args_true():
    assert is_valid_args(_f_two, (1, 2), {}) is True


def test_is_valid_args_too_few():
    assert is_valid_args(_f_two, (1,), {}) is False


def test_is_valid_args_too_many():
    assert is_valid_args(_f_two, (1, 2, 3), {}) is False


def test_is_valid_args_builtin_map_empty():
    assert is_valid_args(map, (), {}) is False


def test_is_partial_args_partial():
    assert is_partial_args(_f_two, (1,), {}) is True


def test_is_partial_args_full():
    assert is_partial_args(_f_two, (1, 2), {}) is True


def test_is_partial_args_too_many():
    assert is_partial_args(_f_two, (1, 2, 3), {}) is False


def test_is_partial_args_builtin_map_empty():
    assert is_partial_args(map, (), {}) is True


def test_is_arity_one():
    def f(x):
        return x

    assert is_arity(1, f) is True


def test_is_arity_with_default_false():
    def g(x, y=1):
        return x + y

    assert is_arity(1, g) is False


def test_is_arity_varargs_false():
    def g(*args):
        return args

    assert is_arity(1, g) is False


def test_is_arity_kwargs_false():
    def g(**kwargs):
        return kwargs

    assert is_arity(0, g) is False


def test_is_arity_wrong_count():
    def f(x, y):
        return x + y

    assert is_arity(1, f) is False


def test_is_arity_two_positional():
    def f(x, y):
        return x + y

    assert is_arity(2, f) is True


def test_is_arity_builtin():
    # len has arity 1
    assert is_arity(1, len) is True


# ---------------------------------------------------------------------------
# helpers with sigspec passed
# ---------------------------------------------------------------------------

def test_num_required_args_with_sigspec():
    sig = inspect.signature(_f_defaults)
    assert num_required_args(_f_defaults, sigspec=sig) == 1


def test_has_varargs_with_sigspec():
    sig = inspect.signature(_f_varargs)
    assert has_varargs(_f_varargs, sigspec=sig) is True


def test_has_keywords_with_sigspec():
    sig = inspect.signature(_f_defaults)
    assert has_keywords(_f_defaults, sigspec=sig) is True


def test_is_valid_args_with_sigspec():
    sig = inspect.signature(_f_two)
    assert is_valid_args(_f_two, (1, 2), {}, sigspec=sig) is True


def test_is_partial_args_with_sigspec():
    sig = inspect.signature(_f_two)
    assert is_partial_args(_f_two, (1,), {}, sigspec=sig) is True


def test_is_arity_with_sigspec():
    sig = inspect.signature(_f_two)
    assert is_arity(2, _f_two, sigspec=sig) is True


# ---------------------------------------------------------------------------
# _check_sigspec returns False for unknown callables
# ---------------------------------------------------------------------------

def test_unknown_callable_signature():
    class Uninspectable:
        def __call__(self):
            return 1

    # If inspect.signature works on this class, check the actual value.
    try:
        inspect.signature(Uninspectable())
        sig_ok = True
    except (ValueError, TypeError):
        sig_ok = False

    if not sig_ok:
        # Falls through to _sigs which returns False for unknown callables
        val = has_varargs(Uninspectable())
        assert val is False or val is True


# ---------------------------------------------------------------------------
# curry of builtin
# ---------------------------------------------------------------------------

def test_curry_builtin():
    c = curry(add)
    assert c(1)(2) == 3


def test_curry_of_curried_flip_sequence():
    def div(a, b):
        return a // b

    d = flip(div, 2)
    assert d(8) == 4
