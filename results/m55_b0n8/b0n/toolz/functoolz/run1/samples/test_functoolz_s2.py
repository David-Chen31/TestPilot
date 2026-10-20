import pickle
import sys
import inspect

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


def test_apply():
    assert apply(lambda x: 2 * x, 5) == 10
    assert apply(lambda x, y: x + y, 3, 4) == 7
    assert apply(lambda x, y=10: x + y, 3, y=20) == 23


def test_apply_requires_func():
    with pytest.raises(TypeError, match='func argument is required'):
        apply()


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def _inc(x):
    return x + 1


def _double(x):
    return 2 * x


def _add(x, y):
    return x + y


def _pow(x, y):
    return x ** y


def test_thread_first_basic():
    assert thread_first(1, _inc, _double) == 4
    assert thread_first(1) == 1


def test_thread_first_tuple_forms():
    assert thread_first(1, (_add, 4), (_pow, 2)) == 25


def test_thread_last_basic():
    assert thread_last(1, _inc, _double) == 4
    assert thread_last(1) == 1


def test_thread_last_tuple_forms():
    assert thread_last(1, (_add, 4), (_pow, 2)) == 32


def test_thread_last_map_filter():
    def iseven(x):
        return x % 2 == 0
    result = list(thread_last([1, 2, 3], (map, _inc), (filter, iseven)))
    assert result == [2, 4]


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_classval():
    class C:
        """class docstring"""

        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return 'obj doc'

        @instanceproperty
        def val(self):
            return 42

    # On the class, the descriptor returns classval
    assert C.__doc__ == 'class docstring'
    assert C.val is None

    c = C()
    assert c.__doc__ == 'obj doc'
    assert c.val == 42


def test_instanceproperty_is_instance_property():
    @instanceproperty
    def f(self):
        return 1

    assert isinstance(f, InstanceProperty)
    assert isinstance(f, property)


def test_instanceproperty_partial_construction():
    # Calling without fget returns a partial that can be given a function
    make = instanceproperty(classval='X')
    prop = make(lambda self: 'instance')
    assert isinstance(prop, InstanceProperty)
    assert prop.classval == 'X'


def test_instanceproperty_setter_getter():
    class C:
        def __init__(self):
            self._x = 0

        @instanceproperty
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

    c = C()
    assert c.x == 0
    c.x = 7
    assert c.x == 7


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def _mul(x, y):
    return x * y


def test_curry_basic():
    mul = curry(_mul)
    double = mul(2)
    assert isinstance(double, curry)
    assert double(10) == 20


def test_curry_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_init_errors():
    with pytest.raises(TypeError):
        curry()
    with pytest.raises(TypeError, match='callable'):
        curry(42)


def test_curry_function_attributes():
    def g(x):
        """doc for g"""
        return x

    c = curry(g)
    assert c.__doc__ == 'doc for g'
    assert c.__name__ == 'g'
    assert c.__module__ == g.__module__
    assert c.__qualname__ == g.__qualname__


def test_curry_str_repr_hash_eq():
    m1 = curry(_mul)
    m2 = curry(_mul)
    assert str(m1) == str(_mul)
    assert repr(m1) == repr(_mul)
    assert m1 == m2
    assert not (m1 != m2)
    assert hash(m1) == hash(m2)

    m3 = m1(2)
    m4 = curry(_mul, 2)
    assert m3 == m4

    m5 = m1(3)
    assert m3 != m5


def test_curry_not_equal_other():
    m = curry(_mul)
    assert m != 5
    assert not (m == 5)


def test_curry_bind_call():
    m = curry(_mul)
    bound = m.bind(2)
    assert bound(5) == 10
    with pytest.raises(TypeError):
        m.call(2)  # missing argument


def test_curry_call_raises_genuine_typeerror():
    def f(x):
        return x + 1

    c = curry(f)
    with pytest.raises(TypeError):
        c(1, 2)  # too many args; cannot curry anymore


def test_curry_varargs_function():
    def f(*args):
        return args

    c = curry(f)
    # Calling with args should work
    assert c(1, 2) == (1, 2)


def test_curry_function_args_keywords_props():
    m = curry(_mul, 2, y=3)
    assert m.func is _mul
    assert m.args == (2,)
    assert m.keywords == {'y': 3}
    assert m.func_name == m.__name__


def test_curry_signature():
    def f(a, b, c=3):
        return a + b + c

    c = curry(f)
    sig = c.__signature__
    assert list(sig.parameters) == ['a', 'b', 'c']

    c2 = curry(f, 1)
    sig2 = c2.__signature__
    assert list(sig2.parameters) == ['b', 'c']


def test_curry_get_descriptor():
    class C:
        @curry
        def meth(self, x, y):
            return x + y

    c = C()
    assert c.meth(1, 2) == 3
    # curried on instance can still partial-apply
    assert c.meth(1)(2) == 3


def test_curry_pickle_roundtrip():
    c = curry(_mul, 2)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(5) == 10
    assert restored.args == (2,)

    restored2 = pickle.loads(pickle.dumps(curry(_mul)))
    assert restored2(3, 4) == 12


def test_curry_pickle_builtin_partial():
    # builtins may fail; use a module-level callable defined here
    from toolz.functoolz import num_required_args as nra
    c = curry(nra)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(_mul) == 2


def test_curry_accepts_partial_like():
    from functools import partial
    p = partial(_mul, 2)
    c = curry(p)
    assert c.func is _mul
    assert c.args == (2,)
    assert c(5) == 10


def test_curry_merge_kwargs_from_partial_like():
    from functools import partial

    def f(a, b, c=0):
        return a + b + c

    p = partial(f, 1, c=10)
    c = curry(p, b=2)
    assert c() == 13


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic():
    calls = []

    @memoize
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_metadata():
    @memoize
    def f(x):
        """mydoc"""
        return x

    assert f.__name__ == 'f'
    assert f.__doc__ == 'mydoc'
    assert f.__wrapped__ is not None


def test_memoize_initial_cache():
    @memoize(cache={(1, 2): 100})
    def add(x, y):
        return x + y

    assert add(1, 2) == 100


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    assert add(1, 2, verbose=False) == 3
    assert add(1, 2, verbose=True) == 3  # ignored by key
    assert len(calls) == 1


def test_memoize_unhashable_raises():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError, match='hashable'):
        f([1, 2, 3])


def test_memoize_unary_key_used():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]


# ---------------------------------------------------------------------------
# compose / compose_left / Compose / pipe
# ---------------------------------------------------------------------------

def test_compose_empty_returns_identity():
    assert compose() is identity
    assert compose_left() is identity


def test_compose_single_returns_func():
    assert compose(_inc) is _inc
    assert compose_left(_inc) is _inc


def test_compose_basic():
    c = compose(str, _inc)
    assert isinstance(c, Compose)
    assert c(3) == '4'


def test_compose_left_basic():
    assert compose_left(_inc, str)(3) == '4'


def test_compose_multiple_args():
    c = compose(_double, _add)
    assert c(1, 2) == 6


def test_compose_doc():
    c = compose(_inc, _double)
    assert 'lambda *args, **kwargs:' in c.__doc__
    assert '_double' in c.__doc__
    assert Compose.__doc__ is not None


def test_compose_name():
    c = compose(_inc, _double)
    assert c.__name__ == '_inc_of__double'


def test_compose_name_no_names():
    class C:
        def __call__(self, x):
            return x
    assert compose(C())(5) == 5


def test_compose_repr():
    c = compose(_inc, _double)
    assert repr(c) == 'Compose(%r, %r)' % (_inc, _double)


def test_compose_eq_hash():
    c1 = compose(_inc, _double)
    c2 = compose(_inc, _double)
    assert c1 == c2
    assert not (c1 != c2)
    assert hash(c1) == hash(c2)
    assert c1 != compose(_double, _inc)


def test_compose_eq_other_type():
    c = compose(_inc, _double)
    assert c != 5
    assert (c == 5) is False


def test_compose_pickle():
    c = compose(_inc, _double)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(2) == 5


def test_compose_signature():
    def f(a, b):
        return a + b

    def g(x) -> int:
        return x

    c = compose(g, f)
    sig = c.__signature__
    assert list(sig.parameters) == ['a', 'b']
    assert sig.return_annotation is int
    assert c.__wrapped__ is f


def test_compose_get_descriptor():
    class C:
        def __init__(self, n):
            self.n = n

        def add(self, x):
            return self.n + x

    inc_by_10 = compose(_double)

    class D:
        f = compose(_inc, _double)

    assert D().f(1) == 3


def test_compose_annotations():
    def f(a: int, b: str) -> float:
        return 1.0

    def g(x: float) -> int:
        return 1

    c = compose(g, f)
    ann = c._combined_annotations()
    assert ann.get('a') is int
    assert ann.get('b') is str
    assert ann.get('return') is int


def test_compose_nested_annotations():
    def f(a: int):
        return a

    def g(x):
        return x

    inner = compose(g, f)
    outer = compose(g, inner)
    ann = outer._combined_annotations()
    assert ann.get('a') is int


def test_compose_annotations_no_signature():
    def no_sig_fn(*args):
        return args

    # A callable with uninspectable signature
    class Bad:
        def __call__(self, x):
            return x

    c = compose(no_sig_fn)
    ann = c._combined_annotations()
    assert isinstance(ann, dict)


def test_pipe():
    assert pipe(3, _double, str) == '6'
    assert pipe(5) == 5


def test_pipe_order():
    calls = []
    pipe(0, calls.append)
    assert calls == [0]


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------

def test_complement():
    isodd = complement(lambda n: n % 2 == 0)
    assert isodd(2) is False
    assert isodd(3) is True


# ---------------------------------------------------------------------------
# juxt
# ---------------------------------------------------------------------------

def test_juxt_multiple():
    j = juxt(_inc, _double)
    assert j(10) == (11, 20)


def test_juxt_list_argument():
    j = juxt([_inc, _double])
    assert j(10) == (11, 20)


def test_juxt_kwargs():
    j = juxt(lambda x, y: x + y, lambda x, y: x * y)
    assert j(2, 3) == (5, 6)


def test_juxt_pickle():
    j = juxt(_inc, _double)
    restored = pickle.loads(pickle.dumps(j))
    assert restored(5) == (6, 10)


def test_juxt_getstate_setstate():
    j = juxt(_inc, _double)
    state = j.__getstate__()
    assert state == (_inc, _double)
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2(5) == (6, 10)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    assert do(log.append, 5) == 5
    assert log == [5]


def test_do_side_effects():
    seen = []

    def record(x):
        seen.append(x)
        return 'ignored'

    result = do(record, 'value')
    assert result == 'value'
    assert seen == ['value']


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------

def test_flip():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3


def test_flip_curried():
    def div(a, b):
        return a // b

    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_flip_isinstance_usage():
    data = [1, 'a', 'b', 2, 1.5, 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none / excepts
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError('x')) is None


def test_excepts_basic():
    excepting = excepts(
        ValueError,
        lambda a: [1, 2].index(a),
        lambda _: -1,
    )
    assert excepting(1) == 0
    assert excepting(3) == -1


def test_excepts_default_handler():
    e = excepts(ValueError, lambda: (_ for _ in ()).throw(ValueError('x')))
    assert e() is None


def test_excepts_tuple_of_exceptions():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_doc():
    e = excepts(
        ValueError,
        lambda a: int(a),
        lambda _: -1,
    )
    doc = e.__doc__
    assert 'ValueError' in doc
    assert 'lambda' in doc or 'Docs for' in doc


def test_excepts_name():
    e = excepts(ValueError, int)
    assert e.__name__ == 'int_excepting_ValueError'


def test_excepts_name_tuple():
    e = excepts((IndexError, KeyError), int)
    assert 'IndexError' in e.__name__
    assert 'KeyError' in e.__name__


def test_excepts_name_without_name():
    class C:
        def __call__(self, x):
            return x

    # __name__ doesn't exist on instance
    e = excepts(C(), C())
    assert e.__name__ == 'excepting'


def test_excepts_doc_without_names():
    class C:
        def __call__(self, x):
            return x

    e = excepts(C(), C())
    assert isinstance(e.__doc__, str)


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def _f_req(x, y, z=3):
    return x + y + z


def _f_varargs(*args):
    return args


def _f_keywords(x, **kwargs):
    return x


def _f_kwonly(x, *, y):
    return x + y


def test_num_required_args():
    assert num_required_args(_f_req) == 2
    assert num_required_args(_f_varargs) == 0
    assert num_required_args(_inc) == 1


def test_num_required_args_sigspec():
    sig = inspect.signature(_f_req)
    assert num_required_args(_f_req, sig) == 2


def test_has_varargs():
    assert has_varargs(_f_varargs) is True
    assert has_varargs(_f_keywords) is False


def test_has_keywords():
    assert has_keywords(_f_keywords) is True
    assert has_keywords(_f_kwonly) is True
    assert has_keywords(_inc) is False


def test_is_valid_args():
    assert is_valid_args(_inc, (1,), {}) is True
    assert is_valid_args(_inc, (), {}) is False
    assert is_valid_args(_add, (1, 2), {}) is True
    assert is_valid_args(_add, (1,), {}) is False
    assert is_valid_args(_f_req, (1, 2), {}) is True


def test_is_valid_args_builtin():
    # map requires at least one argument
    assert is_valid_args(map, (), {}) is False
    assert is_valid_args(map, (lambda x: x, [1]), {}) is True


def test_is_partial_args():
    assert is_partial_args(_add, (1,), {}) is True
    assert is_partial_args(_add, (1, 2), {}) is True
    assert is_partial_args(_add, (1, 2, 3), {}) is False
    assert is_partial_args(map, (), {}) is True


def test_is_arity():
    assert is_arity(1, _inc) is True
    assert is_arity(2, _inc) is False

    def g(x, y=1):
        return x + y

    assert is_arity(1, g) is False
    assert is_arity(2, g) is False  # has keyword arg


def test_is_arity_varargs():
    assert is_arity(1, _f_varargs) is False


# ---------------------------------------------------------------------------
# curry interaction with introspection
# ---------------------------------------------------------------------------

def test_curry_signature_with_partial_args():
    def f(a, b, c):
        return a + b + c

    c = curry(f, 1)
    sig = c.__signature__
    names = list(sig.parameters)
    assert names == ['b', 'c']
    # Defaults should be no_default after partial application
    from toolz.utils import no_default
    assert sig.parameters['b'].default is no_default


def test_curry_bad_partial_args_signature():
    def f(a, b):
        return a + b

    c = curry(f, 1, 2, 3)  # too many args
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_call_method_direct():
    c = curry(_mul)
    assert c.call(3, 4) == 12


def test_curry_should_curry_with_varargs():
    seen = []

    def deco(*args, **kwargs):
        seen.append((args, kwargs))
        return (args, kwargs)

    c = curry(deco)
    # Calling with wrong args may trigger TypeError; should curry
    result = c(1)
    assert result == ((1,), {})


def test_memoize_is_curry_instance():
    assert isinstance(memoize, curry)
    assert isinstance(flip, curry)
