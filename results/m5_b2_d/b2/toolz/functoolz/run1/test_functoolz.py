import builtins
import pickle
import sys
import inspect
from functools import partial

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
# identity
# ---------------------------------------------------------------------------

def test_identity():
    assert identity(3) == 3
    obj = object()
    assert identity(obj) is obj


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def test_apply_basic():
    def double(x):
        return 2 * x

    assert apply(double, 5) == 10


def test_apply_kwargs():
    def f(a, b, c=0):
        return a + b + c

    assert apply(f, 1, 2, c=3) == 6


def test_apply_no_func():
    with pytest.raises(TypeError, match='func argument is required'):
        apply()


def test_apply_map_usage():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_callables():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4


def test_thread_first_tuples():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(7) == 7


def test_thread_last_callables():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4


def test_thread_last_tuples():
    def add(x, y):
        return x + y

    assert thread_last(1, (add, 4)) == 5


def test_thread_last_map_filter():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    assert list(thread_last([1, 2, 3], (map, inc), (filter, iseven))) == [2, 4]


def test_thread_last_no_forms():
    assert thread_last(7) == 7


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_classval():
    class MyClass:
        """The class docstring"""

        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return 'An object docstring'

        @instanceproperty
        def val(self):
            return 42

    assert MyClass.__doc__ == 'The class docstring'
    assert MyClass.val is None
    obj = MyClass()
    assert obj.__doc__ == 'An object docstring'
    assert obj.val == 42


def test_instanceproperty_partial_form():
    class C:
        def _fget(self):
            return 'value'

        p = instanceproperty(fget=_fget, classval='CV')

    assert C.p == 'CV'
    assert C().p == 'value'


def test_instanceproperty_partial_kwargs_forwarded():
    # The `fget is None` branch must create a partial that forwards
    # fset, fdel, doc, and classval. Applying it with fget yields a property.
    store = {}

    def fget(self):
        return store.get('v', 'default')

    def fset(self, val):
        store['v'] = val

    def fdel(self):
        del store['v']

    p = instanceproperty(fset=fset, fdel=fdel, doc='DOCVAL', classval='CV')
    assert isinstance(p, partial)

    prop = p(fget)
    assert isinstance(prop, InstanceProperty)
    assert prop.fget is fget
    assert prop.fset is fset
    assert prop.fdel is fdel
    assert prop.classval == 'CV'

    class C:
        x = prop

    c = C()
    assert c.x == 'default'
    c.x = 5
    assert c.x == 5
    del c.x
    assert 'v' not in store
    assert C.x == 'CV'


def test_instanceproperty_partial_doc_forwarded():
    def fget(self):
        return 1

    prop = instanceproperty(doc='MYDOC', classval='CV')
    assert prop.keywords['doc'] == 'MYDOC'
    assert prop.keywords['classval'] == 'CV'

    class C:
        x = prop(fget)

    assert C.x == 'CV'
    assert C().x == 1


def test_instanceproperty_direct_kwargs():
    # The non-None fget branch must build InstanceProperty with all args.
    def fget(self):
        return 1

    def fset(self, val):
        pass

    def fdel(self):
        pass

    p = instanceproperty(fget=fget, fset=fset, fdel=fdel, doc='D',
                         classval='CV')
    assert isinstance(p, InstanceProperty)
    assert p.fget is fget
    assert p.fset is fset
    assert p.fdel is fdel
    assert isinstance(p.__doc__, str)
    assert p.classval == 'CV'


def test_instanceproperty_is_property_subclass():
    class C:
        @instanceproperty
        def x(self):
            return 1

    assert isinstance(C.__dict__['x'], InstanceProperty)


def test_instanceproperty_setter_deleter():
    store = {}

    class C:
        @instanceproperty
        def x(self):
            return store.get('x')

        @x.setter
        def x(self, value):
            store['x'] = value

    c = C()
    assert c.x is None
    c.x = 99
    assert c.x == 99


def test_instanceproperty_reduce():
    def fget(self):
        return 1

    prop = InstanceProperty(fget=fget, doc='d', classval='cv')
    cls, state = prop.__reduce__()
    assert cls is InstanceProperty
    assert state[0] is fget
    assert state[4] == 'cv'


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert isinstance(double, curry)
    assert double(10) == 20


def test_curry_decorator():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_init_no_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_init_non_callable():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_attributes():
    def f(x):
        "docstring"
        return x

    c = curry(f)
    assert c.func is f
    assert c.args == ()
    assert c.keywords == {}
    assert c.__doc__ == 'docstring'
    assert c.__name__ == 'f'
    assert c.func_name == 'f'
    assert str(c) == str(f)
    assert repr(c) == repr(f)


def test_curry_hash_eq():
    def f(x, y):
        return x + y

    a = curry(f, 1)
    b = curry(f, 1)
    c = curry(f, 2)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert a != 'not curry'


def test_curry_call_genuine_typeerror():
    @curry
    def f(x):
        raise TypeError('real error')

    with pytest.raises(TypeError):
        f(1)


def test_curry_bind_and_call():
    def f(x, y):
        return x + y

    c = curry(f)
    bound = c.bind(1)
    assert bound.call(2) == 3


def test_curry_get_descriptor():
    def f(self, x):
        return x + 1

    class C:
        method = curry(f)

    assert isinstance(C.method, curry)
    c = C()
    bound = c.method
    assert isinstance(bound, curry)
    assert bound(1) == 2


def test_curry_signature():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ['y', 'z']


def test_curry_signature_invalid_args():
    def f(x):
        return x

    c = curry(f, 1, 2)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_merges_partial_like():
    def f(x, y):
        return x + y

    p = partial(f, 1)
    c = curry(p, 2)
    assert c() == 3
    assert c.func is f


def test_curry_pickle():
    c = curry(pow, 2)
    assert c(3) == 8
    restored = pickle.loads(pickle.dumps(c))
    assert isinstance(restored, curry)
    assert restored.args == (2,)
    assert restored(3) == 8


def test_curry_keywords_branch():
    def f(x, y=0):
        return x + y

    c = curry(f, 1, y=2)
    assert c() == 3
    assert c.keywords == {'y': 2}


def test_curry_should_curry_no_sig():
    # A function decorated as functools.wraps with *args; unknown args
    @curry
    def f(*args, **kwargs):
        if len(args) < 2:
            raise TypeError('need 2')
        return sum(args[:2])

    g = f(1)
    assert isinstance(g, curry)
    assert g(2) == 3


def test_curry_should_curry_partial_invalid():
    @curry
    def f(x, y):
        return x + y

    g = f(1)
    assert isinstance(g, curry)
    assert g(2) == 3


def test_curry_should_curry_too_many_args():
    @curry
    def f(x, y):
        return x + y

    with pytest.raises(TypeError):
        f(1, 2, 3)


def test_curry_restore_decorated():
    # memoize is a module-level curry; round-trip preserves identity.
    import toolz.functoolz as fmod
    restored = pickle.loads(pickle.dumps(fmod.memoize))
    assert restored is fmod.memoize


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    madd = memoize(add)
    assert madd(1, 2) == 3
    assert madd(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_decorator_and_cache():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        raise AssertionError('should not be called')

    assert add(1, 2) == 3


def test_memoize_unary():
    calls = []

    def square(x):
        calls.append(x)
        return x * x

    msq = memoize(square)
    assert msq(3) == 9
    assert msq(3) == 9
    assert calls == [3]


def test_memoize_kwargs():
    calls = []

    def f(x, y=1):
        calls.append((x, y))
        return x + y

    mf = memoize(f)
    assert mf(1, y=2) == 3
    assert mf(1, y=2) == 3
    assert mf(1) == 2
    assert len(calls) == 2


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y))
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError, match='Arguments to memoized function must '
                                        'be hashable'):
        f([1, 2, 3])


def test_memoize_wrapping_attrs():
    def f(x):
        "doc"
        return x

    m = memoize(f)
    assert m.__name__ == 'f'
    assert m.__doc__ == 'doc'
    assert m.__wrapped__ is f


def test_memoize_is_curry():
    assert isinstance(memoize, curry)


def test_memoize_varargs_no_kwargs():
    # Function with *args but no keyword args: uses args key.
    calls = []

    @memoize
    def f(*args):
        calls.append(args)
        return sum(args)

    assert f(1, 2, 3) == 6
    assert f(1, 2, 3) == 6
    assert calls == [(1, 2, 3)]


def test_memoize_unary_info():
    # is_unary f takes one positional only
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(4) == 8
    assert f(4) == 8
    assert calls == [4]


def test_memoize_cache_param():
    cache = {}
    calls = []

    def f(x):
        calls.append(x)
        return x

    m = memoize(f, cache=cache)
    m(1)
    m(1)
    assert calls == [1]
    assert cache == {1: 1}


# ---------------------------------------------------------------------------
# compose / compose_left / pipe
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'
    assert compose(str, inc, inc)(1) == '3'


def test_compose_no_funcs():
    assert compose() is identity


def test_compose_one_func():
    def f(x):
        return x

    assert compose(f) is f


def test_compose_left():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'
    assert compose_left() is identity


def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'
    assert pipe(5) == 5


def test_compensate_complement():
    def iseven(n):
        return n % 2 == 0

    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


# ---------------------------------------------------------------------------
# Compose class
# ---------------------------------------------------------------------------

def test_compose_class():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    c = Compose((double, inc))
    assert c(3) == 8
    assert c.first is inc
    assert c.funcs == (double,)


def test_compose_repr():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    c = Compose((double, inc))
    assert repr(c) == 'Compose(%r, %r)' % (double, inc)


def test_compose_name_doc():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    c = Compose((double, inc))
    assert c.__name__ == 'double_of_inc'
    assert 'inc' in c.__doc__
    assert 'double' in c.__doc__


def test_compose_name_missing():
    class NoName:
        def __call__(self, x):
            return x

    c = Compose((NoName(),))
    assert c.__name__ == 'Compose'
    assert c.__doc__ == 'A composition of functions'


def test_compose_eq_hash():
    inc = lambda x: x + 1
    c1 = Compose((inc,))
    c2 = Compose((inc,))
    assert c1 == c2
    assert not (c1 != c2)
    assert hash(c1) == hash(c2)
    assert c1 != ('not', 'compose')
    assert (c1 == 5) is False
    assert (c1 != 5) is True


def test_compose_get_descriptor():
    def inc(self, x):
        return x + 1

    class C:
        method = Compose((inc,))

    assert C.method is C.__dict__['method']
    c = C()
    m = c.method
    assert m(1) == 2


def test_compose_signature():
    def inc(x: int) -> int:
        return x + 1

    def to_str(x: int) -> str:
        return str(x)

    c = Compose((to_str, inc))
    sig = c.__signature__
    assert sig.return_annotation is str
    params = list(sig.parameters.values())
    assert params[0].name == 'x'


def test_compose_wrapped():
    def inc(x):
        return x + 1

    c = Compose((inc,))
    assert c.__wrapped__ is inc


def test_compose_annotations():
    def inc(x: int) -> int:
        return x + 1

    def to_str(x: int) -> str:
        return str(x)

    c = Compose((to_str, inc))
    ann = c.__annotations__
    assert ann['x'] is int
    assert ann['return'] is str


def test_compose_annotations_on_class():
    def inc(x):
        return x + 1

    c = Compose((inc,))
    assert type(c).__annotations__ is not None


def test_compose_getstate_setstate():
    def inc(x):
        return x + 1

    c = Compose((inc,))
    first, funcs = c.__getstate__()
    c2 = Compose.__new__(Compose)
    c2.__setstate__((first, funcs))
    assert c2.first is c.first
    assert c2.funcs == c.funcs


def test_compose_eq_not_implemented():
    inc = lambda x: x + 1
    c = Compose((inc,))
    assert c.__eq__(5) is NotImplemented
    assert c.__ne__(5) is NotImplemented


def test_compose_single_func_call():
    def f(x, y):
        return x + y

    c = Compose((f,))
    assert c(2, 3) == 5


def test_compose_multiple_funcs_call():
    def add1(x):
        return x + 1

    def mul3(x):
        return x * 3

    c = Compose((add1, mul3))  # add1(mul3(x))
    assert c(2) == 7


# ---------------------------------------------------------------------------
# juxt
# ---------------------------------------------------------------------------

def test_juxt_basic():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_list_arg():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_single_callable():
    inc = lambda x: x + 1
    assert juxt(inc)(10) == (11,)


def test_juxt_kwargs():
    def f(x, y=0):
        return x + y

    assert juxt(f)(1, y=2) == (3,)


def test_juxt_state():
    inc = lambda x: x + 1
    j = juxt(inc)
    assert j.__getstate__() == (inc,)
    j2 = juxt.__new__(juxt)
    j2.__setstate__((inc,))
    assert j2.funcs == (inc,)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    assert do(log.append, 5) == 5
    assert log == [5]


def test_do_returns_x():
    def mut(d):
        d['seen'] = True

    data = {}
    result = do(mut, data)
    assert result is data
    assert data == {'seen': True}


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


def test_flip_isinstance():
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
    assert list(filter(flip(isinstance, int), data)) == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError('x')) is None


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


def test_excepts_multiple_default():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_attributes():
    def f(x):
        return x

    def handler(e):
        return 0

    e = excepts(ValueError, f, handler)
    assert e.exc is ValueError
    assert e.func is f
    assert e.handler is handler


def test_excepts_name_and_doc():
    def f(x):
        "func doc"
        return x

    def handler(e):
        "handler doc"
        return 0

    e = excepts(ValueError, f, handler)
    assert e.__name__ == 'f_excepting_ValueError'
    doc = e.__doc__
    assert "'f'" in doc
    assert 'ValueError' in doc
    assert "'handler'" in doc


def test_excepts_multiple_name():
    def f(x):
        return x

    e = excepts((IndexError, KeyError), f)
    assert e.__name__ == 'f_excepting_IndexError_or_KeyError'


def test_excepts_class_doc_attr():
    assert isinstance(excepts.__doc__, str)


def test_excepts_name_fallback():
    class NoName:
        def __call__(self, x):
            return x

    e = excepts((), NoName())
    assert e.__name__ == 'excepting'


# ---------------------------------------------------------------------------
# signature introspection helpers
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2

    def g(*args, **kwargs):
        pass

    assert num_required_args(g) == 0


def test_has_varargs():
    def f(*args):
        return args

    def g(**kwargs):
        return kwargs

    assert has_varargs(f) is True
    assert has_varargs(g) is False
    assert has_varargs(lambda x: x) is False


def test_has_keywords():
    def f(x, y=0):
        return x + y

    def g(x):
        return x

    def h(x, **kwargs):
        return x

    def k(x, *, y):
        return x

    assert has_keywords(f) is True
    assert has_keywords(g) is False
    assert has_keywords(h) is True
    assert has_keywords(k) is True


def test_is_valid_args():
    def add(x, y):
        return x + y

    assert is_valid_args(add, (1,), {}) is False
    assert is_valid_args(add, (1, 2), {}) is True


def test_is_partial_args():
    def add(x, y):
        return x + y

    assert is_partial_args(add, (1,), {}) is True
    assert is_partial_args(add, (1, 2), {}) is True
    assert is_partial_args(add, (1, 2, 3), {}) is False


def test_is_arity():
    def f(x):
        return x

    def g(x, y=1):
        return x + y

    def h(x, *args):
        return x

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False
    assert is_arity(1, h) is False


def test_sigspec_explicit():
    def f(x, y):
        return x + y

    sig = inspect.signature(f)
    assert num_required_args(f, sig) == 2
    assert has_varargs(f, sig) is False
    assert has_keywords(f, sig) is False
    assert is_valid_args(f, (1, 2), {}, sig) is True
    assert is_partial_args(f, (1,), {}, sig) is True
    assert is_arity(2, f, sig) is True
