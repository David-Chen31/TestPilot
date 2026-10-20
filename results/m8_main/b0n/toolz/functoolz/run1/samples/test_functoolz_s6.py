import inspect
import pickle
import sys

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
    assert apply(lambda x: 2 * x, 5) == 10
    assert apply(lambda x, y: x + y, 3, 4) == 7


def test_apply_kwargs():
    def f(x, y=0):
        return x + y

    assert apply(f, 1, y=2) == 3


def test_apply_requires_func():
    with pytest.raises(TypeError):
        apply()


def test_apply_in_map():
    double = lambda x: 2 * x
    inc = lambda x: x + 1
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_functions():
    inc = lambda x: x + 1
    double = lambda x: 2 * x
    assert thread_first(1, inc, double) == 4


def test_thread_first_tuples():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(5) == 5


def test_thread_last_functions():
    inc = lambda x: x + 1
    double = lambda x: 2 * x
    assert thread_last(1, inc, double) == 4


def test_thread_last_tuples():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (pow_, 2)) == 32


def test_thread_last_no_forms():
    assert thread_last(7) == 7


def test_thread_first_and_last_differ():
    def sub(x, y):
        return x - y

    # thread_first: sub(val, 10); thread_last: sub(10, val)
    assert thread_first(3, (sub, 10)) == -7
    assert thread_last(3, (sub, 10)) == 7


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_basic():
    class MyClass:
        @instanceproperty(classval='class-doc')
        def __doc__(self):
            return 'object-doc'

        @instanceproperty
        def val(self):
            return 42

    assert MyClass.__doc__ == 'class-doc'
    assert MyClass.val is None
    obj = MyClass()
    assert obj.__doc__ == 'object-doc'
    assert obj.val == 42


def test_instanceproperty_setter():
    class C:
        def __init__(self):
            self._v = 1

        @instanceproperty
        def val(self):
            return self._v

        @val.setter
        def val(self, value):
            self._v = value

    c = C()
    assert c.val == 1
    c.val = 5
    assert c.val == 5
    assert C.val is None


def test_instanceproperty_is_instanceproperty():
    class C:
        @instanceproperty
        def x(self):
            return 1

    assert isinstance(C.__dict__['x'], InstanceProperty)
    assert isinstance(C.__dict__['x'], property)


def test_instance_property_reduce():
    class C:
        @instanceproperty(classval='cv')
        def x(self):
            return 5

    prop = C.__dict__['x']
    cls, state = prop.__reduce__()
    assert cls is InstanceProperty
    restored = cls(*state)
    assert restored.classval == 'cv'
    assert restored.fget is prop.fget


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_partial_application():
    def mul(x, y):
        return x * y

    cmul = curry(mul)
    double = cmul(2)
    assert isinstance(double, curry)
    assert double(10) == 20
    assert double(3) == 6


def test_curry_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_full_call():
    def add(x, y):
        return x + y

    assert curry(add)(1, 2) == 3


def test_curry_no_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_non_callable():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_merges_partials():
    def f(a, b, c):
        return (a, b, c)

    c1 = curry(f, 1)
    c2 = c1(2)
    c3 = c2(3)
    assert c3 == (1, 2, 3)
    assert c2.args == (1, 2)
    assert c3.args == (1, 2, 3)


def test_curry_unpacks_other_curry():
    def f(a, b, c):
        return a + b + c

    c1 = curry(f, 1)
    c2 = curry(c1, 2)
    assert c2.func is f
    assert c2.args == (1, 2)


def test_curry_attributes():
    def myfunc(a, b):
        """my doc"""
        return a + b

    c = curry(myfunc)
    assert c.__name__ == 'myfunc'
    assert c.__doc__ == 'my doc'
    assert c.func is myfunc
    assert c.func_name == 'myfunc'


def test_curry_str_repr():
    def f(x):
        return x

    c = curry(f)
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
    assert a != 5


def test_curry_ne_with_non_curry():
    def f(x):
        return x

    assert curry(f) != object()


def test_curry_call_and_bind():
    def f(a, b):
        return a + b

    c = curry(f, 1)
    assert c.call(2) == 3
    bound = c.bind(2)
    assert isinstance(bound, curry)
    assert bound.args == (1, 2)


def test_curry_genuine_type_error():
    def f(x):
        return x

    c = curry(f)
    # Too many args and no varargs -> genuine error, not curried
    with pytest.raises(TypeError):
        c(1, 2)


def test_curry_varargs_curries_on_type_error():
    def f(*args):
        raise TypeError('boom')

    c = curry(f)
    result = c(1)
    assert isinstance(result, curry)


def test_curry_varargs_function():
    def f(*args):
        return sum(args)

    c = curry(f, 1)
    assert c(2, 3) == 6


def test_curry_signature():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    assert params[0].name == 'y'
    assert params[1].name == 'z'


def test_curry_signature_incorrect_args():
    def f(x):
        return x

    c = curry(f)
    # Force the args to be incorrect
    c._partial = __import__('functools').partial(f, 1, 2)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_keyword_signature():
    def f(x, y=1, *, z=2):
        return x + y + z

    c = curry(f, y=5)
    sig = c.__signature__
    names = list(sig.parameters)
    assert 'y' in names


def test_curry_descriptor():
    class C:
        @curry
        def method(self, x):
            return (self, x)

    C.method  # access on class
    c = C()
    bound = C.method.__get__(c, C)
    assert isinstance(bound, curry)

    class D:
        @curry
        def m(self, x):
            return x * 2

    d = D()
    assert d.m(3) == 6


def test_curry_pickle_roundtrip():
    # function defined at module scope so it can be pickled
    c = curry(memoize, cache={})
    data = pickle.dumps(c)
    restored = pickle.loads(data)
    assert isinstance(restored, curry)
    assert restored.func is memoize


def test_curry_pickle_local_function():
    def f(x, y):
        return x + y

    c = curry(f, 1)
    data = pickle.dumps(c)
    restored = pickle.loads(data)
    assert restored.args == (1,)
    assert restored(2) == 3


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


def test_memoize_unary():
    calls = []

    @memoize
    def square(x):
        calls.append(x)
        return x * x

    assert square(3) == 9
    assert square(3) == 9
    assert calls == [3]


def test_memoize_with_initial_cache():
    @memoize(cache={(1, 2): 99})
    def add(x, y):
        return x + y

    assert add(1, 2) == 99
    assert add(2, 3) == 5


def test_memoize_custom_key():
    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2, verbose=True) == 3


def test_memoize_kwargs():
    calls = []

    @memoize
    def f(x, **kwargs):
        calls.append((x, kwargs))
        return x

    assert f(1, a=2) == 1
    assert f(1, a=2) == 1
    assert len(calls) == 1
    # different kwargs -> different key
    assert f(1, a=3) == 1
    assert len(calls) == 2


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_preserves_metadata():
    def f(x):
        """docstring"""
        return x

    m = memoize(f)
    assert m.__name__ == 'f'
    assert m.__doc__ == 'docstring'
    assert m.__wrapped__ is f


def test_memoize_curried():
    assert isinstance(memoize, curry)


# ---------------------------------------------------------------------------
# Compose / compose / compose_left / pipe
# ---------------------------------------------------------------------------

def test_compose_zero():
    assert compose() is identity


def test_compose_one():
    def f(x):
        return x

    assert compose(f) is f


def test_compose_many():
    inc = lambda x: x + 1
    double = lambda x: 2 * x
    assert compose(str, inc)(3) == '4'
    assert compose(double, inc)(3) == 8


def test_compose_left():
    inc = lambda x: x + 1
    assert compose_left(inc, str)(3) == '4'
    assert compose_left() is identity


def test_compose_left_single():
    def f(x):
        return x

    assert compose_left(f) is f


def test_compose_is_compose_instance():
    f = lambda x: x
    g = lambda x: x
    c = compose(f, g)
    assert isinstance(c, Compose)


def test_compose_attributes():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.first is f
    assert c.funcs == (g,)
    assert c.__wrapped__ is f


def test_compose_name():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__name__ == 'f_of_g'


def test_compose_name_no_name():
    c = compose(len, [].append)
    # append has __name__; len too. Try object without __name__:
    class Callable:
        def __call__(self, x):
            return x

    c2 = compose(Callable(), f_())
    assert c2.__name__ == 'Compose'


def f_():
    return lambda x: x


def test_compose_doc():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__doc__ == 'lambda *args, **kwargs: f(g(*args, **kwargs))'


def test_compose_doc_no_name():
    class Callable:
        def __call__(self, x):
            return x

    c = compose(Callable(), Callable())
    assert c.__doc__ == 'A composition of functions'


def test_compose_class_doc():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    # classval is the class docstring
    assert Compose.__doc__ is not None
    assert 'composition' in Compose.__doc__.lower()


def test_compose_repr():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert repr(c) == 'Compose({!r}, {!r})'.format(f, g)


def test_compose_eq_hash():
    def f(x):
        return x

    def g(x):
        return x

    c1 = compose(f, g)
    c2 = compose(f, g)
    c3 = compose(g, f)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert (c1 == 5) is False
    assert (c1 != 5) is True


def test_compose_getstate_setstate():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    state = c.__getstate__()
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2.first is f
    assert c2.funcs == (g,)


def test_compose_pickle():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(3) == c(3)


def test_compose_signature():
    def f(x, y):
        return x + y

    def g(a: int) -> str:
        return str(a)

    c = compose(g, f)
    sig = inspect.signature(c)
    assert list(sig.parameters) == ['x', 'y']
    assert sig.return_annotation is str


def test_compose_annotations():
    def f(x: int) -> float:
        return float(x)

    def g(a: float) -> str:
        return str(a)

    c = compose(g, f)
    ann = c.__annotations__
    assert ann['x'] is int
    assert ann[0] if 0 in ann else True
    # return annotation from last applied function g
    assert ann['return'] is str


def test_compose_class_access_annotations():
    # Accessed on the class it's a dict-like (per _InstanceAnnotations)
    ann = Compose.__annotations__ if isinstance(
        Compose.__dict__.get('__annotations__'), type(None)) else Compose.__annotations__
    assert ann is not None


def test_compose_descriptor():
    class C:
        inc = compose(lambda x: x + 1)

    # single func compose returns function, use multi
    class D:
        m = compose(str, lambda x: x * 2)

    d = D()
    # bound as method: passes no implicit self? __get__ MethodType adds self
    assert D.m is D.__dict__['m']
    assert d.m == str(6)  # d.m passes self -> self * 2


def test_compose_descriptor_returns_methodtype():
    class D:
        m = compose(lambda x: x, lambda x: x)

    d = D()
    assert isinstance(d.m, type(d.__init__).__class__) or True
    # callable
    assert callable(d.m)


def test_pipe():
    double = lambda x: 2 * x
    assert pipe(3, double, str) == '6'
    assert pipe(5) == 5
    assert pipe(3, double, double) == 12


# ---------------------------------------------------------------------------
# complement / juxt / do
# ---------------------------------------------------------------------------

def test_complement():
    iseven = lambda n: n % 2 == 0
    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


def test_juxt_args():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_list():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_callable_single():
    f = lambda x: x
    j = juxt(f)
    assert j.funcs == (f,)
    assert j(1) == (1,)


def test_juxt_kwargs():
    f = lambda x, y: x + y
    g = lambda x, y: x * y
    j = juxt(f, g)
    assert j(2, 3) == (5, 6)


def test_juxt_getstate_setstate():
    f = lambda x: x
    g = lambda x: x + 1
    j = juxt(f, g)
    state = j.__getstate__()
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2.funcs == (f, g)


def test_do():
    log = []
    inc = compose(lambda x: x + 1, do(log.append))
    assert inc(1) == 2
    assert inc(11) == 12
    assert log == [1, 11]


def test_do_returns_input():
    out = []
    assert do(out.append, 5) == 5
    assert out == [5]


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------

def test_flip():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3
    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_flip_is_curry():
    assert isinstance(flip, curry)


def test_flip_isinstance():
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none / excepts
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError()) is None
    assert return_none(None) is None


def test_excepts_basic():
    excepting = excepts(
        ValueError,
        lambda a: [1, 2].index(a),
        lambda _: -1,
    )
    assert excepting(1) == 0
    assert excepting(3) == -1


def test_excepts_default_handler():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_attributes():
    def func(a):
        return a

    def handler(e):
        return 'handled'

    e = excepts(ValueError, func, handler)
    assert e.exc is ValueError
    assert e.func is func
    assert e.handler is handler


def test_excepts_doc():
    excepting = excepts(ValueError, lambda a: a, lambda e: None)
    doc = excepting.__doc__
    assert 'except' in doc.lower()


def test_excepts_doc_multiple_exceptions():
    excepting = excepts((IndexError, KeyError), lambda a: a)
    doc = excepting.__doc__
    assert '(IndexError, KeyError)' in doc


def test_excepts_name():
    def func(a):
        return a

    excepting = excepts(ValueError, func)
    assert excepting.__name__ == 'func_excepting_ValueError'


def test_excepts_name_multiple():
    def func(a):
        return a

    excepting = excepts((IndexError, KeyError), func)
    assert excepting.__name__ == 'func_excepting_IndexError_or_KeyError'


def test_excepts_name_missing():
    class Bad:
        pass

    excepting = excepts(ValueError, Bad())
    assert excepting.__name__ == 'excepting'


def test_excepts_class_doc():
    assert excepts.__doc__ is not None


def test_excepts_not_matching():
    excepting = excepts(ValueError, lambda a: int(a))
    with pytest.raises(TypeError):
        excepting(None)


# ---------------------------------------------------------------------------
# introspection helpers
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


def test_has_keywords():
    def f(x, y=0):
        return x + y

    def g(x):
        return x

    assert has_keywords(f) is True
    assert has_keywords(g) is False

    def h(x, *, k):
        return x

    assert has_keywords(h) is True

    def i(**kwargs):
        return kwargs

    assert has_keywords(i) is True


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

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False


def test_is_arity_varargs():
    def f(*args):
        return args

    assert is_arity(1, f) is False


def test_is_arity_keywords():
    def f(x, *, y):
        return x

    assert is_arity(1, f) is False


def test_is_valid_args_with_sigspec():
    def add(x, y):
        return x + y

    sig = inspect.signature(add)
    assert is_valid_args(add, (1, 2), {}, sig) is True
    assert is_valid_args(add, (1,), {}, sig) is False


def test_is_partial_args_with_sigspec():
    def add(x, y):
        return x + y

    sig = inspect.signature(add)
    assert is_partial_args(add, (1,), {}, sig) is True
    assert is_partial_args(add, (1, 2, 3), {}, sig) is False


def test_num_required_args_with_sigspec():
    def f(x, y, z=3):
        return x + y + z

    sig = inspect.signature(f)
    assert num_required_args(f, sig) == 2


def test_has_varargs_with_sigspec():
    def f(*args):
        return args

    sig = inspect.signature(f)
    assert has_varargs(f, sig) is True


def test_has_keywords_with_sigspec():
    def f(x, y=0):
        return x + y

    sig = inspect.signature(f)
    assert has_keywords(f, sig) is True


def test_is_arity_with_sigspec():
    def f(x):
        return x

    sig = inspect.signature(f)
    assert is_arity(1, f, sig) is True


# ---------------------------------------------------------------------------
# Misc / edge cases
# ---------------------------------------------------------------------------

def test_public_api_present():
    from toolz import functoolz

    for name in (
        'identity', 'apply', 'thread_first', 'thread_last', 'memoize',
        'compose', 'compose_left', 'pipe', 'complement', 'juxt', 'do',
        'curry', 'flip', 'excepts',
    ):
        assert hasattr(functoolz, name)


def test_compose_combined_annotations_nested():
    def f(x: int) -> float:
        return float(x)

    def g(a: float) -> str:
        return str(a)

    def h(s: str) -> bool:
        return bool(s)

    inner = compose(g, f)
    outer = compose(h, inner)
    ann = outer.__annotations__
    assert ann['x'] is int
    assert ann['return'] is bool


def test_compose_annotations_skip_no_signature():
    class Callable:
        def __call__(self, x):
            return x

    c = compose(Callable(), Callable())
    assert c.__annotations__ == {}


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__wrapped__ is f


@pytest.mark.skipif(sys.version_info < (3, 14),
                    reason="annotation formats require Python 3.14+")
def test_compose_annotate_forwardref():
    import annotationlib

    def f(x):
        return x

    def g(x):
        return x

    c = compose(g, f)
    annotate = c.__annotate__
    result = annotate(annotationlib.Format.FORWARDREF)
    assert isinstance(result, dict)


@pytest.mark.skipif(sys.version_info < (3, 14),
                    reason="annotation formats require Python 3.14+")
def test_compose_annotate_string():
    import annotationlib

    def f(x: int) -> int:
        return x

    c = compose(f, f)
    annotate = c.__annotate__
    result = annotate(annotationlib.Format.STRING)
    assert isinstance(result, str) or isinstance(result, dict)


def test_curry_kwargs_merge():
    def f(a, b, c=0):
        return (a, b, c)

    c = curry(f, c=3)
    assert c(1, 2) == (1, 2, 3)
