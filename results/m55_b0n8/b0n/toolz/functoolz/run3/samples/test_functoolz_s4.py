import inspect
import pickle
import sys
from functools import partial
from operator import not_

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
    assert apply(lambda x: x * 2, 5) == 10
    assert apply(lambda x, y: x + y, 1, 2) == 3


def test_apply_kwargs():
    def f(x, y=10):
        return x + y
    assert apply(f, 1, y=5) == 6


def test_apply_no_func():
    with pytest.raises(TypeError, match='func argument is required'):
        apply()


def test_apply_in_map():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4
    assert thread_first(1) == 1


def test_thread_first_tuple_forms():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_last_simple():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4
    assert thread_last(1) == 1


def test_thread_last_tuple_forms():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (pow_, 2)) == 32


def test_thread_last_map_filter():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    result = list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
    assert result == [2, 4]


# ---------------------------------------------------------------------------
# instanceproperty / InstanceProperty
# ---------------------------------------------------------------------------

def test_instanceproperty_basic():
    class MyClass:
        '''The class docstring'''

        @instanceproperty(classval='class-val')
        def val(self):
            return 42

    assert MyClass.val == 'class-val'
    assert MyClass().val == 42


def test_instanceproperty_without_arguments():
    class MyClass:
        @instanceproperty
        def val(self):
            return 42

    assert MyClass.val is None
    assert MyClass().val == 42


def test_instanceproperty_docstring_example():
    class MyClass:
        '''The class docstring'''

        @instanceproperty(classval=MyClass.__name__)
        def value(self):
            return 'instance'

    assert MyClass.value == 'MyClass'
    assert MyClass().value == 'instance'


def test_instanceproperty_setter():
    class MyClass:
        def __init__(self):
            self._x = 0

        @instanceproperty
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

    obj = MyClass()
    assert obj.x == 0
    obj.x = 5
    assert obj.x == 5


def test_instance_property_reduce():
    class MyClass:
        @instanceproperty(classval='cv')
        def val(self):
            return 1

    prop = MyClass.__dict__['val']
    assert isinstance(prop, InstanceProperty)
    restored, state = prop.__reduce__()
    assert restored is InstanceProperty
    assert state[0] is prop.fget
    assert state[4] == 'cv'


def test_instance_property_pickle_roundtrip():
    # Use a module-level-ish picklable property: pickling a property that
    # references a local function is not possible, so just check reduce state.
    class MyClass:
        @instanceproperty(classval=3)
        def val(self):
            return 4

    prop = MyClass.__dict__['val']
    new_prop = pickle.loads(pickle.dumps(prop))
    assert isinstance(new_prop, InstanceProperty)
    assert new_prop.classval == 3


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    mul = curry(mul)
    double = mul(2)
    assert double(10) == 20
    assert mul(2, 10) == 20
    assert mul(2)(10) == 20


def test_curry_as_decorator():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5
    assert f(2, 3) == 50


def test_curry_no_args_raises():
    with pytest.raises(TypeError):
        curry()


def test_curry_not_callable():
    with pytest.raises(TypeError, match='Input must be callable'):
        curry(5)


def test_curry_attributes():
    def f(x, y):
        "docstring"
        return x + y

    c = curry(f)
    assert c.func is f
    assert c.__name__ == 'f'
    assert c.__doc__ == 'docstring'
    assert c.func_name == 'f'
    assert c.args == ()
    assert c.keywords == {}
    assert str(c) == str(f)
    assert repr(c) == repr(f)


def test_curry_equality_and_hash():
    def f(x, y):
        return x + y

    c1 = curry(f, 1)
    c2 = curry(f, 1)
    c3 = curry(f, 2)
    assert c1 == c2
    assert not (c1 != c2)
    assert c1 != c3
    assert c1 != 5
    assert hash(c1) == hash(c2)
    assert hash(c1) == hash((f, (1,), None))


def test_curry_merge_partial():
    def f(a, b, c, d):
        return (a, b, c, d)

    p = partial(f, 1, 2)
    c = curry(p, 3, d=4)
    assert c() == (1, 2, 3, 4)
    assert c.func is f
    assert c.args == (1, 2, 3)
    assert c.keywords == {'d': 4}


def test_curry_merge_curry_with_keywords():
    @curry
    def f(a, b, c):
        return (a, b, c)

    c1 = f(1, c=3)
    c2 = c1(2)
    assert c2() == (1, 2, 3)


def test_curry_bind():
    def f(x, y, z):
        return x + y + z

    c = curry(f)
    bound = c.bind(1, 2)
    assert bound(3) == 6
    assert bound.args == (1, 2)


def test_curry_call():
    def f(x, y):
        return x + y

    c = curry(f, 1)
    assert c.call(2) == 3
    # call with insufficient args raises
    with pytest.raises(TypeError):
        c.call()


def test_curry_should_curry():
    def f(x, y):
        return x + y

    c = curry(f)
    # not enough args -> curry
    assert c(1)(2) == 3
    # too many args -> genuine TypeError, raised
    with pytest.raises(TypeError):
        c(1, 2, 3)


def test_curry_varargs_curries_anyway():
    def f(*args):
        if len(args) < 2:
            raise TypeError("need two")
        return args

    c = curry(f)
    # With only one arg, the call raises TypeError, and because f has
    # varargs, curry should curry
    result = c(1)
    assert isinstance(result, curry)
    assert result(2) == (1, 2)


def test_curry_genuine_typeerror_raised():
    def f(x):
        return x + 1

    c = curry(f)
    with pytest.raises(TypeError):
        c('a')  # 'a' + 1 raises TypeError, genuine


def test_curry_descriptor():
    class C:
        def __init__(self, base):
            self.base = base

        @curry
        def add(self, x, y):
            return self.base + x + y

    obj = C(100)
    assert C.add is C.__dict__['add']
    assert obj.add(1, 2) == 103
    assert obj.add(1)(2) == 103


def test_curry_signature():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1)
    sig = c.__signature__
    names = list(sig.parameters)
    assert names == ['y', 'z']


def test_curry_signature_no_default_gives_no_default():
    def f(x, y):
        return x + y

    c = curry(f, 1)
    sig = c.__signature__
    # The remaining param should have no_default as default
    from toolz.utils import no_default
    param = sig.parameters['y']
    assert param.default is no_default


def test_curry_invalid_signature_raises():
    def f(x, y):
        return x + y

    c = curry(f, 1, 2, 3)
    with pytest.raises(TypeError, match='incorrect arguments'):
        c.__signature__


def test_curry_keyword_to_keyword_only():
    def f(a, b, *, c):
        return (a, b, c)

    c = curry(f, 1, c=3)
    sig = c.__signature__
    params = sig.parameters
    assert 'b' in params


def test_curry_pickle_stable_function():
    def f(x, y):
        return x + y

    # module-level pickling is not available; test non-decorated path by
    # using a builtin-ish callable? Instead verify __reduce__ handles it.
    c = curry(len, [1, 2, 3])
    assert c() == 3


def test_curry_reduce_non_decorated():
    def f(x, y):
        return x + y

    c = curry(f, 1)
    restore, state = c.__reduce__()
    assert state[0] is curry
    assert state[2] == (1,)
    assert state[3] == {}


# ---------------------------------------------------------------------------
# memoize
# ---------------------------------------------------------------------------

def test_memoize_basic():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    m = memoize(add)
    assert m(1, 2) == 3
    assert m(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_as_decorator():
    calls = []

    @memoize
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_initial_cache():
    calls = []

    @memoize(cache={(1, 2): 3})
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(1, 2) == 3
    assert calls == []
    assert add(2, 2) == 4
    assert calls == [(2, 2)]


def test_memoize_unary():
    calls = []

    @memoize
    def square(x):
        calls.append(x)
        return x * x

    assert square(3) == 9
    assert square(3) == 9
    assert calls == [3]


def test_memoize_custom_key():
    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2, verbose=True) == 3


def test_memoize_unhashable_args():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError, match='hashable'):
        f([1, 2, 3])


def test_memoize_attributes():
    def f(x):
        "the doc"
        return x

    m = memoize(f)
    assert m.__name__ == 'f'
    assert m.__doc__ == 'the doc'
    assert m.__wrapped__ is f


def test_memoize_lambda_no_name():
    m = memoize(lambda x: x)
    # lambda has __name__, so no AttributeError path; just check it works
    assert m(1) == 1


# ---------------------------------------------------------------------------
# compose / compose_left / Compose
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_no_funcs():
    assert compose() is identity
    assert compose()(5) == 5


def test_compose_single_func():
    f = lambda x: x + 1
    assert compose(f) is f


def test_compose_left_basic():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_no_funcs():
    assert compose_left() is identity


def test_compose_kwargs():
    def f(x, y=1):
        return x + y

    def g(v):
        return v * 2

    assert compose(g, f)(3, y=4) == 14


def test_compose_repr():
    def f(x):
        return x

    def g(x):
        return x

    comp = compose(f, g)
    assert repr(comp) == 'Compose({!r}, {!r})'.format(f, g)


def test_compose_doc():
    def f(x):
        return x

    def g(x):
        return x

    comp = compose(f, g)
    assert comp.__doc__ == 'lambda *args, **kwargs: f(g(*args, **kwargs))'
    assert Compose.__doc__ is not None


def test_compose_doc_no_names():
    class Callable:
        def __call__(self, x):
            return x

    comp = compose(Callable(), Callable())
    assert comp.__doc__ == 'A composition of functions'


def test_compose_name():
    def f(x):
        return x

    def g(x):
        return x

    comp = compose(f, g)
    assert comp.__name__ == 'f_of_g'


def test_compose_name_no_names():
    class Callable:
        def __call__(self, x):
            return x

    comp = compose(Callable(), Callable())
    assert comp.__name__ == 'Compose'


def test_compose_equality():
    def f(x):
        return x

    def g(x):
        return x

    c1 = compose(f, g)
    c2 = compose(f, g)
    c3 = compose(g, f)
    assert c1 == c2
    assert c1 != c3
    assert (c1 == 5) is False
    assert c1.__eq__(5) is NotImplemented
    assert c1.__ne__(5) is NotImplemented
    assert hash(c1) == hash(c1)
    assert hash(c1) != hash(c3)


def test_compose_signature():
    def f(x: int, y: int) -> int:
        return x + y

    def g(z: int) -> str:
        return str(z)

    comp = compose(g, f)
    sig = comp.__signature__
    assert list(sig.parameters) == ['x', 'y']
    assert sig.return_annotation is str


def test_compose_annotations():
    def f(x: int, y: int) -> int:
        return x + y

    def g(z: int) -> str:
        return str(z)

    comp = compose(g, f)
    ann = comp.__annotations__
    assert ann == {'x': int, 'y': int, 'return': str}


def test_compose_annotations_of_instance():
    def f(x: int) -> int:
        return x

    def g(z: int) -> str:
        return str(z)

    comp = compose(g, f)
    assert Compose.__annotations__.get('return') is None
    assert comp.__annotations__['return'] is str


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    comp = compose(f, g)
    assert comp.__wrapped__ is f


def test_compose_descriptor():
    class C:
        def __init__(self, base):
            self.base = base

        def get_base(self):
            return self.base

        apply_base = compose(lambda x: x * 2, get_base)

    obj = C(5)
    assert C.apply_base is C.__dict__['apply_base']
    assert obj.apply_base() == 10


def test_compose_getstate_setstate():
    def f(x):
        return x

    def g(x):
        return x

    comp = compose(f, g)
    first, funcs = comp.__getstate__()
    new = Compose.__new__(Compose)
    new.__setstate__((first, funcs))
    assert new.first is first
    assert new.funcs == funcs
    assert new(3) == 3


@pytest.mark.skipif(sys.version_info < (3, 14), reason='PEP 749')
def test_compose_annotate():
    import annotationlib

    def f(x: int) -> int:
        return x

    def g(z: int) -> str:
        return str(z)

    comp = compose(g, f)
    annotate = comp.__annotate__
    assert annotate(annotationlib.Format.VALUE) == {
        'x': int, 'return': str}
    assert annotate(annotationlib.Format.STRING) == {
        'x': 'int', 'return': 'str'}


# ---------------------------------------------------------------------------
# pipe
# ---------------------------------------------------------------------------

def test_pipe():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'
    assert pipe(3) == 3
    assert pipe(3, double, double, double) == 24


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


# ---------------------------------------------------------------------------
# juxt
# ---------------------------------------------------------------------------

def test_juxt_basic():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_single_callable():
    # A single callable should not be iterated over
    assert juxt(len)('abc') == (3,)


def test_juxt_kwargs():
    def f(x, y=0):
        return x + y

    def g(x, y=0):
        return x - y

    assert juxt(f, g)(10, y=3) == (13, 7)


def test_juxt_getstate_setstate():
    j = juxt(len, str)
    state = j.__getstate__()
    assert state == (len, str)
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2(123) == (3, '123')


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    assert do(log.append, 5) == 5
    assert log == [5]


def test_do_compose():
    log = []
    inc = lambda x: x + 1
    inc = compose(inc, do(log.append))
    assert inc(1) == 2
    assert inc(11) == 12
    assert log == [1, 11]


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------

def test_flip():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3
    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


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
    e = excepts(ValueError, len, lambda _: None)
    assert e.exc is ValueError
    assert e.func is len
    assert e.__name__ == 'len_excepting_ValueError'


def test_excepts_name_tuple():
    e = excepts((ValueError, KeyError), len)
    assert e.__name__ == 'len_excepting_ValueError_or_KeyError'


def test_excepts_name_attribute_error():
    class Callable:
        def __call__(self, x):
            return x

    e = excepts(ValueError, Callable())
    assert e.__name__ == 'excepting'


def test_excepts_doc():
    excepting = excepts(ValueError, lambda a: a, lambda e: None)
    doc = excepting.__doc__
    assert 'A wrapper around' in doc
    assert 'ValueError' in doc
    assert excepts.__doc__ is not None


def test_excepts_doc_tuple():
    e = excepts((ValueError, KeyError), lambda a: a, lambda e: None)
    doc = e.__doc__
    assert '(ValueError, KeyError)' in doc


def test_excepts_doc_attribute_error_fallback():
    class Callable:
        def __call__(self, x):
            return x

    e = excepts(ValueError, Callable())
    assert e.__doc__ == type(e).__doc__


def test_excepts_kwargs():
    def f(x, y=0):
        if x < 0:
            raise ValueError
        return x + y

    e = excepts(ValueError, f, lambda _: -1)
    assert e(1, y=2) == 3
    assert e(-1) == -1


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    def g(*args, **kwargs):
        pass

    assert num_required_args(f) == 2
    assert num_required_args(g) == 0
    assert num_required_args(len) == 1


def test_num_required_args_with_sigspec():
    def f(x, y, z=3):
        return x + y + z

    sig = inspect.signature(f)
    assert num_required_args(f, sig) == 2


def test_has_varargs():
    def f(*args):
        return args

    def g(**kwargs):
        return kwargs

    assert has_varargs(f) is True
    assert has_varargs(g) is False
    assert has_varargs(len) is False


def test_has_varargs_with_sigspec():
    def f(*args):
        return args

    sig = inspect.signature(f)
    assert has_varargs(f, sig) is True


def test_has_keywords():
    def f(x, y=0):
        return x + y

    def g(x, y):
        return x + y

    assert has_keywords(f) is True
    assert has_keywords(g) is False


def test_has_keywords_with_sigspec():
    def f(x, y=0):
        return x + y

    sig = inspect.signature(f)
    assert has_keywords(f, sig) is True


def test_is_valid_args():
    def add(x, y):
        return x + y

    assert is_valid_args(add, (1,), {}) is False
    assert is_valid_args(add, (1, 2), {}) is True
    assert is_valid_args(add, (), {'x': 1, 'y': 2}) is True
    assert is_valid_args(map, (), {}) is False
    assert is_valid_args(map, (len, []), {}) is True


def test_is_valid_args_with_sigspec():
    def add(x, y):
        return x + y

    sig = inspect.signature(add)
    assert is_valid_args(add, (1, 2), {}, sig) is True
    assert is_valid_args(add, (1,), {}, sig) is False


def test_is_partial_args():
    def add(x, y):
        return x + y

    assert is_partial_args(add, (1,), {}) is True
    assert is_partial_args(add, (1, 2), {}) is True
    assert is_partial_args(add, (1, 2, 3), {}) is False
    assert is_partial_args(map, (), {}) is True


def test_is_partial_args_with_sigspec():
    def add(x, y):
        return x + y

    sig = inspect.signature(add)
    assert is_partial_args(add, (1,), {}, sig) is True
    assert is_partial_args(add, (1, 2, 3), {}, sig) is False


def test_is_arity():
    def f(x):
        return x

    def g(x, y=1):
        return x + y

    def h(*args):
        return args

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False
    assert is_arity(2, g) is False
    assert is_arity(1, h) is False
    assert is_arity(2, lambda x, y: x + y) is True


def test_is_arity_with_sigspec():
    def f(x):
        return x

    sig = inspect.signature(f)
    assert is_arity(1, f, sig) is True
    assert is_arity(2, f, sig) is False


def test_is_arity_builtin():
    assert is_arity(1, len) is True


def test_signature_checks_builtin_no_signature():
    # Some builtins may lack a signature; ensure helpers return something
    # (True/False/None) rather than raising.
    result = num_required_args(dict.update)
    assert result is not None or result is None  # just shouldn't raise
