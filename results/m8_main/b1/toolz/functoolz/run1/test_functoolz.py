import inspect
import pickle
import sys
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
# identity / apply
# ---------------------------------------------------------------------------

def test_identity():
    assert identity(3) == 3
    obj = object()
    assert identity(obj) is obj


def test_apply_basic():
    def double(x):
        return 2 * x

    assert apply(double, 5) == 10


def test_apply_kwargs():
    def f(a, b=1):
        return a + b

    assert apply(f, 3, b=10) == 13


def test_apply_no_args_raises():
    with pytest.raises(TypeError):
        apply()


# ---------------------------------------------------------------------------
# thread_first / thread_last
# ---------------------------------------------------------------------------

def test_thread_first_simple():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    assert thread_first(1, inc, double) == 4


def test_thread_first_tuples():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(5) == 5


def test_thread_last_simple():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    assert thread_last(1, inc, double) == 4


def test_thread_last_tuples():
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


def test_instanceproperty_setter_deleter():
    class C:
        def __init__(self):
            self._x = 0

        @instanceproperty
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value * 2

    c = C()
    assert c.x == 0
    c.x = 5
    assert c.x == 10


def test_instanceproperty_classval_none_by_default():
    class C:
        @instanceproperty
        def val(self):
            return 'instance'

    assert C.val is None
    assert C().val == 'instance'


def test_instanceproperty_class_is_instanceproperty():
    class C:
        @instanceproperty
        def x(self):
            return 1

    assert isinstance(inspect.getattr_static(C, 'x'), InstanceProperty)


def test_instanceproperty_reduce():
    class C:
        @instanceproperty
        def x(self):
            return 1

    p = inspect.getattr_static(C, 'x')
    cls, state = p.__reduce__()
    assert cls is InstanceProperty
    fget, fset, fdel, doc, classval = state
    assert fget is p.fget
    assert fset is p.fset
    assert fdel is p.fdel
    assert classval is p.classval


# ---------------------------------------------------------------------------
# curry
# ---------------------------------------------------------------------------

def test_curry_basic():
    def mul(x, y):
        return x * y

    mul_c = curry(mul)
    double = mul_c(2)
    assert double(10) == 20


def test_curry_decorator_with_defaults():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_no_args_raises():
    with pytest.raises(TypeError):
        curry()


def test_curry_non_callable_raises():
    with pytest.raises(TypeError):
        curry(5)


def test_curry_direct_call_enough_args():
    @curry
    def add(x, y):
        return x + y

    assert add(1, 2) == 3


def test_curry_genuine_typeerror_propagates():
    # With all required args supplied via curry, the call runs immediately
    # and a TypeError raised inside the body propagates (it is a real call).
    @curry
    def f(x, y):
        return x + y

    with pytest.raises(TypeError):
        f(1, 'a')  # 1 + 'a' is a TypeError inside the function body


def test_curry_too_many_args_raises():
    @curry
    def add(x, y):
        return x + y

    with pytest.raises(TypeError):
        add(1, 2, 3)


def test_curry_attributes():
    def myfunc(x):
        "docstring"
        return x

    c = curry(myfunc)
    assert c.func is myfunc
    assert c.__name__ == 'myfunc'
    assert c.__doc__ == 'docstring'
    assert c.args == ()
    assert c.keywords == {}
    assert c.func_name == 'myfunc'


def test_curry_str_repr():
    def myfunc(x):
        return x

    c = curry(myfunc)
    assert str(c) == str(myfunc)
    assert repr(c) == repr(myfunc)


def test_curry_eq_hash():
    def f(x, y):
        return x + y

    c1 = curry(f, 1)
    c2 = curry(f, 1)
    c3 = curry(f, 2)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert not (c1 == 'not a curry')
    assert c1 != 'not a curry'


def test_curry_merge_partials():
    def f(a, b, c):
        return (a, b, c)

    p1 = curry(f, 1)
    p2 = p1(2)
    assert isinstance(p2, curry)
    assert p2(3) == (1, 2, 3)


def test_curry_merge_kwargs():
    def f(a, b, c=0):
        return (a, b, c)

    p1 = curry(f, c=5)
    p2 = p1(1, 2)
    assert p2 == (1, 2, 5)


def test_curry_bind_and_call():
    @curry
    def add(x, y):
        return x + y

    c = add(1)
    assert isinstance(c, curry)
    assert c.call(2) == 3


def test_curry_signature():
    def f(a, b, c=3):
        return a + b + c

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ['b', 'c']


def test_curry_signature_varargs():
    def f(a, *args, **kwargs):
        return (a, args, kwargs)

    c = curry(f, 1)
    sig = c.__signature__
    names = list(sig.parameters)
    assert 'a' not in names
    assert 'args' in names


def test_curry_get_descriptor():
    class C:
        @curry
        def method(self, x):
            return self.value + x

        value = 10

    c = C()
    assert c.method(5) == 15


def test_curry_get_descriptor_on_class():
    class C:
        @curry
        def method(self, x):
            return x

    assert isinstance(C.method, curry)


def test_curry_pickle_module_level():
    # __reduce__ tries to resolve the function by module/qualname.  For a
    # module-level function stored in a module attribute it can round-trip;
    # here we use the imported `identity`, giving a stable round trip.
    c = curry(identity, 7)
    restored = pickle.loads(pickle.dumps(c))
    # restoring a curry onto identity with a bound positional arg
    assert isinstance(restored, curry)
    assert restored.func is identity
    assert restored.args == (7,)


def test_curry_pickle_local_raises():
    # Local functions cannot be resolved by module/qualname so pickling fails
    @curry
    def add(x, y):
        return x + y

    with pytest.raises(AttributeError):
        pickle.dumps(add)


def test_curry_should_curry_varargs():
    @curry
    def deco(*args, **kwargs):
        if len(args) < 2:
            raise TypeError("need more")
        return args

    # With *args, curry keeps currying on TypeError
    result = deco(1)
    assert isinstance(result, curry)


def test_curry_keywords_preserved():
    def f(a, b, c=0):
        return (a, b, c)

    c = curry(f, 1, c=3)
    assert c.keywords == {'c': 3}
    assert c(2) == (1, 2, 3)


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
    def sq(x):
        calls.append(x)
        return x * x

    assert sq(3) == 9
    assert sq(3) == 9
    assert sq(4) == 16
    assert calls == [3, 4]


def test_memoize_initial_cache():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        return x + y

    assert add(1, 2) == 3


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y))
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3
    assert calls == [(1, 2)]


def test_memoize_unhashable_raises():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_metadata():
    def f(x):
        """mydoc"""
        return x

    m = memoize(f)
    assert m.__name__ == 'f'
    assert m.__doc__ == 'mydoc'
    assert m.__wrapped__ is f


def test_memoize_kwargs_distinction():
    calls = []

    @memoize
    def f(x, y=0):
        calls.append((x, y))
        return x + y

    assert f(1, y=2) == 3
    assert f(1, y=3) == 4
    assert calls == [(1, 2), (1, 3)]


# ---------------------------------------------------------------------------
# Compose / compose / compose_left / pipe
# ---------------------------------------------------------------------------

def test_compose_basic():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    f = compose(str, inc)
    assert f(3) == '4'

    g = compose(double, inc)
    assert g(1) == 4


def test_compose_zero_funcs_identity():
    f = compose()
    assert f is identity
    assert f(5) == 5


def test_compose_one_func():
    def f(x):
        return x * 2

    assert compose(f) is f


def test_compose_left_basic():
    def inc(x):
        return x + 1

    f = compose_left(inc, str)
    assert f(3) == '4'


def test_compose_left_zero_funcs():
    assert compose_left() is identity


def test_compose_left_order():
    f = compose_left(lambda x: x + 1, lambda x: x * 2)
    assert f(3) == 8  # (3+1)*2


def test_pipe():
    def double(x):
        return 2 * x

    assert pipe(3, double, str) == '6'
    assert pipe(5) == 5


def test_compose_call_with_kwargs():
    def f(a, b=0):
        return a + b

    c = compose(str, f)
    assert c(1, b=2) == '3'


def test_compose_repr_str():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    # repr is 'Compose{!r}'.format(tuple) -> "{!r}".format(tuple) gives
    # 'Compose(<func>, <func>)' (tuples: parens around single here not present)
    assert repr(c) == 'Compose{!r}'.format((f, g))


def test_compose_eq_ne_hash():
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
    assert not (c1 == 'not a compose')
    assert c1 != 'not a compose'


def test_compose_doc():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    doc = c.__doc__
    assert doc.startswith('lambda *args, **kwargs: ')
    assert 'g(' in doc
    assert 'f(' in doc


def test_compose_doc_no_names():
    c = Compose((len, sys.maxsize))
    # sys.maxsize is not callable, but Compose doc catches AttributeError
    doc = c.__doc__
    assert isinstance(doc, str)


def test_compose_name():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__name__ == 'f_of_g'


def test_compose_name_no_names():
    c = Compose((len,))
    assert c.__name__ == 'len'


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__wrapped__ is g


def test_compose_get_descriptor():
    def base(self, x):
        return self.value + x

    def double(x):
        return x * 2

    class C:
        value = 5
        method = Compose((double, base))

    c = C()
    assert c.method(3) == 16  # double(base(c, 3)) = double(8)
    # On class, returns Compose itself
    assert isinstance(C.method, Compose)


def test_compose_signature():
    def f(a, b):
        return a + b

    def g(x) -> int:
        return x

    c = compose(f, g)
    sig = c.__signature__
    assert list(sig.parameters) == ['x']
    # return annotation comes from last function applied, which for
    # compose(f, g) is f (first in the funcs list), not g.
    assert sig.return_annotation is inspect.Parameter.empty


def test_compose_signature_with_return():
    def f(a, b) -> str:
        return ''

    def g(x) -> int:
        return x

    c = compose(f, g)
    sig = c.__signature__
    assert list(sig.parameters) == ['x']
    # The last function applied (f) provides the return annotation
    assert sig.return_annotation is str


def test_compose_annotations():
    def f(a: int, b: str) -> float:
        return 0.0

    def g(x: bool) -> str:
        return ''

    c = compose(f, g)
    annotations = c.__annotations__
    assert annotations == {'x': bool, 'return': float}


def test_compose_annotations_no_return_on_first():
    def f(a):
        return a

    def g(x: int) -> str:
        return ''

    c = compose(f, g)
    # Return annotation only comes from the last function applied (f),
    # which has no return annotation here.
    assert c.__annotations__ == {'x': int}


def test_compose_annotations_class_access():
    def f(a: int):
        return a

    def g(x: int):
        return x

    c = compose(f, g)
    # Accessed via class attribute on Compose type... actually accessing
    # type(c).__annotations__ returns the descriptor object
    desc = inspect.getattr_static(Compose, '__annotations__')
    assert desc.__get__(None, Compose) is desc


def test_compose_annotations_nested_compose():
    def f(a: int) -> str:
        return ''

    def g(x: int) -> int:
        return x

    def h(y: str) -> bool:
        return True

    inner = compose(g, h)
    outer = compose(f, inner)
    ann = outer.__annotations__
    # return annotation comes from last function applied, which is 'f' -> str
    assert ann['return'] == str


def test_compose_annotations_signature_error_skipped():
    class NoSig:
        def __call__(self, *a, **k):
            pass

    def f(a: int) -> str:
        return ''

    c = Compose((f, NoSig()))
    annotations = c.__annotations__
    # NoSig is callable and has a signature; its params override.  'a' from
    # f is replaced by NoSig's *a/**k (which have no annotations).
    assert 'return' in annotations
    assert annotations['return'] == str


def test_compose_pickle_state():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    # roundtrip via getstate/setstate
    state = c.__getstate__()
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2 == c


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

def test_juxt():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_list():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(a, b=0):
        return a + b

    def g(a, b=0):
        return a * b

    j = juxt(f, g)
    assert j(3, b=4) == (7, 12)


def test_juxt_state():
    def f(x):
        return x

    def g(x):
        return x

    j = juxt(f, g)
    state = j.__getstate__()
    assert state == (f, g)
    j2 = juxt(lambda: None)
    j2.__setstate__(state)
    assert j2.funcs == (f, g)


# ---------------------------------------------------------------------------
# do
# ---------------------------------------------------------------------------

def test_do():
    log = []
    result = do(log.append, 5)
    assert result == 5
    assert log == [5]


def test_do_side_effect_only():
    log = []

    def inc(x):
        return x + 1

    def logged(x):
        log.append(x)
        return x

    f = compose(inc, partial(do, logged))
    assert f(1) == 2
    assert log == [1]


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
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------------------------------------------------------------------------
# return_none / excepts
# ---------------------------------------------------------------------------

def test_return_none():
    assert return_none(ValueError()) is None


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


def test_excepts_reraises_other():
    excepting = excepts(ValueError, lambda a: a[0])
    with pytest.raises(TypeError):
        excepting(5)


def test_excepts_name():
    excepting = excepts(ValueError, lambda a: a)
    assert excepting.__name__ == '<lambda>_excepting_ValueError'


def test_excepts_name_tuple():
    excepting = excepts((ValueError, KeyError), lambda a: a)
    assert excepting.__name__ == '<lambda>_excepting_ValueError_or_KeyError'


def test_excepts_name_no_name():
    class NotCallable:
        def __call__(self, x):
            return x

    obj = NotCallable()
    excepting = excepts(ValueError, obj)
    assert excepting.__name__ == 'excepting'


def test_excepts_doc_class_access():
    assert excepts.__doc__ is not None
    assert 'functional try/except block' in excepts.__doc__


def test_excepts_doc_instance_access():
    excepting = excepts(ValueError, lambda a: a, lambda e: None)
    doc = excepting.__doc__
    assert 'ValueError' in doc
    assert '<lambda>' in doc


def test_excepts_doc_named():
    def myfunc(x):
        """func doc"""
        return x

    def handler(e):
        """handler doc"""
        return None

    excepting = excepts(ValueError, myfunc, handler)
    doc = excepting.__doc__
    assert 'myfunc' in doc
    assert 'handler' in doc
    assert 'func doc' in doc
    assert 'handler doc' in doc


def test_excepts_doc_no_attribute():
    class NoName:
        def __call__(self, x):
            return x

    obj = NoName()
    excepting = excepts(ValueError, obj)
    doc = excepting.__doc__
    assert doc is not None


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2

    def g(*args, **kwargs):
        pass

    assert num_required_args(g) == 0


def test_num_required_args_positional_only():
    def f(x, y, /, z=3):
        return x + y + z

    assert num_required_args(f) == 2


def test_has_varargs():
    def f(*args):
        return args

    assert has_varargs(f) is True

    def g(**kwargs):
        return kwargs

    assert has_varargs(g) is False


def test_has_keywords():
    def f(x, y=0):
        return x + y

    assert has_keywords(f) is True

    def g(x, y):
        return x + y

    assert has_keywords(g) is False


def test_has_keywords_kwonly():
    def f(x, *, y):
        return x + y

    assert has_keywords(f) is True


def test_is_valid_args():
    def add(x, y):
        return x + y

    assert is_valid_args(add, (1,), {}) is False
    assert is_valid_args(add, (1, 2), {}) is True
    assert is_valid_args(map, (), {}) is False


def test_is_partial_args():
    def add(x, y):
        return x + y

    assert is_partial_args(add, (1,), {}) is True
    assert is_partial_args(add, (1, 2), {}) is True
    assert is_partial_args(add, (1, 2, 3), {}) is False
    assert is_partial_args(map, (), {}) is True


def test_is_arity():
    def f(x):
        return x

    assert is_arity(1, f) is True

    def g(x, y=1):
        return x + y

    assert is_arity(1, g) is False

    def h(*args):
        return args

    assert is_arity(1, h) is False


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

def test_all_exports():
    import toolz.functoolz as ft
    expected = {
        'identity', 'apply', 'thread_first', 'thread_last', 'memoize',
        'compose', 'compose_left', 'pipe', 'complement', 'juxt', 'do',
        'curry', 'flip', 'excepts',
    }
    assert set(ft.__all__) == expected
