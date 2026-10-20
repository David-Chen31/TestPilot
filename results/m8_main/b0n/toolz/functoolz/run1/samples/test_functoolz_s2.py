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


# ---------- identity ----------

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3
    assert identity(None) is None


# ---------- apply ----------

def test_apply_basic():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert apply(double, 5) == 10
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)


def test_apply_kwargs():
    def f(a, b=10):
        return a + b

    assert apply(f, 1, b=2) == 3


def test_apply_no_func():
    with pytest.raises(TypeError, match='func argument is required'):
        apply()


# ---------- thread_first / thread_last ----------

def test_thread_first_plain():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4
    assert thread_first(5) == 5


def test_thread_first_tuple_forms():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pow_, 2)) == 25
    assert thread_first(0, (add, 1), (add, 2), (add, 3)) == 6


def test_thread_last_plain():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4
    assert thread_last(5) == 5


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


# ---------- instanceproperty / InstanceProperty ----------

def test_instanceproperty_basic():
    class MyClass:
        '''The class docstring'''

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
            self._v = 1

        @instanceproperty
        def v(self):
            return self._v

        @v.setter
        def v(self, value):
            self._v = value

        @v.deleter
        def v(self):
            self._v = -1

    c = C()
    assert c.v == 1
    c.v = 5
    assert c.v == 5
    del c.v
    assert c.v == -1


def test_instanceproperty_partial_form():
    # Without fget, instanceproperty returns a partial
    decorator = instanceproperty(classval='cv')
    assert isinstance(decorator, type(instanceproperty(classval='cv')))

    def fget(self):
        return 'instance'

    prop = decorator(fget)
    assert isinstance(prop, InstanceProperty)

    class C:
        __doc__ = prop

    # Verify via a normal attribute path on the class
    assert prop.classval == 'cv'


def test_instanceproperty_reduce_roundtrip():
    class C:
        @instanceproperty(classval='CLS')
        def v(self):
            return 'inst'

    # Pickling requires the function to be picklable; a locally-defined
    # class won't pickle. Just exercise __reduce__ directly.
    prop = C.__dict__['v']
    assert isinstance(prop, InstanceProperty)
    func, state = prop.__reduce__()
    assert func is InstanceProperty
    assert state == (prop.fget, prop.fset, prop.fdel, prop.__doc__,
                     prop.classval)
    restored = InstanceProperty(*state)
    assert restored.classval == 'CLS'
    assert restored.fget is prop.fget


# ---------- curry ----------

def test_curry_simple_partial():
    def mul(x, y):
        return x * y

    mul = curry(mul)
    double = mul(2)
    assert isinstance(double, curry)
    assert double(10) == 20


def test_curry_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5
    assert f(2, 3) == 50
    assert f(2, 3, a=2) == 10


def test_curry_no_args_init():
    with pytest.raises(TypeError):
        curry()


def test_curry_non_callable():
    with pytest.raises(TypeError, match='Input must be callable'):
        curry(42)


def test_curry_attributes():
    def f(x, y):
        """docstring here"""
        return x + y

    c = curry(f)
    assert c.func is f
    assert c.args == ()
    assert c.keywords == {}
    assert c.__doc__ == 'docstring here'
    assert c.__name__ == 'f'
    assert c.func_name == 'f'
    assert str(c) == str(f)
    assert repr(c) == repr(f)


def test_curry_equality_and_hash():
    def f(x, y):
        return x + y

    c1 = curry(f, 1)
    c2 = curry(f, 1)
    c3 = curry(f, 2)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert c1 != 'notacurry'


def test_curry_bind_and_call():
    def f(x, y, z=0):
        return x + y + z

    c = curry(f)
    bound = c.bind(1)
    assert isinstance(bound, curry)
    assert bound(2) == 3
    assert bound(2, 3) == 6
    assert c.call(1, 2, 3) == 6


def test_curry_genuine_typeerror_propagates():
    def f(x):
        return x + 'not-a-number'

    c = curry(f)
    with pytest.raises(TypeError):
        c(1)


def test_curry_descriptor():
    class C:
        def __init__(self, a):
            self.a = a

        @curry
        def method(self, x, y):
            return self.a + x + y

    c = C(10)
    assert c.method(1, 2) == 13
    assert c.method(1)(2) == 13
    # Accessing on the class returns the curry itself
    assert isinstance(C.__dict__['method'], curry)


def test_curry_sigspec():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ['y', 'z']
    # y is required (no_default sentinel), z keeps its default
    assert params[0].default is not inspect.Parameter.empty
    assert params[1].default == 3


def test_curry_unpack_existing_partial():
    import functools

    def f(x, y, z):
        return x + y + z

    p = functools.partial(f, 1)
    c = curry(p, 2)
    assert c.func is f
    assert c.args == (1, 2)
    assert c(3) == 6


def test_curry_unpack_existing_curry():
    def f(x, y, z):
        return x + y + z

    c1 = curry(f, 1)
    c2 = curry(c1, 2)
    assert c2.args == (1, 2)
    assert c2(3) == 6


def test_curry_reduce_roundtrip():
    # Use a module-level-ish function by relying on functools.partial-like
    # behavior through pickle of a curry wrapping a builtin.
    c = curry(pow, 2)
    assert c(3) == 8

    def recur():
        return None
    # Exercise __reduce__ structure without requiring picklability
    reduce_fn, state = c.__reduce__()
    from toolz.functoolz import _restore_curry
    assert reduce_fn is _restore_curry
    restored = reduce_fn(*state)
    assert restored(3) == 8


def test_curry_restore_decorated():
    # Create a decorated curry and restore it via the string path.
    @curry
    def base(x, y):
        return x + y

    c = base(1)
    reduce_fn, state = c.__reduce__()
    # base isn't importable by module/qualname, so restore via func object
    restored = reduce_fn(*state)
    assert restored(2) == 3


def test_curry_wrong_args_to_signature():
    def f(x, y):
        return x + y

    # Both args bound, but explicit kwargs conflict?
    c = curry(f, 1, 2)
    # __signature__ may raise if args are incorrect
    with pytest.raises(TypeError):
        curry(f, 1, 2, 3).__signature__


# ---------- memoize ----------

def test_memoize_basic():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    m = memoize(add)
    assert m(1, 2) == 3
    assert m(1, 2) == 3
    assert calls == [(1, 2)]
    assert m.__name__ == 'add'
    assert m.__wrapped__ is add


def test_memoize_kwargs():
    calls = []

    def add(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    m = memoize(add)
    assert m(1, 2) == 3
    assert m(1, 2, verbose=False) == 3
    assert m(1, 2, verbose=True) == 3
    assert len(calls) == 2


def test_memoize_custom_cache():
    cache = {(1, 2): 3}
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    m = memoize(add, cache=cache)
    assert m(1, 2) == 3
    assert calls == []
    assert m(2, 3) == 5
    assert cache[(2, 3)] == 5


def test_memoize_custom_key():
    def add(x, y, verbose=False):
        return x + y

    m = memoize(add, key=lambda args, kwargs: args)
    assert m(1, 2, verbose=True) == 3
    assert m(1, 2, verbose=False) == 3


def test_memoize_decorator_form():
    calls = []

    @memoize
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_decorator_with_cache():
    @memoize(cache={(1, 2): 99})
    def add(x, y):
        return x + y

    assert add(1, 2) == 99
    assert add(5, 6) == 11


def test_memoize_unhashable():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError, match='must be hashable'):
        f([1, 2, 3])


def test_memoize_unary_key():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]


def test_memoize_varargs_no_kwargs():
    # Exercises the "else" key branch: no keywords, has varargs
    calls = []

    def f(*args):
        calls.append(args)
        return sum(args)

    m = memoize(f)
    assert m(1, 2, 3) == 6
    assert m(1, 2, 3) == 6
    assert m(1, 2) == 3
    assert len(calls) == 2


# ---------- compose / Compose ----------

def inc(x):
    return x + 1


def double(x):
    return x * 2


def test_compose_basic():
    f = compose(str, inc)
    assert f(3) == '4'


def test_compose_zero_funcs():
    assert compose() is identity


def test_compose_one_func():
    assert compose(inc) is inc


def test_compose_multiple():
    f = compose(inc, double)
    assert f(3) == 7
    assert isinstance(f, Compose)


def test_compose_with_kwargs():
    def add(a, b):
        return a + b

    f = compose(inc, add)
    assert f(1, b=2) == 4


def test_compose_left():
    f = compose_left(inc, str)
    assert f(3) == '4'
    assert compose_left() is identity
    assert compose_left(inc) is inc


def test_compose_class_attributes():
    f = compose(inc, double)
    assert isinstance(f, Compose)
    assert f.first is double
    assert f.funcs == (inc,)
    assert f.__name__ == 'inc_of_double'
    assert repr(f).startswith('Compose(')
    assert f.__wrapped__ is double


def test_compose_doc():
    f = compose(inc, double)
    assert f.__doc__.startswith('lambda *args, **kwargs:')
    assert 'inc(double(' in f.__doc__


def test_compose_doc_no_name():
    f = compose(inc, object())
    # Object instance has no __name__, falls back
    assert f.__doc__ == 'A composition of functions'


def test_compose_name_no_name():
    f = compose(inc, object())
    assert f.__name__ == 'Compose'


def test_compose_eq_hash():
    f1 = compose(inc, double)
    f2 = compose(inc, double)
    f3 = compose(double, inc)
    assert f1 == f2
    assert f1 != f3
    assert hash(f1) == hash(f2)
    assert (f1 == 'x') is False or not (f1 == 'x')


def test_compose_getstate_setstate():
    f = compose(inc, double)
    state = f.__getstate__()
    assert state == (double, (inc,))
    g = Compose.__new__(Compose)
    g.__setstate__(state)
    assert g.first is double
    assert g.funcs == (inc,)


def test_compose_signature():
    def f(x: int) -> int:
        return x

    def g(y: int) -> str:
        return str(y)

    c = compose(g, f)
    sig = c.__signature__
    assert list(sig.parameters) == ['x']
    assert sig.return_annotation is str


def test_compose_descriptor():
    class C:
        def method(self, x):
            return x + 1

        comp = compose(str, do)

    # Binding a Compose as descriptor: call as method
    class D:
        def base(self, x):
            return x

        f = compose(inc)

    # compose with single function returns the raw function, so use two
    class E:
        def base(self, x):
            return x + 100

        comp = compose(inc, base)

    e = E()
    assert e.comp(5) == 106


def test_compose_descriptor_class_access():
    class E:
        def base(self, x):
            return x + 1

        comp = compose(inc, base)

    assert isinstance(E.__dict__['comp'], Compose)


def test_compose_annotations():
    def f(x: int) -> int:
        return x

    def g(y: int) -> str:
        return str(y)

    c = compose(g, f)
    ann = c._combined_annotations()
    assert ann['x'] is int
    assert ann['return'] is str


def test_compose_nested_annotations():
    def f(x: int) -> int:
        return x

    def g(y: int) -> str:
        return str(y)

    inner = compose(g, f)

    def h(z: str) -> float:
        return float(z)

    outer = compose(h, inner)
    ann = outer._combined_annotations()
    assert ann['x'] is int
    assert ann['return'] is float


# ---------- pipe ----------

def test_pipe():
    assert pipe(3, double, str) == '6'
    assert pipe(3) == 3
    assert pipe('a', str.upper) == 'A'


# ---------- complement ----------

def test_complement():
    def iseven(n):
        return n % 2 == 0

    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


# ---------- juxt ----------

def test_juxt_basic():
    assert juxt(inc, double)(10) == (11, 20)
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_single_callable():
    j = juxt(inc)
    assert j(1) == (2,)


def test_juxt_kwargs():
    def f(a, b=1):
        return a + b

    def g(a, b=1):
        return a * b

    j = juxt(f, g)
    assert j(3, b=2) == (5, 6)


def test_juxt_pickle_roundtrip():
    j = juxt(inc, double)
    state = j.__getstate__()
    assert state == (inc, double)
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2(5) == (6, 10)


# ---------- do ----------

def test_do():
    log = []
    inc_do = compose(inc, do(log.append))
    assert inc_do(1) == 2
    assert inc_do(11) == 12
    assert log == [1, 11]


def test_do_returns_x():
    result = do(lambda x: None, 42)
    assert result == 42


# ---------- flip ----------

def test_flip():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3
    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_flip_with_builtin():
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


# ---------- return_none ----------

def test_return_none():
    assert return_none(ValueError('x')) is None
    assert return_none(None) is None


# ---------- excepts ----------

def test_excepts_basic():
    excepting = excepts(
        ValueError,
        lambda a: [1, 2].index(a),
        lambda _: -1,
    )
    assert excepting(1) == 0
    assert excepting(3) == -1


def test_excepts_multiple_default_handler():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_no_catch():
    excepting = excepts(ValueError, lambda x: x + 1)
    assert excepting(1) == 2
    with pytest.raises(TypeError):
        excepting('a')


def test_excepts_name():
    excepting = excepts(ValueError, lambda a: a)
    assert excepting.__name__ == '<lambda>_excepting_ValueError'


def test_excepts_name_tuple():
    excepting = excepts((ValueError, KeyError), lambda a: a)
    assert excepting.__name__ == '<lambda>_excepting_ValueError_or_KeyError'


def test_excepts_name_fallback():
    class Callable:
        def __call__(self, *a):
            return None

    excepting = excepts(ValueError, Callable())
    assert excepting.__name__ == 'excepting'


def test_excepts_doc():
    excepting = excepts(ValueError, lambda a: a)
    doc = excepting.__doc__
    assert "A wrapper around '<lambda>' that will except:" in doc
    assert 'ValueError' in doc


def test_excepts_doc_class():
    excepting = excepts(ValueError, lambda a: a)
    # class access gives original docstring
    assert excepts.__doc__ is not None
    assert 'A wrapper around a function' in excepts.__doc__


def test_excepts_doc_fallback():
    class Callable:
        def __call__(self, *a):
            return None

    excepting = excepts(ValueError, Callable())
    # func has no __name__, so doc falls back to class doc
    assert excepting.__doc__ == excepts.__doc__


# ---------- introspection helpers ----------

def test_num_required_args():
    def f(x, y, z=3):
        return x + y + z

    def g(*args, **kwargs):
        pass

    assert num_required_args(f) == 2
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

    def g(x, y):
        return x + y

    assert has_keywords(f) is True
    assert has_keywords(g) is False


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

    def h(*args):
        return args

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False
    assert is_arity(1, h) is False


def test_is_arity_none():
    def f(x, y):
        return x + y

    assert is_arity(3, f) is False


def test_introspection_with_sigspec():
    sig = inspect.signature(lambda x, y: x + y)
    assert num_required_args(lambda x, y: x + y, sig) == 2
    assert has_varargs(lambda x, y: x + y, sig) is False
    assert has_keywords(lambda x, y: x + y, sig) is False
    assert is_valid_args(lambda x, y: x + y, (1, 2), {}, sig) is True
    assert is_partial_args(lambda x, y: x + y, (1,), {}, sig) is True
    assert is_arity(2, lambda x, y: x + y, sig) is True


def test_num_required_args_builtin():
    # Builtins without signature fall back to the registry
    assert num_required_args(map) == 2


def test_is_valid_args_builtin():
    assert is_valid_args(map, (), {}) is False
    assert is_partial_args(map, (), {}) is True
