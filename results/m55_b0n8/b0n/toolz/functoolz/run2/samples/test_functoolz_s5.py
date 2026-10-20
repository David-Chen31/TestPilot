import pickle
import inspect
import sys
from operator import add, not_

import pytest

from toolz.functoolz import (
    identity, apply, thread_first, thread_last, instanceproperty,
    InstanceProperty, curry, memoize, Compose, compose, compose_left, pipe,
    complement, juxt, do, flip, return_none, excepts, num_required_args,
    has_varargs, has_keywords, is_valid_args, is_partial_args, is_arity,
)
from toolz.utils import no_default


# ---------------------------------------------------------------------------
# identity / apply / thread_* / pipe / do / return_none
# ---------------------------------------------------------------------------

def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3


def test_apply_basic_and_kwargs():
    assert apply(lambda x: 2 * x, 5) == 10
    assert apply(lambda x, y, z=0: x + y + z, 1, 2, z=3) == 6


def test_apply_no_func_raises():
    with pytest.raises(TypeError):
        apply()


def test_apply_map():
    assert tuple(map(apply, [lambda x: x + 1, lambda x: x * 2], [3, 4])) == (4, 8)


def test_thread_first():
    assert thread_first(1, lambda x: x + 1, lambda x: 2 * x) == 4
    assert thread_first(1, (add, 4), (pow, 2)) == 25
    # no forms
    assert thread_first(5) == 5


def test_thread_last():
    assert thread_last(1, lambda x: x + 1, lambda x: 2 * x) == 4
    assert thread_last(1, (add, 4), (pow, 2)) == 32
    assert thread_last(5) == 5


def test_pipe():
    assert pipe(3, lambda i: 2 * i, str) == '6'
    assert pipe(3) == 3


def test_do_returns_x_and_side_effect():
    log = []
    assert do(log.append, 5) == 5
    assert log == [5]


def test_return_none():
    assert return_none(ValueError("x")) is None


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


def test_instanceproperty_is_property():
    @instanceproperty
    def f(self):
        return 1

    assert isinstance(f, InstanceProperty)
    assert isinstance(f, property)


def test_instanceproperty_partial_application():
    # Calling with no fget returns a partial that can later be used as decorator
    factory = instanceproperty()
    assert isinstance(factory, type(instanceproperty))  # functools.partial
    prop = factory(lambda self: 7)
    assert isinstance(prop, InstanceProperty)


def test_instanceproperty_classval_default_none():
    @instanceproperty
    def f(self):
        return 1

    assert f.classval is None


def test_instanceproperty_reduce_roundtrip():
    @instanceproperty(classval="classval")
    def f(self):
        return "instance"

    restored = pickle.loads(pickle.dumps(f))
    assert isinstance(restored, InstanceProperty)
    assert restored.classval == "classval"
    assert restored.fget is f.fget


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
    assert cmul(2, 10) == 20


def test_curry_kwargs():
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


def test_curry_type_error_genuine():
    @curry
    def f(x):
        return x
    # Calling with too many args should raise since nothing can fix it
    with pytest.raises(TypeError):
        f(1, 2)


def test_curry_type_error_inside_body():
    @curry
    def f(x):
        raise TypeError("boom")  # real TypeError from body when args are enough

    with pytest.raises(TypeError, match="boom"):
        f(1)


def test_curry_bind_and_call():
    def f(x, y):
        return x + y

    cf = curry(f)
    bound = cf.bind(1)
    assert isinstance(bound, curry)
    assert bound.call(2) == 3


def test_curry_flatten_partial():
    from functools import partial

    def f(x, y, z):
        return x + y + z

    p = partial(f, 1)
    cp = curry(p, 2)
    assert cp(3) == 6


def test_curry_str_repr():
    def f():
        return 1

    cf = curry(f)
    assert str(cf) == str(f)
    assert repr(cf) == repr(f)


def test_curry_eq_hash():
    def f(x, y):
        return x + y

    a = curry(f, 1)
    b = curry(f, 1)
    c = curry(f, 2)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    # not equal to non-curry
    assert a != f


def test_curry_metadata():
    def f():
        "docstring"
        return 1

    cf = curry(f)
    assert cf.__doc__ == "docstring"
    assert cf.__name__ == f.__name__
    assert cf.__module__ == f.__module__
    assert cf.func_name == f.__name__


def test_curry_func_args_keywords():
    def f(x, y=2):
        return x + y

    cf = curry(f, 1, y=5)
    assert cf.func is f
    assert cf.args == (1,)
    assert cf.keywords == {'y': 5}


def test_curry_signature_partial_applied():
    def f(x, y, z=3):
        return x + y + z

    cf = curry(f, 1)
    sig = cf.__signature__
    params = list(sig.parameters.values())
    # x satisfied; remaining y (no default -> no_default) and z (default 3)
    names = [p.name for p in params]
    assert names == ['y', 'z']
    assert params[0].default is no_default
    assert params[1].default == 3


def test_curry_signature_invalid_args():
    def f(x, y):
        return x + y

    cf = curry(f, 1, 2)  # fully applied, valid
    # But curry with too many args and a signature mismatch
    cf_bad = curry(f, 1, 2, 3)
    with pytest.raises(TypeError):
        cf_bad.__signature__


def test_curry_descriptor():
    class C:
        def __init__(self, mul):
            self.mul = mul

        @curry
        def method(self, x, y):
            return self.mul * x + y

    c = C(2)
    # Accessing on class returns curry
    assert isinstance(C.method, curry)
    bound = c.method(3)
    assert bound(4) == 10


def test_curry_get_descriptor_returns_self_on_class():
    def f(self):
        return 1

    cf = curry(f)
    assert cf.__get__(None, object) is cf


def test_curry_call_method():
    @curry
    def f(x, y):
        return x + y

    part = f(1)  # partial via curry
    assert part.call(2) == 3


def test_curry_varargs_on_error_curries():
    # Function with *args: TypeError inside might be curried anyway
    @curry
    def f(*args):
        if len(args) < 2:
            raise TypeError("need more")
        return sum(args)

    # Calling with 1 arg raises TypeError inside, but curry should return a curry
    result = f(1)
    assert isinstance(result, curry)


def test_curry_pickle_decorated():
    # Decorated at module level style: curried function can be pickled
    import toolz.functoolz as cf_mod

    # Use a known curried function from the module
    restored = pickle.loads(pickle.dumps(cf_mod.flip))
    assert isinstance(restored, curry)


def test_curry_pickle_plain():
    # A curried function of a module-level function
    c = curry(num_required_args, 1)
    restored = pickle.loads(pickle.dumps(c))
    # can't compare curry of functions easily, but behave the same
    assert isinstance(restored, curry)
    assert restored.__name__ == c.__name__


def test_curry_eq_with_non_curry():
    def f():
        return 1

    assert (curry(f) == 3) is False
    assert (curry(f) != 3) is True


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


def test_memoize_with_cache():
    @memoize(cache={(1, 2): 100})
    def add(x, y):
        return x + y

    assert add(1, 2) == 100


def test_memoize_key_function():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3
    assert len(calls) == 1  # kwargs ignored by key


def test_memoize_unhashable_args_raises():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_unary():
    calls = []

    @memoize
    def inc(x):
        calls.append(x)
        return x + 1

    assert inc(1) == 2
    assert inc(1) == 2
    assert calls == [1]


def test_memoize_metadata():
    def f(x):
        "the doc"
        return x

    m = memoize(f)
    assert m.__name__ == 'f'
    assert m.__doc__ == 'the doc'
    assert m.__wrapped__ is f


def test_memoize_no_kwargs_func():
    calls = []

    @memoize
    def f(x, y):
        calls.append((x, y))
        return x + y

    assert f(1, 2) == 3
    assert f(1, 2) == 3
    assert calls == [(1, 2)]


# ---------------------------------------------------------------------------
# compose / compose_left / Compose
# ---------------------------------------------------------------------------

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'
    assert compose(inc, str)(3) == '4'


def test_compose_empty():
    assert compose() is identity


def test_compose_single():
    f = lambda x: x + 1
    assert compose(f) is f


def test_compose_return_type():
    c = compose(str, int)
    assert isinstance(c, Compose)


def test_compose_left_basic():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'
    assert compose_left() is identity


def test_compose_left_single():
    f = lambda x: x
    assert compose_left(f) is f


def test_compose_call_kwargs():
    def f(x, y):
        return x + y

    def g(x):
        return x * 10

    c = compose(g, f)
    assert c(1, y=2) == 30


def test_compose_eq():
    f = lambda x: x
    g = lambda x: x
    c1 = compose(f, g)
    c2 = compose(f, g)
    assert c1 == c2
    assert c1 != compose(g, f)
    assert (c1 == 5) is False
    assert (c1.__eq__(5)) is NotImplemented


def test_compose_hash():
    f = lambda x: x
    g = lambda x: x
    c1 = compose(f, g)
    c2 = compose(f, g)
    assert hash(c1) == hash(c2)


def test_compose_repr():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    r = repr(c)
    assert r.startswith('Compose(')
    assert 'f' in r and 'g' in r


def test_compose_doc_and_name():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    c = compose(inc, double)
    assert c.__name__ == 'inc_of_double'
    assert 'inc(double(' in c.__doc__


def test_compose_doc_missing_name():
    c = compose(lambda x: x, object())
    assert c.__doc__ == 'A composition of functions'
    assert c.__name__ == 'Compose'


def test_compose_signature():
    def f(x: int) -> str:
        return str(x)

    def g(y: str) -> float:
        return float(y)

    c = compose(f, g)
    sig = c.__signature__
    assert sig.return_annotation is float


def test_compose_wrapped():
    def f(x):
        return x

    def g(x):
        return x

    c = compose(f, g)
    assert c.__wrapped__ is f


def test_compose_descriptor():
    class C:
        func = compose(lambda self: self.value + 1, lambda x: x * 2)

        def __init__(self, value):
            self.value = value

    c = C(3)
    # Bound as method: self.func is a MethodType
    assert c.func() == (6 + 1)


def test_compose_get_descriptor_on_class():
    c = compose(lambda x: x, lambda x: x)
    assert c.__get__(None, object) is c


def test_compose_getstate_setstate():
    f = lambda x: x
    g = lambda x: x * 2
    c = compose(f, g)
    state = c.__getstate__()
    new = Compose.__new__(Compose)
    new.__setstate__(state)
    assert new.first is c.first
    assert new.funcs == c.funcs


def test_compose_pickle():
    c = compose(abs, add)
    restored = pickle.loads(pickle.dumps(c))
    assert restored(3, -4) == c(3, -4)


def test_compose_annotations():
    def f(x: int, y: str) -> str:
        return x + y

    def g(z: float) -> bool:
        return bool(z)

    c = compose(f, g)
    ann = c.__annotations__
    assert isinstance(ann, dict)


def test_compose_annotations_class_access_is_dict():
    def f(x: int) -> int:
        return x

    def g(x: int) -> int:
        return x

    c = compose(f, g)
    # Accessing annotations on the class (via type) returns empty dict
    # (the _InstanceAnnotations descriptor with obj=None returns self)
    val = c.__class__.__dict__['__annotations__'].__get__(None, type(c))
    assert isinstance(val, dict)


def test_compose_combined_annotations_nested():
    def f(x: int) -> str:
        return str(x)

    def g(y: str) -> float:
        return float(y)

    def h(z: float) -> bool:
        return bool(z)

    inner = compose(f, g)
    outer = compose(inner, h)
    ann = outer._combined_annotations()
    # parameter from first (int), return from last (bool)
    assert ann.get('x') is int
    assert ann.get('return') is bool


def test_compose_combined_annotations_no_signature():
    # object() has no signature -> contributes nothing
    c = compose(lambda x: x, object())
    ann = c._combined_annotations()
    # The first's annotations come through; the last contributes nothing
    assert 'return' not in ann or ann['return'] is not inspect.Parameter.empty


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------

def test_complement():
    iseven = lambda n: n % 2 == 0
    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True
    assert isodd(3) is not_ is not False or True


# ---------------------------------------------------------------------------
# juxt
# ---------------------------------------------------------------------------

def test_juxt_variadic():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)


def test_juxt_list():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(x, y):
        return x + y

    def g(x, y):
        return x - y

    assert juxt(f, g)(10, y=3) == (13, 7)


def test_juxt_getstate_setstate():
    j = juxt(abs, neg if False else (lambda x: -x))
    state = j.__getstate__()
    new = juxt.__new__(juxt)
    new.__setstate__(state)
    assert new.funcs == j.funcs


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------

def test_flip():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3
    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_flip_builtin():
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


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


def test_excepts_default_handler():
    excepting = excepts(ValueError, lambda a: [1, 2].index(a))
    assert excepting(3) is None


def test_excepts_multiple():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_excepts_doc():
    def f(x):
        "docs for f"
        return x

    def handler(e):
        "docs for handler"
        return None

    e = excepts(ValueError, f, handler)
    doc = e.__doc__
    assert "A wrapper around 'f'" in doc
    assert "ValueError" in doc
    assert "docs for f" in doc
    assert "docs for handler" in doc


def test_excepts_doc_tuple():
    e = excepts((ValueError, KeyError), lambda x: x)
    assert "(ValueError, KeyError)" in e.__doc__


def test_excepts_doc_fallback_attribute_error():
    # func without __name__ -> AttributeError -> class doc
    class Callable:
        def __call__(self, x):
            return x

    e = excepts(ValueError, Callable(), Callable())
    assert e.__doc__ == excepts.__doc__


def test_excepts_name():
    def f(x):
        return x

    e = excepts(ValueError, f)
    assert e.__name__ == 'f_excepting_ValueError'


def test_excepts_name_tuple():
    e = excepts((ValueError, KeyError), lambda x: x)
    assert '__name__' not in e.__name__  # lambdas are '<lambda>'
    assert e.__name__ == '<lambda>_excepting_ValueError_or_KeyError'


def test_excepts_name_fallback():
    class Callable:
        def __call__(self, x):
            return x

    e = excepts(ValueError, Callable(), Callable())
    assert e.__name__ == 'excepting'


def test_excepts_class_doc_on_class_access():
    # classval=__doc__ means accessing on the class returns the class docstring
    assert excepts.__doc__ is not None
    e = excepts(ValueError, lambda x: x)
    # accessing via type returns classval
    assert type(e).__dict__['__doc__'].classval == excepts.__doc__


# ---------------------------------------------------------------------------
# introspectors
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

    def h(*args, **kwargs):
        pass

    assert has_keywords(h) is True


def test_is_valid_args():
    def add_(x, y):
        return x + y

    assert is_valid_args(add_, (1,), {}) is False
    assert is_valid_args(add_, (1, 2), {}) is True
    assert is_valid_args(add_, (1, 2, 3), {}) is False


def test_is_partial_args():
    def add_(x, y):
        return x + y

    assert is_partial_args(add_, (1,), {}) is True
    assert is_partial_args(add_, (1, 2), {}) is True
    assert is_partial_args(add_, (1, 2, 3), {}) is False


def test_is_arity():
    def f(x):
        return x

    assert is_arity(1, f) is True
    assert is_arity(2, f) is False

    def g(x, y=1):
        return x + y

    assert is_arity(1, g) is False


def test_is_arity_varargs():
    def f(*args):
        return args

    assert is_arity(1, f) is False


def test_num_required_args_with_sigspec():
    def f(x, y, z=3):
        return x + y + z

    sig = inspect.signature(f)
    assert num_required_args(f, sig) == 2


def test_is_valid_args_with_sigspec():
    def f(x, y):
        return x + y

    sig = inspect.signature(f)
    assert is_valid_args(f, (1, 2), {}, sig) is True


def test_is_arity_with_sigspec():
    def f(x):
        return x

    sig = inspect.signature(f)
    assert is_arity(1, f, sig) is True


def test_builtin_signatures():
    # map and isinstance have registered signatures in _signatures
    assert is_partial_args(map, (), {}) in (True, None)
    assert is_valid_args(map, (), {}) is False
