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
from toolz.utils import no_default


# ---------------------------------------------------------------- identity

def test_identity():
    assert identity(3) == 3
    obj = object()
    assert identity(obj) is obj


# ---------------------------------------------------------------- apply

def test_apply_basic():
    def double(x):
        return 2 * x

    assert apply(double, 5) == 10


def test_apply_kwargs():
    def f(x, y=0):
        return x + y

    assert apply(f, 1, y=10) == 11


def test_apply_no_args():
    with pytest.raises(TypeError):
        apply()


# ---------------------------------------------------------------- threading

def test_thread_first():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_first(1, inc, double) == 4

    def add(x, y):
        return x + y

    def pw(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pw, 2)) == 25


def test_thread_first_no_forms():
    assert thread_first(5) == 5


def test_thread_last():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert thread_last(1, inc, double) == 4

    def add(x, y):
        return x + y

    def pw(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (pw, 2)) == 32


def test_thread_last_map_filter():
    def inc(x):
        return x + 1

    def iseven(x):
        return x % 2 == 0

    result = list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
    assert result == [2, 4]


# ---------------------------------------------------------------- instanceproperty

def test_instanceproperty_basic():
    class MyClass:
        """The class docstring"""

        @instanceproperty(classval=__doc__)
        def __doc__(self):
            return "An object docstring"

        @instanceproperty
        def val(self):
            return 42

    assert MyClass.__doc__ == 'The class docstring'
    assert MyClass.val is None
    obj = MyClass()
    assert obj.__doc__ == 'An object docstring'
    assert obj.val == 42


def test_instanceproperty_lazy():
    @instanceproperty(classval="classval")
    def getter(self):
        return "instanceval"

    class C:
        attr = getter

    assert C.attr == "classval"
    assert C().attr == "instanceval"


def test_instanceproperty_setter_and_deleter():
    @instanceproperty
    def val(self):
        return self._val

    @val.setter
    def val(self, v):
        self._val = v

    @val.deleter
    def val(self):
        del self._val

    class C:
        val = val

    c = C()
    c.val = 10
    assert c.val == 10
    del c.val
    assert not hasattr(c, '_val')


def test_instance_property_reduce():
    def getter(self):
        return 1

    prop = InstanceProperty(fget=getter, classval="cv")
    cls, state = prop.__reduce__()
    assert cls is InstanceProperty
    new = cls(*state)
    assert new.fget is getter
    assert new.classval == "cv"


# ---------------------------------------------------------------- curry

def test_curry_basic():
    def mul(x, y):
        return x * y

    mul = curry(mul)
    double = mul(2)
    assert double(10) == 20


def test_curry_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_curry_no_args():
    with pytest.raises(TypeError):
        curry()


def test_curry_not_callable():
    with pytest.raises(TypeError):
        curry(3)


def test_curry_merges_partial_like():
    def mul(x, y):
        return x * y

    c1 = curry(mul, 2)
    c2 = curry(c1, 3)
    assert c2.func is mul
    assert c2.args == (2, 3)
    assert c2() == 6


def test_curry_merges_keywords():
    def f(x, y, z=0):
        return x + y + z

    c1 = curry(f, 1, z=5)
    c2 = curry(c1, y=2)
    assert c2() == 8


def test_curry_str_repr():
    def f(x):
        return x

    c = curry(f)
    assert str(c) == str(f)
    assert repr(c) == repr(f)


def test_curry_doc_name_module_qualname():
    def f(x):
        """docstring"""
        return x

    c = curry(f)
    assert c.__doc__ == "docstring"
    assert c.__name__ == "f"
    assert c.__module__ == f.__module__
    assert c.__qualname__ == f.__qualname__
    assert c.func_name == "f"


def test_curry_of_callable_without_name():
    class C:
        def __call__(self, x):
            return x

    c = curry(C())
    assert c.__name__ == '<curry>'


def test_curry_hash_eq():
    def f(x, y):
        return x + y

    c1 = curry(f, 1)
    c2 = curry(f, 1)
    c3 = curry(f, 2)
    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)


def test_curry_ne_non_curry():
    def f(x):
        return x

    c = curry(f)
    assert c != 42


def test_curry_bind():
    def f(x, y):
        return x + y

    c = curry(f, 1)
    c2 = c.bind(2)
    assert isinstance(c2, curry)
    assert c2() == 3


def test_curry_call_forces_call():
    def f(x, y):
        return x + y

    c = curry(f, 1)
    with pytest.raises(TypeError):
        c.call()
    assert c.call(2) == 3


def test_curry_genuine_typeerror_raised():
    def f(x):
        raise TypeError("real error")

    c = curry(f)
    with pytest.raises(TypeError, match="real error"):
        c(1)


def test_curry_should_curry_returns_bind():
    # Calling with incomplete args triggers curry via failed call
    def f(x, y, z):
        return x + y + z

    c = curry(f, 1)
    result = c(2)
    assert isinstance(result, curry)
    assert result(3) == 6


def test_curry_with_varargs():
    def f(*args):
        return sum(args)

    c = curry(f, 1)
    assert c(2, 3) == 6


def test_curry_descriptor():
    class C:
        @curry
        def method(self, x, y):
            return self.base + x + y

        base = 100

    assert isinstance(C.method, curry)
    c = C()
    assert c.method(1, 2) == 103


def test_curry_signature():
    def f(x, y, z=3):
        return x + y + z

    c = curry(f, 1)
    sig = c.__signature__
    params = list(sig.parameters.values())
    assert params[0].name == "y"
    assert params[1].name == "z"
    assert params[0].default is no_default


def test_curry_signature_incorrect_args():
    def f(x):
        return x

    c = curry(f, 1, 2)
    with pytest.raises(TypeError):
        c.__signature__


def test_curry_keywords_property():
    def f(x, y):
        return x + y

    c = curry(f, 1, y=2)
    assert c.args == (1,)
    assert c.keywords == {'y': 2}


@pytest.mark.skipif(sys.version_info < (3, 0), reason="py3")
def test_curry_pickle_module_level():
    import toolz.functoolz as ft
    # memoize is a module-level curry, should be picklable by name
    assert isinstance(ft.memoize, curry)


# ---------------------------------------------------------------- memoize

def test_memoize_basic():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    m = memoize(add)
    assert m(1, 2) == 3
    assert m(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_decorator():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]


def test_memoize_initial_cache():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        return x + y

    assert add(1, 2) == 3
    assert add(1, 3) == 4


def test_memoize_custom_key():
    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add(x, y, verbose=False):
        calls.append((x, y, verbose))
        return x + y

    assert add(1, 2, verbose=True) == 3
    assert add(1, 2, verbose=False) == 3
    assert calls == [(1, 2, True)]


def test_memoize_unhashable_args():
    @memoize
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_memoize_preserves_metadata():
    def f(x):
        """mydoc"""
        return x

    m = memoize(f)
    assert m.__name__ == "f"
    assert m.__doc__ == "mydoc"
    assert m.__wrapped__ is f


def test_memoize_unary_key():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(1) == 1
    assert calls == [1]


def test_memoize_keyword_key():
    calls = []

    @memoize
    def f(x, y=0):
        calls.append((x, y))
        return x + y

    assert f(1, y=2) == 3
    assert f(1, y=2) == 3
    assert calls == [(1, 2)]


# ---------------------------------------------------------------- compose

def test_compose_basic():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == '4'


def test_compose_no_args():
    assert compose() is identity


def test_compose_single():
    f = lambda x: x + 1
    assert compose(f) is f


def test_compose_left_basic():
    inc = lambda i: i + 1
    assert compose_left(inc, str)(3) == '4'


def test_compose_left_no_args():
    assert compose_left() is identity


def test_compose_multiple_args():
    def add(x, y):
        return x + y

    assert compose(str, add)(1, 2) == '3'


def test_compose_class():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    c = Compose((double, inc))  # apply inc first, then double
    assert c(3) == 8
    assert c.first is inc
    assert c.funcs == (double,)


def test_compose_doc():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    c = compose(double, inc)
    assert c.__doc__ == 'lambda *args, **kwargs: double(inc(*args, **kwargs))'
    assert Compose.__doc__ is not None


def test_compose_doc_no_names():
    class C:
        def __call__(self, x):
            return x

    c = Compose((C(), C()))
    assert c.__doc__ == 'A composition of functions'


def test_compose_name():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    c = compose(double, inc)
    assert c.__name__ == 'double_of_inc'


def test_compose_name_no_names():
    class C:
        def __call__(self, x):
            return x

    c = Compose((C(), C()))
    assert c.__name__ == 'Compose'


def test_compose_repr():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    c = compose(double, inc)
    assert repr(c) == 'Compose({!r}, {!r})'.format(double, inc)


def test_compose_eq():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert compose(double, inc) == compose(double, inc)
    assert compose(double, inc) != compose(inc, double)
    assert compose(double, inc) != 42


def test_compose_hash():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    assert hash(compose(double, inc)) == hash(compose(double, inc))


def test_compose_get_descriptor():
    def inc(x):
        return x + 1

    def double(x):
        return 2 * x

    class C:
        method = compose(double, inc)

    c = C()
    assert C.method(1) == 4
    assert c.method(1) == 4


def test_compose_signature():
    def inc(x: int) -> int:
        return x + 1

    def to_str(x: int) -> str:
        return str(x)

    c = compose(to_str, inc)
    sig = c.__signature__
    assert sig.return_annotation is str
    assert list(sig.parameters) == ['x']


def test_compose_getstate_setstate():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    c = compose(double, inc)
    state = c.__getstate__()
    c2 = Compose.__new__(Compose)
    c2.__setstate__(state)
    assert c2(1) == 4


def test_compose_wrapped():
    def double(x):
        return 2 * x

    def inc(x):
        return x + 1

    c = compose(double, inc)
    assert c.__wrapped__ is inc


# ---------------------------------------------------------------- pipe

def test_pipe_basic():
    double = lambda i: 2 * i
    assert pipe(3, double, str) == '6'


def test_pipe_no_funcs():
    assert pipe(5) == 5


# ---------------------------------------------------------------- complement

def test_complement():
    def iseven(n):
        return n % 2 == 0

    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


# ---------------------------------------------------------------- juxt

def test_juxt_basic():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)
    assert juxt([inc, double])(10) == (11, 20)


def test_juxt_kwargs():
    def f(x, y=0):
        return x + y

    def g(x, y=0):
        return x - y

    assert juxt(f, g)(5, y=1) == (6, 4)


def test_juxt_getstate_setstate():
    def inc(x):
        return x + 1

    j = juxt(inc)
    state = j.__getstate__()
    j2 = juxt.__new__(juxt)
    j2.__setstate__(state)
    assert j2(1) == (2,)


# ---------------------------------------------------------------- do

def test_do_returns_x():
    log = []
    result = do(log.append, 5)
    assert result == 5
    assert log == [5]


# ---------------------------------------------------------------- flip

def test_flip_basic():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3
    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_flip_isinstance():
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


def test_flip_is_curried():
    assert isinstance(flip, curry)


# ---------------------------------------------------------------- return_none

def test_return_none():
    assert return_none(ValueError()) is None


# ---------------------------------------------------------------- excepts

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


def test_excepts_kwargs():
    def f(x, y=0):
        return x + y

    e = excepts(ValueError, f)
    assert e(1, y=2) == 3


def test_excepts_doc():
    def f(x):
        """f doc"""
        return [1, 2].index(x)

    def handler(e):
        """handler doc"""
        return -1

    e = excepts(ValueError, f, handler)
    doc = e.__doc__
    assert "'f'" in doc
    assert "ValueError" in doc
    assert "'handler'" in doc
    assert "f doc" in doc
    assert "handler doc" in doc


def test_excepts_doc_tuple():
    e = excepts((ValueError, KeyError), lambda x: x)
    assert "(ValueError, KeyError)" in e.__doc__


def test_excepts_doc_fallback_attribute_error():
    # A callable without __name__ triggers AttributeError -> fallback
    class C:
        def __call__(self, x):
            return x

    e = excepts(ValueError, C())
    # __doc__ access falls back to type(self).__doc__
    assert e.__doc__ == excepts.__doc__


def test_excepts_name():
    excepting = excepts(ValueError, lambda a: a)
    assert excepting.__name__ == '<lambda>_excepting_ValueError'


def test_excepts_name_tuple():
    excepting = excepts((ValueError, KeyError), lambda a: a)
    assert excepting.__name__ == '<lambda>_excepting_ValueError_or_KeyError'


def test_excepts_name_fallback():
    class C:
        def __call__(self, x):
            return x

    e = excepts(ValueError, C())
    assert e.__name__ == 'excepting'


# ---------------------------------------------------------------- introspection helpers

def _three(x, y, z=3):
    return x + y + z


def _varargs(*args):
    return args


def _keywords(**kwargs):
    return kwargs


def test_num_required_args():
    assert num_required_args(_three) == 2
    assert num_required_args(_varargs) == 0

    def g(*args, **kwargs):
        pass

    assert num_required_args(g) == 0

    def f(x, y, z=3):
        return x + y + z

    assert num_required_args(f) == 2


def test_num_required_args_positional_only():
    def f(x, y, /, z):
        return x

    assert num_required_args(f) == 3


def test_has_varargs():
    assert has_varargs(_varargs) is True
    assert has_varargs(_keywords) is False

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
    assert has_keywords(_three) is True
    assert has_keywords(_varargs) is False


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


def test_is_arity():
    def f(x):
        return x

    assert is_arity(1, f) is True

    def g(x, y=1):
        return x + y

    assert is_arity(1, g) is False

    def h(x, y):
        return x + y

    assert is_arity(2, h) is True
    assert is_arity(1, h) is False

    def i(*args):
        return args

    assert is_arity(1, i) is False


# ---------------------------------------------------------------- builtins / signatures fallback

def test_builtin_introspection():
    # sum is unary-ish from sigs perspective; just check it doesn't blow up
    assert is_valid_args(sum, ([1, 2],), {}) is True or is_valid_args(
        sum, ([1, 2],), {}) is False
    # More concrete: len requires 1 arg
    assert is_valid_args(len, ([],), {}) is True
    assert is_valid_args(len, (), {}) is False


def test_num_required_args_builtin():
    assert num_required_args(len) in (1, None)


def test_compose_combined_annotations():
    def inc(x: int) -> int:
        return x + 1

    def to_str(s: int) -> str:
        return str(s)

    c = compose(to_str, inc)
    ann = c._combined_annotations()
    assert ann.get('x') is int
    assert ann.get('return') is str


def test_compose_annotations_descriptor():
    def inc(x: int) -> int:
        return x + 1

    def to_str(s: int) -> str:
        return str(s)

    c = compose(to_str, inc)
    # On an instance, __annotations__ resolves to combined dict
    ann = c.__annotations__
    assert ann.get('x') is int
    assert ann.get('return') is str


def test_compose_combined_annotations_ignores_unsignatured():
    c = compose(str)
    # single func: compose returns the func itself; use Compose directly
    cc = Compose((object(), ))
    ann = cc._combined_annotations()
    assert ann == {}


# ---------------------------------------------------------------- curried attribute reuse

def test_curry_of_lambda_doc_none():
    c = curry(lambda x: x)
    assert c.__doc__ is None
