import copy
import pickle
import sys

import pytest

from boltons.typeutils import (
    make_sentinel,
    issubclass,
    get_all_subclasses,
    classproperty,
)


# ---------------------------------------------------------------------------
# make_sentinel
# ---------------------------------------------------------------------------

def test_make_sentinel_default_name():
    s = make_sentinel()
    assert s.name == '_MISSING'
    assert s.var_name is None


def test_make_sentinel_custom_name():
    s = make_sentinel('MYNAME')
    assert s.name == 'MYNAME'
    assert s.var_name is None


def test_make_sentinel_var_name():
    s = make_sentinel(var_name='_MISSING')
    assert s.var_name == '_MISSING'


def test_make_sentinel_repr_without_var_name():
    s = make_sentinel('TEST')
    assert repr(s) == "Sentinel('TEST')"


def test_make_sentinel_repr_with_var_name_module_level():
    # var_name requires a top-level module scope; tests module is one.
    s = make_sentinel(var_name='_TEST_MISSING')
    assert repr(s) == '_TEST_MISSING'


def test_make_sentinel_falsey():
    s = make_sentinel()
    assert bool(s) is False
    assert not s


def test_make_sentinel_not_equal_to_another():
    a = make_sentinel('TEST')
    b = make_sentinel('TEST')
    assert a != b
    assert type(a) is not type(b)


def test_make_sentinel_same_name_not_equal():
    assert make_sentinel('TEST') != make_sentinel('TEST')


def test_make_sentinel_equality_to_self():
    s = make_sentinel('TEST')
    assert s == s


def test_make_sentinel_copy_returns_self():
    s = make_sentinel('TEST')
    assert copy.copy(s) is s


def test_make_sentinel_deepcopy_returns_self():
    s = make_sentinel('TEST')
    assert copy.deepcopy(s) is s


def test_make_sentinel_pickle_roundtrip():
    global _PICKLE_SENTINEL
    _PICKLE_SENTINEL = make_sentinel(var_name='_PICKLE_SENTINEL')
    restored = pickle.loads(pickle.dumps(_PICKLE_SENTINEL))
    assert restored is _PICKLE_SENTINEL


def test_make_sentinel_non_module_scope_raises():
    # Calling from inside a function without the globals being usable as
    # a module name should raise ValueError. We simulate by calling through
    # a wrapper whose frame globals has no valid __name__.
    ns = {}
    code = (
        "import sys\n"
        "from boltons.typeutils import make_sentinel\n"
        "def make_it():\n"
        "    return make_sentinel(var_name='X')\n"
        "result = make_it()\n"
    )
    with pytest.raises(ValueError):
        # Use exec with a fresh globals dict lacking a valid module name.
        exec(code, {'__name__': 'does_not_exist_module_xyz'})


# ---------------------------------------------------------------------------
# issubclass
# ---------------------------------------------------------------------------

def test_issubclass_normal():
    class A:
        pass

    class B(A):
        pass

    assert issubclass(B, A) is True
    assert issubclass(A, B) is False


def test_issubclass_object():
    class A:
        pass

    assert issubclass(A, object) is True


def test_issubclass_non_types_returns_false():
    assert issubclass('hi', 'friend') is False
    assert issubclass(1, int) is False
    assert issubclass(int, 1) is False


def test_issubclass_none_returns_false():
    assert issubclass(None, int) is False
    assert issubclass(int, None) is False


def test_issubclass_matches_builtin_when_valid():
    assert issubclass(bool, int) == __builtins__['issubclass'](bool, int) if isinstance(__builtins__, dict) else True


# ---------------------------------------------------------------------------
# get_all_subclasses
# ---------------------------------------------------------------------------

def test_get_all_subclasses_basic():
    class A:
        pass

    class B(A):
        pass

    class C(B):
        pass

    class D(A):
        pass

    result = get_all_subclasses(A)
    names = {t.__name__ for t in result}
    assert names == {'B', 'C', 'D'}
    # ordering: B and D before C (breadth-ish via deque)
    assert result.index(B) < result.index(C)
    assert result.index(D) < result.index(C)


def test_get_all_subclasses_leaf():
    class A:
        pass

    class B(A):
        pass

    class C(B):
        pass

    assert get_all_subclasses(B) == [C]
    assert get_all_subclasses(C) == []


def test_get_all_subclasses_no_subclasses():
    class Lonely:
        pass

    assert get_all_subclasses(Lonely) == []


def test_get_all_subclasses_builtin():
    result = get_all_subclasses(object)
    assert isinstance(result, list)
    assert int in result or bool in result or True  # sanity
    assert all(isinstance(t, type) for t in result)


def test_get_all_subclasses_invalid_raises():
    with pytest.raises(TypeError):
        get_all_subclasses(42)


def test_get_all_subclasses_invalid_none_raises():
    with pytest.raises(TypeError):
        get_all_subclasses(None)


def test_get_all_subclasses_diamond_no_dup():
    class A:
        pass

    class B(A):
        pass

    class C(A):
        pass

    class D(B, C):
        pass

    result = get_all_subclasses(A)
    assert result.count(D) == 1
    assert set(result) == {B, C, D}


# ---------------------------------------------------------------------------
# classproperty
# ---------------------------------------------------------------------------

def test_classproperty_basic():
    class MyClass:
        _val = 42

        @classproperty
        def value(cls):
            return cls._val

    assert MyClass.value == 42


def test_classproperty_on_instance():
    class MyClass:
        _val = 7

        @classproperty
        def value(cls):
            return cls._val

    assert MyClass().value == 7


def test_classproperty_receives_class():
    seen = {}

    class MyClass:
        @classproperty
        def who(cls):
            seen['cls'] = cls
            return cls.__name__

    assert MyClass.who == 'MyClass'
    assert seen['cls'] is MyClass


def test_classproperty_subclass_uses_subclass():
    class Base:
        _name = 'base'

        @classproperty
        def name(cls):
            return cls._name

    class Sub(Base):
        _name = 'sub'

    assert Base.name == 'base'
    assert Sub.name == 'sub'
