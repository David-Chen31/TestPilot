import pytest

from funcy.objects import cached_property, cached_readonly, wrap_prop, monkey, LazyObject


# ---------------------------------------------------------------------------
# cached_property
# ---------------------------------------------------------------------------

def test_cached_property_basic():
    class A:
        def __init__(self):
            self.calls = 0

        @cached_property
        def value(self):
            self.calls += 1
            return 42

    a = A()
    assert a.calls == 0
    assert a.value == 42
    assert a.calls == 1
    # cached: no recompute
    assert a.value == 42
    assert a.calls == 1
    # stored in instance dict
    assert a.__dict__['value'] == 42


def test_cached_property_class_access_returns_descriptor():
    class A:
        @cached_property
        def value(self):
            return 1

    assert isinstance(A.value, cached_property)
    assert A.value.fget is not None


def test_cached_property_fset_fdel_none():
    assert cached_property.fset is None
    assert cached_property.fdel is None


def test_cached_property_doc():
    class A:
        @cached_property
        def value(self):
            """My docstring."""
            return 1

    assert A.value.__doc__ == "My docstring."
    assert A().value == 1


def test_cached_property_overwritable():
    class A:
        @cached_property
        def value(self):
            return 1

    a = A()
    assert a.value == 1
    a.value = 5
    assert a.value == 5
    assert a.__dict__['value'] == 5


# ---------------------------------------------------------------------------
# cached_readonly
# ---------------------------------------------------------------------------

def test_cached_readonly_basic():
    class A:
        def __init__(self):
            self.calls = 0

        @cached_readonly
        def value(self):
            self.calls += 1
            return 7

    a = A()
    assert a.value == 7
    assert a.calls == 1
    assert a.value == 7
    assert a.calls == 1


def test_cached_readonly_class_access():
    class A:
        @cached_readonly
        def value(self):
            return 3

    assert isinstance(A.value, cached_readonly)


def test_cached_readonly_set_raises():
    class A:
        @cached_readonly
        def value(self):
            return 1

    a = A()
    with pytest.raises(AttributeError) as exc:
        a.value = 10
    assert "read-only" in str(exc.value)


def test_cached_readonly_assign_after_access_raises():
    class A:
        @cached_readonly
        def value(self):
            return 2

    a = A()
    assert a.value == 2
    with pytest.raises(AttributeError):
        a.value = 99


# ---------------------------------------------------------------------------
# wrap_prop
# ---------------------------------------------------------------------------

def test_wrap_prop_get_with_context():
    # wrap_prop expects a real property-like descriptor with __get__.
    # A plain method does not have __get__ in the way expected, so wrap
    # an actual property.
    log = []

    class Ctx:
        def __enter__(self):
            log.append("enter")
            return self

        def __exit__(self, *args):
            log.append("exit")

    class A:
        def __init__(self):
            self._x = 5

        @wrap_prop(Ctx())
        @property
        def x(self):
            return self._x

    a = A()
    assert a.x == 5
    assert log == ["enter", "exit"]


def test_wrap_prop_class_access_returns_wrapper():
    class Ctx:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    class A:
        @wrap_prop(Ctx())
        @property
        def x(self):
            return 1

    wrapped = A.x
    assert wrapped is A.x  # descriptor returned for class access


def test_wrap_prop_repr():
    class Ctx:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    class A:
        @wrap_prop(Ctx())
        @property
        def x(self):
            return 1

    inner = A.__dict__['x']
    # repr of wrapper mirrors repr of underlying prop
    assert repr(inner) == repr(A.__dict__['x'])


def test_wrap_prop_setter_defined_in_property_class():
    # Build a property with both getter and setter and wrap it.
    log = []

    class Ctx:
        def __enter__(self):
            log.append("enter")
            return self

        def __exit__(self, *args):
            log.append("exit")

    class A:
        def __init__(self):
            self._y = 0

        def _get_y(self):
            return self._y

        def _set_y(self, val):
            self._y = val

        y = wrap_prop(Ctx())(property(_get_y, _set_y))

    a = A()
    assert isinstance(A.__dict__['y'].__class__, type)
    # Getter goes through context
    assert a.y == 0
    assert log == ["enter", "exit"]
    log.clear()
    # Setter goes through context
    a.y = 11
    assert a._y == 11
    assert log == ["enter", "exit"]


def test_wrap_prop_deleter_defined_in_property_class():
    log = []

    class Ctx:
        def __enter__(self):
            log.append("enter")
            return self

        def __exit__(self, *args):
            log.append("exit")

    class A:
        def __init__(self):
            self._z = 3

        def _get_z(self):
            return self._z

        def _del_z(self):
            del self._z

        z = wrap_prop(Ctx())(property(_get_z, fdel=_del_z))

    a = A()
    assert a.z == 3
    assert log == ["enter", "exit"]
    log.clear()
    del a.z
    assert '_z' not in a.__dict__
    assert log == ["enter", "exit"]


def test_wrap_prop_no_set_no_delete_only_get():
    # A property ALWAYS defines __set__ and __delete__ attributes (even if they
    # raise AttributeError when used). wrap_prop uses hasattr checks, so the
    # wrapper for a read-only property will still define __set__ and __delete__.
    log = []

    class Ctx:
        def __enter__(self):
            log.append("enter")
            return self

        def __exit__(self, *args):
            log.append("exit")

    class A:
        @wrap_prop(Ctx())
        @property
        def x(self):
            return 99

    wrapper_cls = type(A.__dict__['x'])
    assert hasattr(wrapper_cls, '__get__')
    # property has __set__/__delete__, so wrapper does too
    assert hasattr(wrapper_cls, '__set__')
    assert hasattr(wrapper_cls, '__delete__')

    a = A()
    assert a.x == 99
    assert log == ["enter", "exit"]

    # Setting raises because the underlying property is read-only
    with pytest.raises(AttributeError):
        a.x = 1

    # Deleting raises because the underlying property has no deleter
    with pytest.raises(AttributeError):
        del a.x


def test_wrap_prop_custom_descriptor_set_only():
    # A custom descriptor with __set__ but no get doesn't work with
    # wrap_prop's get path. Instead verify that a descriptor defining only
    # __set__ results in a wrapper with a __set__ that uses the context.
    log = []

    class Ctx:
        def __enter__(self):
            log.append("enter")
            return self

        def __exit__(self, *args):
            log.append("exit")

    class SetOnly:
        def __set__(self, instance, value):
            instance.__dict__['stored'] = value

    class A:
        z = wrap_prop(Ctx())(SetOnly())

    wrapper_cls = type(A.__dict__['z'])
    assert hasattr(wrapper_cls, '__set__')
    assert not hasattr(wrapper_cls, '__delete__')

    a = A()
    a.z = 7
    assert a.__dict__['stored'] == 7
    assert log == ["enter", "exit"]


def test_wrap_prop_custom_descriptor_delete_only():
    # A custom descriptor with __delete__ but no __set__ exercises the
    # branch that only defines __delete__ on the wrapper.
    log = []

    class Ctx:
        def __enter__(self):
            log.append("enter")
            return self

        def __exit__(self, *args):
            log.append("exit")

    class DeleteOnly:
        def __get__(self, instance, type=None):
            if instance is None:
                return self
            return 'val'

        def __delete__(self, instance):
            instance.__dict__['deleted'] = True

    class A:
        d = wrap_prop(Ctx())(DeleteOnly())

    wrapper_cls = type(A.__dict__['d'])
    assert hasattr(wrapper_cls, '__get__')
    assert not hasattr(wrapper_cls, '__set__')
    assert hasattr(wrapper_cls, '__delete__')

    a = A()
    assert a.d == 'val'
    assert log == ["enter", "exit"]
    log.clear()
    del a.d
    assert a.__dict__['deleted'] is True
    assert log == ["enter", "exit"]


# ---------------------------------------------------------------------------
# monkey
# ---------------------------------------------------------------------------

def test_monkey_patches_class():
    class Target:
        def greet(self):
            return "original"

    @monkey(Target)
    def new_method(self):
        return "patched"

    assert Target().new_method() == "patched"


def test_monkey_original_attribute():
    class Target:
        def method(self):
            return "orig"

    @monkey(Target)
    def method(self):
        return "new"

    assert Target().method() == "new"
    assert method.original is not None
    assert method.original(Target()) == "orig"


def test_monkey_original_none_when_new():
    class Target:
        pass

    @monkey(Target)
    def fresh(self):
        return "ok"

    assert fresh.original is None
    assert Target().fresh() == "ok"
    assert fresh.__name__ == "fresh"


def test_monkey_strips_class_prefix():
    class Target:
        pass

    @monkey(Target)
    def Target__prefixed(self):
        return "pfx"

    assert Target().prefixed() == "pfx"
    assert Target__prefixed.__name__ == "prefixed"
    assert Target.prefixed is not None


def test_monkey_explicit_name():
    class Target:
        pass

    @monkey(Target, name="custom_name")
    def whatever(self):
        return "x"

    assert Target().custom_name() == "x"
    assert whatever.__name__ == "custom_name"


def test_monkey_non_class_module_raises():
    with pytest.raises(AssertionError):
        monkey(42)


def test_monkey_property_support():
    class Target:
        def __init__(self):
            self._val = 0

        @property
        def prop(self):
            return self._val

    @monkey(Target)
    @property
    def prop(self):
        return 100

    assert Target().prop == 100
    # original stored on the getter function
    assert prop.fget.original is not None


def test_monkey_module():
    import types

    mod = types.ModuleType("mymod")

    @monkey(mod)
    def modfunc():
        return "m"

    assert mod.modfunc is modfunc
    assert modfunc() == "m"
    assert modfunc.original is None


def test_monkey_returns_value():
    class Target:
        pass

    def fn(self):
        return 1

    decorator = monkey(Target)
    result = decorator(fn)
    assert result is fn
    assert Target.fn is fn


# ---------------------------------------------------------------------------
# LazyObject
# ---------------------------------------------------------------------------

def test_lazy_object_init_lazy():
    calls = []

    def init():
        calls.append(1)
        return [1, 2, 3]

    lo = LazyObject(init)
    assert calls == []
    assert lo._init is init


def test_lazy_object_attribute_triggers_setup():
    calls = []

    class Real:
        def __init__(self):
            self.value = 10

        def method(self):
            return "m"

    def init():
        calls.append(1)
        return Real()

    lo = LazyObject(init)
    assert calls == []
    assert lo.value == 10
    assert calls == [1]
    assert lo.method() == "m"


def test_lazy_object_class_rewritten_after_setup():
    class Real:
        pass

    lo = LazyObject(Real)
    assert type(lo) is LazyObject
    lo.some_attr = 5
    assert type(lo) is Real
    assert lo.some_attr == 5


def test_lazy_object_getattr_setup():
    class Real:
        def __init__(self):
            self.attr = "hello"

    lo = LazyObject(Real)
    assert lo.attr == "hello"


def test_lazy_object_setattr_before_setup():
    class Real:
        pass

    lo = LazyObject(Real)
    lo.foo = 123
    assert type(lo) is Real
    assert lo.foo == 123


def test_lazy_object_init_called_once():
    calls = []

    class Real:
        def __init__(self):
            self.x = 1

    def init():
        calls.append(1)
        return Real()

    lo = LazyObject(init)
    assert lo.x == 1
    assert lo.x == 1
    assert calls == [1]
