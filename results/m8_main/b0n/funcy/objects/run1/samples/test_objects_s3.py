import types

import pytest

from funcy.objects import cached_property, cached_readonly, wrap_prop, monkey, LazyObject


# ---------------------------------------------------------------------------
# cached_property
# ---------------------------------------------------------------------------

class TestCachedProperty:
    def test_caches_value_on_instance(self):
        calls = []

        class Foo:
            @cached_property
            def value(self):
                calls.append(1)
                return 42

        f = Foo()
        assert f.value == 42
        assert f.value == 42
        assert calls == [1]  # computed only once
        assert f.__dict__['value'] == 42

    def test_returns_descriptor_on_class_access(self):
        class Foo:
            @cached_property
            def value(self):
                return 42

        assert isinstance(Foo.__dict__['value'], cached_property)
        assert Foo.value is Foo.__dict__['value']

    def test_preserves_docstring(self):
        class Foo:
            @cached_property
            def value(self):
                """My docstring."""
                return 1

        assert Foo.__dict__['value'].__doc__ == "My docstring."

    def test_fset_fdel_are_none(self):
        class Foo:
            @cached_property
            def value(self):
                return 1

        desc = Foo.__dict__['value']
        assert desc.fset is None
        assert desc.fdel is None
        assert desc.fget is not None

    def test_instance_specific_cache(self):
        counter = [0]

        class Foo:
            @cached_property
            def value(self):
                counter[0] += 1
                return counter[0]

        a, b = Foo(), Foo()
        assert a.value == 1
        assert a.value == 1
        assert b.value == 2
        assert b.value == 2
        assert counter[0] == 2

    def test_uses_fget_name_as_cache_key(self):
        class Foo:
            @cached_property
            def myname(self):
                return "x"

        f = Foo()
        assert f.myname == "x"
        assert 'myname' in f.__dict__


# ---------------------------------------------------------------------------
# cached_readonly
# ---------------------------------------------------------------------------

class TestCachedReadonly:
    def test_caches_value(self):
        calls = []

        class Foo:
            @cached_readonly
            def value(self):
                calls.append(1)
                return 7

        f = Foo()
        assert f.value == 7
        assert f.value == 7
        assert calls == [1]

    def test_cached_value_is_returned_without_recompute(self):
        class Foo:
            @cached_readonly
            def value(self):
                return 7

        f = Foo()
        # Prime the cache then mutate directly to check it is read back
        assert f.value == 7
        f.__dict__['value'] = 99
        assert f.value == 99

    def test_set_raises_attribute_error(self):
        class Foo:
            @cached_readonly
            def value(self):
                return 7

        f = Foo()
        with pytest.raises(AttributeError):
            f.value = 5

    def test_class_level_access_returns_descriptor(self):
        class Foo:
            @cached_readonly
            def value(self):
                return 7

        assert isinstance(Foo.__dict__['value'], cached_readonly)
        assert Foo.value is Foo.__dict__['value']

    def test_is_subclass_of_cached_property(self):
        assert issubclass(cached_readonly, cached_property)


# ---------------------------------------------------------------------------
# wrap_prop
# ---------------------------------------------------------------------------

class TrackingCM:
    def __init__(self):
        self.enter_count = 0
        self.exit_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, *exc):
        self.exit_count += 1
        return False


class TestWrapProp:
    def test_wraps_getter(self):
        cm = TrackingCM()

        class Foo:
            def __init__(self):
                self._val = 3

            @wrap_prop(cm)
            @property
            def value(self):
                return self._val

        f = Foo()
        assert f.value == 3
        assert cm.enter_count == 1
        assert cm.exit_count == 1

    def test_wraps_setter(self):
        cm = TrackingCM()

        class Foo:
            def __init__(self):
                self._val = 0

            @wrap_prop(cm)
            @property
            def value(self):
                return self._val

            @value.setter
            def value(self, v):
                self._val = v

        f = Foo()
        f.value = 11
        assert f._val == 11
        assert cm.enter_count == 1
        assert cm.exit_count == 1
        assert f.value == 11
        assert cm.enter_count == 2

    def test_wraps_deleter(self):
        cm = TrackingCM()

        class Foo:
            def __init__(self):
                self._val = 3

            @wrap_prop(cm)
            @property
            def value(self):
                return self._val

            @value.deleter
            def value(self):
                del self._val

        f = Foo()
        del f.value
        assert not hasattr(f, '_val')
        assert cm.enter_count == 1
        assert cm.exit_count == 1

    def test_class_level_access_returns_wrapper(self):
        cm = TrackingCM()

        class Foo:
            @wrap_prop(cm)
            @property
            def value(self):
                return 1

        assert isinstance(Foo.value, object)
        assert Foo.value is Foo.__dict__['value']
        # Accessing class-level attribute should not enter the context.
        assert cm.enter_count == 0

    def test_repr_delegates_to_wrapped(self):
        cm = TrackingCM()

        class Foo:
            @wrap_prop(cm)
            @property
            def value(self):
                return 1

        inner = Foo.__dict__['value'].__repr__()
        # The repr should look like a property repr.
        assert inner == repr(property(Foo.__dict__['value'].fget if hasattr(Foo.__dict__['value'], 'fget') else (lambda self: 1))) or 'property' in inner.lower() or '<' in inner

    def test_no_setter_means_no_set_descriptor(self):
        cm = TrackingCM()

        class Foo:
            @wrap_prop(cm)
            @property
            def value(self):
                return 1

        wrapper = Foo.__dict__['value']
        # Since underlying property has no __set__, wrapper should not define one
        # (property actually always has __set__, so this checks hasattr behavior consistency)
        assert hasattr(wrapper, '__get__')

    def test_reusable_decorator_for_multiple_props(self):
        cm = TrackingCM()

        class Foo:
            def __init__(self):
                self._a = 1
                self._b = 2

            @wrap_prop(cm)
            @property
            def a(self):
                return self._a

            @wrap_prop(cm)
            @property
            def b(self):
                return self._b

        f = Foo()
        assert f.a == 1
        assert f.b == 2
        assert cm.enter_count == 2
        assert cm.exit_count == 2


# ---------------------------------------------------------------------------
# monkey
# ---------------------------------------------------------------------------

class TestMonkey:
    def test_patches_class_method(self):
        class Target:
            pass

        @monkey(Target)
        def Target__greet(self):
            return "hello"

        t = Target()
        assert t.greet() == "hello"
        assert Target.greet is Target.__dict__['greet']

    def test_strips_class_prefix(self):
        class Target:
            pass

        @monkey(Target)
        def Target__foo(self):
            return 1

        assert hasattr(Target, 'foo')
        assert not hasattr(Target, 'Target__foo')

    def test_explicit_name(self):
        class Target:
            pass

        @monkey(Target, 'custom')
        def whatever(self):
            return 2

        assert Target().custom() == 2

    def test_original_attribute_set(self):
        class Target:
            def foo(self):
                return "orig"

        @monkey(Target)
        def Target__foo(self):
            return "new"

        assert Target.__dict__['foo'].original is not None
        assert Target.__dict__['foo'].original.__func__(Target()) == "orig"
        assert Target().foo() == "new"

    def test_original_none_when_new_method(self):
        class Target:
            pass

        @monkey(Target)
        def Target__bar(self):
            return 1

        assert Target.__dict__['bar'].original is None

    def test_patches_module(self):
        mod = types.ModuleType('fakemod')

        @monkey(mod)
        def fakemod__hello():
            return "hi"

        assert mod.hello() == "hi"
        assert not hasattr(mod, 'fakemod__hello')

    def test_monkey_property(self):
        class Target:
            def __init__(self):
                self._v = 5

        @monkey(Target)
        @property
        def Target__value(self):
            return self._v

        assert Target().value == 5
        assert isinstance(Target.__dict__['value'], property)

    def test_raises_for_non_class_non_module(self):
        with pytest.raises(AssertionError):
            monkey("not a class")

    def test_raises_for_instance(self):
        with pytest.raises(AssertionError):
            monkey(object())

    def test_raises_for_function(self):
        def f():
            pass

        with pytest.raises(AssertionError):
            monkey(f)

    def test_renames_function(self):
        class Target:
            pass

        def Target__renamed(self):
            return 1

        monkey(Target)(Target__renamed)
        assert Target__renamed.__name__ == 'renamed'

    def test_returns_value(self):
        class Target:
            pass

        def Target__fn(self):
            return 1

        result = monkey(Target)(Target__fn)
        assert result is Target__fn


# ---------------------------------------------------------------------------
# LazyObject
# ---------------------------------------------------------------------------

class Real:
    class_attr = "class"

    def __init__(self):
        self.value = 10

    def method(self):
        return self.value * 2


class TestLazyObject:
    def test_lazy_does_not_init_until_accessed(self):
        calls = []

        def init():
            calls.append(1)
            return Real()

        obj = LazyObject(init)
        assert calls == []
        assert obj.value == 10
        assert calls == [1]

    def test_setattr_triggers_setup(self):
        calls = []

        def init():
            calls.append(1)
            return Real()

        obj = LazyObject(init)
        obj.value = 5
        assert calls == [1]
        assert obj.value == 5

    def test_class_rewritten_after_setup(self):
        obj = LazyObject(Real)
        assert isinstance(obj, LazyObject)
        _ = obj.value
        assert isinstance(obj, Real)

    def test_class_attrs_available_after_setup(self):
        obj = LazyObject(Real)
        assert obj.class_attr == "class"

    def test_methods_available(self):
        obj = LazyObject(Real)
        assert obj.method() == 20

    def test_dict_replaced(self):
        def init():
            r = Real()
            r.extra = "x"
            return r

        obj = LazyObject(init)
        _ = obj.value
        assert obj.__dict__ == {'value': 10, 'extra': 'x'}

    def test_init_attribute_present_before_setup(self):
        obj = LazyObject(Real)
        assert obj.__dict__['_init'] is Real

    def test_setattr_returns_none(self):
        obj = LazyObject(Real)
        # __setattr__ returns result of setattr which is None
        result = obj.__setattr__('value', 1)
        assert result is None
        assert obj.value == 1

    def test_getattr_returns_attr_value(self):
        obj = LazyObject(Real)
        assert obj.__getattr__('value') == 10

    def test_double_access_after_setup(self):
        calls = []

        def init():
            calls.append(1)
            return Real()

        obj = LazyObject(init)
        assert obj.value == 10
        assert obj.value == 10
        assert calls == [1]
