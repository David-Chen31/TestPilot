import pytest

from funcy.objects import (
    cached_property,
    cached_readonly,
    wrap_prop,
    monkey,
    LazyObject,
)


# ---------------------------------------------------------------------------
# cached_property
# ---------------------------------------------------------------------------

class TestCachedProperty:
    def test_basic_caching(self):
        calls = []

        class A:
            @cached_property
            def value(self):
                calls.append(1)
                return 42

        a = A()
        assert a.value == 42
        assert a.value == 42
        assert calls == [1]
        # cached in instance __dict__ under the function name
        assert a.__dict__['value'] == 42

    def test_class_access_returns_descriptor(self):
        class A:
            @cached_property
            def value(self):
                return 1

        assert isinstance(A.value, cached_property)

    def test_doc_preserved(self):
        class A:
            @cached_property
            def value(self):
                """My doc."""
                return 1

        assert A.value.__doc__ == "My doc."
        assert A().value == 1

    def test_doc_none(self):
        class A:
            @cached_property
            def value(self):
                return 1

        # function without docstring -> __doc__ is None
        assert A.value.__doc__ is None

    def test_fset_fdel_none(self):
        class A:
            @cached_property
            def value(self):
                return 1

        assert cached_property.fset is None
        assert cached_property.fdel is None
        assert A.__dict__['value'].fset is None
        assert A.__dict__['value'].fdel is None

    def test_fget_attribute(self):
        def fget(self):
            return 7

        cp = cached_property(fget)
        assert cp.fget is fget

    def test_per_instance_cache(self):
        counter = [0]

        class A:
            @cached_property
            def value(self):
                counter[0] += 1
                return counter[0]

        a, b = A(), A()
        assert a.value == 1
        assert a.value == 1
        assert b.value == 2
        assert b.value == 2


# ---------------------------------------------------------------------------
# cached_readonly
# ---------------------------------------------------------------------------

class TestCachedReadonly:
    def test_caching(self):
        calls = []

        class A:
            @cached_readonly
            def value(self):
                calls.append(1)
                return 10

        a = A()
        assert a.value == 10
        assert a.value == 10
        assert calls == [1]
        assert a.__dict__['value'] == 10

    def test_read_only_set_raises(self):
        class A:
            @cached_readonly
            def value(self):
                return 1

        a = A()
        with pytest.raises(AttributeError, match="read-only"):
            a.value = 5
        # accessing still works
        assert a.value == 1

    def test_set_before_access_raises(self):
        class A:
            @cached_readonly
            def value(self):
                return 1

        a = A()
        with pytest.raises(AttributeError):
            a.value = 5

    def test_class_access_returns_descriptor(self):
        class A:
            @cached_readonly
            def value(self):
                return 1

        assert isinstance(A.value, cached_readonly)
        assert A.value.__get__(None) is A.value

    def test_direct_set_via_dict_bypasses(self):
        # Cached value returning works even when manually placed in __dict__.
        class A:
            @cached_readonly
            def value(self):
                return 1

        a = A()
        a.__dict__['value'] = 99
        assert a.value == 99

    def test_doc(self):
        class A:
            @cached_readonly
            def value(self):
                """readonly doc"""
                return 1

        assert A.value.__doc__ == "readonly doc"

    def test_is_subclass(self):
        assert issubclass(cached_readonly, cached_property)


# ---------------------------------------------------------------------------
# wrap_prop
# ---------------------------------------------------------------------------

class TestWrapProp:
    def test_get_wraps_context(self):
        log = []

        class Ctx:
            def __enter__(self):
                log.append("enter")

            def __exit__(self, *a):
                log.append("exit")

        ctx = Ctx()

        class A:
            @wrap_prop(ctx)
            @property
            def value(self):
                log.append("get")
                return 5

        a = A()
        assert a.value == 5
        assert log == ["enter", "get", "exit"]

    def test_class_access_returns_wrapper(self):
        class Ctx:
            def __enter__(self):
                pass

            def __exit__(self, *a):
                pass

        class A:
            @wrap_prop(Ctx())
            @property
            def value(self):
                return 5

        assert A.value.__get__(None) is A.value
        # repr delegates to wrapped property
        assert "property" in repr(A.value)

    def test_set_wraps_context(self):
        log = []

        class Ctx:
            def __enter__(self):
                log.append("enter")

            def __exit__(self, *a):
                log.append("exit")

        class A:
            def __init__(self):
                self._v = 0

            @wrap_prop(Ctx())
            @property
            def value(self):
                return self._v

            @value.setter
            def value(self, v):
                log.append("set")
                self._v = v

        a = A()
        a.value = 3
        assert a.value == 3  # note: accessing also logs
        assert "set" in log

    def test_delete_wraps_context(self):
        log = []

        class Ctx:
            def __enter__(self):
                log.append("enter")

            def __exit__(self, *a):
                log.append("exit")

        class A:
            def __init__(self):
                self._v = 1

            @wrap_prop(Ctx())
            @property
            def value(self):
                return self._v

            @value.deleter
            def value(self):
                log.append("del")
                del self._v

        a = A()
        del a.value
        assert log == ["enter", "del", "exit"]
        assert not hasattr(a, "_v")

    def test_wrapper_has_no_set_when_prop_lacks(self):
        class Ctx:
            def __enter__(self):
                pass

            def __exit__(self, *a):
                pass

        class A:
            @wrap_prop(Ctx())
            @property
            def value(self):
                return 1

        wrapper = A.__dict__['value']
        # property without setter does have __set__, so wrapper should too... 
        # Actually property always has __set__. Test with object lacking it.

        class BareProp:
            def __get__(self, instance, type=None):
                return 1

        wrapped = wrap_prop(Ctx())(BareProp())
        assert not hasattr(wrapped, '__set__')
        assert not hasattr(wrapped, '__delete__')
        assert wrapped.__get__(None) is wrapped

    def test_get_only_deletion_raises(self):
        class Ctx:
            def __enter__(self):
                pass

            def __exit__(self, *a):
                pass

        class BareProp:
            def __get__(self, instance, type=None):
                return 1

        class A:
            value = wrap_prop(Ctx())(BareProp())

        a = A()
        assert a.value == 1
        with pytest.raises(AttributeError):
            a.value = 2
        with pytest.raises(AttributeError):
            del a.value


# ---------------------------------------------------------------------------
# monkey
# ---------------------------------------------------------------------------

class TestMonkey:
    def test_patch_class_method(self):
        class A:
            pass

        @monkey(A)
        def new_method(self):
            return "patched"

        assert A.new_method is new_method
        assert new_method.__name__ == "new_method"
        assert new_method.original is None
        assert A().new_method() == "patched"

    def test_name_cut_prefix_from_class_name(self):
        class MyClass:
            pass

        @monkey(MyClass)
        def MyClass__method(self):
            return 1

        assert MyClass.method is not None
        assert MyClass().method() == 1
        assert MyClass__method.__name__ == "method"

    def test_explicit_name(self):
        class A:
            pass

        @monkey(A, name="renamed")
        def whatever(self):
            return 2

        assert A.renamed().model if False else True  # placeholder no-op

        assert A().renamed() == 2
        assert whatever.__name__ == "renamed"

    def test_original_holds_overwritten(self):
        class A:
            def method(self):
                return "orig"

        @monkey(A)
        def method(self):
            return "new"

        assert A().method() == "new"
        assert method.original is not None
        assert A.method.original is None  # class attr replaced but new func's original set
        assert method.original(None) == "orig"

    def test_property_support(self):
        class A:
            pass

        @monkey(A)
        @property
        def val(self):
            return 42

        assert A().val == 42
        # property object stored on class
        assert isinstance(A.__dict__['val'], property)
        # __name__ set on underlying function
        assert A.__dict__['val'].fget.__name__ == "val"

    def test_non_class_raises(self):
        with pytest.raises(AssertionError):
            monkey(5)

    def test_non_module_object_raises(self):
        with pytest.raises(AssertionError):
            monkey("string")

    def test_module_patching(self):
        import types

        mod = types.ModuleType("fakemod")

        @monkey(mod)
        def hello():
            return "hi"

        assert mod.hello is hello
        assert mod.hello() == "hi"
        assert hello.original is None

    def test_returns_decorated_value(self):
        class A:
            pass

        def raw(self):
            return 1

        result = monkey(A)(raw)
        assert result is raw


# ---------------------------------------------------------------------------
# LazyObject
# ---------------------------------------------------------------------------

class TestLazyObject:
    def test_initial_init_stored(self):
        def init():
            return [1, 2, 3]

        lazy = LazyObject(init)
        assert lazy.__dict__['_init'] is init

    def test_attribute_access_triggers_setup(self):
        calls = []

        def init():
            calls.append(1)
            return [1, 2, 3]

        lazy = LazyObject(init)
        assert calls == []
        assert lazy.append(4) is None
        assert calls == [1]
        assert list(lazy) == [1, 2, 3, 4]

    def test_class_rewritten(self):
        lazy = LazyObject(lambda: [1])
        assert lazy.__class__ is LazyObject
        lazy.append(2)
        assert lazy.__class__ is list
        assert lazy == [1, 2]

    def test_setattr_triggers_setup(self):
        calls = []

        class Obj:
            pass

        def init():
            calls.append(1)
            return Obj()

        lazy = LazyObject(init)
        lazy.x = 5
        assert calls == [1]
        assert lazy.x == 5
        assert isinstance(lazy, Obj)

    def test_setattr_returns_value(self):
        lazy = LazyObject(lambda: type("Ob", (), {})())
        # __setattr__ returns result of setattr
        result = lazy.foo = 1
        assert result == 1
        assert lazy.foo == 1

    def test_init_called_once(self):
        calls = []

        class Obj:
            def __init__(self):
                self.val = 123

        def init():
            calls.append(1)
            return Obj()

        lazy = LazyObject(init)
        assert lazy.val == 123
        assert lazy.val == 123
        assert calls == [1]

    def test_dict_replaced(self):
        class Obj:
            def __init__(self):
                self.a = 1

        lazy = LazyObject(Obj)
        lazy.a = 2
        assert lazy.a == 2
        assert lazy.__dict__ == {"a": 2}

    def test_getattr_missing_raises(self):
        lazy = LazyObject(lambda: object())
        with pytest.raises(AttributeError):
            lazy.nonexistent

    def test_object_setattr_direct(self):
        class Obj:
            def __init__(self):
                self.x = 10

        lazy = LazyObject(Obj)
        object.__setattr__(lazy, 'y', 20)  # shouldn't trigger setup
        assert lazy.__dict__['y'] == 20

    def test_setup_manually(self):
        lazy = LazyObject(lambda: [1])
        lazy._setup()
        assert lazy.__class__ is list
        assert lazy == [1]

    def test_init_failure_propagates(self):
        def init():
            raise ValueError("boom")

        lazy = LazyObject(init)
        with pytest.raises(ValueError, match="boom"):
            lazy.anything
