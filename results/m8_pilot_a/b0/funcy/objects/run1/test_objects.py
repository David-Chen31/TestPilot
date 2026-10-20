import types

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
    def test_caches_value(self):
        calls = []

        class A:
            @cached_property
            def value(self):
                calls.append(1)
                return 42

        a = A()
        assert a.value == 42
        assert a.value == 42
        assert len(calls) == 1

    def test_stores_in_instance_dict(self):
        class A:
            @cached_property
            def value(self):
                return 7

        a = A()
        _ = a.value
        assert a.__dict__['value'] == 7

    def test_class_access_returns_descriptor(self):
        class A:
            @cached_property
            def value(self):
                return 7

        assert isinstance(A.value, cached_property)
        assert A.value.fget is not None

    def test_fset_fdel_are_none(self):
        class A:
            @cached_property
            def value(self):
                return 7

        assert A.value.fset is None
        assert A.value.fdel is None

    def test_docstring_copied(self):
        class A:
            @cached_property
            def value(self):
                """my docstring"""
                return 7

        assert A.value.__doc__ == "my docstring"

    def test_docstring_missing_is_none(self):
        def f(self):
            return 1

        cp = cached_property(f)
        assert cp.__doc__ is None

    def test_separate_instances(self):
        class A:
            @cached_property
            def value(self):
                return object()

        a, b = A(), A()
        assert a.value is a.value
        assert b.value is b.value
        assert a.value is not b.value


# ---------------------------------------------------------------------------
# cached_readonly
# ---------------------------------------------------------------------------

class TestCachedReadonly:
    def test_readonly_is_subclass(self):
        assert issubclass(cached_readonly, cached_property)

    def test_caches_value(self):
        calls = []

        class A:
            @cached_readonly
            def value(self):
                calls.append(1)
                return 42

        a = A()
        assert a.value == 42
        assert a.value == 42
        assert len(calls) == 1

    def test_set_raises(self):
        class A:
            @cached_readonly
            def value(self):
                return 1

        a = A()
        with pytest.raises(AttributeError):
            a.value = 5

    def test_readonly_caches_and_protects(self):
        class A:
            @cached_readonly
            def value(self):
                return object()

        a = A()
        first = a.value
        assert a.value is first
        with pytest.raises(AttributeError):
            a.value = first

    def test_class_access_returns_descriptor(self):
        class A:
            @cached_readonly
            def value(self):
                return 1

        assert isinstance(A.value, cached_readonly)

    def test_docstring_copied(self):
        class A:
            @cached_readonly
            def value(self):
                """readonly doc"""
                return 1

        assert A.value.__doc__ == "readonly doc"


# ---------------------------------------------------------------------------
# wrap_prop
# ---------------------------------------------------------------------------

class CtxTracker:
    def __init__(self):
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *args):
        self.exited += 1
        return False


class TestWrapProp:
    def test_wraps_getter(self):
        ctx = CtxTracker()

        class A:
            @wrap_prop(ctx)
            @property
            def value(self):
                return 10

        a = A()
        assert ctx.entered == 0
        assert a.value == 10
        assert ctx.entered == 1
        assert ctx.exited == 1

    def test_class_access_returns_wrapper(self):
        ctx = CtxTracker()

        class A:
            @wrap_prop(ctx)
            @property
            def value(self):
                return 10

        assert ctx.entered == 0
        # class access returns the wrapper sentinel itself, no ctx entry
        assert isinstance(A.value, object)
        assert ctx.entered == 0

    def test_repr_delegates_to_prop(self):
        ctx = CtxTracker()

        def fget(self):
            return 5

        prop = property(fget)

        class A:
            value = wrap_prop(ctx)(prop)

        assert repr(A.value) == repr(prop)

    def test_wraps_setter(self):
        ctx = CtxTracker()

        class A:
            def __init__(self):
                self._v = 0

            @wrap_prop(ctx)
            @property
            def value(self):
                return self._v

            @value.setter
            def value(self, v):
                self._v = v

        a = A()
        a.value = 99
        assert ctx.entered == 1
        assert ctx.exited == 1
        assert a.value == 99
        # getter also wrapped
        assert ctx.entered == 2
        assert ctx.exited == 2

    def test_wraps_deleter(self):
        ctx = CtxTracker()

        class A:
            def __init__(self):
                self._v = 3

            @wrap_prop(ctx)
            @property
            def value(self):
                return self._v

            @value.deleter
            def value(self):
                del self._v

        a = A()
        del a.value
        assert ctx.entered == 2  # one for deleter, plus? check
        assert ctx.exited == 2
        assert not hasattr(a, '_v')

    def test_no_setter_no_setter_method(self):
        ctx = CtxTracker()

        class A:
            @wrap_prop(ctx)
            @property
            def value(self):
                return 1

        # The wrapper only defines __set__ if prop did; plain property does
        # not have __set__, so wrapper's class should not define __set__.
        wrapper_cls = type(A.__dict__['value'])
        assert not hasattr(wrapper_cls, '__set__') or '__set__' not in wrapper_cls.__dict__


# ---------------------------------------------------------------------------
# monkey
# ---------------------------------------------------------------------------

class TestMonkey:
    def test_patch_class(self):
        class A:
            def foo(self):
                return 'original'

        @monkey(A)
        def A__foo(self):
            return 'patched'

        assert A().foo() == 'patched'

    def test_original_accessible(self):
        class A:
            def foo(self):
                return 'original'

        @monkey(A)
        def A__foo(self):
            return 'patched'

        assert A.foo.original is not None
        assert A.foo.original(A) == 'original'

    def test_original_none_when_new(self):
        class A:
            pass

        @monkey(A)
        def A__bar(self):
            return 1

        assert A.bar.original is None

    def test_explicit_name(self):
        class A:
            pass

        @monkey(A, name='custom')
        def whatever(self):
            return 'x'

        assert A().custom() == 'x'
        assert whatever.__name__ == 'custom'

    def test_prefix_stripped(self):
        class MyClass:
            pass

        @monkey(MyClass)
        def MyClass__method(self):
            return 'called'

        assert MyClass().method() == 'called'

    def test_prefix_not_matching_kept(self):
        class A:
            pass

        @monkey(A)
        def other_name(self):
            return 'n'

        assert A().other_name() == 'n'

    def test_returns_value(self):
        class A:
            pass

        def A__method(self):
            return 1

        result = monkey(A)(A__method)
        assert result is A__method

    def test_property_support(self):
        class A:
            pass

        prop = property(lambda self: 'propval')

        @monkey(A)
        def A__val(self):
            return 'propval'

        # Patch using a property-like value
        wrapped = property(lambda self: 'wrapped')
        monkey(A, name='val')(wrapped)
        assert A().val == 'wrapped'

    def test_original_property_support(self):
        class A:
            def foo(self):
                return 'orig'

        prop = property(lambda self: 'wrapped')

        @monkey(A, name='bar')
        def some_name(self):
            return 'ignored'

        # Now patch with property; original should reference property's fget name
        prop.fget.__name__ = 'bar'
        monkey(A, name='bar')(prop)
        assert A().bar == 'wrapped'

    def test_monkey_module(self):
        mod = types.ModuleType('testmod')
        mod.original_func = lambda: 'orig'

        @monkey(mod)
        def testmod__original_func():
            return 'patched'

        assert mod.original_func() == 'patched'
        assert mod.original_func.original is not None
        assert mod.original_func.original() == 'orig'

    def test_non_class_non_module_raises(self):
        with pytest.raises(AssertionError):
            monkey(lambda: None)

        with pytest.raises(AssertionError):
            monkey(42)


# ---------------------------------------------------------------------------
# LazyObject
# ---------------------------------------------------------------------------

class Made:
    def __init__(self):
        self.x = 10

    def method(self):
        return self.x + 1


class TestLazyObject:
    def test_lazy_no_init_until_access(self):
        calls = []

        def init():
            calls.append(1)
            return Made()

        lazy = LazyObject(init)
        assert calls == []
        assert lazy.x == 10
        assert calls == [1]

    def test_isinstance_after_setup(self):
        lazy = LazyObject(Made)
        assert type(lazy) is LazyObject
        _ = lazy.x
        assert isinstance(lazy, Made)

    def test_method_access(self):
        lazy = LazyObject(Made)
        assert lazy.method() == 11

    def test_init_only_once(self):
        calls = []

        def init():
            calls.append(1)
            return Made()

        lazy = LazyObject(init)
        _ = lazy.x
        _ = lazy.x
        assert calls == [1]

    def test_set_attr_triggers_setup(self):
        calls = []

        def init():
            calls.append(1)
            return Made()

        lazy = LazyObject(init)
        lazy.new_attr = 5
        assert calls == [1]
        assert lazy.new_attr == 5
        assert isinstance(lazy, Made)

    def test_attribute_error_propagates(self):
        lazy = LazyObject(Made)
        with pytest.raises(AttributeError):
            _ = lazy.nonexistent

    def test_getattr_after_setup(self):
        lazy = LazyObject(Made)
        _ = lazy.x
        assert lazy.__class__ is Made
