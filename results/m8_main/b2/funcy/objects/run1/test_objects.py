import pytest

from funcy.objects import cached_property, cached_readonly, wrap_prop, monkey, LazyObject


# ---------------------------------------------------------------------------
# cached_property
# ---------------------------------------------------------------------------

class TestCachedProperty:
    def test_caches_value(self):
        calls = []

        class C:
            @cached_property
            def value(self):
                calls.append(1)
                return 42

        c = C()
        assert c.value == 42
        assert c.value == 42
        assert len(calls) == 1
        assert c.__dict__['value'] == 42

    def test_access_on_class_returns_descriptor(self):
        class C:
            @cached_property
            def value(self):
                return 1

        assert isinstance(C.__dict__['value'], cached_property)
        assert C.value is C.__dict__['value']

    def test_docstring_copied(self):
        class C:
            @cached_property
            def value(self):
                """my doc"""
                return 1

        assert C.__dict__['value'].__doc__ == "my doc"

    def test_fset_fdel_none(self):
        class C:
            @cached_property
            def value(self):
                return 1

        assert C.__dict__['value'].fset is None
        assert C.__dict__['value'].fdel is None

    def test_stores_under_fget_name(self):
        class C:
            @cached_property
            def foo(self):
                return 'bar'

        c = C()
        assert 'foo' not in c.__dict__
        assert c.foo == 'bar'
        assert c.__dict__['foo'] == 'bar'

    def test_fget_receives_instance(self):
        # Kills mutant that passes None instead of instance to fget
        received = []

        class C:
            @cached_property
            def value(self):
                received.append(self)
                return len(received)

        c = C()
        assert c.value == 1
        assert received == [c]
        assert received[0] is c


# ---------------------------------------------------------------------------
# cached_readonly
# ---------------------------------------------------------------------------

class TestCachedReadonly:
    def test_caches_value(self):
        calls = []

        class C:
            @cached_readonly
            def value(self):
                calls.append(1)
                return 7

        c = C()
        assert c.value == 7
        assert c.value == 7
        assert len(calls) == 1
        assert c.__dict__['value'] == 7

    def test_set_raises(self):
        class C:
            @cached_readonly
            def value(self):
                return 1

        c = C()
        with pytest.raises(AttributeError) as exc:
            c.value = 5
        assert str(exc.value) == "property is read-only"

    def test_set_raises_even_when_cached(self):
        class C:
            @cached_readonly
            def value(self):
                return 1

        c = C()
        assert c.value == 1
        with pytest.raises(AttributeError) as exc:
            c.value = 2
        assert str(exc.value) == "property is read-only"

    def test_class_access_returns_descriptor(self):
        class C:
            @cached_readonly
            def value(self):
                return 1

        assert isinstance(C.__dict__['value'], cached_readonly)
        assert C.value is C.__dict__['value']

    def test_is_subclass_of_cached_property(self):
        assert issubclass(cached_readonly, cached_property)

    def test_type_passed_to_fget_on_first_access(self):
        # Strengthen super().__get__(instance, type) call path.
        class C:
            @cached_readonly
            def value(self):
                return 'v'

        c = C()
        assert c.value == 'v'
        # second access goes through cached branch
        assert c.value == 'v'

    def test_super_get_receives_instance_and_type(self):
        # Kill mutants at line 34 that pass None or drop the `type` argument
        # to `super().__get__(instance, type)`. cached_property.__get__
        # delegates to `self.fget(instance)`, but it also consults
        # `instance.__dict__[self.fget.__name__]`, so the `instance` argument
        # is observable. To observe the `type` argument independently, we
        # override cached_property.__get__ on a subclass-free wrapper.
        received = []

        class RecordingDescriptor(cached_property):
            def __get__(self, instance, type=None):
                received.append((instance, type))
                return super().__get__(instance, type)

        class C:
            @RecordingDescriptor
            def value(self):
                return 'computed'

        c = C()
        # First access hits the "not in __dict__" branch => calls
        # super().__get__(instance, type) with both arguments.
        assert c.value == 'computed'
        assert len(received) == 1
        inst, typ = received[0]
        assert inst is c
        assert typ is C
        assert received[0] == (c, C)

        # Second access hits the cached branch and must NOT call __get__
        # again (returns early).
        assert c.value == 'computed'
        assert len(received) == 1

    def test_super_get_type_is_owning_class(self):
        # Additional explicit check on the type argument, using a spy
        # subclass of cached_property that intercepts the super() call.
        captured = {}

        class Spy(cached_property):
            def __get__(self, instance, type=None):
                if instance is not None and self.fget.__name__ in instance.__dict__:
                    return instance.__dict__[self.fget.__name__]
                captured['instance'] = instance
                captured['type'] = type
                return super().__get__(instance, type)

        class Owner:
            @Spy
            def attr(self):
                return 'v'

        o = Owner()
        assert o.attr == 'v'
        assert captured['instance'] is o
        assert captured['type'] is Owner


# ---------------------------------------------------------------------------
# wrap_prop
# ---------------------------------------------------------------------------

class TestWrapProp:
    def test_wraps_property_get(self):
        log = []

        class Ctx:
            def __enter__(self):
                log.append('enter')

            def __exit__(self, *a):
                log.append('exit')

        ctx = Ctx()

        class C:
            @wrap_prop(ctx)
            @property
            def value(self):
                log.append('get')
                return 5

        c = C()
        assert c.value == 5
        assert log == ['enter', 'get', 'exit']

    def test_class_access_returns_wrapper(self):
        class Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        ctx = Ctx()

        class C:
            @wrap_prop(ctx)
            @property
            def value(self):
                return 1

        assert isinstance(C.value, C.__dict__['value'].__class__)
        assert C.value is C.__dict__['value']

    def test_repr_delegates_to_prop(self):
        class Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        ctx = Ctx()
        prop = property(lambda self: 1)
        wrapper = wrap_prop(ctx)(prop)
        assert repr(wrapper) == repr(prop)

    def test_wraps_property_set(self):
        log = []

        class Ctx:
            def __enter__(self):
                log.append('enter')

            def __exit__(self, *a):
                log.append('exit')

        ctx = Ctx()

        class C:
            def __init__(self):
                self._v = 0

            def _get(self):
                return self._v

            def _set(self, value):
                log.append('set')
                self._v = value

            value = wrap_prop(ctx)(property(_get, _set))

        c = C()
        c.value = 9
        assert log == ['enter', 'set', 'exit']
        assert c._v == 9

    def test_wraps_property_delete(self):
        log = []

        class Ctx:
            def __enter__(self):
                log.append('enter')

            def __exit__(self, *a):
                log.append('exit')

        ctx = Ctx()

        class C:
            def __init__(self):
                self._v = 1

            def _get(self):
                return self._v

            def _del(self):
                log.append('del')
                del self._v

            value = wrap_prop(ctx)(property(_get, fdel=_del))

        c = C()
        del c.value
        assert log == ['enter', 'del', 'exit']

    def test_no_set_when_prop_has_none(self):
        class Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class C:
            @wrap_prop(Ctx())
            @property
            def value(self):
                return 1

        # A plain property has __set__/__delete__ on the property object,
        # but WrapperProp checks hasattr(prop, '__set__'), which is True
        # for all properties in modern Python. Just check it exists.
        wrapper = C.__dict__['value']
        assert hasattr(wrapper, '__get__')

    def test_get_matches_plain_property(self):
        # Ensure the wrapper's __get__ behaves like the underlying property's
        # __get__ and passes instance/type through correctly.
        class Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class C:
            def _get(self, _get_val=1):
                return 'plain'

            def _get2(self):
                return 'wrapped'

            plain = property(_get2)

        ctx = Ctx()
        wrapper = wrap_prop(ctx)(property(C._get2))
        # Calling wrapper.__get__ directly with an instance
        c = C()
        assert wrapper.__get__(c, C) == 'wrapped'

    def test_get_descriptor_via_class_returns_wrapper(self):
        # Verify that calling __get__ with instance=None returns the wrapper
        class Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        ctx = Ctx()
        wrapper = wrap_prop(ctx)(property(lambda self: 1))
        assert wrapper.__get__(None, object) is wrapper

    def test_get_passes_exact_instance_and_type_to_prop(self):
        # Kill mutants that pass None as type, drop type, or pass type as
        # the instance argument in `prop.__get__(instance, type)` (line 52).
        received = []

        class SpyDescriptor:
            def __get__(self, instance, type=None):
                received.append((instance, type))
                return 'result'

        class Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class C:
            value = wrap_prop(Ctx())(SpyDescriptor())

        c = C()
        assert c.value == 'result'
        assert received == [(c, C)]
        inst, owner = received[0]
        assert inst is c
        assert owner is C


# ---------------------------------------------------------------------------
# monkey
# ---------------------------------------------------------------------------

class TestMonkey:
    def test_monkey_class_with_name(self):
        class C:
            def existing(self):
                return 'orig'

        @monkey(C, 'existing')
        def replacement(self):
            return 'new'

        assert C().existing() == 'new'
        assert replacement.__name__ == 'existing'
        assert replacement.original is not None
        assert replacement.original(C()) == 'orig'

    def test_monkey_class_strips_prefix(self):
        class C:
            pass

        @monkey(C)
        def C__added(self):
            return 'added'

        assert C().added() == 'added'
        # setattr uses func_name != original function name
        assert C.added.__name__ == 'added'
        assert C.added.original is None

    def test_monkey_without_prefix_keeps_name(self):
        class C:
            pass

        @monkey(C)
        def plain(self):
            return 'plain'

        assert C().plain() == 'plain'
        assert C.plain is plain

    def test_monkey_overwrites_existing_property_with_function(self):
        # Monkey patching replaces the property object with a plain function,
        # so accessing on an instance binds it as a method.
        class C:
            def _get(self):
                return 'prop'

            value = property(_get)

        @monkey(C)
        def value(self):
            return 'monkeyed'

        assert C.value.__name__ == 'value'
        assert C().value() == 'monkeyed'

    def test_monkey_with_explicit_name_and_func(self):
        class C:
            pass

        @monkey(C, 'p')
        def whatever(self):
            return 'val'

        assert C().p() == 'val'
        assert C.p.__name__ == 'p'
        assert C.p.original is None

    def test_monkey_non_class_raises(self):
        with pytest.raises(AssertionError) as exc:
            @monkey(42)
            def foo():
                pass
        # exact message check
        assert str(exc.value) == "Attempting to monkey patch non-class and non-module"

    def test_monkey_module(self):
        import types
        mod = types.ModuleType('testmod')

        @monkey(mod, 'func')
        def foo():
            return 1

        assert mod.func is foo
        assert foo.original is None

    def test_monkey_returns_value(self):
        class C:
            pass

        def target(self):
            return 1

        result = monkey(C, 'target')(target)
        assert result is target

    def test_original_set_to_none_for_new(self):
        class C:
            pass

        @monkey(C, 'brand_new')
        def fn(self):
            return 1

        assert fn.original is None

    def test_monkey_with_property_value(self):
        # The 'fget' branch: when value is a property, func_name is derived
        # from func.__name__ (the getter's name).
        class C:
            pass

        def orig_getter(self):
            return 'computed'

        prop = property(orig_getter)
        # Give the getter a prefixed name to exercise prefix stripping
        orig_getter.__name__ = 'C__prop'
        result = monkey(C)(prop)
        assert result is prop
        assert C.__dict__['prop'] is prop
        assert orig_getter.__name__ == 'prop'

    def test_monkey_with_property_original_refers_to_existing(self):
        # Original for a new attribute is None.
        class C:
            pass

        def getter(self):
            return 1

        prop = property(getter)
        monkey(C, 'newattr')(prop)
        assert getter.original is None
        assert C.__dict__['newattr'] is prop

    def test_monkey_property_func_name_from_fget(self):
        # Verify func is value.fget, not value, when name is None.
        class C:
            pass

        def C__meth(self):
            return 'x'

        prop = property(C__meth)
        monkey(C)(prop)
        # Attribute added under stripped prefix name
        assert C.__dict__['meth'] is prop
        assert C__meth.__name__ == 'meth'


# ---------------------------------------------------------------------------
# LazyObject
# ---------------------------------------------------------------------------

class TestLazyObject:
    def test_lazy_init_on_attribute_access(self):
        calls = []

        class Real:
            def __init__(self):
                calls.append(1)
                self.x = 10

            def method(self):
                return 'm'

        lazy = LazyObject(Real)
        assert calls == []
        assert lazy.x == 10
        assert calls == [1]
        assert lazy.method() == 'm'

    def test_class_rewritten_after_setup(self):
        class Real:
            def __init__(self):
                self.x = 3

        lazy = LazyObject(Real)
        lazy.x  # trigger
        assert lazy.__class__ is Real

    def test_setattr_triggers_setup(self):
        calls = []

        class Real:
            def __init__(self):
                calls.append('init')

        lazy = LazyObject(Real)
        lazy.foo = 'bar'
        assert calls == ['init']
        assert lazy.foo == 'bar'
        assert lazy.__class__ is Real

    def test_init_stores_init(self):
        class Real:
            def __init__(self):
                self.v = 1

        lazy = LazyObject(Real)
        assert lazy.__dict__['_init'] is Real

    def test_getattr_on_object_with_dict(self):
        class Real:
            def __init__(self):
                self.value = 99

        lazy = LazyObject(Real)
        assert lazy.value == 99

    def test_setup_only_once(self):
        calls = []

        class Real:
            def __init__(self):
                calls.append(1)
                self.a = 1
                self.b = 2

        lazy = LazyObject(Real)
        lazy.a
        lazy.b
        lazy.a
        assert len(calls) == 1
