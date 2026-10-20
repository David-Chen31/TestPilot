import pytest

from funcy.objects import cached_property, cached_readonly, wrap_prop, monkey, LazyObject


# ---------------------------------------------------------------------------
# cached_property
# ---------------------------------------------------------------------------

class CPBase:
    def __init__(self):
        self.calls = 0

    @cached_property
    def value(self):
        """Docstring for value."""
        self.calls += 1
        return self.calls * 10


def test_cached_property_computes_and_caches():
    obj = CPBase()
    assert obj.value == 10
    assert obj.value == 10
    assert obj.value == 10
    assert obj.calls == 1
    # value stored in instance __dict__
    assert obj.__dict__['value'] == 10


def test_cached_property_distinct_instances():
    a = CPBase()
    b = CPBase()
    assert a.value == 10
    assert b.value == 10
    assert a.value == 10
    assert b.value == 10
    assert a.calls == 1
    assert b.calls == 1


def test_cached_property_class_access_returns_descriptor():
    assert isinstance(CPBase.__dict__['value'], cached_property)
    assert CPBase.value is CPBase.__dict__['value']
    assert CPBase.value.fget.__name__ == 'value'


def test_cached_property_doc():
    assert CPBase.value.__doc__ == "Docstring for value."


def test_cached_property_no_setter_or_deleter():
    assert cached_property.fset is None
    assert cached_property.fdel is None


def test_cached_property_stores_result_in_instance_dict():
    obj = CPBase()
    assert 'value' not in obj.__dict__
    obj.value
    assert 'value' in obj.__dict__
    # Cached value then short-circuits fget
    obj.__dict__['value'] = 999
    assert obj.value == 999
    assert obj.calls == 1


# ---------------------------------------------------------------------------
# cached_readonly
# ---------------------------------------------------------------------------

class CROBase:
    def __init__(self):
        self.calls = 0

    @cached_readonly
    def value(self):
        self.calls += 1
        return self.calls * 5


def test_cached_readonly_caches():
    obj = CROBase()
    assert obj.value == 5
    assert obj.value == 5
    assert obj.calls == 1
    assert obj.__dict__['value'] == 5


def test_cached_readonly_returns_cached_from_dict():
    obj = CROBase()
    obj.__dict__['value'] = 123
    assert obj.value == 123
    assert obj.calls == 0


def test_cached_readonly_set_raises():
    obj = CROBase()
    with pytest.raises(AttributeError):
        obj.value = 42
    assert obj.calls == 0


def test_cached_readonly_set_raises_even_after_access():
    obj = CROBase()
    assert obj.value == 5
    with pytest.raises(AttributeError):
        obj.value = 7
    # value remains unchanged
    assert obj.value == 5


def test_cached_readonly_class_access():
    descriptor = CROBase.__dict__['value']
    assert isinstance(descriptor, cached_readonly)
    assert isinstance(descriptor, cached_property)
    assert CROBase.value is descriptor


# ---------------------------------------------------------------------------
# wrap_prop
# ---------------------------------------------------------------------------

class Ctx:
    def __init__(self):
        self.events = []

    def __enter__(self):
        self.events.append('enter')
        return self

    def __exit__(self, *exc):
        self.events.append('exit')
        return False


class WrappedClass:
    def __init__(self):
        self._value = 10
        self.ctx = Ctx()

    @wrap_prop(Ctx())  # separate ctx captured but not easily inspected, fine
    def prop(self):
        return self._value


def test_wrap_prop_get():
    obj = WrappedClass()
    # access through instance
    assert obj.prop == 10


def test_wrap_prop_class_access_returns_wrapper():
    wrapper = WrappedClass.__dict__['prop']
    assert WrappedClass.prop is wrapper
    # repr delegates
    assert 'property' in repr(wrapper)


def test_wrap_prop_repr():
    prop = property(lambda self: 1)
    wrapper = wrap_prop(Ctx())(prop)
    assert repr(wrapper) == repr(prop)


def test_wrap_prop_enters_and_exits_context():
    ctx = Ctx()

    class A:
        def __init__(self):
            self.x = 1

        getter = wrap_prop(ctx)(property(lambda self: self.x))

    a = A()
    assert a.getter == 1
    assert ctx.events == ['enter', 'exit']


def test_wrap_prop_setter_and_deleter():
    ctx = Ctx()
    store = {}

    class A:
        def __init__(self):
            self._v = 0

        def get(self):
            return self._v

        def set(self, value):
            self._v = value
            store['set'] = value

        def delete(self):
            del self._v
            store['deleted'] = True

        prop = wrap_prop(ctx)(property(get, set, delete))

    a = A()
    a.prop = 99
    assert a._v == 99
    assert store['set'] == 99
    with pytest.raises(AttributeError):
        _ = a.prop
    del a.prop
    assert store['deleted'] is True
    with pytest.raises(AttributeError):
        _ = a._v
    # Each interaction used the context
    assert ctx.events.count('enter') == 3
    assert ctx.events.count('exit') == 3


def test_wrap_prop_readonly_no_setter():
    ctx = Ctx()

    class A:
        prop = wrap_prop(ctx)(property(lambda self: 1))

    a = A()
    assert a.prop == 1
    with pytest.raises(AttributeError):
        a.prop = 5


def test_wrap_prop_no_deleter():
    ctx = Ctx()

    class A:
        prop = wrap_prop(ctx)(property(lambda self: 1, lambda self, v: None))

    a = A()
    del_called = {'v': False}
    with pytest.raises(AttributeError):
        del a.prop
    assert del_called['v'] is False


# ---------------------------------------------------------------------------
# monkey
# ---------------------------------------------------------------------------

def test_monkey_class_basic():
    class Target:
        def existing(self):
            return 'orig'

    @monkey(Target)
    def new_method(self):
        return 'new'

    assert Target().new_method() == 'new'
    assert new_method.__name__ == 'new_method'


def test_monkey_class_prefix_strip():
    class Target:
        pass

    @monkey(Target)
    def Target__prefixed(self):
        return 'hello'

    assert hasattr(Target, 'prefixed')
    assert Target().prefixed() == 'hello'


def test_monkey_explicit_name():
    class Target:
        pass

    @monkey(Target, name='renamed')
    def whatever(self):
        return 'value'

    assert Target().renamed() == 'value'
    assert whatever.__name__ == 'renamed'


def test_monkey_original_attribute_when_overwriting():
    class Target:
        def method(self):
            return 'old'

    @monkey(Target)
    def method(self):
        return 'new'

    assert Target().method() == 'new'
    assert method.original is not None
    assert method.original(Target()) == 'old'


def test_monkey_original_none_when_new():
    class Target:
        pass

    @monkey(Target)
    def fresh(self):
        return 1

    assert fresh.original is None


def test_monkey_returns_value():
    class Target:
        pass

    def fn(self):
        return 'x'

    result = monkey(Target)(fn)
    assert result is fn


def test_monkey_on_module():
    import types
    mod = types.ModuleType('fake_mod')

    @monkey(mod)
    def fake_mod__hello():
        return 'hi'

    assert mod.hello() == 'hi'
    assert mod.hello.__name__ == 'hello'


def test_monkey_on_module_original_none():
    import types
    mod = types.ModuleType('fake_mod2')

    @monkey(mod)
    def fake_mod2__foo():
        return 1

    assert mod.foo.original is None


def test_monkey_property_support():
    class Target:
        def __init__(self):
            self._x = 3

    @monkey(Target)
    @property
    def Target__x(self):
        return self._x

    assert Target().x == 3
    # The returned value is the property, but its fget got patched
    assert Target.x.fget.__name__ == 'x'
    assert Target.x.fget.original is None


def test_monkey_invalid_target_raises():
    with pytest.raises(AssertionError):
        monkey(42)
    with pytest.raises(AssertionError):
        monkey('notaclass')


def test_monkey_invalid_target_instance():
    class Foo:
        pass

    with pytest.raises(AssertionError):
        monkey(Foo())


# ---------------------------------------------------------------------------
# LazyObject
# ---------------------------------------------------------------------------

class Real:
    class_attr = 'class-level'

    def __init__(self):
        self.a = 1
        self.b = 2

    def method(self):
        return self.a + self.b


def test_lazy_object_attribute_access_triggers_setup():
    created = []

    def init():
        created.append(True)
        return Real()

    lazy = LazyObject(init)
    assert created == []
    assert lazy.a == 1
    assert created == [True]
    assert lazy.b == 2
    assert created == [True]


def test_lazy_object_becomes_real_class():
    lazy = LazyObject(Real)
    assert type(lazy) is LazyObject
    assert lazy.a == 1
    assert type(lazy) is Real


def test_lazy_object_method():
    lazy = LazyObject(Real)
    assert lazy.method() == 3
    assert type(lazy) is Real


def test_lazy_object_set_attr_triggers_setup():
    created = []

    def init():
        created.append(True)
        return Real()

    lazy = LazyObject(init)
    assert created == []
    lazy.c = 5
    assert created == [True]
    assert lazy.c == 5
    assert type(lazy) is Real


def test_lazy_object_init_stored():
    lazy = LazyObject(Real)
    assert lazy.__dict__['_init'] is Real


def test_lazy_object_class_attribute():
    lazy = LazyObject(Real)
    assert lazy.class_attr == 'class-level'
    assert type(lazy) is Real


def test_lazy_object_init_called_once():
    calls = []

    def init():
        calls.append(1)
        return Real()

    lazy = LazyObject(init)
    lazy.a
    lazy.b
    lazy.method()
    assert calls == [1]
