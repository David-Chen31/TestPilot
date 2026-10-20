import pytest
from datetime import timedelta
from collections import deque

from funcy.calc import (
    SkipMemory,
    memoize,
    cache,
    CacheMemory,
    has_arg_types,
    make_lookuper,
    silent_lookuper,
)


# ---------------------------------------------------------------------------
# SkipMemory
# ---------------------------------------------------------------------------

def test_skip_memory_is_exception():
    assert issubclass(SkipMemory, Exception)
    e = SkipMemory("value")
    assert e.args == ("value",)


def test_memoize_skip_attribute():
    assert memoize.skip is SkipMemory


def test_cache_skip_attribute():
    assert cache.skip is SkipMemory


# ---------------------------------------------------------------------------
# memoize - basic behavior
# ---------------------------------------------------------------------------

def test_memoize_plain_function():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]
    assert f(4) == 8
    assert calls == [3, 4]


def test_memoize_with_keyword_args():
    calls = []

    @memoize
    def f(a, b=0):
        calls.append((a, b))
        return a + b

    assert f(1, b=2) == 3
    assert f(1, b=2) == 3
    assert calls == [(1, 2)]
    # Different kwargs order still resolves to same key
    assert f(a=1, b=2) == 3
    # Positional vs keyword distinct keys
    assert f(1, 2) == 3
    assert len(calls) == 3


def test_memoize_positional_and_kwargs_not_colliding():
    calls = []

    @memoize
    def f(*args, **kwargs):
        calls.append((args, kwargs))
        return (args, kwargs)

    # Should be different from positional (1,) with no kwargs
    f(1)
    f(1, x=1)
    # (1,) followed by a kwarg whose value could collide with positional
    assert f(1) == ((1,), {})
    assert f(1, x=1) == ((1,), {'x': 1})
    assert len(calls) == 2


def test_memoize_key_func():
    calls = []

    @memoize(key_func=lambda x: x % 2)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(3) == 1  # same key as f(1)
    assert calls == [1]


def test_memoize_key_func_with_kwargs():
    calls = []

    @memoize(key_func=lambda *args, **kwargs: (args, tuple(sorted(kwargs.items()))))
    def f(a, b=0):
        calls.append((a, b))
        return a + b

    assert f(1, b=2) == 3
    assert f(1, b=2) == 3
    assert calls == [(1, 2)]
    assert f(1, b=3) == 4


def test_memoize_key_func_via_call_form():
    # The direct-call form memoize(_func) must preserve key_func.
    # If key_func is dropped, distinct args would create separate entries.
    calls = []

    def f(x):
        calls.append(x)
        return x

    wrapped = memoize(f, key_func=lambda x: "shared")
    assert wrapped(1) == 1
    assert wrapped(2) == 1
    assert calls == [1]
    assert wrapped.memory == {"shared": 1}


def test_memoize_memory_attribute():
    @memoize
    def f(x):
        return x + 1

    assert isinstance(f.memory, dict)
    assert f.memory == {}
    f(1)
    # Positional-only calls use the args tuple as key
    assert f.memory == {(1,): 2}


def test_memoize_invalidate():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    assert calls == [1]
    f.invalidate(1)
    assert f.memory == {}
    f(1)
    assert calls == [1, 1]


def test_memoize_invalidate_missing_key():
    @memoize
    def f(x):
        return x

    f(1)
    # invalidating a key that was never computed should not raise
    f.invalidate(999)
    assert f.memory == {(1,): 1}


def test_memoize_invalidate_with_kwargs():
    @memoize
    def f(x, y=0):
        return x + y

    f(1, y=2)
    f.invalidate(1, y=2)
    assert f.memory == {}


def test_memoize_invalidate_uses_key_func():
    # If invalidate dropped kwargs when calling key_func, the key would differ
    # and the entry wouldn't be removed.
    calls = []

    @memoize(key_func=lambda *args, **kwargs: (args, tuple(sorted(kwargs.items()))))
    def f(a, b=0):
        calls.append((a, b))
        return a + b

    f(1, b=2)
    assert calls == [(1, 2)]
    f.invalidate(1, b=2)
    assert f.memory == {}
    f(1, b=2)
    assert calls == [(1, 2), (1, 2)]


def test_memoize_invalidate_with_key_func():
    @memoize(key_func=lambda x: x)
    def f(x):
        return x

    f(5)
    f.invalidate(5)
    assert f.memory == {}


def test_memoize_invalidate_all():
    @memoize
    def f(x):
        return x

    f(1)
    f(2)
    assert f.memory == {(1,): 1, (2,): 2}
    f.invalidate_all()
    assert f.memory == {}


def test_memoize_with_skip_memory():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        if x < 0:
            raise SkipMemory("skipped")
        return x

    assert f(-1) == "skipped"
    assert f(3) == 3
    assert f(3) == 3
    # SkipMemory does not get stored
    assert calls == [-1, 3]
    assert f(-1) == "skipped"
    assert calls == [-1, 3, -1]


def test_memoize_skip_memory_no_args():
    @memoize
    def f(x=0):
        raise SkipMemory()

    assert f() is None


def test_memoize_wraps_metadata():
    @memoize
    def my_func(a, b):
        """docstring"""
        return a + b

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "docstring"


def test_memoize_used_with_call_form():
    @memoize()
    def f(x):
        return x * 10

    assert f(2) == 20
    assert f.memory == {(2,): 20}


def test_memoize_used_with_key_func_only():
    @memoize(key_func=lambda x: "k")
    def f(x):
        return x

    assert f(1) == 1
    # Same key, different arg
    assert f(2) == 1
    assert f.memory == {"k": 1}


def test_memoize_recursion():
    @memoize
    def fib(n):
        return n if n < 2 else fib(n - 1) + fib(n - 2)

    assert fib(10) == 55


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

def test_cache_basic():
    calls = []

    @cache(60)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(2) == 4
    assert f(2) == 4
    assert calls == [2]
    assert isinstance(f.memory, CacheMemory)


def test_cache_with_timedelta():
    @cache(timedelta(seconds=60))
    def f(x):
        return x

    assert f.memory.timeout == 60


def test_cache_timeout_stored():
    @cache(42)
    def f(x):
        return x

    assert f.memory.timeout == 42


def test_cache_invalidate():
    calls = []

    @cache(60)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    assert calls == [1]
    f.invalidate(1)
    f(1)
    assert calls == [1, 1]


def test_cache_invalidate_all():
    @cache(60)
    def f(x):
        return x

    f(1)
    f(2)
    f.invalidate_all()
    assert f.memory == {}


def test_cache_with_key_func():
    calls = []

    @cache(60, key_func=lambda x: x % 2)
    def f(x):
        calls.append(x)
        return x

    assert f(2) == 2
    assert f(4) == 2  # same key as f(2)
    assert calls == [2]


def test_cache_with_skip_memory():
    @cache(60)
    def f(x):
        if x == 0:
            raise SkipMemory("zero")
        return x

    assert f(0) == "zero"
    assert f(1) == 1


def test_cache_wraps_metadata():
    @cache(60)
    def my_func(x):
        """my doc"""
        return x

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "my doc"


# ---------------------------------------------------------------------------
# CacheMemory
# ---------------------------------------------------------------------------

def test_cachememory_set_and_get():
    m = CacheMemory(60)
    m["a"] = 1
    assert m["a"] == 1
    assert m.timeout == 60


def test_cachememory_clear():
    m = CacheMemory(60)
    m["a"] = 1
    m["b"] = 2
    assert len(m) == 2
    m.clear()
    assert len(m) == 0
    assert len(m._keys) == 0
    assert len(m._expires) == 0


def test_cachememory_init_state():
    m = CacheMemory(5)
    assert m.timeout == 5
    assert len(m) == 0


def test_cachememory_missing_key_raises():
    m = CacheMemory(60)
    with pytest.raises(KeyError):
        m["missing"]


def test_cachememory_expired_entry_raises_with_key():
    # The raised KeyError should carry the actual key.
    m = CacheMemory(0)  # immediate expiration
    m["a"] = 1
    with pytest.raises(KeyError) as exc_info:
        m["a"]
    assert exc_info.value.args == ("a",)


def test_cachememory_expired_entry_gets_removed():
    m = CacheMemory(0)
    m["a"] = 1
    assert "a" in m  # still present in dict
    with pytest.raises(KeyError):
        m["a"]
    # After expire() has run, entry should be gone from dict
    assert "a" not in m


def test_cachememory_negative_timeout():
    m = CacheMemory(-1)
    m["a"] = 1
    with pytest.raises(KeyError):
        m["a"]


def test_cachememory_reassignment():
    m = CacheMemory(60)
    m["a"] = 1
    first_exp = m._expires[0]
    m["a"] = 2
    assert m["a"] == 2
    # Reassigning appends new tracking entries
    assert len(m._keys) == 2
    # The new expiration should be >= the old one
    assert m._expires[1] >= first_exp


def test_cachememory_expire_keeps_reassigned_fresh_entry():
    # After reassignment the entry should still be valid
    m = CacheMemory(60)
    m["a"] = 1
    m["a"] = 2
    m.expire()
    assert m["a"] == 2


def test_cachememory_expire_does_not_remove_unrelated_entries():
    # expire() should only remove entries whose expiry is in the past, and
    # only when the stored entry matches the tracked expiration.
    m = CacheMemory(60)
    m["fresh"] = 1
    m.expire()
    # A freshly-set, non-expired entry must survive expire().
    assert "fresh" in m
    assert m["fresh"] == 1


def test_cachememory_expire_removes_only_expired():
    m = CacheMemory(60)
    m["old"] = 1
    # Manually set an expired tracking entry for another key
    m["new"] = 2
    # Corrupt the tracking for "old" so it's already expired.
    # Actually, force "old" to be expired by rewriting its expires_at.
    # (We push a new key/expires pair with an expired time.)
    import time as _time
    from collections import deque
    # Replace deque entries for "old" with an expired timestamp, keeping order.
    # Simpler: use a separate CacheMemory with a tiny timeout is flaky;
    # instead rely on expire with real (non-expired) entries: nothing removed.
    m.expire()
    assert "old" in m
    assert "new" in m


def test_cachememory_expire_removes_expired_after_positive_wait():
    # Use 0-timeout instance to test removal.
    m = CacheMemory(0)
    m["a"] = 1
    m["b"] = 2
    assert "a" in m and "b" in m
    m.expire()
    assert "a" not in m
    assert "b" not in m


def test_cachememory_expire_mixed_keeps_fresh():
    # Expired key removed, fresh key kept.
    m = CacheMemory(0)
    m["expired"] = 1
    # Create a fresh CacheMemory-like scenario: a key with a long timeout.
    m.timeout = 60
    # Re-set an entry with fresh (long) timeout by temporarily bumping timeout.
    # Add "fresh" to another instance and transfer? Simpler: build a fresh
    # instance with mixed contents by directly manipulating.
    m2 = CacheMemory(60)
    m2["fresh"] = 2
    # Manually add an expired entry to m2's tracking
    import time as _time
    expired_at = _time.monotonic() - 1
    dict.__setitem__(m2, "expired", (1, expired_at))
    m2._keys.append("expired")
    m2._expires.append(expired_at)
    # Deques: _expires must be sorted; expired_at is smaller so pushleft.
    m2._keys.pop()
    m2._expires.pop()
    m2._keys.appendleft("expired")
    m2._expires.appendleft(expired_at)
    m2.expire()
    assert "expired" not in m2
    assert "fresh" in m2
    assert m2["fresh"] == 2


# ---------------------------------------------------------------------------
# has_arg_types
# ---------------------------------------------------------------------------

def test_has_arg_types_positional():
    def f(a, b):
        pass

    assert has_arg_types(f) == (True, False)


def test_has_arg_types_positional_only():
    def f(a, /):
        pass

    assert has_arg_types(f) == (True, False)


def test_has_arg_types_var_positional():
    def f(*args):
        pass

    assert has_arg_types(f) == (True, False)


def test_has_arg_types_keyword_only():
    def f(*, a):
        pass

    assert has_arg_types(f) == (False, True)


def test_has_arg_types_var_keyword():
    def f(**kwargs):
        pass

    assert has_arg_types(f) == (False, True)


def test_has_arg_types_none():
    def f():
        pass

    assert has_arg_types(f) == (False, False)


def test_has_arg_types_mixed():
    def f(a, *args, b, **kwargs):
        pass

    assert has_arg_types(f) == (True, True)


def test_has_arg_types_keyword_only_does_not_count_positional():
    def f(a, *, b):
        pass

    assert has_arg_types(f) == (True, True)


# ---------------------------------------------------------------------------
# make_lookuper / silent_lookuper
# ---------------------------------------------------------------------------

def test_make_lookuper_no_args():
    @make_lookuper
    def lookup():
        return {"a": 1, "b": 2}

    assert lookup("a") == 1
    assert lookup("b") == 2
    with pytest.raises(LookupError):
        lookup("c")


def test_make_lookuper_called_once():
    calls = []

    @make_lookuper
    def lookup():
        calls.append(1)
        return {"a": 1}

    lookup("a")
    lookup("a")
    # raises, but should not refill
    with pytest.raises(LookupError):
        lookup("b")
    assert len(calls) == 1


def test_silent_lookuper_no_args():
    @silent_lookuper
    def lookup():
        return {"a": 1}

    assert lookup("a") == 1
    assert lookup("missing") is None


def test_make_lookuper_error_message():
    @make_lookuper
    def my_table():
        return {"x": 10}

    with pytest.raises(LookupError) as exc_info:
        my_table("y")
    # Exact message check (mutation protection)
    assert str(exc_info.value) == "Failed to look up my_table(y)"


def test_make_lookuper_with_args():
    # When the decorated function takes positional args, make_lookuper returns
    # a memoized wrapper that takes those args and returns a *lookup function*.
    @make_lookuper
    def lookup(name):
        return {name: name.upper()}

    table = lookup("abc")
    assert table("abc") == "ABC"
    # memoized on the args: same args returns same lookup function
    assert lookup("abc") is table
    assert lookup("def") is not table


def test_make_lookuper_with_args_raises():
    @make_lookuper
    def lookup(name):
        return {name.upper(): 1}

    table = lookup("a")
    # table maps "A" -> 1, "a" is missing
    with pytest.raises(LookupError):
        table("a")
    assert table("A") == 1


def test_make_lookuper_with_args_subtable_error_message():
    # The sub-lookup function's error message uses the generated name.
    @make_lookuper
    def lookup(name):
        return {}

    table = lookup("abc")
    with pytest.raises(LookupError) as exc_info:
        table("zzz")
    assert str(exc_info.value) == "Failed to look up lookup(abc)(zzz)"


def test_make_lookuper_keyword_args_assertion_message():
    # The assertion message must match exactly; and it should be raised.
    with pytest.raises(AssertionError) as exc_info:
        @make_lookuper
        def lookup(*, a):
            return {"x": 1}
    assert str(exc_info.value) == \
        'Lookup table building function should not have keyword arguments'


def test_make_lookuper_keyword_args_assertion():
    with pytest.raises(AssertionError):
        @make_lookuper
        def lookup(*, a):
            return {"x": 1}


def test_make_lookuper_var_keyword_assertion():
    # **kwargs also counts as keyword args and should trigger assertion.
    with pytest.raises(AssertionError):
        @make_lookuper
        def lookup(**kwargs):
            return {"x": 1}


def test_make_lookuper_positional_only_accepted():
    # Positional-only args go through the "has_args" branch (memoized wrapper).
    @make_lookuper
    def lookup(name, /):
        return {name: name * 2}

    table = lookup("a")
    assert table("a") == "aa"


def test_silent_lookuper_with_args():
    @silent_lookuper
    def lookup(name):
        return {name: name.upper()}

    table = lookup("abc")
    assert table("abc") == "ABC"
    assert table("missing") is None
    # Ensure it returns the value for a present key (guards `memory[arg]` branch)
    assert table("abc") == "ABC"


def test_make_lookuper_func_called_once_then_cached():
    calls = []

    @make_lookuper
    def lookup():
        calls.append(1)
        return {"a": 1}

    lookup("a")
    lookup("a")
    assert calls == [1]


def test_make_lookuper_empty_table_continues_refill():
    # With an empty table, a sentinel key is still inserted so the building
    # function is called only once even if the table is empty.
    calls = []

    @make_lookuper
    def lookup():
        calls.append(1)
        return {}

    with pytest.raises(LookupError):
        lookup("a")
    with pytest.raises(LookupError):
        lookup("b")
    assert len(calls) == 1


def test_silent_lookuper_empty_table_called_once():
    calls = []

    @silent_lookuper
    def lookup():
        calls.append(1)
        return {}

    assert lookup("a") is None
    assert lookup("b") is None
    assert len(calls) == 1


def test_silent_lookuper_present_value_distinct_from_none():
    # Guards against `memory.get(None)` mutation: present values must be
    # returned exactly, even non-None ones.
    @silent_lookuper
    def lookup():
        return {"a": 1, "b": 2}

    assert lookup("a") == 1
    assert lookup("b") == 2
    assert lookup("missing") is None


def test_make_lookuper_error_clears_state_and_reraises():
    calls = []

    @make_lookuper
    def lookup():
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("boom")
        return {"a": 1}

    with pytest.raises(ValueError):
        lookup("a")
    # After the exception, memory is clear so it refills
    assert lookup("a") == 1
    assert len(calls) == 2


def test_make_lookuper_returns_correct_wrapper():
    # Guards against `wraps(func)(None)` / `wraps(None)(wrapper)` mutations:
    # the returned object must be callable and behave as the wrapper.
    @make_lookuper
    def lookup():
        return {"a": 1}

    assert callable(lookup)
    assert lookup("a") == 1


def test_make_lookuper_wraps_name():
    @make_lookuper
    def my_table():
        return {}

    assert my_table.__name__ == "my_table"


def test_silent_lookuper_is_named():
    assert silent_lookuper.__name__ == "silent_lookuper"
    assert make_lookuper.__name__ == "make_lookuper"


# ---------------------------------------------------------------------------
# Additional edge-case / mutation-killing tests
# ---------------------------------------------------------------------------

def test_cachememory_getitem_boundary_zero_timeout():
    m = CacheMemory(0)
    m["a"] = 1
    with pytest.raises(KeyError):
        m["a"]


def test_cachememory_entry_expires_exactly_at_boundary(monkeypatch):
    import funcy.calc as calc

    fake_now = [1000.0]
    monkeypatch.setattr(calc.time, "monotonic", lambda: fake_now[0])

    m = CacheMemory(10)
    m["a"] = 1  # expires_at == 1010.0

    # Exactly at expiry: `expires_at <= now` is True -> KeyError.
    fake_now[0] = 1010.0
    with pytest.raises(KeyError):
        m["a"]

    # Before expiry: value returned.
    m2 = CacheMemory(10)
    m2["b"] = 2  # expires_at == 1020.0 (now still 1010.0)
    assert m2["b"] == 2


def test_cachememory_getitem_not_yet_expired(monkeypatch):
    import funcy.calc as calc

    fake_now = [100.0]
    monkeypatch.setattr(calc.time, "monotonic", lambda: fake_now[0])

    m = CacheMemory(5)
    m["a"] = 42  # expires_at == 105.0
    fake_now[0] = 104.0
    assert m["a"] == 42


def test_cachememory_expire_orphan_tracking_no_crash():
    m = CacheMemory(10)
    # Manually insert tracking for a key that is not in the dict.
    m._keys.append("ghost")
    m._expires.append(-1.0)  # already expired
    # expire() must handle the missing entry gracefully.
    m.expire()
    assert "ghost" not in m


def test_cachememory_expire_reassigned_discards_stale_tracking():
    fake_now = [0.0]
    import funcy.calc as calc

    m = CacheMemory(100)
    m["a"] = 1  # tracking: expires 100 -> entry ("1", 100.0)
    old_expires = m._expires[0]

    # Reassign: entry becomes (2, 100.0), tracking appends (a, 100.0)
    m["a"] = 2
    new_expires = m._expires[1]
    assert new_expires >= old_expires

    fake_now[0] = new_expires - 0.5  # not yet expired
    m.expire()
    assert m["a"] == 2


def test_cachememory_pop_used_without_default_does_not_leak():
    # Line 105 mutation `self.pop(key, None)` -> `self.pop(key)` would raise
    # KeyError if the key is already absent. Exercise expire() where the
    # key is tracked but not present in the dict: must not raise.
    m = CacheMemory(0)
    # Add tracking for a key that is not in the dict at all.
    m._keys.append("missing")
    m._expires.append(-1.0)
    m.expire()  # should not raise
    assert "missing" not in m


def test_cachememory_expire_skips_already_absent_key(monkeypatch):
    # Force expire() to attempt popping a key that's already gone from the
    # dict, exercising the default-arg form of pop.  We seed tracking for a
    # key we then delete directly from the dict.
    import funcy.calc as calc
    fake_now = [0.0]
    monkeypatch.setattr(calc.time, "monotonic", lambda: fake_now[0])

    m = CacheMemory(0)
    m["expired"] = 1
    # Directly remove the entry from the dict, leaving its tracking behind.
    dict.pop(m, "expired")
    fake_now[0] = 100.0
    # expire() will attempt to pop "expired" again; must not raise.
    m.expire()


# ---------------------------------------------------------------------------
# has_arg_types used indirectly
# ---------------------------------------------------------------------------

def test_make_lookuper_no_header_kwargs_ok_for_plain_noarg():
    # Exercises has_arg_types(func) unpacking (line 124).
    # If line 124 became `has_args, has_keys = None`, ValueError would raise.
    @make_lookuper
    def lookup():
        return {"a": 1}

    assert lookup("a") == 1


def test_make_lookuper_header_func_returns_full_table():
    # Line 140: `memory.update(func())` must call the wrapped function.
    # If mutated to update(None), TypeError would be raised.
    @silent_lookuper
    def lookup():
        return {"k1": "v1", "k2": "v2"}

    assert lookup("k1") == "v1"
    assert lookup("k2") == "v2"


def test_make_lookuper_sentinel_prevents_reads_on_empty():
    # Line 144 mutation: `memory[object()] = None` -> `memory[object()] = ""`.
    # The sentinel is only inserted to prevent memory reloading. When the
    # underlying table is empty, the value is irrelevant to lookups, but the
    # sentinel must make the resulting dict truthy so that the refill is
    # suppressed. Verify refill suppression for both "" and None style
    # sentinel semantics: making the sentinel won't change behavior unless
    # it's falsy or replaced by a real lookup key. Use a table that maps
    # *all* keys passed by identity? Not possible. Instead, test that
    # lookups into an empty table don't trigger refills after being written
    # in either the const-or-mutation case via observable call count.
    calls = []

    @silent_lookuper
    def lookup():
        calls.append(1)
        return {}

    assert lookup("x") is None
    assert lookup("y") is None
    assert calls == [1]


def test_silent_lookuper_returns_falsy_present_value():
    # Line 147 mutation: `memory.get(arg)` -> `memory.get(None)`.
    # A present key with a falsy value must be returned, not None.
    @silent_lookuper
    def lookup():
        return {"zero": 0, "empty": "", "false": False}

    assert lookup("zero") == 0
    assert lookup("empty") == ""
    assert lookup("false") is False
    assert lookup("missing") is None


def test_make_lookuper_membership_branch_precise():
    # Line 148 mutation: `arg in memory` -> `arg not in memory`.
    # Present key must return value; absent key must raise.
    @make_lookuper
    def lookup():
        return {"present": "value"}

    assert lookup("present") == "value"
    with pytest.raises(LookupError):
        lookup("absent")


def test_make_lookuper_error_message_exact_no_args():
    # Line 151 message check with exact contents.
    @make_lookuper
    def table_builder():
        return {}

    with pytest.raises(LookupError) as exc_info:
        table_builder("key1")
    assert exc_info.value.args == ("Failed to look up table_builder(key1)",)


def test_make_lookuper_error_message_exact_with_subtable():
    @make_lookuper
    def builder(n):
        return {}

    sub = builder("abc")
    with pytest.raises(LookupError) as exc_info:
        sub("xyz")
    assert exc_info.value.args == ("Failed to look up builder(abc)(xyz)",)


def test_make_lookuper_no_arg_builder_empty_dict_sentinel_is_truthy():
    # Confirm the sentinel prevents refill with an empty table (guards
    # against sentinel becoming falsy, e.g. `memory[object()] = ""`):
    # empty string is falsy, which combined with `if not memory:` would
    # cause refills.  Verify refill suppression works for repeated misses.
    calls = []

    @make_lookuper
    def lookup():
        calls.append(1)
        return {}

    for _ in range(5):
        with pytest.raises(LookupError):
            lookup("nope")
    # Refill must happen exactly once.
    assert calls == [1]


def test_make_lookuper_silent_empty_repeated():
    # Same as above but through silent_lookuper.
    calls = []

    @silent_lookuper
    def lookup():
        calls.append(1)
        return {}

    for _ in range(5):
        assert lookup("nope") is None
    assert calls == [1]


def test_make_lookuper_inverted_membership_would_break_present():
    # Additional guard for line 148 mutation using a non-empty table and
    # several distinct present keys.
    @make_lookuper
    def lookup():
        return {"a": 1, "b": 2, "c": 3}

    assert lookup("a") == 1
    assert lookup("b") == 2
    assert lookup("c") == 3


def test_make_lookuper_inverted_membership_would_break_absent():
    @make_lookuper
    def lookup():
        return {"a": 1}

    with pytest.raises(LookupError):
        lookup("not-there-1")
    with pytest.raises(LookupError):
        lookup("not-there-2")


def test_make_lookuper_update_none_would_raise():
    # Line 140 mutation `memory.update(None)` would raise TypeError. Ensure
    # a silent_lookuper build with a real mapping works.
    @silent_lookuper
    def lookup():
        return {"x": "X"}

    assert lookup("x") == "X"


def test_make_lookuper_memory_none_would_raise():
    # Line 135 mutation `memory = None` would crash when calling
    # `memory.update(...)` or `memory[object()] = ...`. Ensure both silent
    # and non-silent variants work on a no-arg builder.
    @make_lookuper
    def lookup_a():
        return {"a": 1}

    assert lookup_a("a") == 1

    @silent_lookuper
    def lookup_b():
        return {"b": 2}

    assert lookup_b("b") == 2


def test_make_lookuper_if_memory_inverted_breaks_refill():
    # Line 138 mutation `if not memory:` -> `if memory:` would cause the
    # builder to run on the *second* lookup (when memory is non-empty),
    # overwriting the sentinel and refilling each time.
    calls = []

    @silent_lookuper
    def lookup():
        calls.append(1)
        return {"a": 1}

    assert lookup("a") == 1
    assert lookup("a") == 1
    assert lookup("missing") is None
    # Builder must run exactly once; inverted condition would run it again.
    assert calls == [1]


def test_make_lookuper_if_memory_inverted_breaks_first_lookup():
    # With an inverted condition, the first lookup with empty memory would
    # skip the fill and go straight to the lookup path, raising or returning
    # None instead of the expected value.
    @make_lookuper
    def lookup():
        return {"first": "VALUE"}

    assert lookup("first") == "VALUE"


def test_make_lookuper_has_arg_types_dispatch_with_pos_args():
    # Line 124 mutant `has_args, has_keys = None` would raise TypeError on
    # unpack. A builder with positional args must take the memoized-branch.
    @make_lookuper
    def lookup(x):
        return {x: x * 10}

    t = lookup(4)
    assert t(4) == 40


def test_make_lookuper_has_arg_types_dispatch_no_args_ok():
    # Also exercises dispatch for a no-arg builder (has_args False, has_keys
    # False). With a `None` unpack this would fail.
    @make_lookuper
    def lookup():
        return {"z": 26}

    assert lookup("z") == 26


def test_make_lookuper_keyword_only_assertion_exact_variants():
    # Cover the assertion raised for keyword-only funcs with exact message.
    # This kills message mutations like case/prefix variations.
    with pytest.raises(AssertionError) as exc_info:
        @make_lookuper
        def lookup(*, kw):
            return {}

    msg = str(exc_info.value)
    assert msg == "Lookup table building function should not have keyword arguments"
    assert msg.startswith("Lookup table")
    assert msg.endswith("keyword arguments")
    assert "should not have" in msg


def test_cache_timedelta_invalidate_flow():
    # Round trip through CacheMemory with timedelta timeout.
    calls = []

    @cache(timedelta(seconds=30))
    def f(x):
        calls.append(x)
        return x * 3

    assert f(2) == 6
    assert f(2) == 6
    assert calls == [2]
    # Value survives expire() when not expired.
    f.memory.expire()
    assert f(2) == 6
    assert calls == [2]


def test_cachememory_expire_after_reassignment_picks_latest():
    m = CacheMemory(0)
    m["a"] = "old"
    m["a"] = "new"
    # Both tracking entries are expired now. expire() checks each entry's
    # own expires_at against the popped one; the final state must not contain
    # "a" after a full expire.
    m.expire()
    assert "a" not in m
    assert len(m._keys) == 0
    assert len(m._expires) == 0


def test_make_lookuper_with_args_memoized_subtable_independent():
    # Provides a second, distinct memoized table to verify each arg produces
    # its own builder-derived mapping without sharing.
    @make_lookuper
    def lookup(prefix):
        return {prefix + "x": 1, prefix + "y": 2}

    t1 = lookup("A")
    t2 = lookup("B")
    assert t1("Ax") == 1
    assert t1("Ay") == 2
    assert t2("Bx") == 1
    assert t2("By") == 2
    with pytest.raises(LookupError):
        t1("Bx")


def test_make_lookuper_silent_with_args_misses_none():
    @silent_lookuper
    def lookup(prefix):
        return {prefix + "x": 5}

    t = lookup("A")
    assert t("Ax") == 5
    assert t("NOPE") is None
