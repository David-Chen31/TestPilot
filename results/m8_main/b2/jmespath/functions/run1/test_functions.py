import pytest

from jmespath import exceptions
from jmespath.functions import (
    TYPES_MAP,
    REVERSE_TYPES_MAP,
    signature,
    FunctionRegistry,
    Functions,
)


# ---------------------------------------------------------------------------
# A helper expref whose type name is "_Expression" so the type checker accepts
# it as an "expref".
#
# NOTE: the metapath `map`, `sort_by`, `min_by`, `max_by` functions dispatch
# through `Functions.call_function`, whose `_type_check_single` uses
# `type(current).__name__`.  Because `_type_` for `expref` maps only to
# `'_Expression'`, subclasses (whose `__name__` is the subclass name) are not
# accepted.  We therefore use `_Expression` instances directly.
# ---------------------------------------------------------------------------

_Expression = type('_Expression', (object,), {})


class _Expref(_Expression):
    def __init__(self, key=None):
        self.expression = key

    def visit(self, expression, element):
        return element[expression]


def make_expref(key):
    return _Expref(key)


# ---------------------------------------------------------------------------
# signature decorator
# ---------------------------------------------------------------------------

def test_signature_attaches_signature_attribute():
    @signature({'types': ['number']})
    def foo(self, arg):
        return arg

    assert foo.signature == ({'types': ['number']},)


def test_signature_preserves_returned_function():
    def foo(self, arg):
        return arg

    decorated = signature({'types': []})(foo)
    assert decorated is foo


# ---------------------------------------------------------------------------
# Type maps
# ---------------------------------------------------------------------------

def test_types_map_entries():
    assert TYPES_MAP['bool'] == 'boolean'
    assert TYPES_MAP['list'] == 'array'
    assert TYPES_MAP['dict'] == 'object'
    assert TYPES_MAP['NoneType'] == 'null'
    assert TYPES_MAP['str'] == 'string'
    assert TYPES_MAP['int'] == 'number'
    assert TYPES_MAP['float'] == 'number'
    assert TYPES_MAP['_Expression'] == 'expref'


def test_reverse_types_map_entries():
    assert REVERSE_TYPES_MAP['boolean'] == ('bool',)
    assert 'list' in REVERSE_TYPES_MAP['array']
    assert 'dict' in REVERSE_TYPES_MAP['object']
    assert REVERSE_TYPES_MAP['null'] == ('NoneType',)
    assert 'str' in REVERSE_TYPES_MAP['string']
    assert 'int' in REVERSE_TYPES_MAP['number']
    assert REVERSE_TYPES_MAP['expref'] == ('_Expression',)


# ---------------------------------------------------------------------------
# FunctionRegistry
# ---------------------------------------------------------------------------

def test_function_registry_creates_function_table():
    class MyFunctions(metaclass=FunctionRegistry):
        @signature({'types': ['number']})
        def _func_double(self, arg):
            return arg * 2

        def _func_missing_signature(self, arg):
            return arg

        def helper(self, arg):
            return arg

    assert 'double' in MyFunctions.FUNCTION_TABLE
    assert 'missing_signature' not in MyFunctions.FUNCTION_TABLE
    assert 'helper' not in MyFunctions.FUNCTION_TABLE
    entry = MyFunctions.FUNCTION_TABLE['double']
    assert entry['signature'] == ({'types': ['number']},)
    assert entry['function'].__name__ == '_func_double'


def test_function_registry_subclass_inherits_is_rebuilt():
    class Base(metaclass=FunctionRegistry):
        @signature({'types': ['number']})
        def _func_base(self, arg):
            return arg

    class Child(Base):
        @signature({'types': ['string']})
        def _func_child(self, arg):
            return arg

    # get_methods uses inspect.getmembers which only returns functions
    # defined on the class itself (including inherited ones via the MRO).
    assert 'child' in Child.FUNCTION_TABLE


# ---------------------------------------------------------------------------
# Functions.call_function
# ---------------------------------------------------------------------------

def test_call_function_unknown():
    f = Functions()
    with pytest.raises(exceptions.UnknownFunctionError) as exc:
        f.call_function('nope', [])
    assert 'Unknown function: nope()' in str(exc.value)


def test_call_function_dispatches():
    f = Functions()
    assert f.call_function('abs', [-5]) == 5


# ---------------------------------------------------------------------------
# Argument validation: arity
# ---------------------------------------------------------------------------

def test_arity_error_too_few():
    f = Functions()
    with pytest.raises(exceptions.ArityError):
        f.call_function('abs', [])


def test_arity_error_too_many():
    f = Functions()
    with pytest.raises(exceptions.ArityError):
        f.call_function('abs', [1, 2])


def test_variadic_too_few():
    f = Functions()
    with pytest.raises(exceptions.VariadictArityError):
        f.call_function('not_null', [])


def test_variadic_ok():
    f = Functions()
    assert f.call_function('not_null', [None, None, 5, 6]) == 5


def test_variadic_all_none():
    f = Functions()
    assert f.call_function('not_null', [None, None]) is None


# ---------------------------------------------------------------------------
# Type checking
# ---------------------------------------------------------------------------

def test_type_check_error():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError) as exc:
        f.call_function('abs', ['notanumber'])
    assert isinstance(exc.value, exceptions.JMESPathTypeError)


def test_bool_not_accepted_as_number():
    # bool is int in python, but jmespath checks type name explicitly.
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('abs', [True])


def test_subtype_check_single():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('sum', [['a', 'b']])


def test_subtype_check_dynamic_mismatch():
    f = Functions()
    # max accepts array-number or array-string, dynamic based on first elem.
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('max', [[1, 'a']])


def test_subtype_check_dynamic_all_strings():
    f = Functions()
    assert f.call_function('max', [['a', 'b', 'c']]) == 'c'


def test_subtype_check_dynamic_all_numbers():
    f = Functions()
    assert f.call_function('max', [[1, 2, 3]]) == 3


def test_subtype_check_empty_list_ok():
    f = Functions()
    assert f.call_function('max', [[]]) is None


def test_get_allowed_pytypes():
    f = Functions()
    allowed, subtypes = f._get_allowed_pytypes(['array-number', 'string'])
    assert 'list' in allowed
    assert 'str' in allowed
    assert subtypes == [REVERSE_TYPES_MAP['number']]


def test_convert_to_jmespath_type_unknown():
    f = Functions()
    assert f._convert_to_jmespath_type('BogusType') == 'unknown'
    assert f._convert_to_jmespath_type('int') == 'number'


# ---------------------------------------------------------------------------
# Individual functions
# ---------------------------------------------------------------------------

def test_abs():
    f = Functions()
    assert f.call_function('abs', [-3]) == 3
    assert f.call_function('abs', [3]) == 3
    assert f.call_function('abs', [3.5]) == 3.5


def test_avg():
    f = Functions()
    assert f.call_function('avg', [[1, 2, 3]]) == 2
    assert f.call_function('avg', [[]]) is None


def test_not_null():
    f = Functions()
    assert f.call_function('not_null', ['a', None]) == 'a'
    assert f.call_function('not_null', [None]) is None


def test_to_array():
    f = Functions()
    assert f.call_function('to_array', [[1, 2]]) == [1, 2]
    assert f.call_function('to_array', [1]) == [1]
    assert f.call_function('to_array', ['x']) == ['x']
    assert f.call_function('to_array', [None]) == [None]


def test_to_string():
    f = Functions()
    assert f.call_function('to_string', ['abc']) == 'abc'
    assert f.call_function('to_string', [1]) == '1'
    assert f.call_function('to_string', [[1, 2]]) == '[1,2]'
    assert f.call_function('to_string', [{'a': 1}]) == '{"a":1}'


def test_to_number_from_int_float():
    f = Functions()
    assert f.call_function('to_number', [5]) == 5
    assert f.call_function('to_number', [5.5]) == 5.5


def test_to_number_from_string():
    f = Functions()
    assert f.call_function('to_number', ['5']) == 5
    assert f.call_function('to_number', ['5.5']) == 5.5


def test_to_number_failures():
    f = Functions()
    assert f.call_function('to_number', ['abc']) is None
    assert f.call_function('to_number', [None]) is None
    assert f.call_function('to_number', [[1]]) is None
    assert f.call_function('to_number', [{'a': 1}]) is None
    assert f.call_function('to_number', [True]) is None


def test_contains():
    f = Functions()
    assert f.call_function('contains', ['foobar', 'oba']) is True
    assert f.call_function('contains', ['foobar', 'zzz']) is False
    assert f.call_function('contains', [[1, 2, 3], 2]) is True
    assert f.call_function('contains', [[1, 2, 3], 5]) is False


def test_length():
    f = Functions()
    assert f.call_function('length', ['abc']) == 3
    assert f.call_function('length', [[1, 2]]) == 2
    assert f.call_function('length', [{'a': 1, 'b': 2}]) == 2


def test_ends_with():
    f = Functions()
    assert f.call_function('ends_with', ['foobar', 'bar']) is True
    assert f.call_function('ends_with', ['foobar', 'foo']) is False


def test_starts_with():
    f = Functions()
    assert f.call_function('starts_with', ['foobar', 'foo']) is True
    assert f.call_function('starts_with', ['foobar', 'bar']) is False


def test_reverse():
    f = Functions()
    assert f.call_function('reverse', ['abc']) == 'cba'
    assert f.call_function('reverse', [[1, 2, 3]]) == [3, 2, 1]


def test_ceil_floor():
    f = Functions()
    assert f.call_function('ceil', [1.2]) == 2
    assert f.call_function('floor', [1.8]) == 1


def test_join():
    f = Functions()
    assert f.call_function('join', [',', ['a', 'b', 'c']]) == 'a,b,c'
    assert f.call_function('join', ['-', []]) == ''


def test_join_subtype_error():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('join', [',', [1, 2]])


def test_map():
    class FakeExpref(_Expression):
        def __init__(self):
            self.expression = 'expr'

        def visit(self, expression, element):
            assert expression == 'expr'
            return element * 2

    f = Functions()
    # `map` takes an expref.  Direct instances of `_Expression` are accepted;
    # we give it the visit/expression attributes it needs.
    expref = _Expression()
    expref.expression = 'expr'

    def visit(expression, element):
        assert expression == 'expr'
        return element * 2

    expref.visit = visit
    assert f.call_function('map', [expref, [1, 2, 3]]) == [2, 4, 6]


def test_map_subclass_rejected():
    # Subclasses have their own __name__ and are not recognized as exprefs.
    class FakeExpref(_Expression):
        def __init__(self):
            self.expression = 'expr'

        def visit(self, expression, element):
            return element * 2

    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('map', [FakeExpref(), [1, 2, 3]])


def test_max_min():
    f = Functions()
    assert f.call_function('max', [[1, 5, 3]]) == 5
    assert f.call_function('min', [[1, 5, 3]]) == 1
    assert f.call_function('max', [[]]) is None
    assert f.call_function('min', [[]]) is None
    assert f.call_function('max', [['a', 'c', 'b']]) == 'c'
    assert f.call_function('min', [['a', 'c', 'b']]) == 'a'


def test_merge():
    f = Functions()
    result = f.call_function('merge', [{'a': 1}, {'b': 2}, {'a': 3}])
    assert result == {'a': 3, 'b': 2}


def test_sort():
    f = Functions()
    assert f.call_function('sort', [[3, 1, 2]]) == [1, 2, 3]
    assert f.call_function('sort', [['c', 'a', 'b']]) == ['a', 'b', 'c']


def test_sum():
    f = Functions()
    assert f.call_function('sum', [[1, 2, 3]]) == 6
    assert f.call_function('sum', [[]]) == 0


def test_keys_values():
    f = Functions()
    d = {'a': 1, 'b': 2}
    assert f.call_function('keys', [d]) == ['a', 'b']
    assert f.call_function('values', [d]) == [1, 2]


@pytest.mark.parametrize('value, expected', [
    ('abc', 'string'),
    (True, 'boolean'),
    (False, 'boolean'),
    ([1, 2], 'array'),
    ({'a': 1}, 'object'),
    (1, 'number'),
    (1.5, 'number'),
    (None, 'null'),
])
def test_type(value, expected):
    f = Functions()
    assert f.call_function('type', [value]) == expected


# ---------------------------------------------------------------------------
# sort_by / min_by / max_by with exprefs.
#
# See note at top of file: only direct `_Expression` instances are accepted by
# call_function's type checker, so we build them directly.
# ---------------------------------------------------------------------------

def _expref_for_key(key):
    expref = _Expression()

    def visit(expression, element):
        return element[expression]

    expref.expression = key
    expref.visit = visit
    return expref


def test_sort_by_numbers():
    f = Functions()
    array = [{'v': 3}, {'v': 1}, {'v': 2}]
    result = f.call_function('sort_by', [array, _expref_for_key('v')])
    assert result == [{'v': 1}, {'v': 2}, {'v': 3}]


def test_sort_by_strings():
    f = Functions()
    array = [{'v': 'c'}, {'v': 'a'}, {'v': 'b'}]
    result = f.call_function('sort_by', [array, _expref_for_key('v')])
    assert [x['v'] for x in result] == ['a', 'b', 'c']


def test_sort_by_empty():
    f = Functions()
    assert f.call_function('sort_by', [[], _expref_for_key('v')]) == []


def test_sort_by_invalid_type():
    f = Functions()
    array = [{'v': True}]
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('sort_by', [array, _expref_for_key('v')])


def test_min_by():
    f = Functions()
    array = [{'v': 3}, {'v': 1}, {'v': 2}]
    assert f.call_function('min_by', [array, _expref_for_key('v')]) == {'v': 1}
    assert f.call_function('min_by', [[], _expref_for_key('v')]) is None


def test_max_by():
    f = Functions()
    array = [{'v': 3}, {'v': 1}, {'v': 2}]
    assert f.call_function('max_by', [array, _expref_for_key('v')]) == {'v': 3}
    assert f.call_function('max_by', [[], _expref_for_key('v')]) is None


# Mixed-type arrays are not rejected by the key function ahead of time: the
# keyfunc only checks that each key is a number or string, not that they all
# agree.  Python's min()/max() therefore raise a TypeError when comparing the
# resolved keys, which is what the current implementation does.

def test_min_by_mixed_types_raises_type_error():
    f = Functions()
    array = [{'v': 1}, {'v': 'a'}]
    with pytest.raises(TypeError):
        f.call_function('min_by', [array, _expref_for_key('v')])


def test_max_by_mixed_types_raises_type_error():
    f = Functions()
    array = [{'v': 1}, {'v': 'a'}]
    with pytest.raises(TypeError):
        f.call_function('max_by', [array, _expref_for_key('v')])


def test_create_key_func_type_error():
    f = Functions()
    keyfunc = f._create_key_func(_expref_for_key('v'), ['number', 'string'],
                                 'test')
    with pytest.raises(exceptions.JMESPathTypeError):
        keyfunc({'v': [1, 2]})
