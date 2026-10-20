import json
import math

import pytest

from jmespath import exceptions
from jmespath.functions import (
    signature,
    FunctionRegistry,
    Functions,
    TYPES_MAP,
    REVERSE_TYPES_MAP,
)


# ---------------------------------------------------------------------------
# signature decorator
# ---------------------------------------------------------------------------

def test_signature_records_signature_attribute():
    @signature({'types': ['number']}, {'types': ['string']})
    def myfunc(a, b):
        return a, b

    assert myfunc.signature == ({'types': ['number']}, {'types': ['string']})
    assert myfunc(1, 'a') == (1, 'a')


def test_signature_no_arguments():
    @signature()
    def myfunc():
        return 42

    assert myfunc.signature == ()
    assert myfunc() == 42


# ---------------------------------------------------------------------------
# Type maps
# ---------------------------------------------------------------------------

def test_types_map_entries():
    assert TYPES_MAP['bool'] == 'boolean'
    assert TYPES_MAP['list'] == 'array'
    assert TYPES_MAP['dict'] == 'object'
    assert TYPES_MAP['NoneType'] == 'null'
    assert TYPES_MAP['str'] == 'string'
    assert TYPES_MAP['float'] == 'number'
    assert TYPES_MAP['int'] == 'number'
    assert TYPES_MAP['_Expression'] == 'expref'


def test_reverse_types_map_entries():
    assert REVERSE_TYPES_MAP['boolean'] == ('bool',)
    assert REVERSE_TYPES_MAP['null'] == ('NoneType',)
    assert 'list' in REVERSE_TYPES_MAP['array']
    assert 'dict' in REVERSE_TYPES_MAP['object']
    assert 'int' in REVERSE_TYPES_MAP['number']


# ---------------------------------------------------------------------------
# FunctionRegistry / FUNCTION_TABLE
# ---------------------------------------------------------------------------

def test_function_table_contains_known_functions():
    table = Functions.FUNCTION_TABLE
    assert 'abs' in table
    assert 'avg' in table
    assert 'max_by' in table
    assert 'sort_by' in table
    assert 'not_null' in table
    # Function names have the _func_ prefix stripped.
    assert '_func_abs' not in table
    for name, spec in table.items():
        assert 'function' in spec
        assert 'signature' in spec


def test_function_table_skips_undecorated_methods():
    assert '_create_key_func' not in Functions.FUNCTION_TABLE
    assert 'create_key_func' not in Functions.FUNCTION_TABLE


def test_custom_registry_subclass():
    class Custom(Functions):
        @signature({'types': ['number']})
        def _func_double(self, arg):
            return arg * 2

    assert 'double' in Custom.FUNCTION_TABLE
    assert 'abs' in Custom.FUNCTION_TABLE
    result = Custom().call_function('double', [21])
    assert result == 42


def test_registry_is_metaclass():
    assert isinstance(Functions, FunctionRegistry)


# ---------------------------------------------------------------------------
# call_function / error handling
# ---------------------------------------------------------------------------

def test_unknown_function_raises():
    f = Functions()
    with pytest.raises(exceptions.UnknownFunctionError) as exc:
        f.call_function('nonexistent', [])
    assert 'Unknown function' in str(exc.value)
    assert 'nonexistent' in str(exc.value)


def test_arity_error_too_few():
    f = Functions()
    with pytest.raises(exceptions.ArityError):
        f.call_function('abs', [])


def test_arity_error_too_many():
    f = Functions()
    with pytest.raises(exceptions.ArityError):
        f.call_function('abs', [1, 2])


def test_variadic_arity_error():
    f = Functions()
    with pytest.raises(exceptions.VariadictArityError):
        f.call_function('not_null', [])


def test_type_error_bad_argument_type():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('abs', ['not-a-number'])


def test_type_check_allows_bool_not_as_number():
    # bool is a subclass of int in Python, but jmespath treats it distinctly.
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('abs', [True])


# ---------------------------------------------------------------------------
# Number functions
# ---------------------------------------------------------------------------

def test_abs():
    f = Functions()
    assert f.call_function('abs', [-5]) == 5
    assert f.call_function('abs', [5]) == 5
    assert f.call_function('abs', [-2.5]) == 2.5


def test_avg():
    f = Functions()
    assert f.call_function('avg', [[1, 2, 3]]) == 2
    assert f.call_function('avg', [[1.0, 2.0]]) == 1.5
    assert f.call_function('avg', [[]]) is None


def test_ceil_floor():
    f = Functions()
    assert f.call_function('ceil', [1.2]) == 2
    assert f.call_function('ceil', [1.0]) == 1
    assert f.call_function('floor', [1.8]) == 1
    assert f.call_function('floor', [2.0]) == 2


def test_sum():
    f = Functions()
    assert f.call_function('sum', [[1, 2, 3]]) == 6
    assert f.call_function('sum', [[]]) == 0


def test_max_min():
    f = Functions()
    assert f.call_function('max', [[1, 3, 2]]) == 3
    assert f.call_function('max', [['a', 'c', 'b']]) == 'c'
    assert f.call_function('max', [[]]) is None
    assert f.call_function('min', [[1, 3, 2]]) == 1
    assert f.call_function('min', [['a', 'c', 'b']]) == 'a'
    assert f.call_function('min', [[]]) is None


def test_sort():
    f = Functions()
    assert f.call_function('sort', [[3, 1, 2]]) == [1, 2, 3]
    assert f.call_function('sort', [['b', 'c', 'a']]) == ['a', 'b', 'c']
    assert f.call_function('sort', [[]]) == []


def test_sort_bad_mixed_types():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('sort', [[1, 'a']])


# ---------------------------------------------------------------------------
# String functions
# ---------------------------------------------------------------------------

def test_contains():
    f = Functions()
    assert f.call_function('contains', ['foobar', 'bar']) is True
    assert f.call_function('contains', ['foobar', 'baz']) is False
    assert f.call_function('contains', [[1, 2, 3], 2]) is True
    assert f.call_function('contains', [[1, 2, 3], 9]) is False


def test_length():
    f = Functions()
    assert f.call_function('length', ['abcd']) == 4
    assert f.call_function('length', [[1, 2, 3]]) == 3
    assert f.call_function('length', [{'a': 1, 'b': 2}]) == 2


def test_ends_with_starts_with():
    f = Functions()
    assert f.call_function('ends_with', ['foobar', 'bar']) is True
    assert f.call_function('ends_with', ['foobar', 'foo']) is False
    assert f.call_function('starts_with', ['foobar', 'foo']) is True
    assert f.call_function('starts_with', ['foobar', 'bar']) is False


def test_reverse():
    f = Functions()
    assert f.call_function('reverse', ['abc']) == 'cba'
    assert f.call_function('reverse', [[1, 2, 3]]) == [3, 2, 1]
    assert f.call_function('reverse', [[]]) == []


def test_join():
    f = Functions()
    assert f.call_function('join', [', ', ['a', 'b', 'c']]) == 'a, b, c'
    assert f.call_function('join', ['', ['a', 'b']]) == 'ab'
    assert f.call_function('join', ['-', []]) == ''


def test_join_requires_array_of_strings():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError):
        f.call_function('join', [',', [1, 2, 3]])


# ---------------------------------------------------------------------------
# Type conversion functions
# ---------------------------------------------------------------------------

def test_to_array():
    f = Functions()
    assert f.call_function('to_array', [[1, 2]]) == [1, 2]
    assert f.call_function('to_array', ['abc']) == ['abc']
    assert f.call_function('to_array', [1]) == [1]
    assert f.call_function('to_array', [None]) == [None]


def test_to_string():
    f = Functions()
    assert f.call_function('to_string', ['abc']) == 'abc'
    assert f.call_function('to_string', [1]) == '1'
    assert f.call_function('to_string', [1.5]) == '1.5'
    assert f.call_function('to_string', [True]) == 'true'
    assert f.call_function('to_string', [None]) == 'null'
    assert f.call_function('to_string', [[1, 2]]) == '[1,2]'
    assert f.call_function('to_string', [{'a': 1}]) == '{"a":1}'


def test_to_number():
    f = Functions()
    assert f.call_function('to_number', ['42']) == 42
    assert f.call_function('to_number', ['3.14']) == 3.14
    assert f.call_function('to_number', [7]) == 7
    assert f.call_function('to_number', [1.5]) == 1.5
    assert f.call_function('to_number', ['not-a-number']) is None
    assert f.call_function('to_number', [[1, 2]]) is None
    assert f.call_function('to_number', [{'a': 1}]) is None
    assert f.call_function('to_number', [True]) is None
    assert f.call_function('to_number', [False]) is None
    assert f.call_function('to_number', [None]) is None


# ---------------------------------------------------------------------------
# Object functions
# ---------------------------------------------------------------------------

def test_keys_values():
    f = Functions()
    assert f.call_function('keys', [{'a': 1, 'b': 2}]) == ['a', 'b']
    assert f.call_function('values', [{'a': 1, 'b': 2}]) == [1, 2]
    assert f.call_function('keys', [{}]) == []
    assert f.call_function('values', [{}]) == []


def test_merge():
    f = Functions()
    assert f.call_function('merge', [{'a': 1}, {'b': 2}]) == {'a': 1, 'b': 2}
    assert f.call_function('merge', [{'a': 1}, {'a': 2}]) == {'a': 2}
    assert f.call_function('merge', [{'a': 1}]) == {'a': 1}
    assert f.call_function('merge', [{'a': 1}, {'b': 2}, {'c': 3}]) == {
        'a': 1, 'b': 2, 'c': 3}


def test_not_null():
    f = Functions()
    assert f.call_function('not_null', [None, None, 'x']) == 'x'
    assert f.call_function('not_null', [1, 2]) == 1
    assert f.call_function('not_null', [None]) is None
    # Multiple args all None.
    assert f.call_function('not_null', [None, None]) is None


# ---------------------------------------------------------------------------
# type()
# ---------------------------------------------------------------------------

def test_type_of_values():
    f = Functions()
    assert f.call_function('type', ['abc']) == 'string'
    assert f.call_function('type', [True]) == 'boolean'
    assert f.call_function('type', [False]) == 'boolean'
    assert f.call_function('type', [[1, 2]]) == 'array'
    assert f.call_function('type', [{'a': 1}]) == 'object'
    assert f.call_function('type', [1]) == 'number'
    assert f.call_function('type', [1.5]) == 'number'
    assert f.call_function('type', [None]) == 'null'


# ---------------------------------------------------------------------------
# _convert_to_jmespath_type helper
# ---------------------------------------------------------------------------

def test_convert_to_jmespath_type():
    f = Functions()
    assert f._convert_to_jmespath_type('int') == 'number'
    assert f._convert_to_jmespath_type('str') == 'string'
    assert f._convert_to_jmespath_type('bool') == 'boolean'
    assert f._convert_to_jmespath_type('list') == 'array'
    assert f._convert_to_jmespath_type('NoneType') == 'null'
    assert f._convert_to_jmespath_type('SomeUnknownType') == 'unknown'


# ---------------------------------------------------------------------------
# expref-using functions: _func_map, _func_sort_by, _func_min_by, _func_max_by
#
# The expref type check is keyed on the python type name "_Expression"
# (see TYPES_MAP and REVERSE_TYPES_MAP).  We therefore need a fake expref
# whose class name is exactly "_Expression" so it passes type validation.
# ---------------------------------------------------------------------------

def _make_expref(fn):
    expref_cls = type('_Expression', (object,), {
        '__init__': lambda self, f: setattr(self, '_fn', f),
        'expression': 'expr',
        'visit': lambda self, expression, value: self._fn(value),
    })
    return expref_cls(fn)


def test_map_with_expref():
    f = Functions()
    expref = _make_expref(lambda x: x['foo'])
    result = f.call_function('map', [expref, [{'foo': 1}, {'foo': 2}]])
    assert result == [1, 2]


def test_map_empty_array():
    f = Functions()
    expref = _make_expref(lambda x: x)
    assert f.call_function('map', [expref, []]) == []


def test_map_non_expref_raises():
    f = Functions()
    with pytest.raises(exceptions.JMESPathTypeError) as exc:
        f.call_function('map', [object(), []])
    assert exc.value.function_name == 'map'


def test_sort_by_numbers():
    f = Functions()
    expref = _make_expref(lambda x: x['age'])
    result = f.call_function('sort_by', [[{'age': 3}, {'age': 1}, {'age': 2}], expref])
    assert result == [{'age': 1}, {'age': 2}, {'age': 3}]


def test_sort_by_strings():
    f = Functions()
    expref = _make_expref(lambda x: x['name'])
    result = f.call_function('sort_by', [[{'name': 'c'}, {'name': 'a'}], expref])
    assert result == [{'name': 'a'}, {'name': 'c'}]


def test_sort_by_empty_array():
    f = Functions()
    expref = _make_expref(lambda x: x)
    arr = []
    result = f.call_function('sort_by', [arr, expref])
    assert result == []


def test_sort_by_wrong_first_type():
    f = Functions()
    expref = _make_expref(lambda x: x['x'])
    with pytest.raises(exceptions.JMESPathTypeError) as exc:
        f.call_function('sort_by', [[{'x': [1, 2]}], expref])
    assert exc.value.function_name == 'sort_by'


def test_sort_by_inconsistent_types():
    f = Functions()
    expref = _make_expref(lambda x: x['x'])
    with pytest.raises(exceptions.JMESPathTypeError) as exc:
        f.call_function('sort_by', [[{'x': 1}, {'x': 'a'}], expref])
    assert exc.value.function_name == 'sort_by'


def test_min_by():
    f = Functions()
    expref = _make_expref(lambda x: x['age'])
    result = f.call_function('min_by', [[{'age': 3}, {'age': 1}], expref])
    assert result == {'age': 1}


def test_min_by_empty():
    f = Functions()
    expref = _make_expref(lambda x: x['age'])
    result = f.call_function('min_by', [[], expref])
    assert result is None


def test_max_by():
    f = Functions()
    expref = _make_expref(lambda x: x['age'])
    result = f.call_function('max_by', [[{'age': 3}, {'age': 1}], expref])
    assert result == {'age': 3}


def test_max_by_empty():
    f = Functions()
    expref = _make_expref(lambda x: x['age'])
    result = f.call_function('max_by', [[], expref])
    assert result is None


def test_max_by_invalid_key_type():
    f = Functions()
    expref = _make_expref(lambda x: x['age'])
    with pytest.raises(exceptions.JMESPathTypeError) as exc:
        f.call_function('max_by', [[{'age': [1, 2]}], expref])
    assert exc.value.function_name == 'max_by'


# ---------------------------------------------------------------------------
# _get_allowed_pytypes / subtype checks
# ---------------------------------------------------------------------------

def test_get_allowed_pytypes():
    f = Functions()
    allowed, subtypes = f._get_allowed_pytypes(['string', 'array'])
    assert 'str' in allowed
    assert 'list' in allowed
    assert subtypes == []


def test_get_allowed_pytypes_with_subtype():
    f = Functions()
    allowed, subtypes = f._get_allowed_pytypes(['array-number'])
    assert 'list' in allowed
    assert ('float', 'int', 'long') in subtypes


def test_subtype_check_dynamic_multiple():
    f = Functions()
    # sort accepts array-number or array-string (dynamic subtype check)
    # Directly call sort with valid inputs to exercise the >1 subtype branch.
    assert f.call_function('sort', [[3, 1, 2]]) == [1, 2, 3]
    assert f.call_function('sort', [['c', 'a', 'b']]) == ['a', 'b', 'c']
