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

class TestSignature:
    def test_signature_attaches_arguments_attribute(self):
        @signature({'types': ['number']}, {'types': ['string']})
        def func(a, b):
            return (a, b)

        assert func.signature == ({'types': ['number']}, {'types': ['string']})

    def test_signature_returns_function(self):
        def func():
            return 42

        result = signature({'types': []})(func)
        assert result is func

    def test_signature_with_no_arguments(self):
        @signature()
        def func():
            return None

        assert func.signature == ()


# ---------------------------------------------------------------------------
# FunctionRegistry
# ---------------------------------------------------------------------------

class TestFunctionRegistry:
    def test_populates_function_table_from_func_methods(self):
        class MyFunctions(metaclass=FunctionRegistry):
            @signature({'types': ['number']})
            def _func_double(self, arg):
                return arg * 2

        assert set(MyFunctions.FUNCTION_TABLE.keys()) == {'double'}
        spec = MyFunctions.FUNCTION_TABLE['double']
        assert spec['signature'] == ({'types': ['number']},)
        assert spec['function'].__name__ == '_func_double'

    def test_ignores_methods_not_starting_with_func(self):
        class MyFunctions(metaclass=FunctionRegistry):
            @signature({'types': []})
            def helper(self, arg):
                return arg

            def plain(self):
                return None

        assert MyFunctions.FUNCTION_TABLE == {}

    def test_ignores_func_methods_without_signature(self):
        class MyFunctions(metaclass=FunctionRegistry):
            def _func_nosig(self, arg):
                return arg

        assert MyFunctions.FUNCTION_TABLE == {}

    def test_multiple_functions_registered(self):
        class MyFuncs(metaclass=FunctionRegistry):
            @signature({'types': []})
            def _func_a(self, arg):
                return 'a'

            @signature({'types': []})
            def _func_b(self, arg):
                return 'b'

        assert set(MyFuncs.FUNCTION_TABLE.keys()) == {'a', 'b'}

    def test_builtin_functions_present(self):
        # The Functions class uses this metaclass and registers many functions.
        table = Functions.FUNCTION_TABLE
        for name in ['abs', 'avg', 'not_null', 'to_array', 'to_string',
                     'to_number', 'contains', 'length', 'ends_with',
                     'starts_with', 'reverse', 'ceil', 'floor', 'join',
                     'map', 'max', 'merge', 'min', 'sort', 'sum', 'keys',
                     'values', 'type', 'sort_by', 'min_by', 'max_by']:
            assert name in table


# ---------------------------------------------------------------------------
# call_function dispatch and error handling
# ---------------------------------------------------------------------------

class TestCallFunction:
    def setup_method(self):
        self.funcs = Functions()

    def test_unknown_function_raises(self):
        with pytest.raises(exceptions.UnknownFunctionError) as excinfo:
            self.funcs.call_function('does_not_exist', [])
        assert 'Unknown function: does_not_exist()' in str(excinfo.value)

    def test_arity_error(self):
        with pytest.raises(exceptions.ArityError):
            self.funcs.call_function('abs', [])

    def test_arity_error_too_many(self):
        with pytest.raises(exceptions.ArityError):
            self.funcs.call_function('abs', [1, 2])

    def test_variadic_arity_error(self):
        # merge requires at least one object argument
        with pytest.raises(exceptions.VariadictArityError):
            self.funcs.call_function('merge', [])

    def test_variadic_arity_ok(self):
        result = self.funcs.call_function('merge', [{'a': 1}])
        assert result == {'a': 1}

    def test_type_error(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.funcs.call_function('abs', ['foo'])

    def test_type_error_reports_info(self):
        with pytest.raises(exceptions.JMESPathTypeError) as excinfo:
            self.funcs.call_function('abs', ['foo'])
        err = excinfo.value
        assert err.function_name == 'abs'
        assert err.current_value == 'foo'
        assert err.actual_type == 'string'
        assert err.expected_types == ['number']

    def test_call_valid_function(self):
        assert self.funcs.call_function('abs', [-3]) == 3

    def test_call_dispatches_with_correct_args(self):
        assert self.funcs.call_function('join', ['-', ['a', 'b', 'c']]) == 'a-b-c'


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class TestValidation:
    def setup_method(self):
        self.funcs = Functions()

    def test_validate_non_variadic_exact_match(self):
        sig = ({'types': ['number']},)
        # No exception.
        assert self.funcs._validate_arguments([1], sig, 'f') is None

    def test_validate_variadic_min_arity(self):
        sig = ({'types': ['object'], 'variadic': True},)
        with pytest.raises(exceptions.VariadictArityError):
            self.funcs._validate_arguments([], sig, 'f')

    def test_validate_variadic_more_than_min(self):
        sig = ({'types': ['object'], 'variadic': True},)
        assert self.funcs._validate_arguments([{}, {}, {}], sig, 'f') is None

    def test_get_allowed_pytypes_simple(self):
        allowed, subtypes = self.funcs._get_allowed_pytypes(['number'])
        assert set(allowed) == {'float', 'int', 'long'}
        assert subtypes == []

    def test_get_allowed_pytypes_with_subtype(self):
        allowed, subtypes = self.funcs._get_allowed_pytypes(['array-number'])
        assert set(allowed) == {'list', '_Projection'}
        assert set(subtypes[0]) == {'float', 'int', 'long'}

    def test_subtype_check_single_type_ok(self):
        # allowed_subtypes = [('float', 'int', 'long')]
        self.funcs._subtype_check([1, 2, 3.0], [('float', 'int', 'long')],
                                  ['array-number'], 'f')

    def test_subtype_check_single_type_bad(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.funcs._subtype_check([1, 'a'], [('float', 'int', 'long')],
                                      ['array-number'], 'f')

    def test_subtype_check_dynamic_ok(self):
        allowed = [('float', 'int', 'long'), ('unicode', 'str')]
        self.funcs._subtype_check([1, 2, 3], allowed, ['t'], 'f')

    def test_subtype_check_dynamic_mixed_bad(self):
        allowed = [('float', 'int', 'long'), ('unicode', 'str')]
        with pytest.raises(exceptions.JMESPathTypeError):
            self.funcs._subtype_check([1, 'a'], allowed, ['t'], 'f')

    def test_subtype_check_dynamic_unknown_first(self):
        allowed = [('float', 'int', 'long'), ('unicode', 'str')]
        with pytest.raises(exceptions.JMESPathTypeError):
            self.funcs._subtype_check([True, 1], allowed, ['t'], 'f')

    def test_convert_to_jmespath_type(self):
        assert self.funcs._convert_to_jmespath_type('int') == 'number'
        assert self.funcs._convert_to_jmespath_type('str') == 'string'
        assert self.funcs._convert_to_jmespath_type('list') == 'array'
        assert self.funcs._convert_to_jmespath_type('dict') == 'object'
        assert self.funcs._convert_to_jmespath_type('bool') == 'boolean'
        assert self.funcs._convert_to_jmespath_type('NoneType') == 'null'
        assert self.funcs._convert_to_jmespath_type('nothing') == 'unknown'


# ---------------------------------------------------------------------------
# Individual function implementations
# ---------------------------------------------------------------------------

class TestAbs:
    def test_abs_positive(self):
        assert Functions().call_function('abs', [5]) == 5

    def test_abs_negative(self):
        assert Functions().call_function('abs', [-5.5]) == 5.5


class TestAvg:
    def test_avg(self):
        assert Functions().call_function('avg', [[1, 2, 3]]) == 2

    def test_avg_empty(self):
        assert Functions().call_function('avg', [[]]) is None

    def test_avg_float(self):
        assert Functions().call_function('avg', [[1.0, 2.0]]) == 1.5


class TestNotNull:
    def test_not_null_first(self):
        assert Functions().call_function('not_null', [1, 2, 3]) == 1

    def test_not_null_skips_none(self):
        assert Functions().call_function('not_null', [None, None, 'x']) == 'x'

    def test_not_null_all_none(self):
        assert Functions().call_function('not_null', [None, None]) is None

    def test_not_null_false(self):
        # False is not None, so it is returned.
        assert Functions().call_function('not_null', [False]) is False


class TestToArray:
    def test_to_array_from_list_returns_same(self):
        funcs = Functions()
        arg = [1, 2]
        result = funcs.call_function('to_array', [arg])
        assert result is arg

    def test_to_array_from_scalar(self):
        assert Functions().call_function('to_array', [1]) == [1]

    def test_to_array_from_none(self):
        assert Functions().call_function('to_array', [None]) == [None]

    def test_to_array_from_string(self):
        assert Functions().call_function('to_array', ['foo']) == ['foo']


class TestToString:
    def test_to_string_returns_string(self):
        assert Functions().call_function('to_string', ['foo']) == 'foo'

    def test_to_string_number(self):
        assert Functions().call_function('to_string', [1]) == '1'

    def test_to_string_list_compact(self):
        assert Functions().call_function('to_string', [[1, 2]]) == '[1,2]'

    def test_to_string_dict_compact(self):
        result = Functions().call_function('to_string', [{'a': 1, 'b': 2}])
        assert result == '{"a":1,"b":2}'

    def test_to_string_bool(self):
        assert Functions().call_function('to_string', [True]) == 'true'

    def test_to_string_null(self):
        assert Functions().call_function('to_string', [None]) == 'null'


class TestToNumber:
    def test_to_number_list_returns_none(self):
        assert Functions().call_function('to_number', [[]]) is None

    def test_to_number_dict_returns_none(self):
        assert Functions().call_function('to_number', [{}]) is None

    def test_to_number_bool_returns_none(self):
        assert Functions().call_function('to_number', [True]) is None

    def test_to_number_none_returns_none(self):
        assert Functions().call_function('to_number', [None]) is None

    def test_to_number_int(self):
        assert Functions().call_function('to_number', [3]) == 3

    def test_to_number_float(self):
        assert Functions().call_function('to_number', [3.5]) == 3.5

    def test_to_number_string_int(self):
        result = Functions().call_function('to_number', ['3'])
        assert result == 3
        assert isinstance(result, int)

    def test_to_number_string_float(self):
        result = Functions().call_function('to_number', ['3.5'])
        assert result == 3.5
        assert isinstance(result, float)

    def test_to_number_invalid_string(self):
        assert Functions().call_function('to_number', ['abc']) is None


class TestContains:
    def test_contains_string_true(self):
        assert Functions().call_function('contains', ['foobar', 'oba']) is True

    def test_contains_string_false(self):
        assert Functions().call_function('contains', ['foobar', 'xy']) is False

    def test_contains_array_true(self):
        assert Functions().call_function('contains', [[1, 2, 3], 2]) is True

    def test_contains_array_false(self):
        assert Functions().call_function('contains', [[1, 2, 3], 9]) is False


class TestLength:
    def test_length_string(self):
        assert Functions().call_function('length', ['abcd']) == 4

    def test_length_array(self):
        assert Functions().call_function('length', [[1, 2, 3]]) == 3

    def test_length_object(self):
        assert Functions().call_function('length', [{'a': 1, 'b': 2}]) == 2


class TestEndsStartsWith:
    def test_ends_with_true(self):
        assert Functions().call_function('ends_with', ['foobar', 'bar']) is True

    def test_ends_with_false(self):
        assert Functions().call_function('ends_with', ['foobar', 'foo']) is False

    def test_starts_with_true(self):
        assert Functions().call_function('starts_with', ['foobar', 'foo']) is True

    def test_starts_with_false(self):
        assert Functions().call_function('starts_with', ['foobar', 'bar']) is False


class TestReverse:
    def test_reverse_string(self):
        assert Functions().call_function('reverse', ['abc']) == 'cba'

    def test_reverse_array(self):
        assert Functions().call_function('reverse', [[1, 2, 3]]) == [3, 2, 1]

    def test_reverse_empty_array(self):
        assert Functions().call_function('reverse', [[]]) == []


class TestCeilFloor:
    def test_ceil(self):
        assert Functions().call_function('ceil', [1.2]) == 2

    def test_ceil_int(self):
        assert Functions().call_function('ceil', [2]) == 2

    def test_floor(self):
        assert Functions().call_function('floor', [1.8]) == 1

    def test_floor_int(self):
        assert Functions().call_function('floor', [2]) == 2


class TestJoin:
    def test_join(self):
        assert Functions().call_function('join', [',', ['a', 'b', 'c']]) == 'a,b,c'

    def test_join_empty(self):
        assert Functions().call_function('join', ['-', []]) == ''


class TestMinMaxSumSort:
    def test_max_numbers(self):
        assert Functions().call_function('max', [[1, 3, 2]]) == 3

    def test_max_strings(self):
        assert Functions().call_function('max', [['a', 'c', 'b']]) == 'c'

    def test_max_empty(self):
        assert Functions().call_function('max', [[]]) is None

    def test_min_numbers(self):
        assert Functions().call_function('min', [[1, 3, 2]]) == 1

    def test_min_strings(self):
        assert Functions().call_function('min', [['a', 'c', 'b']]) == 'a'

    def test_min_empty(self):
        assert Functions().call_function('min', [[]]) is None

    def test_sort(self):
        assert Functions().call_function('sort', [[3, 1, 2]]) == [1, 2, 3]

    def test_sort_strings(self):
        assert Functions().call_function('sort', [['b', 'a', 'c']]) == ['a', 'b', 'c']

    def test_sum(self):
        assert Functions().call_function('sum', [[1, 2, 3]]) == 6

    def test_sum_empty(self):
        assert Functions().call_function('sum', [[]]) == 0


class TestMerge:
    def test_merge_single(self):
        assert Functions().call_function('merge', [{'a': 1}]) == {'a': 1}

    def test_merge_multiple(self):
        result = Functions().call_function(
            'merge', [{'a': 1}, {'b': 2}, {'a': 3}])
        assert result == {'a': 3, 'b': 2}


class TestKeysValues:
    def test_keys(self):
        assert Functions().call_function('keys', [{'a': 1, 'b': 2}]) == ['a', 'b']

    def test_values(self):
        assert Functions().call_function('values', [{'a': 1, 'b': 2}]) == [1, 2]


class TestType:
    def test_type_string(self):
        assert Functions().call_function('type', ['foo']) == 'string'

    def test_type_boolean(self):
        assert Functions().call_function('type', [True]) == 'boolean'

    def test_type_array(self):
        assert Functions().call_function('type', [[1, 2]]) == 'array'

    def test_type_object(self):
        assert Functions().call_function('type', [{'a': 1}]) == 'object'

    def test_type_number_int(self):
        assert Functions().call_function('type', [1]) == 'number'

    def test_type_number_float(self):
        assert Functions().call_function('type', [1.5]) == 'number'

    def test_type_null(self):
        assert Functions().call_function('type', [None]) == 'null'


# ---------------------------------------------------------------------------
# ExpRef-based functions
# ---------------------------------------------------------------------------

class FakeExpref:
    """Minimal stand-in for an _Expression object.

    ``visit`` ignores its first argument (the expression node) and applies
    ``func`` to the current value.
    """

    def __init__(self, func, expression=None):
        self._func = func
        self.expression = expression

    def visit(self, expression, value):
        return self._func(value)


class TestMap:
    def test_map(self):
        expref = FakeExpref(lambda x: x * 2)
        result = Functions().call_function('map', [expref, [1, 2, 3]])
        assert result == [2, 4, 6]

    def test_map_empty(self):
        expref = FakeExpref(lambda x: x)
        assert Functions().call_function('map', [expref, []]) == []


class TestSortBy:
    def test_sort_by_numbers(self):
        expref = FakeExpref(lambda x: x['age'])
        arr = [{'age': 30}, {'age': 10}, {'age': 20}]
        result = Functions().call_function('sort_by', [arr, expref])
        assert [x['age'] for x in result] == [10, 20, 30]

    def test_sort_by_strings(self):
        expref = FakeExpref(lambda x: x['name'])
        arr = [{'name': 'c'}, {'name': 'a'}, {'name': 'b'}]
        result = Functions().call_function('sort_by', [arr, expref])
        assert [x['name'] for x in result] == ['a', 'b', 'c']

    def test_sort_by_empty_returns_same(self):
        expref = FakeExpref(lambda x: x)
        arr = []
        result = Functions().call_function('sort_by', [arr, expref])
        assert result == []

    def test_sort_by_invalid_first_type(self):
        expref = FakeExpref(lambda x: x['val'])
        arr = [{'val': [1]}, {'val': [2]}]
        with pytest.raises(exceptions.JMESPathTypeError) as excinfo:
            Functions().call_function('sort_by', [arr, expref])
        assert excinfo.value.function_name == 'sort_by'

    def test_sort_by_mixed_types_in_array(self):
        expref = FakeExpref(lambda x: x['val'])
        arr = [{'val': 1}, {'val': 'a'}]
        with pytest.raises(exceptions.JMESPathTypeError):
            Functions().call_function('sort_by', [arr, expref])


class TestMinByMaxBy:
    def test_min_by(self):
        expref = FakeExpref(lambda x: x['age'])
        arr = [{'age': 30, 'n': 'a'}, {'age': 10, 'n': 'b'},
               {'age': 20, 'n': 'c'}]
        result = Functions().call_function('min_by', [arr, expref])
        assert result == {'age': 10, 'n': 'b'}

    def test_max_by(self):
        expref = FakeExpref(lambda x: x['age'])
        arr = [{'age': 30, 'n': 'a'}, {'age': 10, 'n': 'b'},
               {'age': 20, 'n': 'c'}]
        result = Functions().call_function('max_by', [arr, expref])
        assert result == {'age': 30, 'n': 'a'}

    def test_min_by_empty(self):
        expref = FakeExpref(lambda x: x['age'])
        assert Functions().call_function('min_by', [[], expref]) is None

    def test_max_by_empty(self):
        expref = FakeExpref(lambda x: x['age'])
        assert Functions().call_function('max_by', [[], expref]) is None

    def test_min_by_invalid_type(self):
        expref = FakeExpref(lambda x: x['val'])
        arr = [{'val': [1]}, {'val': [2]}]
        with pytest.raises(exceptions.JMESPathTypeError) as excinfo:
            Functions().call_function('min_by', [arr, expref])
        assert excinfo.value.function_name == 'min_by'

    def test_max_by_invalid_type(self):
        expref = FakeExpref(lambda x: x['val'])
        arr = [{'val': [1]}, {'val': [2]}]
        with pytest.raises(exceptions.JMESPathTypeError) as excinfo:
            Functions().call_function('max_by', [arr, expref])
        assert excinfo.value.function_name == 'max_by'


# ---------------------------------------------------------------------------
# Module-level constant sanity checks
# ---------------------------------------------------------------------------

class TestTypeMaps:
    def test_types_map_basic(self):
        assert TYPES_MAP['int'] == 'number'
        assert TYPES_MAP['str'] == 'string'
        assert TYPES_MAP['list'] == 'array'
        assert TYPES_MAP['dict'] == 'object'
        assert TYPES_MAP['bool'] == 'boolean'
        assert TYPES_MAP['NoneType'] == 'null'
        assert TYPES_MAP['_Expression'] == 'expref'

    def test_reverse_types_map_basic(self):
        assert REVERSE_TYPES_MAP['number'] == ('float', 'int', 'long')
        assert REVERSE_TYPES_MAP['string'] == ('unicode', 'str')
        assert REVERSE_TYPES_MAP['array'] == ('list', '_Projection')
        assert REVERSE_TYPES_MAP['object'] == ('dict', 'OrderedDict')
        assert REVERSE_TYPES_MAP['null'] == ('NoneType',)
        assert REVERSE_TYPES_MAP['expref'] == ('_Expression',)
