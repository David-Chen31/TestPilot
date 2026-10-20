import math

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
# Tests for the ``signature`` decorator
# ---------------------------------------------------------------------------

class TestSignatureDecorator:
    def test_signature_attaches_arguments_tuple(self):
        sig = ({'types': ['number']},)
        def func(self, arg):
            return arg

        decorated = signature(*sig)(func)
        assert decorated is func
        assert func.signature == sig

    def test_signature_no_args_gives_empty_tuple(self):
        def func(self):
            return None
        signature()(func)
        assert func.signature == ()

    def test_signature_multiple_decorations_last_wins(self):
        def func(self, a):
            return a
        signature({'types': ['number']})(func)
        signature({'types': ['string']})(func)
        assert func.signature == ({'types': ['string']},)


# ---------------------------------------------------------------------------
# Tests for the FunctionRegistry metaclass
# ---------------------------------------------------------------------------

class TestFunctionRegistry:
    def test_registers_decorated_func_methods(self):
        class Sample(metaclass=FunctionRegistry):
            @signature({'types': ['number']})
            def _func_double(self, arg):
                return arg * 2

            def not_a_function(self):
                return 1

            def _func_undecorated(self):
                return 2

        table = Sample.FUNCTION_TABLE
        assert set(table.keys()) == {'double'}
        assert table['double']['signature'] == ({'types': ['number']},)
        assert table['double']['function'].__name__ == '_func_double'

    def test_multiple_functions_registered_with_name_stripping(self):
        class Sample(metaclass=FunctionRegistry):
            @signature({'types': []})
            def _func_a(self):
                return 'a'

            @signature({'types': []})
            def _func_b(self):
                return 'b'

        assert set(Sample.FUNCTION_TABLE.keys()) == {'a', 'b'}

    def test_empty_subclass_has_empty_table(self):
        class Empty(metaclass=FunctionRegistry):
            pass

        assert Empty.FUNCTION_TABLE == {}

    def test_registry_inherits_from_base_table(self):
        class Base(metaclass=FunctionRegistry):
            @signature({'types': []})
            def _func_base_fn(self):
                return 'base'

        class Child(Base):
            @signature({'types': []})
            def _func_child_fn(self):
                return 'child'

        # get_methods uses inspect.getmembers which includes inherited members.
        assert 'base_fn' in Child.FUNCTION_TABLE
        assert 'child_fn' in Child.FUNCTION_TABLE


# ---------------------------------------------------------------------------
# Helper: an expression reference stub for map/sort_by/min_by/max_by
# ---------------------------------------------------------------------------

class _ExprRef:
    def __init__(self, expression, visitor):
        self.expression = expression
        self._visitor = visitor

    def visit(self, expression, element):
        return self._visitor(expression, element)


# ---------------------------------------------------------------------------
# call_function dispatching & validation
# ---------------------------------------------------------------------------

class TestCallFunction:
    def setup_method(self):
        self.funcs = Functions()

    def test_dispatch_known_function(self):
        assert self.funcs.call_function('abs', [-5]) == 5

    def test_unknown_function_raises(self):
        with pytest.raises(exceptions.UnknownFunctionError) as exc:
            self.funcs.call_function('does_not_exist', [])
        assert 'does_not_exist' in str(exc.value)

    def test_arity_error_wrong_count(self):
        with pytest.raises(exceptions.ArityError):
            self.funcs.call_function('abs', [1, 2])

    def test_variadic_arity_error_too_few(self):
        # not_null requires at least 1 variadic arg.
        with pytest.raises(exceptions.VariadictArityError):
            self.funcs.call_function('not_null', [])

    def test_variadic_accepts_more(self):
        assert self.funcs.call_function(
            'not_null', [None, None, 3, 4]) == 3

    def test_type_error_wrong_type(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.funcs.call_function('abs', ['not a number'])


# ---------------------------------------------------------------------------
# Tests for the individual function implementations
# ---------------------------------------------------------------------------

class TestAbs:
    def test_positive(self):
        assert Functions()._func_abs(5) == 5

    def test_negative_float(self):
        assert Functions()._func_abs(-2.5) == 2.5

    def test_zero(self):
        assert Functions()._func_abs(0) == 0


class TestAvg:
    def test_avg_values(self):
        assert Functions()._func_avg([1, 2, 3, 4]) == 2.5

    def test_avg_empty_returns_none(self):
        assert Functions()._func_avg([]) is None


class TestNotNull:
    def test_returns_first_non_none(self):
        assert Functions()._func_not_null(None, None, 'x', 'y') == 'x'

    def test_all_none_returns_none(self):
        assert Functions()._func_not_null(None, None) is None

    def test_falsy_non_none_returned(self):
        assert Functions()._func_not_null(None, 0, '') == 0


class TestToArray:
    def test_list_passes_through(self):
        lst = [1, 2]
        assert Functions()._func_to_array(lst) is lst

    def test_scalar_wrapped(self):
        assert Functions()._func_to_array(1) == [1]

    def test_string_wrapped(self):
        assert Functions()._func_to_array('ab') == ['ab']

    def test_none_wrapped(self):
        assert Functions()._func_to_array(None) == [None]


class TestToString:
    def test_string_passthrough(self):
        assert Functions()._func_to_string('hello') == 'hello'

    def test_number(self):
        assert Functions()._func_to_string(5) == '5'

    def test_list_compact_json(self):
        assert Functions()._func_to_string([1, 2, 3]) == '[1,2,3]'

    def test_dict_compact_json(self):
        assert Functions()._func_to_string({'a': 1}) == '{"a":1}'

    def test_none(self):
        assert Functions()._func_to_string(None) == 'null'

    def test_bool(self):
        assert Functions()._func_to_string(True) == 'true'


class TestToNumber:
    def test_int_passthrough(self):
        assert Functions()._func_to_number(5) == 5

    def test_float_passthrough(self):
        assert Functions()._func_to_number(2.5) == 2.5

    def test_numeric_string_int(self):
        result = Functions()._func_to_number('42')
        assert result == 42
        assert isinstance(result, int)

    def test_numeric_string_float(self):
        result = Functions()._func_to_number('2.5')
        assert result == 2.5
        assert isinstance(result, float)

    def test_non_numeric_string(self):
        assert Functions()._func_to_number('abc') is None

    def test_bool_returns_none(self):
        assert Functions()._func_to_number(True) is None

    def test_list_returns_none(self):
        assert Functions()._func_to_number([1]) is None

    def test_dict_returns_none(self):
        assert Functions()._func_to_number({'a': 1}) is None

    def test_none_returns_none(self):
        assert Functions()._func_to_number(None) is None


class TestContains:
    def test_string_substring(self):
        assert Functions()._func_contains('foobar', 'oba') is True

    def test_string_not_found(self):
        assert Functions()._func_contains('foobar', 'xyz') is False

    def test_array_contains(self):
        assert Functions()._func_contains([1, 2, 3], 2) is True

    def test_array_not_contains(self):
        assert Functions()._func_contains([1, 2, 3], 9) is False


class TestLength:
    def test_string(self):
        assert Functions()._func_length('hello') == 5

    def test_list(self):
        assert Functions()._func_length([1, 2, 3]) == 3

    def test_dict(self):
        assert Functions()._func_length({'a': 1, 'b': 2}) == 2

    def test_empty(self):
        assert Functions()._func_length('') == 0


class TestEndsWith:
    def test_true(self):
        assert Functions()._func_ends_with('foobar', 'bar') is True

    def test_false(self):
        assert Functions()._func_ends_with('foobar', 'foo') is False


class TestStartsWith:
    def test_true(self):
        assert Functions()._func_starts_with('foobar', 'foo') is True

    def test_false(self):
        assert Functions()._func_starts_with('foobar', 'bar') is False


class TestReverse:
    def test_string(self):
        assert Functions()._func_reverse('abc') == 'cba'

    def test_list(self):
        result = Functions()._func_reverse([1, 2, 3])
        assert result == [3, 2, 1]
        assert isinstance(result, list)

    def test_empty_string(self):
        assert Functions()._func_reverse('') == ''


class TestCeil:
    def test_positive(self):
        assert Functions()._func_ceil(1.2) == 2

    def test_negative(self):
        assert Functions()._func_ceil(-1.2) == -1

    def test_integer(self):
        assert Functions()._func_ceil(3) == 3


class TestFloor:
    def test_positive(self):
        assert Functions()._func_floor(1.8) == 1

    def test_negative(self):
        assert Functions()._func_floor(-1.2) == -2

    def test_integer(self):
        assert Functions()._func_floor(3) == 3


class TestJoin:
    def test_join_comma(self):
        assert Functions()._func_join(',', ['a', 'b', 'c']) == 'a,b,c'

    def test_join_empty_list(self):
        assert Functions()._func_join('-', []) == ''

    def test_join_single(self):
        assert Functions()._func_join('', ['x']) == 'x'


class TestMap:
    def test_map_applies_expr(self):
        expref = _ExprRef('id', lambda expr, element: element[expr])
        data = [{'id': 1}, {'id': 2}]
        assert Functions()._func_map(expref, data) == [1, 2]

    def test_map_empty(self):
        expref = _ExprRef('id', lambda expr, element: element[expr])
        assert Functions()._func_map(expref, []) == []


class TestMax:
    def test_max_numbers(self):
        assert Functions()._func_max([1, 5, 3]) == 5

    def test_max_strings(self):
        assert Functions()._func_max(['a', 'c', 'b']) == 'c'

    def test_max_empty_returns_none(self):
        assert Functions()._func_max([]) is None


class TestMin:
    def test_min_numbers(self):
        assert Functions()._func_min([4, 2, 9]) == 2

    def test_min_strings(self):
        assert Functions()._func_min(['b', 'a', 'c']) == 'a'

    def test_min_empty_returns_none(self):
        assert Functions()._func_min([]) is None


class TestMerge:
    def test_merge_two(self):
        result = Functions()._func_merge({'a': 1}, {'b': 2})
        assert result == {'a': 1, 'b': 2}

    def test_merge_overrides(self):
        result = Functions()._func_merge({'a': 1}, {'a': 2})
        assert result == {'a': 2}

    def test_merge_single(self):
        assert Functions()._func_merge({'a': 1}) == {'a': 1}

    def test_merge_does_not_mutate_inputs(self):
        first = {'a': 1}
        second = {'b': 2}
        Functions()._func_merge(first, second)
        assert first == {'a': 1}
        assert second == {'b': 2}


class TestSort:
    def test_sort_numbers(self):
        assert Functions()._func_sort([3, 1, 2]) == [1, 2, 3]

    def test_sort_strings(self):
        assert Functions()._func_sort(['c', 'a', 'b']) == ['a', 'b', 'c']

    def test_sort_empty(self):
        assert Functions()._func_sort([]) == []

    def test_sort_returns_new_list(self):
        original = [2, 1]
        result = Functions()._func_sort(original)
        assert result == [1, 2]
        assert original == [2, 1]


class TestSum:
    def test_sum(self):
        assert Functions()._func_sum([1, 2, 3]) == 6

    def test_sum_empty(self):
        assert Functions()._func_sum([]) == 0

    def test_sum_floats(self):
        assert Functions()._func_sum([1.5, 2.5]) == 4.0


class TestKeys:
    def test_keys(self):
        assert Functions()._func_keys({'a': 1, 'b': 2}) == ['a', 'b']

    def test_keys_empty(self):
        assert Functions()._func_keys({}) == []


class TestValues:
    def test_values(self):
        assert Functions()._func_values({'a': 1, 'b': 2}) == [1, 2]

    def test_values_empty(self):
        assert Functions()._func_values({}) == []


class TestType:
    def test_string(self):
        assert Functions()._func_type('x') == 'string'

    def test_boolean(self):
        assert Functions()._func_type(True) == 'boolean'

    def test_array(self):
        assert Functions()._func_type([1, 2]) == 'array'

    def test_object(self):
        assert Functions()._func_type({'a': 1}) == 'object'

    def test_number_int(self):
        assert Functions()._func_type(5) == 'number'

    def test_number_float(self):
        assert Functions()._func_type(5.5) == 'number'

    def test_null(self):
        assert Functions()._func_type(None) == 'null'


class TestSortBy:
    def test_sort_by_number(self):
        expref = _ExprRef('age', lambda expr, element: element[expr])
        data = [{'age': 30}, {'age': 10}, {'age': 20}]
        result = Functions()._func_sort_by(data, expref)
        assert [d['age'] for d in result] == [10, 20, 30]

    def test_sort_by_string(self):
        expref = _ExprRef('name', lambda expr, element: element[expr])
        data = [{'name': 'c'}, {'name': 'a'}, {'name': 'b'}]
        result = Functions()._func_sort_by(data, expref)
        assert [d['name'] for d in result] == ['a', 'b', 'c']

    def test_sort_by_empty_array(self):
        expref = _ExprRef('x', lambda expr, element: element[expr])
        assert Functions()._func_sort_by([], expref) == []

    def test_sort_by_invalid_key_type_raises(self):
        expref = _ExprRef('x', lambda expr, element: element[expr])
        data = [{'x': [1, 2]}, {'x': [3]}]
        with pytest.raises(exceptions.JMESPathTypeError):
            Functions()._func_sort_by(data, expref)

    def test_sort_by_mixed_types_raises(self):
        expref = _ExprRef('x', lambda expr, element: element[expr])
        data = [{'x': 1}, {'x': 'a'}]
        with pytest.raises(exceptions.JMESPathTypeError):
            Functions()._func_sort_by(data, expref)


class TestMinBy:
    def test_min_by_number(self):
        expref = _ExprRef('age', lambda expr, element: element[expr])
        data = [{'age': 30, 'n': 'a'}, {'age': 10, 'n': 'b'}]
        assert Functions()._func_min_by(data, expref) == {'age': 10, 'n': 'b'}

    def test_min_by_string(self):
        expref = _ExprRef('name', lambda expr, element: element[expr])
        data = [{'name': 'b'}, {'name': 'a'}]
        assert Functions()._func_min_by(data, expref) == {'name': 'a'}

    def test_min_by_empty_returns_none(self):
        expref = _ExprRef('x', lambda expr, element: element[expr])
        assert Functions()._func_min_by([], expref) is None

    def test_min_by_invalid_type_raises(self):
        expref = _ExprRef('x', lambda expr, element: element[expr])
        data = [{'x': [1]}, {'x': [2]}]
        with pytest.raises(exceptions.JMESPathTypeError):
            Functions()._func_min_by(data, expref)


class TestMaxBy:
    def test_max_by_number(self):
        expref = _ExprRef('age', lambda expr, element: element[expr])
        data = [{'age': 30, 'n': 'a'}, {'age': 10, 'n': 'b'}]
        assert Functions()._func_max_by(data, expref) == {'age': 30, 'n': 'a'}

    def test_max_by_string(self):
        expref = _ExprRef('name', lambda expr, element: element[expr])
        data = [{'name': 'b'}, {'name': 'c'}]
        assert Functions()._func_max_by(data, expref) == {'name': 'c'}

    def test_max_by_empty_returns_none(self):
        expref = _ExprRef('x', lambda expr, element: element[expr])
        assert Functions()._func_max_by([], expref) is None

    def test_max_by_invalid_type_raises(self):
        expref = _ExprRef('x', lambda expr, element: element[expr])
        data = [{'x': None}, {'x': None}]
        with pytest.raises(exceptions.JMESPathTypeError):
            Functions()._func_max_by(data, expref)


# ---------------------------------------------------------------------------
# Type-checking helpers
# ---------------------------------------------------------------------------

class TestTypeCheck:
    def setup_method(self):
        self.funcs = Functions()

    def test_get_allowed_pytypes_simple(self):
        allowed, subtypes = self.funcs._get_allowed_pytypes(['number'])
        assert set(allowed) == {'float', 'int', 'long'}
        assert subtypes == []

    def test_get_allowed_pytypes_with_subtype(self):
        allowed, subtypes = self.funcs._get_allowed_pytypes(['array-number'])
        assert 'list' in allowed
        assert len(subtypes) == 1
        assert set(subtypes[0]) == {'float', 'int', 'long'}

    def test_get_allowed_pytypes_multiple(self):
        allowed, subtypes = self.funcs._get_allowed_pytypes(
            ['array-string', 'array-number'])
        assert len(subtypes) == 2

    def test_convert_to_jmespath_type(self):
        f = self.funcs._convert_to_jmespath_type
        assert f('str') == 'string'
        assert f('int') == 'number'
        assert f('bool') == 'boolean'
        assert f('dict') == 'object'
        assert f('NoneType') == 'null'
        assert f('SomethingWeird') == 'unknown'

    def test_type_check_single_ok(self):
        # No exception.
        self.funcs._type_check_single(5, ['number'], 'myfunc')

    def test_type_check_single_wrong(self):
        with pytest.raises(exceptions.JMESPathTypeError) as exc:
            self.funcs._type_check_single('x', ['number'], 'myfunc')
        assert exc.value.function_name == 'myfunc'

    def test_type_check_skips_empty_types(self):
        # types == [] means no type restriction; should not raise.
        self.funcs._type_check([object()], [{'types': []}], 'fn')

    def test_subtype_check_single_all_ok(self):
        self.funcs._subtype_check(
            [1, 2, 3], [('float', 'int', 'long')], ['array-number'], 'fn')

    def test_subtype_check_single_bad(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.funcs._subtype_check(
                [1, 'x'], [('float', 'int', 'long')], ['array-number'], 'fn')

    def test_subtype_check_dynamic_mixed_bad(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.funcs._subtype_check(
                [1, 'x'],
                [('float', 'int', 'long'), ('str', 'unicode')],
                ['array-number', 'array-string'],
                'fn')

    def test_subtype_check_dynamic_first_invalid(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.funcs._subtype_check(
                [True],
                [('float', 'int', 'long'), ('str', 'unicode')],
                ['array-number', 'array-string'],
                'fn')

    def test_subtype_check_dynamic_empty_current(self):
        # Empty list with multiple subtypes should not raise.
        self.funcs._subtype_check(
            [], [('float', 'int', 'long'), ('str', 'unicode')],
            ['array-number', 'array-string'], 'fn')

    def test_subtype_bool_is_not_number(self):
        with pytest.raises(exceptions.JMESPathTypeError):
            self.funcs._subtype_check(
                [True], [('float', 'int', 'long')], ['array-number'], 'fn')


# ---------------------------------------------------------------------------
# Sanity checks on module constants
# ---------------------------------------------------------------------------

class TestTypeMaps:
    def test_types_map_entries(self):
        assert TYPES_MAP['bool'] == 'boolean'
        assert TYPES_MAP['int'] == 'number'
        assert TYPES_MAP['str'] == 'string'
        assert TYPES_MAP['NoneType'] == 'null'

    def test_reverse_types_map_entries(self):
        assert REVERSE_TYPES_MAP['boolean'] == ('bool',)
        assert 'list' in REVERSE_TYPES_MAP['array']
        assert 'int' in REVERSE_TYPES_MAP['number']

    def test_all_registered_functions_present(self):
        expected = {
            'abs', 'avg', 'not_null', 'to_array', 'to_string', 'to_number',
            'contains', 'length', 'ends_with', 'starts_with', 'reverse',
            'ceil', 'floor', 'join', 'map', 'max', 'merge', 'min', 'sort',
            'sum', 'keys', 'values', 'type', 'sort_by', 'min_by', 'max_by',
        }
        assert set(Functions.FUNCTION_TABLE.keys()) == expected
