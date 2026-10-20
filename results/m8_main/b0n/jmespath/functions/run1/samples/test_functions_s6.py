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


# ---------------------------------------------------------------- signature


class TestSignature:
    def test_signature_sets_attribute(self):
        @signature({'types': ['number']}, {'types': ['string']})
        def func(self, a, b):
            return a, b

        assert func.signature == ({'types': ['number']}, {'types': ['string']})

    def test_signature_returns_same_function(self):
        def func(self):
            pass

        decorated = signature({'types': []})(func)
        assert decorated is func

    def test_signature_with_no_arguments(self):
        @signature()
        def func(self):
            pass

        assert func.signature == ()


# ------------------------------------------------------ FunctionRegistry


class TestFunctionRegistry:
    def test_registry_populates_function_table(self):
        class MyFunctions(metaclass=FunctionRegistry):
            @signature({'types': ['number']})
            def _func_double(self, arg):
                return arg * 2

        assert 'double' in MyFunctions.FUNCTION_TABLE
        entry = MyFunctions.FUNCTION_TABLE['double']
        assert entry['signature'] == ({'types': ['number']},)
        assert entry['function'].__name__ == '_func_double'

    def test_registry_ignores_non_prefixed_methods(self):
        class MyFunctions(metaclass=FunctionRegistry):
            @signature({'types': []})
            def not_a_func(self, arg):
                return arg

        assert MyFunctions.FUNCTION_TABLE == {}

    def test_registry_ignores_undecorated_methods(self):
        class MyFunctions(metaclass=FunctionRegistry):
            def _func_nosig(self, arg):
                return arg

        assert MyFunctions.FUNCTION_TABLE == {}

    def test_registry_multiple_functions(self):
        class MyFunctions(metaclass=FunctionRegistry):
            @signature({'types': ['number']})
            def _func_a(self, arg):
                return 'a'

            @signature({'types': ['string']})
            def _func_b(self, arg):
                return 'b'

        assert set(MyFunctions.FUNCTION_TABLE.keys()) == {'a', 'b'}

    def test_subclass_inherits_but_repopulates(self):
        class Base(metaclass=FunctionRegistry):
            @signature({'types': []})
            def _func_base_fn(self, arg):
                return arg

        class Child(Base):
            @signature({'types': []})
            def _func_child_fn(self, arg):
                return arg

        # Because it's a metaclass over the class, the child gets its own table
        # populated from getmembers, which includes inherited functions.
        assert 'base_fn' in Child.FUNCTION_TABLE
        assert 'child_fn' in Child.FUNCTION_TABLE


# -------------------------------------------------------------- Functions


@pytest.fixture
def fns():
    return Functions()


class TestCallFunction:
    def test_unknown_function(self, fns):
        with pytest.raises(exceptions.UnknownFunctionError) as excinfo:
            fns.call_function('nope', [])
        assert 'Unknown function: nope()' in str(excinfo.value)

    def test_call_abs(self, fns):
        assert fns.call_function('abs', [-5]) == 5

    def test_call_abs_wrong_type(self, fns):
        with pytest.raises(exceptions.JMESPathTypeError):
            fns.call_function('abs', ['foo'])

    def test_call_abs_wrong_arity(self, fns):
        with pytest.raises(exceptions.ArityError):
            fns.call_function('abs', [])


class TestValidateArguments:
    def test_arity_error(self, fns):
        with pytest.raises(exceptions.ArityError):
            fns.call_function('contains', ['a'])

    def test_variadic_arity_error(self, fns):
        with pytest.raises(exceptions.VariadictArityError):
            fns.call_function('not_null', [])

    def test_variadic_ok(self, fns):
        assert fns.call_function('not_null', [None, None, 5]) == 5


class TestTypeChecking:
    def test_type_error_contains(self, fns):
        with pytest.raises(exceptions.JMESPathTypeError) as excinfo:
            fns.call_function('contains', [5, 1])
        err = excinfo.value
        assert err.function_name == 'contains'
        assert err.current_value == 5
        assert err.actual_type == 'number'

    def test_subtype_single(self, fns):
        # array-number requires all elements be numbers.
        with pytest.raises(exceptions.JMESPathTypeError):
            fns.call_function('sum', [[1, 2, 'x']])

    def test_subtype_dynamic(self, fns):
        # max accepts array-number or array-string; mixing fails.
        with pytest.raises(exceptions.JMESPathTypeError):
            fns.call_function('max', [[1, 'b']])

    def test_get_allowed_pytypes(self, fns):
        allowed, subtypes = fns._get_allowed_pytypes(['array-number'])
        assert 'list' in allowed
        assert 'int' in subtypes[0]

    def test_convert_to_jmespath_type_known(self, fns):
        assert fns._convert_to_jmespath_type('int') == 'number'
        assert fns._convert_to_jmespath_type('str') == 'string'

    def test_convert_to_jmespath_type_unknown(self, fns):
        assert fns._convert_to_jmespath_type('set') == 'unknown'


class TestSimpleFunctions:
    def test_avg(self, fns):
        assert fns.call_function('avg', [[1, 2, 3]]) == 2

    def test_avg_empty(self, fns):
        assert fns.call_function('avg', [[]]) is None

    def test_not_null(self, fns):
        assert fns.call_function('not_null', [None, 'a', 'b']) == 'a'

    def test_not_null_all_none(self, fns):
        assert fns.call_function('not_null', [None, None]) is None

    def test_to_array_list(self, fns):
        assert fns.call_function('to_array', [[1, 2]]) == [1, 2]

    def test_to_array_scalar(self, fns):
        assert fns.call_function('to_array', [5]) == [5]

    def test_to_string_passthrough(self, fns):
        assert fns.call_function('to_string', ['hi']) == 'hi'

    def test_to_string_dict(self, fns):
        assert fns.call_function('to_string', [{'a': 1}]) == '{"a":1}'

    def test_to_string_list(self, fns):
        assert fns.call_function('to_string', [[1, 2]]) == '[1,2]'

    def test_to_number_from_string_int(self, fns):
        assert fns.call_function('to_number', ['42']) == 42
        assert isinstance(fns.call_function('to_number', ['42']), int)

    def test_to_number_from_string_float(self, fns):
        result = fns.call_function('to_number', ['3.5'])
        assert result == 3.5
        assert isinstance(result, float)

    def test_to_number_invalid_string(self, fns):
        assert fns.call_function('to_number', ['abc']) is None

    def test_to_number_bool(self, fns):
        assert fns.call_function('to_number', [True]) is None

    def test_to_number_none(self, fns):
        assert fns.call_function('to_number', [None]) is None

    def test_to_number_list(self, fns):
        assert fns.call_function('to_number', [[1]]) is None

    def test_to_number_dict(self, fns):
        assert fns.call_function('to_number', [{'a': 1}]) is None

    def test_to_number_passthrough_int(self, fns):
        result = fns.call_function('to_number', [7])
        assert result == 7
        assert isinstance(result, int)

    def test_to_number_passthrough_float(self, fns):
        result = fns.call_function('to_number', [7.5])
        assert result == 7.5
        assert isinstance(result, float)

    def test_contains_string(self, fns):
        assert fns.call_function('contains', ['foobar', 'oba']) is True

    def test_contains_list(self, fns):
        assert fns.call_function('contains', [[1, 2, 3], 2]) is True

    def test_length_string(self, fns):
        assert fns.call_function('length', ['abcd']) == 4

    def test_length_list(self, fns):
        assert fns.call_function('length', [[1, 2, 3]]) == 3

    def test_length_object(self, fns):
        assert fns.call_function('length', [{'a': 1, 'b': 2}]) == 2

    def test_ends_with(self, fns):
        assert fns.call_function('ends_with', ['foobar', 'bar']) is True
        assert fns.call_function('ends_with', ['foobar', 'foo']) is False

    def test_starts_with(self, fns):
        assert fns.call_function('starts_with', ['foobar', 'foo']) is True
        assert fns.call_function('starts_with', ['foobar', 'bar']) is False

    def test_reverse_string(self, fns):
        assert fns.call_function('reverse', ['abc']) == 'cba'

    def test_reverse_list(self, fns):
        assert fns.call_function('reverse', [[1, 2, 3]]) == [3, 2, 1]

    def test_ceil(self, fns):
        assert fns.call_function('ceil', [1.2]) == 2

    def test_floor(self, fns):
        assert fns.call_function('floor', [1.8]) == 1

    def test_join(self, fns):
        assert fns.call_function('join', [',', ['a', 'b', 'c']]) == 'a,b,c'

    def test_join_empty(self, fns):
        assert fns.call_function('join', [',', []]) == ''

    def test_max(self, fns):
        assert fns.call_function('max', [[1, 5, 3]]) == 5

    def test_max_empty(self, fns):
        assert fns.call_function('max', [[]]) is None

    def test_max_strings(self, fns):
        assert fns.call_function('max', [['a', 'c', 'b']]) == 'c'

    def test_min(self, fns):
        assert fns.call_function('min', [[1, 5, 3]]) == 1

    def test_min_empty(self, fns):
        assert fns.call_function('min', [[]]) is None

    def test_sort(self, fns):
        assert fns.call_function('sort', [[3, 1, 2]]) == [1, 2, 3]

    def test_sort_does_not_mutate(self, fns):
        original = [3, 1, 2]
        fns.call_function('sort', [original])
        assert original == [3, 1, 2]

    def test_sum(self, fns):
        assert fns.call_function('sum', [[1, 2, 3]]) == 6

    def test_sum_empty(self, fns):
        assert fns.call_function('sum', [[]]) == 0

    def test_keys(self, fns):
        result = fns.call_function('keys', [{'a': 1, 'b': 2}])
        assert sorted(result) == ['a', 'b']

    def test_values(self, fns):
        result = fns.call_function('values', [{'a': 1, 'b': 2}])
        assert sorted(result) == [1, 2]

    def test_merge(self, fns):
        assert fns.call_function('merge', [{'a': 1}, {'b': 2}]) == {'a': 1, 'b': 2}

    def test_merge_overrides(self, fns):
        assert fns.call_function('merge', [{'a': 1}, {'a': 2}]) == {'a': 2}


class TestTypeFunction:
    def test_type_string(self, fns):
        assert fns.call_function('type', ['x']) == 'string'

    def test_type_boolean(self, fns):
        assert fns.call_function('type', [True]) == 'boolean'

    def test_type_array(self, fns):
        assert fns.call_function('type', [[1]]) == 'array'

    def test_type_object(self, fns):
        assert fns.call_function('type', [{'a': 1}]) == 'object'

    def test_type_number(self, fns):
        assert fns.call_function('type', [1]) == 'number'
        assert fns.call_function('type', [1.5]) == 'number'

    def test_type_null(self, fns):
        assert fns.call_function('type', [None]) == 'null'


class FakeExpref:
    def __init__(self, expression, accessor):
        self.expression = expression
        self.accessor = accessor

    def visit(self, expression, element):
        return self.accessor(element)


class TestSortBy:
    def test_sort_by(self, fns):
        data = [{'a': 3}, {'a': 1}, {'a': 2}]
        expref = FakeExpref('a', lambda x: x['a'])
        result = fns.call_function('sort_by', [data, expref])
        assert result == [{'a': 1}, {'a': 2}, {'a': 3}]

    def test_sort_by_empty(self, fns):
        expref = FakeExpref('a', lambda x: x['a'])
        assert fns.call_function('sort_by', [[], expref]) == []

    def test_sort_by_invalid_type(self, fns):
        data = [{'a': True}, {'a': False}]
        expref = FakeExpref('a', lambda x: x['a'])
        with pytest.raises(exceptions.JMESPathTypeError) as excinfo:
            fns.call_function('sort_by', [data, expref])
        assert excinfo.value.function_name == 'sort_by'

    def test_sort_by_inconsistent_types(self, fns):
        # First element number, second element string -> keyfunc raises.
        data = [{'a': 1}, {'a': 'x'}]
        expref = FakeExpref('a', lambda x: x['a'])
        with pytest.raises(exceptions.JMESPathTypeError):
            fns.call_function('sort_by', [data, expref])


class TestMinMaxBy:
    def test_min_by(self, fns):
        data = [{'a': 3}, {'a': 1}, {'a': 2}]
        expref = FakeExpref('a', lambda x: x['a'])
        assert fns.call_function('min_by', [data, expref]) == {'a': 1}

    def test_min_by_empty(self, fns):
        expref = FakeExpref('a', lambda x: x['a'])
        assert fns.call_function('min_by', [[], expref]) is None

    def test_min_by_bad_type(self, fns):
        data = [{'a': 1}, {'a': None}]
        expref = FakeExpref('a', lambda x: x['a'])
        with pytest.raises(exceptions.JMESPathTypeError):
            fns.call_function('min_by', [data, expref])

    def test_max_by(self, fns):
        data = [{'a': 3}, {'a': 1}, {'a': 2}]
        expref = FakeExpref('a', lambda x: x['a'])
        assert fns.call_function('max_by', [data, expref]) == {'a': 3}

    def test_max_by_empty(self, fns):
        expref = FakeExpref('a', lambda x: x['a'])
        assert fns.call_function('max_by', [[], expref]) is None

    def test_max_by_bad_type(self, fns):
        data = [{'a': 1}, {'a': []}]
        expref = FakeExpref('a', lambda x: x['a'])
        with pytest.raises(exceptions.JMESPathTypeError):
            fns.call_function('max_by', [data, expref])


class TestMap:
    def test_map(self, fns):
        data = [{'a': 1}, {'a': 2}]
        expref = FakeExpref('a', lambda x: x['a'])
        assert fns.call_function('map', [expref, data]) == [1, 2]

    def test_map_empty(self, fns):
        expref = FakeExpref('a', lambda x: x['a'])
        assert fns.call_function('map', [expref, []]) == []


class TestCreateKeyFunc:
    def test_allowed(self, fns):
        expref = FakeExpref('a', lambda x: x['a'])
        keyfunc = fns._create_key_func(expref, ['number', 'string'], 'f')
        assert keyfunc({'a': 5}) == 5

    def test_not_allowed(self, fns):
        expref = FakeExpref('a', lambda x: x['a'])
        keyfunc = fns._create_key_func(expref, ['number', 'string'], 'f')
        with pytest.raises(exceptions.JMESPathTypeError):
            keyfunc({'a': [1]})
