import pytest

from boltons.formatutils import (
    construct_format_field_str,
    split_format_str,
    infer_positional_format_args,
    get_format_args,
    tokenize_format_str,
    BaseFormatField,
    DeferredValue,
)


# ---------------------------------------------------------------------------
# construct_format_field_str
# ---------------------------------------------------------------------------

def test_construct_format_field_str_basic():
    assert construct_format_field_str('name', '', None) == '{name}'


def test_construct_format_field_str_none_fname():
    assert construct_format_field_str(None, 'spec', 'r') == ''


def test_construct_format_field_str_with_conv_and_spec():
    assert construct_format_field_str('name', '>10', 'r') == '{name!r:>10}'


def test_construct_format_field_str_with_conv_only():
    assert construct_format_field_str('x', '', 's') == '{x!s}'


def test_construct_format_field_str_with_spec_only():
    assert construct_format_field_str('x', '.2f', None) == '{x:.2f}'


# ---------------------------------------------------------------------------
# split_format_str
# ---------------------------------------------------------------------------

def test_split_format_str_simple():
    result = split_format_str('Hello {name}!')
    assert result == [('Hello ', '{name}'), ('!', None)]


def test_split_format_str_with_spec_and_conv():
    result = split_format_str('{value:>8} and {other!r}')
    assert result == [('', '{value:>8}'), (' and ', '{other!r}')]


def test_split_format_str_no_fields():
    assert split_format_str('just text') == [('just text', None)]


def test_split_format_str_escaped_braces():
    result = split_format_str('{{literal}}')
    assert result == [('{', None), ('literal}', None)]


# ---------------------------------------------------------------------------
# infer_positional_format_args
# ---------------------------------------------------------------------------

def test_infer_positional_format_args_empty_fstring():
    assert infer_positional_format_args('{}') == '{0}'
    assert infer_positional_format_args('{:d}') == '{0:d}'
    assert infer_positional_format_args('{!r}') == '{0!r}'
    assert infer_positional_format_args('{.2f}') == '{0.2f}'
    assert infer_positional_format_args('{[0]}') == '{0[0]}'


def test_infer_positional_format_args_multiple():
    assert infer_positional_format_args('{} {} {}') == '{0} {1} {2}'


def test_infer_positional_format_args_with_named():
    assert infer_positional_format_args('{name} and {}') == '{name} and {0}'


def test_infer_positional_format_args_escaped_braces_preserved():
    assert infer_positional_format_args('{{}}') == '{{}}'
    assert infer_positional_format_args('{{{}}}') == '{{{0}}}'


def test_infer_positional_format_args_noop_for_numbered():
    assert infer_positional_format_args('{0} {1}') == '{0} {1}'


# ---------------------------------------------------------------------------
# get_format_args
# ---------------------------------------------------------------------------

def test_get_format_args_docstring_example():
    fargs, fkwargs = get_format_args("{noun} is {1:d} years old{punct}")
    assert fargs == [(1, int)]
    assert fkwargs == [('noun', str), ('punct', str)]


def test_get_format_args_types():
    fargs, fkwargs = get_format_args("{a:f} {b:x} {c:s} {0}")
    assert fargs == [(0, str)]
    assert fkwargs == [('a', float), ('b', int), ('c', str)]


def test_get_format_args_dedup():
    fargs, fkwargs = get_format_args("{a} {a} {0} {0}")
    assert fargs == [(0, str)]
    assert fkwargs == [('a', str)]


def test_get_format_args_nested_spec():
    fargs, fkwargs = get_format_args("{value:{width}}")
    assert fkwargs == [('value', str), ('width', str)]
    assert fargs == []


def test_get_format_args_compound_raises():
    with pytest.raises(ValueError):
        get_format_args("{a.b}")


def test_get_format_args_compound_index_raises():
    with pytest.raises(ValueError):
        get_format_args("{a[0]}")


def test_get_format_args_anonymous_raises():
    with pytest.raises(ValueError):
        get_format_args("{}")


def test_get_format_args_empty():
    assert get_format_args("") == ([], [])


# ---------------------------------------------------------------------------
# tokenize_format_str
# ---------------------------------------------------------------------------

def test_tokenize_format_str_basic():
    result = tokenize_format_str("Hello {name}!")
    assert len(result) == 3
    assert result[0] == "Hello "
    assert isinstance(result[1], BaseFormatField)
    assert result[1].fname == "name"
    assert result[2] == "!"


def test_tokenize_format_str_resolve_pos():
    result = tokenize_format_str("{} {:d}")
    assert isinstance(result[0], BaseFormatField)
    assert result[0].fname == "0"
    assert result[1] == " "
    assert isinstance(result[2], BaseFormatField)
    assert result[2].fname == "1"
    assert result[2].fspec == "d"


def test_tokenize_format_str_no_resolve_pos():
    result = tokenize_format_str("{}", resolve_pos=False)
    assert isinstance(result[0], BaseFormatField)
    assert result[0].fname == ""


def test_tokenize_format_str_only_literal():
    assert tokenize_format_str("plain") == ["plain"]


def test_tokenize_format_str_empty():
    assert tokenize_format_str("") == []


def test_tokenize_format_str_escaped_braces():
    result = tokenize_format_str("{{}}")
    assert "{".__str__() in result
    # Escaped braces produce literal strings, not fields
    assert all(not isinstance(x, BaseFormatField) for x in result)


# ---------------------------------------------------------------------------
# BaseFormatField
# ---------------------------------------------------------------------------

def test_baseformatfield_basic():
    f = BaseFormatField("name")
    assert f.fname == "name"
    assert f.base_name == "name"
    assert f.subpath == []
    assert f.is_positional is False
    assert f.fspec == ""
    assert f.conv is None
    assert f.type_char == ""
    assert f.type_func is str
    assert f.fstr == "{name}"


def test_baseformatfield_positional_digit():
    f = BaseFormatField("0")
    assert f.is_positional is True


def test_baseformatfield_positional_empty():
    f = BaseFormatField("")
    assert f.is_positional is True
    assert f.base_name == ""


def test_baseformatfield_subpath():
    f = BaseFormatField("obj.attr")
    assert f.base_name == "obj"
    assert f.subpath == ["attr"]
    assert f.is_positional is False


def test_baseformatfield_subpath_index():
    f = BaseFormatField("items[0]")
    assert f.base_name == "items"
    assert f.subpath == ["0]"]


def test_baseformatfield_spec_types():
    assert BaseFormatField("x", "f").type_func is float
    assert BaseFormatField("x", "d").type_func is int
    assert BaseFormatField("x", "s").type_func is str


def test_baseformatfield_subfields():
    f = BaseFormatField("x", ">{width}")
    assert f.subfields == ["width"]


def test_baseformatfield_str_and_repr():
    f = BaseFormatField("name", ">8", "r")
    assert str(f) == "{name!r:>8}"
    assert repr(f) == "BaseFormatField('name', '>8', 'r')"


def test_baseformatfield_repr_no_conv():
    f = BaseFormatField("name", ">8")
    assert repr(f) == "BaseFormatField('name', '>8')"

    f2 = BaseFormatField("name")
    assert repr(f2) == "BaseFormatField('name')"


def test_baseformatfield_set_fname():
    f = BaseFormatField("a")
    f.set_fname("b.c")
    assert f.base_name == "b"
    assert f.subpath == ["c"]
    assert f.fname == "b.c"


def test_baseformatfield_set_fspec():
    f = BaseFormatField("x")
    f.set_fspec(".2f")
    assert f.fspec == ".2f"
    assert f.type_char == "f"
    assert f.type_func is float


def test_baseformatfield_set_conv():
    f = BaseFormatField("x")
    f.set_conv("r")
    assert f.conv == "r"
    assert f.conv_func is None


# ---------------------------------------------------------------------------
# DeferredValue
# ---------------------------------------------------------------------------

def test_deferred_value_caches_by_default():
    calls = []

    def compute():
        calls.append(1)
        return 42

    dv = DeferredValue(compute)
    assert dv.get_value() == 42
    assert dv.get_value() == 42
    assert len(calls) == 1


def test_deferred_value_no_cache():
    calls = []

    def compute():
        calls.append(1)
        return 42

    dv = DeferredValue(compute, cache_value=False)
    assert dv.get_value() == 42
    assert dv.get_value() == 42
    assert len(calls) == 2


def test_deferred_value_str_int_float_repr():
    dv = DeferredValue(lambda: 10)
    assert str(dv) == "10"
    assert int(dv) == 10
    assert float(dv) == 10.0
    assert repr(dv) == "10"


def test_deferred_value_format():
    dv = DeferredValue(lambda: 3.14159)
    assert format(dv, ".2f") == "3.14"
    assert "{:.1f}".format(dv) == "3.1"


def test_deferred_value_format_with_conv():
    dv = DeferredValue(lambda: 42)
    assert format(dv, "d") == "42"


def test_deferred_value_format_fallback():
    # fmt type 'x' with a str value: value.__format__ raises -> falls back
    dv = DeferredValue(lambda: "abc")
    assert format(dv, "") == "abc"


def test_deferred_value_uses_cached_in_format():
    calls = []

    def compute():
        calls.append(1)
        return 7

    dv = DeferredValue(compute)
    assert "{}".format(dv) == "7"
    assert "{}".format(dv) == "7"
    assert len(calls) == 1


def test_deferred_value_custom_value_type():
    class Custom:
        def __format__(self, spec):
            return "custom:" + spec

    dv = DeferredValue(lambda: Custom())
    assert format(dv, "x") == "custom:x"
