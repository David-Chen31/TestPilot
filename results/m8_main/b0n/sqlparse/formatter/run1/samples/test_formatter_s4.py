import pytest

from sqlparse import filters
from sqlparse.exceptions import SQLParseError
from sqlparse.formatter import validate_options, build_filter_stack
from sqlparse.filters import FilterStack


# ---------------------------------------------------------------------------
# validate_options
# ---------------------------------------------------------------------------

def test_validate_options_returns_same_dict():
    options = {}
    result = validate_options(options)
    assert result is options


def test_validate_options_defaults():
    options = {}
    result = validate_options(options)
    # Defaults set by the function
    assert result['indent_columns'] is False
    assert result['indent_after_first'] is False
    assert result['indent_char'] == ' '
    assert result['indent_width'] == 2
    assert result['wrap_after'] == 0
    assert result['comma_first'] is False
    assert result['compact'] is False
    assert result['right_margin'] is None
    # Not set unless provided
    assert 'keyword_case' not in result
    assert 'identifier_case' not in result
    assert 'output_format' not in result
    assert 'truncate_strings' not in result
    assert 'truncate_char' not in result
    assert 'strip_comments' not in result
    assert 'use_space_around_operators' not in result
    assert 'strip_whitespace' not in result
    assert 'reindent' not in result
    assert 'reindent_aligned' not in result


@pytest.mark.parametrize('value', [None, 'upper', 'lower', 'capitalize'])
def test_validate_options_keyword_case_valid(value):
    options = {'keyword_case': value}
    validate_options(options)
    assert options['keyword_case'] == value


def test_validate_options_keyword_case_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'keyword_case': 'title'})
    assert 'keyword_case' in str(exc.value)


@pytest.mark.parametrize('value', [None, 'upper', 'lower', 'capitalize'])
def test_validate_options_identifier_case_valid(value):
    options = {'identifier_case': value}
    validate_options(options)
    assert options['identifier_case'] == value


def test_validate_options_identifier_case_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'identifier_case': 'mixed'})
    assert 'identifier_case' in str(exc.value)


@pytest.mark.parametrize('value', [None, 'sql', 'python', 'php'])
def test_validate_options_output_format_valid(value):
    options = {'output_format': value}
    validate_options(options)
    assert options['output_format'] == value


def test_validate_options_output_format_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'output_format': 'ruby'})
    assert 'output format' in str(exc.value).lower()


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_strip_comments_valid(value):
    options = {'strip_comments': value}
    validate_options(options)
    assert options['strip_comments'] == value


def test_validate_options_strip_comments_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'strip_comments': 'yes'})
    assert 'strip_comments' in str(exc.value)


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_use_space_around_operators_valid(value):
    options = {'use_space_around_operators': value}
    validate_options(options)
    assert options['use_space_around_operators'] == value


def test_validate_options_use_space_around_operators_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'use_space_around_operators': 1})
    assert 'use_space_around_operators' in str(exc.value)


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_strip_whitespace_valid(value):
    options = {'strip_whitespace': value}
    validate_options(options)
    assert options['strip_whitespace'] == value


def test_validate_options_strip_whitespace_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'strip_whitespace': 'no'})
    assert 'strip_whitespace' in str(exc.value)


def test_validate_options_truncate_strings_none():
    options = {'truncate_strings': None}
    result = validate_options(options)
    assert 'truncate_strings' not in result
    assert 'truncate_char' not in result


@pytest.mark.parametrize('value, expected', [('10', 10), (10, 10), (2, 2), (100, 100)])
def test_validate_options_truncate_strings_valid(value, expected):
    options = {'truncate_strings': value}
    result = validate_options(options)
    assert result['truncate_strings'] == expected
    assert isinstance(result['truncate_strings'], int)
    assert result['truncate_char'] == '[...]'


def test_validate_options_truncate_strings_custom_char():
    options = {'truncate_strings': 5, 'truncate_char': '...'}
    result = validate_options(options)
    assert result['truncate_strings'] == 5
    assert result['truncate_char'] == '...'


@pytest.mark.parametrize('value', ['abc', object()])
def test_validate_options_truncate_strings_invalid_type(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'truncate_strings': value})
    assert 'truncate_strings' in str(exc.value)


@pytest.mark.parametrize('value', [0, 1, -5, '1', '0'])
def test_validate_options_truncate_strings_out_of_range(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'truncate_strings': value})
    assert 'truncate_strings' in str(exc.value)


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_indent_columns_valid(value):
    options = {'indent_columns': value}
    result = validate_options(options)
    assert result['indent_columns'] is value


def test_validate_options_indent_columns_enforces_reindent():
    options = {'indent_columns': True}
    result = validate_options(options)
    assert result['reindent'] is True


def test_validate_options_indent_columns_no_reindent_when_false():
    options = {'indent_columns': False}
    result = validate_options(options)
    assert 'reindent' not in result


def test_validate_options_indent_columns_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_columns': 'yes'})
    assert 'indent_columns' in str(exc.value)


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_reindent_valid(value):
    options = {'reindent': value}
    result = validate_options(options)
    assert result['reindent'] is value


def test_validate_options_reindent_enforces_strip_whitespace():
    options = {'reindent': True}
    result = validate_options(options)
    assert result['strip_whitespace'] is True


def test_validate_options_reindent_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'reindent': 'yes'})
    assert 'reindent' in str(exc.value)


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_reindent_aligned_valid(value):
    options = {'reindent_aligned': value}
    result = validate_options(options)
    # reindent_aligned not explicitly set back on the dict
    if value:
        assert result['strip_whitespace'] is True


def test_validate_options_reindent_aligned_enforces_strip_whitespace():
    options = {'reindent_aligned': True}
    result = validate_options(options)
    assert result['strip_whitespace'] is True


def test_validate_options_reindent_aligned_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'reindent_aligned': 'yes'})
    assert 'reindent' in str(exc.value)


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_indent_after_first_valid(value):
    options = {'indent_after_first': value}
    result = validate_options(options)
    assert result['indent_after_first'] is value


def test_validate_options_indent_after_first_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_after_first': 'yes'})
    assert 'indent_after_first' in str(exc.value)


def test_validate_options_indent_tabs_true_sets_tab():
    options = {'indent_tabs': True}
    result = validate_options(options)
    assert result['indent_char'] == '\t'


def test_validate_options_indent_tabs_false_sets_space():
    options = {'indent_tabs': False}
    result = validate_options(options)
    assert result['indent_char'] == ' '


def test_validate_options_indent_tabs_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_tabs': 'yes'})
    assert 'indent_tabs' in str(exc.value)


@pytest.mark.parametrize('value, expected', [(4, 4), ('6', 6), (2, 2)])
def test_validate_options_indent_width_valid(value, expected):
    options = {'indent_width': value}
    result = validate_options(options)
    assert result['indent_width'] == expected
    assert isinstance(result['indent_width'], int)


@pytest.mark.parametrize('value', ['abc', None, object()])
def test_validate_options_indent_width_invalid_type(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_width': value})
    assert 'indent_width requires an integer' in str(exc.value)


@pytest.mark.parametrize('value', [0, -1, '-3'])
def test_validate_options_indent_width_non_positive(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_width': value})
    assert 'indent_width requires a positive integer' in str(exc.value)


@pytest.mark.parametrize('value, expected', [(0, 0), (50, 50), ('100', 100)])
def test_validate_options_wrap_after_valid(value, expected):
    options = {'wrap_after': value}
    result = validate_options(options)
    assert result['wrap_after'] == expected
    assert isinstance(result['wrap_after'], int)


@pytest.mark.parametrize('value', ['abc', None, object()])
def test_validate_options_wrap_after_invalid_type(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'wrap_after': value})
    assert 'wrap_after requires an integer' in str(exc.value)


@pytest.mark.parametrize('value', [-1, '-5'])
def test_validate_options_wrap_after_negative(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'wrap_after': value})
    assert 'wrap_after requires a positive integer' in str(exc.value)


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_comma_first_valid(value):
    options = {'comma_first': value}
    result = validate_options(options)
    assert result['comma_first'] is value


def test_validate_options_comma_first_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'comma_first': 'yes'})
    assert 'comma_first requires a boolean value' in str(exc.value)


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_compact_valid(value):
    options = {'compact': value}
    result = validate_options(options)
    assert result['compact'] is value


def test_validate_options_compact_invalid():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'compact': 'yes'})
    assert 'compact requires a boolean value' in str(exc.value)


def test_validate_options_right_margin_none():
    options = {'right_margin': None}
    result = validate_options(options)
    assert result['right_margin'] is None


@pytest.mark.parametrize('value, expected', [(10, 10), (50, 50), ('20', 20)])
def test_validate_options_right_margin_valid(value, expected):
    options = {'right_margin': value}
    result = validate_options(options)
    assert result['right_margin'] == expected
    assert isinstance(result['right_margin'], int)


@pytest.mark.parametrize('value', ['abc', object()])
def test_validate_options_right_margin_invalid_type(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'right_margin': value})
    assert 'right_margin requires an integer' in str(exc.value)


@pytest.mark.parametrize('value', [0, 5, 9, '-10'])
def test_validate_options_right_margin_too_small(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'right_margin': value})
    assert 'right_margin requires an integer > 10' in str(exc.value)


# ---------------------------------------------------------------------------
# build_filter_stack
# ---------------------------------------------------------------------------

def _validated(options):
    return validate_options(options)


def test_build_filter_stack_empty_options():
    stack = FilterStack()
    options = _validated({})
    result = build_filter_stack(stack, options)
    assert result is stack
    assert stack.preprocess == []
    assert stack.stmtprocess == []
    assert stack.postprocess == []


def test_build_filter_stack_keyword_case():
    stack = FilterStack()
    result = build_filter_stack(stack, _validated({'keyword_case': 'upper'}))
    assert result is stack
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)


def test_build_filter_stack_identifier_case():
    stack = FilterStack()
    build_filter_stack(stack, _validated({'identifier_case': 'lower'}))
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.IdentifierCaseFilter)


def test_build_filter_stack_truncate_strings():
    stack = FilterStack()
    build_filter_stack(stack, _validated({'truncate_strings': 5}))
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.TruncateStringFilter)


def test_build_filter_stack_space_around_operators():
    stack = FilterStack()
    build_filter_stack(stack, _validated({'use_space_around_operators': True}))
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.SpacesAroundOperatorsFilter)


def test_build_filter_stack_strip_comments():
    stack = FilterStack()
    build_filter_stack(stack, _validated({'strip_comments': True}))
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripCommentsFilter)


def test_build_filter_stack_strip_whitespace():
    stack = FilterStack()
    build_filter_stack(stack, _validated({'strip_whitespace': True}))
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)


def test_build_filter_stack_reindent_adds_strip_and_reindent():
    stack = FilterStack()
    build_filter_stack(stack, _validated({'reindent': True}))
    assert len(stack.stmtprocess) == 2
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)
    assert isinstance(stack.stmtprocess[1], filters.ReindentFilter)


def test_build_filter_stack_reindent_aligned():
    stack = FilterStack()
    build_filter_stack(stack, _validated({'reindent_aligned': True}))
    # AlignedIndentFilter goes after StripWhitespaceFilter
    assert len(stack.stmtprocess) == 2
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)
    assert isinstance(stack.stmtprocess[1], filters.AlignedIndentFilter)


def test_build_filter_stack_right_margin():
    stack = FilterStack()
    build_filter_stack(stack, _validated({'right_margin': 50}))
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.RightMarginFilter)


def test_build_filter_stack_output_format_php():
    stack = FilterStack()
    build_filter_stack(stack, _validated({'output_format': 'php'}))
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPHPFilter)


def test_build_filter_stack_output_format_python():
    stack = FilterStack()
    build_filter_stack(stack, _validated({'output_format': 'python'}))
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)


def test_build_filter_stack_output_format_sql_no_postprocessor():
    stack = FilterStack()
    build_filter_stack(stack, _validated({'output_format': 'sql'}))
    assert stack.postprocess == []


def test_build_filter_stack_output_format_uppercase_php():
    stack = FilterStack()
    build_filter_stack(stack, _validated({'output_format': 'PHP'}))
    # validate_options allows 'php' only; 'PHP' would raise.
    # This test documents the actual behavior of build_filter_stack looking up
    # options which we inject directly rather than through validate_options.
    assert isinstance(stack.postprocess[0], filters.OutputPHPFilter)


def test_build_filter_stack_combined_options():
    stack = FilterStack()
    options = _validated({
        'keyword_case': 'upper',
        'identifier_case': 'lower',
        'truncate_strings': 3,
        'strip_comments': True,
        'strip_whitespace': True,
        'right_margin': 40,
        'output_format': 'python',
    })
    build_filter_stack(stack, options)
    assert len(stack.preprocess) == 3
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)
    assert isinstance(stack.preprocess[1], filters.IdentifierCaseFilter)
    assert isinstance(stack.preprocess[2], filters.TruncateStringFilter)
    assert len(stack.stmtprocess) == 2
    assert isinstance(stack.stmtprocess[0], filters.StripCommentsFilter)
    assert isinstance(stack.stmtprocess[1], filters.StripWhitespaceFilter)
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)
