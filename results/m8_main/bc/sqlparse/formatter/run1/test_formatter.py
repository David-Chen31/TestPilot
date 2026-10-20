"""Unit tests for sqlparse.formatter."""

import pytest

from sqlparse.formatter import validate_options, build_filter_stack
from sqlparse.exceptions import SQLParseError
from sqlparse import filters


# ---------------------------------------------------------------------------
# validate_options
# ---------------------------------------------------------------------------

def test_validate_options_empty_dict_adds_defaults():
    options = {}
    result = validate_options(options)
    assert result is options
    assert options['indent_columns'] is False
    assert options['indent_char'] == ' '
    assert options['indent_width'] == 2
    assert options['wrap_after'] == 0
    assert options['comma_first'] is False
    assert options['compact'] is False
    assert options['indent_after_first'] is False
    assert options['right_margin'] is None


@pytest.mark.parametrize("case", [None, 'upper', 'lower', 'capitalize'])
def test_validate_options_valid_keyword_case(case):
    options = {'keyword_case': case}
    validate_options(options)
    assert options.get('keyword_case') == case


@pytest.mark.parametrize("case", ['UPPER', 'title', '', 1, 'mixed'])
def test_validate_options_invalid_keyword_case(case):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'keyword_case': case})
    assert 'Invalid value for keyword_case' in str(exc.value)


@pytest.mark.parametrize("case", [None, 'upper', 'lower', 'capitalize'])
def test_validate_options_valid_identifier_case(case):
    options = {'identifier_case': case}
    validate_options(options)
    assert options.get('identifier_case') == case


@pytest.mark.parametrize("case", ['UPPER', 'title', '', 2])
def test_validate_options_invalid_identifier_case(case):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'identifier_case': case})
    assert 'Invalid value for identifier_case' in str(exc.value)


@pytest.mark.parametrize("fmt", [None, 'sql', 'python', 'php'])
def test_validate_options_valid_output_format(fmt):
    options = {'output_format': fmt}
    validate_options(options)
    assert options.get('output_format') == fmt


@pytest.mark.parametrize("fmt", ['json', 'SQL', '', 'xml'])
def test_validate_options_invalid_output_format(fmt):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'output_format': fmt})
    assert 'Unknown output format' in str(exc.value)


@pytest.mark.parametrize("value", [True, False])
def test_validate_options_strip_comments_valid(value):
    validate_options({'strip_comments': value})


# The implementation checks membership in [True, False] with `in`, so
# 0 and 1 (which compare equal to False and True) are accepted.
@pytest.mark.parametrize("value", [0, 1])
def test_validate_options_strip_comments_int_equivalent(value):
    validate_options({'strip_comments': value})


@pytest.mark.parametrize("value", ['yes', None, 2])
def test_validate_options_strip_comments_invalid(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'strip_comments': value})
    assert 'Invalid value for strip_comments' in str(exc.value)


def test_validate_options_strip_comments_default_false():
    options = {}
    validate_options(options)
    # default not explicitly set, but does not raise
    assert 'strip_comments' not in options


@pytest.mark.parametrize("value", [True, False])
def test_validate_options_use_space_around_operators_valid(value):
    validate_options({'use_space_around_operators': value})


@pytest.mark.parametrize("value", [0, 1])
def test_validate_options_use_space_around_operators_int_equivalent(value):
    validate_options({'use_space_around_operators': value})


@pytest.mark.parametrize("value", ['yes', None])
def test_validate_options_use_space_around_operators_invalid(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'use_space_around_operators': value})
    assert 'Invalid value for use_space_around_operators' in str(exc.value)


@pytest.mark.parametrize("value", [True, False])
def test_validate_options_strip_whitespace_valid(value):
    validate_options({'strip_whitespace': value})


@pytest.mark.parametrize("value", [0, 1])
def test_validate_options_strip_whitespace_int_equivalent(value):
    validate_options({'strip_whitespace': value})


@pytest.mark.parametrize("value", ['x', None])
def test_validate_options_strip_whitespace_invalid(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'strip_whitespace': value})
    assert 'Invalid value for strip_whitespace' in str(exc.value)


def test_validate_options_truncate_strings_none_noop():
    options = {'truncate_strings': None}
    validate_options(options)
    assert options['truncate_strings'] is None
    assert 'truncate_char' not in options


@pytest.mark.parametrize("value,expected", [('5', 5), (10, 10), (2, 2),
                                            (100, 100)])
def test_validate_options_truncate_strings_valid(value, expected):
    options = {'truncate_strings': value}
    validate_options(options)
    assert options['truncate_strings'] == expected
    assert options['truncate_char'] == '[...]'


def test_validate_options_truncate_strings_keeps_custom_char():
    options = {'truncate_strings': 5, 'truncate_char': '...'}
    validate_options(options)
    assert options['truncate_strings'] == 5
    assert options['truncate_char'] == '...'


@pytest.mark.parametrize("value", ['abc', [], {}, object()])
def test_validate_options_truncate_strings_not_int(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'truncate_strings': value})
    assert 'Invalid value for truncate_strings' in str(exc.value)


@pytest.mark.parametrize("value", [1, 0, -1, '1', '0', '-5'])
def test_validate_options_truncate_strings_too_small(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'truncate_strings': value})
    assert 'Invalid value for truncate_strings' in str(exc.value)


def test_validate_options_indent_columns_true_enforces_reindent():
    options = {'indent_columns': True}
    validate_options(options)
    assert options['indent_columns'] is True
    assert options['reindent'] is True


def test_validate_options_indent_columns_false():
    options = {'indent_columns': False}
    validate_options(options)
    assert options['indent_columns'] is False
    assert 'reindent' not in options


# 0 and 1 compare equal to False/True so are accepted by the `in` check.
@pytest.mark.parametrize("value", [0, 1])
def test_validate_options_indent_columns_int_equivalent(value):
    validate_options({'indent_columns': value})


@pytest.mark.parametrize("value", ['yes', None])
def test_validate_options_indent_columns_invalid(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_columns': value})
    assert 'Invalid value for indent_columns' in str(exc.value)


def test_validate_options_reindent_true_sets_strip_whitespace():
    options = {'reindent': True}
    validate_options(options)
    assert options['strip_whitespace'] is True


def test_validate_options_reindent_false():
    options = {'reindent': False}
    validate_options(options)
    assert options['reindent'] is False
    assert 'strip_whitespace' not in options


@pytest.mark.parametrize("value", [0, 1])
def test_validate_options_reindent_int_equivalent(value):
    validate_options({'reindent': value})


@pytest.mark.parametrize("value", ['yes', None])
def test_validate_options_reindent_invalid(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'reindent': value})
    assert 'Invalid value for reindent' in str(exc.value)


def test_validate_options_reindent_aligned_true_sets_strip_whitespace():
    options = {'reindent_aligned': True}
    validate_options(options)
    assert options['strip_whitespace'] is True


def test_validate_options_reindent_aligned_false():
    options = {'reindent_aligned': False}
    validate_options(options)
    assert options['reindent_aligned'] is False
    assert 'strip_whitespace' not in options


@pytest.mark.parametrize("value", [0, 1])
def test_validate_options_reindent_aligned_int_equivalent(value):
    validate_options({'reindent_aligned': value})


@pytest.mark.parametrize("value", ['yes', None])
def test_validate_options_reindent_aligned_invalid(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'reindent_aligned': value})
    # message reuses reindent value
    assert 'Invalid value for reindent' in str(exc.value)


@pytest.mark.parametrize("value", [True, False])
def test_validate_options_indent_after_first_valid(value):
    options = {'indent_after_first': value}
    validate_options(options)
    assert options['indent_after_first'] is value


@pytest.mark.parametrize("value", [0, 1])
def test_validate_options_indent_after_first_int_equivalent(value):
    validate_options({'indent_after_first': value})


@pytest.mark.parametrize("value", ['yes', None])
def test_validate_options_indent_after_first_invalid(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_after_first': value})
    assert 'Invalid value for indent_after_first' in str(exc.value)


def test_validate_options_indent_tabs_true_sets_tab_char():
    options = {'indent_tabs': True}
    validate_options(options)
    assert options['indent_char'] == '\t'


def test_validate_options_indent_tabs_false_sets_space_char():
    options = {'indent_tabs': False}
    validate_options(options)
    assert options['indent_char'] == ' '


@pytest.mark.parametrize("value", [0, 1])
def test_validate_options_indent_tabs_int_equivalent(value):
    validate_options({'indent_tabs': value})


@pytest.mark.parametrize("value", ['yes', None])
def test_validate_options_indent_tabs_invalid(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_tabs': value})
    assert 'Invalid value for indent_tabs' in str(exc.value)


@pytest.mark.parametrize("value,expected", [('4', 4), (1, 1), (8, 8)])
def test_validate_options_indent_width_valid(value, expected):
    options = {'indent_width': value}
    validate_options(options)
    assert options['indent_width'] == expected


@pytest.mark.parametrize("value", ['abc', [], {}])
def test_validate_options_indent_width_not_int(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_width': value})
    assert 'indent_width requires an integer' in str(exc.value)


@pytest.mark.parametrize("value", [0, -1, '-3'])
def test_validate_options_indent_width_non_positive(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_width': value})
    assert 'indent_width requires a positive integer' in str(exc.value)


@pytest.mark.parametrize("value,expected", [('10', 10), (0, 0), (5, 5)])
def test_validate_options_wrap_after_valid(value, expected):
    options = {'wrap_after': value}
    validate_options(options)
    assert options['wrap_after'] == expected


@pytest.mark.parametrize("value", ['abc', [], {}])
def test_validate_options_wrap_after_not_int(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'wrap_after': value})
    assert 'wrap_after requires an integer' in str(exc.value)


@pytest.mark.parametrize("value", [-1, '-5'])
def test_validate_options_wrap_after_negative(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'wrap_after': value})
    assert 'wrap_after requires a positive integer' in str(exc.value)


@pytest.mark.parametrize("value", [True, False])
def test_validate_options_comma_first_valid(value):
    options = {'comma_first': value}
    validate_options(options)
    assert options['comma_first'] is value


@pytest.mark.parametrize("value", [0, 1])
def test_validate_options_comma_first_int_equivalent(value):
    validate_options({'comma_first': value})


@pytest.mark.parametrize("value", ['yes', None])
def test_validate_options_comma_first_invalid(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'comma_first': value})
    assert 'comma_first requires a boolean value' in str(exc.value)


@pytest.mark.parametrize("value", [True, False])
def test_validate_options_compact_valid(value):
    options = {'compact': value}
    validate_options(options)
    assert options['compact'] is value


@pytest.mark.parametrize("value", [0, 1])
def test_validate_options_compact_int_equivalent(value):
    validate_options({'compact': value})


@pytest.mark.parametrize("value", ['yes', None])
def test_validate_options_compact_invalid(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'compact': value})
    assert 'compact requires a boolean value' in str(exc.value)


@pytest.mark.parametrize("value,expected", [('15', 15), (10, 10), (100, 100)])
def test_validate_options_right_margin_valid(value, expected):
    options = {'right_margin': value}
    validate_options(options)
    assert options['right_margin'] == expected


@pytest.mark.parametrize("value", ['abc', [], {}])
def test_validate_options_right_margin_not_int(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'right_margin': value})
    assert 'right_margin requires an integer' in str(exc.value)


@pytest.mark.parametrize("value", [0, 5, 9, '-20'])
def test_validate_options_right_margin_too_small(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'right_margin': value})
    assert 'right_margin requires an integer > 10' in str(exc.value)


def test_validate_options_right_margin_none():
    options = {}
    validate_options(options)
    assert options['right_margin'] is None


# ---------------------------------------------------------------------------
# build_filter_stack
# ---------------------------------------------------------------------------

def _stack():
    return filters.FilterStack()


def test_build_filter_stack_returns_stack():
    stack = _stack()
    result = build_filter_stack(stack, {})
    assert result is stack


def test_build_filter_stack_empty_options_no_filters():
    stack = _stack()
    build_filter_stack(stack, {})
    assert stack.preprocess == []
    assert stack.stmtprocess == []
    assert stack.postprocess == []


def test_build_filter_stack_keyword_case():
    stack = _stack()
    build_filter_stack(stack, {'keyword_case': 'upper'})
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)


def test_build_filter_stack_identifier_case():
    stack = _stack()
    build_filter_stack(stack, {'identifier_case': 'lower'})
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.IdentifierCaseFilter)


def test_build_filter_stack_truncate_strings():
    stack = _stack()
    build_filter_stack(stack, {'truncate_strings': 5,
                                'truncate_char': '[...]'})
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.TruncateStringFilter)


def test_build_filter_stack_truncate_strings_falsy_not_added():
    stack = _stack()
    build_filter_stack(stack, {'truncate_strings': 0})
    assert stack.preprocess == []


def test_build_filter_stack_multiple_preprocess():
    stack = _stack()
    build_filter_stack(stack, {
        'keyword_case': 'upper',
        'identifier_case': 'lower',
        'truncate_strings': 5,
        'truncate_char': '[...]',
    })
    assert len(stack.preprocess) == 3
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)
    assert isinstance(stack.preprocess[1], filters.IdentifierCaseFilter)
    assert isinstance(stack.preprocess[2], filters.TruncateStringFilter)


def test_build_filter_stack_space_around_operators():
    stack = _stack()
    build_filter_stack(stack, {'use_space_around_operators': True})
    assert stack.grouping is True
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.SpacesAroundOperatorsFilter)


def test_build_filter_stack_strip_comments():
    stack = _stack()
    build_filter_stack(stack, {'strip_comments': True})
    assert stack.grouping is True
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripCommentsFilter)


def test_build_filter_stack_strip_whitespace():
    stack = _stack()
    build_filter_stack(stack, {'strip_whitespace': True})
    assert stack.grouping is True
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)


def test_build_filter_stack_reindent_adds_strip_and_reindent():
    stack = _stack()
    build_filter_stack(stack, {
        'reindent': True,
        'indent_char': ' ',
        'indent_width': 2,
        'indent_after_first': False,
        'indent_columns': False,
        'wrap_after': 0,
        'comma_first': False,
        'compact': False,
    })
    assert stack.grouping is True
    assert len(stack.stmtprocess) == 2
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)
    assert isinstance(stack.stmtprocess[1], filters.ReindentFilter)


def test_build_filter_stack_reindent_aligned():
    stack = _stack()
    build_filter_stack(stack, {'reindent_aligned': True,
                                'indent_char': ' '})
    assert stack.grouping is True
    # reindent_aligned triggers strip_whitespace path too
    assert len(stack.stmtprocess) == 2
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)
    assert isinstance(stack.stmtprocess[1], filters.AlignedIndentFilter)


def test_build_filter_stack_right_margin():
    stack = _stack()
    build_filter_stack(stack, {'right_margin': 30})
    assert stack.grouping is True
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.RightMarginFilter)


def test_build_filter_stack_output_php():
    stack = _stack()
    build_filter_stack(stack, {'output_format': 'php'})
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPHPFilter)


def test_build_filter_stack_output_python():
    stack = _stack()
    build_filter_stack(stack, {'output_format': 'python'})
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)


@pytest.mark.parametrize("fmt", ['sql', None])
def test_build_filter_stack_output_sql_or_none_no_postprocess(fmt):
    stack = _stack()
    build_filter_stack(stack, {'output_format': fmt} if fmt else {})
    assert stack.postprocess == []


def test_build_filter_stack_output_format_case_insensitive():
    stack = _stack()
    build_filter_stack(stack, {'output_format': 'PHP'})
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPHPFilter)


def test_build_filter_stack_symptoms_ordering():
    # Full option set - assert structural ordering of filters
    stack = _stack()
    build_filter_stack(stack, {
        'keyword_case': 'upper',
        'identifier_case': 'lower',
        'truncate_strings': 5,
        'truncate_char': '[...]',
        'use_space_around_operators': True,
        'strip_comments': True,
        'strip_whitespace': True,
        'reindent': True,
        'indent_char': ' ',
        'indent_width': 2,
        'indent_after_first': False,
        'indent_columns': False,
        'wrap_after': 0,
        'comma_first': False,
        'compact': False,
        'reindent_aligned': False,
        'right_margin': 30,
        'output_format': 'python',
    })
    assert len(stack.preprocess) == 3
    stmt_types = [type(f) for f in stack.stmtprocess]
    assert filters.SpacesAroundOperatorsFilter in stmt_types
    assert filters.StripCommentsFilter in stmt_types
    assert filters.StripWhitespaceFilter in stmt_types
    assert filters.ReindentFilter in stmt_types
    assert filters.RightMarginFilter in stmt_types
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)
