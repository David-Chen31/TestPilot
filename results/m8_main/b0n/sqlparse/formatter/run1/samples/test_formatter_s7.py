import pytest

from sqlparse import filters
from sqlparse.exceptions import SQLParseError
from sqlparse.formatter import validate_options, build_filter_stack


# ---------------------------------------------------------------------------
# validate_options: happy paths / defaults
# ---------------------------------------------------------------------------

def test_validate_options_empty_returns_same_dict_and_sets_defaults():
    opts = {}
    result = validate_options(opts)
    assert result is opts
    # Defaults filled in by validation.
    assert opts['indent_columns'] is False
    assert opts['indent_after_first'] is False
    assert opts['indent_char'] == ' '
    assert opts['indent_width'] == 2
    assert opts['wrap_after'] == 0
    assert opts['comma_first'] is False
    assert opts['compact'] is False
    assert opts['right_margin'] is None
    # These were not set because they defaulted to None/absent.
    assert 'keyword_case' not in opts
    assert 'output_format' not in opts


def test_validate_options_accepts_valid_keyword_case_values():
    for val in (None, 'upper', 'lower', 'capitalize'):
        opts = {'keyword_case': val}
        result = validate_options(opts)
        assert result['keyword_case'] == val


def test_validate_options_accepts_valid_identifier_case_values():
    for val in (None, 'upper', 'lower', 'capitalize'):
        opts = {'identifier_case': val}
        result = validate_options(opts)
        assert result['identifier_case'] == val


def test_validate_options_accepts_valid_output_format_values():
    for val in (None, 'sql', 'python', 'php'):
        opts = {'output_format': val}
        result = validate_options(opts)
        assert result['output_format'] == val


# ---------------------------------------------------------------------------
# validate_options: error handling
# ---------------------------------------------------------------------------

def test_validate_options_invalid_keyword_case():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'keyword_case': 'title'})
    assert "Invalid value for keyword_case: 'title'" in str(exc.value)


def test_validate_options_invalid_identifier_case():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'identifier_case': 'TITLE'})
    assert "Invalid value for identifier_case: 'TITLE'" in str(exc.value)


def test_validate_options_invalid_output_format():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'output_format': 'ruby'})
    assert "Unknown output format: 'ruby'" in str(exc.value)


@pytest.mark.parametrize('key', [
    'strip_comments',
    'use_space_around_operators',
    'strip_whitespace',
    'indent_columns',
    'reindent',
    'reindent_aligned',
    'indent_after_first',
    'indent_tabs',
    'comma_first',
    'compact',
])
def test_validate_options_boolean_fields_reject_non_boolean(key):
    with pytest.raises(SQLParseError) as exc:
        validate_options({key: 'yes'})
    assert key in str(exc.value)


# ---------------------------------------------------------------------------
# validate_options: strip_comments / whitespace flags
# ---------------------------------------------------------------------------

def test_validate_options_strip_comments_true():
    opts = {'strip_comments': True}
    result = validate_options(opts)
    assert result['strip_comments'] is True


def test_validate_options_strip_whitespace_true():
    opts = {'strip_whitespace': True}
    result = validate_options(opts)
    assert result['strip_whitespace'] is True


# ---------------------------------------------------------------------------
# validate_options: truncate_strings
# ---------------------------------------------------------------------------

def test_validate_options_truncate_strings_sets_char_default():
    opts = {'truncate_strings': 5}
    result = validate_options(opts)
    assert result['truncate_strings'] == 5
    assert result['truncate_char'] == '[...]'


def test_validate_options_truncate_strings_keeps_custom_char():
    opts = {'truncate_strings': '10', 'truncate_char': '...'}
    result = validate_options(opts)
    assert result['truncate_strings'] == 10
    assert result['truncate_char'] == '...'


def test_validate_options_truncate_strings_string_int_accepted():
    opts = {'truncate_strings': '3'}
    result = validate_options(opts)
    assert result['truncate_strings'] == 3
    assert result['truncate_char'] == '[...]'


@pytest.mark.parametrize('value', ['abc', 1.5, object()])
def test_validate_options_truncate_strings_non_convertible(value):
    with pytest.raises(SQLParseError):
        validate_options({'truncate_strings': value})


@pytest.mark.parametrize('value', [1, 0, -5])
def test_validate_options_truncate_strings_too_small(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'truncate_strings': value})
    assert 'Invalid value for truncate_strings' in str(exc.value)


# ---------------------------------------------------------------------------
# validate_options: indent_columns enforces reindent
# ---------------------------------------------------------------------------

def test_validate_options_indent_columns_enforces_reindent():
    opts = {'indent_columns': True}
    result = validate_options(opts)
    assert result['indent_columns'] is True
    assert result['reindent'] is True
    # reindent -> strip_whitespace
    assert result['strip_whitespace'] is True


# ---------------------------------------------------------------------------
# validate_options: reindent / reindent_aligned set strip_whitespace
# ---------------------------------------------------------------------------

def test_validate_options_reindent_sets_strip_whitespace():
    opts = {'reindent': True}
    result = validate_options(opts)
    assert result['reindent'] is True
    assert result['strip_whitespace'] is True


def test_validate_options_reindent_aligned_sets_strip_whitespace():
    opts = {'reindent_aligned': True}
    result = validate_options(opts)
    assert result['reindent_aligned'] is True
    assert result['strip_whitespace'] is True


# ---------------------------------------------------------------------------
# validate_options: indent_tabs / indent_char
# ---------------------------------------------------------------------------

def test_validate_options_indent_tabs_sets_tab_char():
    opts = {'indent_tabs': True}
    result = validate_options(opts)
    assert result['indent_char'] == '\t'


def test_validate_options_indent_tabs_false_sets_space_char():
    opts = {'indent_tabs': False}
    result = validate_options(opts)
    assert result['indent_char'] == ' '


# ---------------------------------------------------------------------------
# validate_options: indent_width
# ---------------------------------------------------------------------------

def test_validate_options_indent_width_coerced_from_string():
    opts = {'indent_width': '8'}
    result = validate_options(opts)
    assert result['indent_width'] == 8


@pytest.mark.parametrize('value', ['abc', None, object()])
def test_validate_options_indent_width_non_integer(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_width': value})
    assert 'indent_width requires an integer' in str(exc.value)


@pytest.mark.parametrize('value', [0, -1])
def test_validate_options_indent_width_not_positive(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'indent_width': value})
    assert 'indent_width requires a positive integer' in str(exc.value)


# ---------------------------------------------------------------------------
# validate_options: wrap_after
# ---------------------------------------------------------------------------

def test_validate_options_wrap_after_coerced_from_string():
    opts = {'wrap_after': '80'}
    result = validate_options(opts)
    assert result['wrap_after'] == 80


@pytest.mark.parametrize('value', ['abc', None, object()])
def test_validate_options_wrap_after_non_integer(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'wrap_after': value})
    assert 'wrap_after requires an integer' in str(exc.value)


def test_validate_options_wrap_after_negative():
    with pytest.raises(SQLParseError) as exc:
        validate_options({'wrap_after': -1})
    assert 'wrap_after requires a positive integer' in str(exc.value)


# ---------------------------------------------------------------------------
# validate_options: right_margin
# ---------------------------------------------------------------------------

def test_validate_options_right_margin_none_ok():
    opts = {'right_margin': None}
    result = validate_options(opts)
    assert result['right_margin'] is None


def test_validate_options_right_margin_ok():
    opts = {'right_margin': '80'}
    result = validate_options(opts)
    assert result['right_margin'] == 80


@pytest.mark.parametrize('value', ['abc', object()])
def test_validate_options_right_margin_non_integer(value):
    with pytest.raises(SQLParseError) as exc:
        validate_options({'right_margin': value})
    assert 'right_margin requires an integer' in str(exc.value)


@pytest.mark.parametrize('value', [10, 9, 0, -1])
def test_validate_options_right_margin_too_small(value):
    # Note: implementation uses `< 10`, so 10 is accepted at bound? Let's check.
    # Actually `right_margin < 10` raises; so 10 passes.
    if value >= 10:
        opts = {'right_margin': value}
        result = validate_options(opts)
        assert result['right_margin'] == value
    else:
        with pytest.raises(SQLParseError) as exc:
            validate_options({'right_margin': value})
        assert 'right_margin requires an integer > 10' in str(exc.value)


# ---------------------------------------------------------------------------
# build_filter_stack
# ---------------------------------------------------------------------------

def _make_stack():
    # FilterStack is a simple container; build one via sqlparse.filters.
    return filters.FilterStack()


def test_build_filter_stack_empty_options_returns_same_stack():
    stack = _make_stack()
    result = build_filter_stack(stack, {})

    assert result is stack
    # No processors should have been registered.
    assert stack.preprocess == []
    assert stack.stmtprocess == []
    assert stack.postprocess == []
    assert stack.grouping is False


def test_build_filter_stack_keyword_case():
    stack = _make_stack()
    build_filter_stack(stack, {'keyword_case': 'upper'})

    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)
    assert stack.postprocess == []


def test_build_filter_stack_identifier_case():
    stack = _make_stack()
    build_filter_stack(stack, {'identifier_case': 'lower'})

    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.IdentifierCaseFilter)


def test_build_filter_stack_truncate_strings():
    stack = _make_stack()
    options = {
        'truncate_strings': 5,
        'truncate_char': '[...]',
    }
    build_filter_stack(stack, options)

    assert len(stack.preprocess) == 1
    fltr = stack.preprocess[0]
    assert isinstance(fltr, filters.TruncateStringFilter)
    assert fltr.width == 5
    assert fltr.char == '[...]'


def test_build_filter_stack_use_space_around_operators():
    stack = _make_stack()
    build_filter_stack(stack, {'use_space_around_operators': True})

    assert stack.grouping is True
    assert len(stack.preprocess) == 0
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.SpacesAroundOperatorsFilter)


def test_build_filter_stack_strip_comments():
    stack = _make_stack()
    build_filter_stack(stack, {'strip_comments': True})

    assert stack.grouping is True
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripCommentsFilter)


def test_build_filter_stack_strip_whitespace():
    stack = _make_stack()
    build_filter_stack(stack, {'strip_whitespace': True})

    assert stack.grouping is True
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)


def test_build_filter_stack_reindent_adds_strip_and_reindent():
    stack = _make_stack()
    options = {
        'reindent': True,
        'indent_char': ' ',
        'indent_width': 4,
        'indent_after_first': False,
        'indent_columns': False,
        'wrap_after': 0,
        'comma_first': False,
        'compact': False,
    }
    build_filter_stack(stack, options)

    assert stack.grouping is True
    assert len(stack.stmtprocess) == 2
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)
    fltr = stack.stmtprocess[1]
    assert isinstance(fltr, filters.ReindentFilter)
    assert fltr.char == ' '
    assert fltr.width == 4
    assert fltr.indent_after_first is False
    assert fltr.indent_columns is False
    assert fltr.wrap_after == 0
    assert fltr.comma_first is False
    assert fltr.compact is False


def test_build_filter_stack_reindent_aligned():
    stack = _make_stack()
    options = {'reindent_aligned': True, 'indent_char': '\t'}
    build_filter_stack(stack, options)

    assert stack.grouping is True
    assert len(stack.stmtprocess) == 1
    fltr = stack.stmtprocess[0]
    assert isinstance(fltr, filters.AlignedIndentFilter)
    assert fltr.char == '\t'


def test_build_filter_stack_right_margin():
    stack = _make_stack()
    build_filter_stack(stack, {'right_margin': 80})

    assert stack.grouping is True
    assert len(stack.stmtprocess) == 1
    fltr = stack.stmtprocess[0]
    assert isinstance(fltr, filters.RightMarginFilter)
    assert fltr.width == 80


def test_build_filter_stack_output_python():
    stack = _make_stack()
    build_filter_stack(stack, {'output_format': 'python'})

    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)


def test_build_filter_stack_output_php():
    stack = _make_stack()
    build_filter_stack(stack, {'output_format': 'php'})

    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPHPFilter)


def test_build_filter_stack_output_sql_no_filter():
    stack = _make_stack()
    build_filter_stack(stack, {'output_format': 'sql'})

    assert stack.postprocess == []


def test_build_filter_stack_output_format_case_insensitive():
    stack = _make_stack()
    build_filter_stack(stack, {'output_format': 'PHP'})

    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPHPFilter)


def test_build_filter_stack_combined():
    stack = _make_stack()
    options = {
        'keyword_case': 'upper',
        'identifier_case': 'lower',
        'truncate_strings': 3,
        'truncate_char': '..',
        'use_space_around_operators': True,
        'strip_comments': True,
        'strip_whitespace': True,
        'reindent': True,
        'indent_char': ' ',
        'indent_width': 4,
        'indent_after_first': True,
        'indent_columns': True,
        'wrap_after': 10,
        'comma_first': True,
        'compact': True,
        'right_margin': 100,
        'output_format': 'python',
    }
    build_filter_stack(stack, options)

    assert stack.grouping is True
    assert len(stack.preprocess) == 3
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)
    assert isinstance(stack.preprocess[1], filters.IdentifierCaseFilter)
    assert isinstance(stack.preprocess[2], filters.TruncateStringFilter)

    # stmtprocess order: spaces, strip comments, strip whitespace,
    # reindent, right margin
    assert len(stack.stmtprocess) == 5
    assert isinstance(stack.stmtprocess[0], filters.SpacesAroundOperatorsFilter)
    assert isinstance(stack.stmtprocess[1], filters.StripCommentsFilter)
    assert isinstance(stack.stmtprocess[2], filters.StripWhitespaceFilter)
    assert isinstance(stack.stmtprocess[3], filters.ReindentFilter)
    assert isinstance(stack.stmtprocess[4], filters.RightMarginFilter)

    reindent = stack.stmtprocess[3]
    assert reindent.char == ' '
    assert reindent.width == 4
    assert reindent.indent_after_first is True
    assert reindent.indent_columns is True
    assert reindent.wrap_after == 10
    assert reindent.comma_first is True
    assert reindent.compact is True

    assert stack.stmtprocess[4].width == 100

    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)
