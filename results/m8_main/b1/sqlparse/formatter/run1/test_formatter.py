import pytest

from sqlparse import filters
from sqlparse.exceptions import SQLParseError
from sqlparse.formatter import validate_options, build_filter_stack


# ---------------------------------------------------------------------------
# validate_options
# ---------------------------------------------------------------------------

def test_validate_options_empty_returns_dict():
    opts = {}
    result = validate_options(opts)
    assert result is opts


def test_validate_options_defaults_applied():
    opts = {}
    result = validate_options(opts)
    assert result['indent_columns'] is False
    assert result['indent_after_first'] is False
    assert result['indent_char'] == ' '
    assert result['indent_width'] == 2
    assert result['wrap_after'] == 0
    assert result['comma_first'] is False
    assert result['compact'] is False
    assert result['right_margin'] is None
    # defaults that are not explicitly set are simply absent
    assert 'keyword_case' not in result
    assert 'identifier_case' not in result
    assert 'output_format' not in result


@pytest.mark.parametrize('value', [None, 'upper', 'lower', 'capitalize'])
def test_validate_options_keyword_case_valid(value):
    opts = {}
    if value is not None:
        opts['keyword_case'] = value
    validate_options(opts)


@pytest.mark.parametrize('value', ['UPPER', 'title', 'foo', 1])
def test_validate_options_keyword_case_invalid(value):
    with pytest.raises(SQLParseError, match='Invalid value for keyword_case'):
        validate_options({'keyword_case': value})


@pytest.mark.parametrize('value', [None, 'upper', 'lower', 'capitalize'])
def test_validate_options_identifier_case_valid(value):
    opts = {}
    if value is not None:
        opts['identifier_case'] = value
    validate_options(opts)


@pytest.mark.parametrize('value', ['UPPER', 'title', 'foo', 1])
def test_validate_options_identifier_case_invalid(value):
    with pytest.raises(SQLParseError, match='Invalid value for identifier_case'):
        validate_options({'identifier_case': value})


@pytest.mark.parametrize('value', [None, 'sql', 'python', 'php'])
def test_validate_options_output_format_valid(value):
    opts = {}
    if value is not None:
        opts['output_format'] = value
    validate_options(opts)


@pytest.mark.parametrize('value', ['SQL', 'rust', 'foo', 1])
def test_validate_options_output_format_invalid(value):
    with pytest.raises(SQLParseError, match='Unknown output format'):
        validate_options({'output_format': value})


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_strip_comments_valid(value):
    validate_options({'strip_comments': value})


# NB: the implementation compares against [True, False] with `in`, so integer
# 1 and 0 are accepted (they compare equal to True/False).
def test_validate_options_strip_comments_invalid():
    with pytest.raises(SQLParseError, match='Invalid value for strip_comments'):
        validate_options({'strip_comments': 'yes'})


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_use_space_around_operators_valid(value):
    validate_options({'use_space_around_operators': value})


def test_validate_options_use_space_around_operators_invalid():
    with pytest.raises(SQLParseError,
                       match='Invalid value for use_space_around_operators'):
        validate_options({'use_space_around_operators': 'yes'})


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_strip_whitespace_valid(value):
    validate_options({'strip_whitespace': value})


def test_validate_options_strip_whitespace_invalid():
    with pytest.raises(SQLParseError,
                       match='Invalid value for strip_whitespace'):
        validate_options({'strip_whitespace': 'yes'})


def test_validate_options_truncate_strings_none_absent():
    opts = {}
    validate_options(opts)
    assert 'truncate_strings' not in opts
    assert 'truncate_char' not in opts


def test_validate_options_truncate_strings_valid_converts_to_int():
    opts = {'truncate_strings': '10'}
    validate_options(opts)
    assert opts['truncate_strings'] == 10
    assert opts['truncate_char'] == '[...]'


def test_validate_options_truncate_strings_default_char():
    opts = {'truncate_strings': 5}
    validate_options(opts)
    assert opts['truncate_strings'] == 5
    assert opts['truncate_char'] == '[...]'


def test_validate_options_truncate_strings_custom_char_preserved():
    opts = {'truncate_strings': 5, 'truncate_char': '...'}
    validate_options(opts)
    assert opts['truncate_char'] == '...'


@pytest.mark.parametrize('value', ['abc', object(), [1]])
def test_validate_options_truncate_strings_invalid_type(value):
    with pytest.raises(SQLParseError,
                       match='Invalid value for truncate_strings'):
        validate_options({'truncate_strings': value})


@pytest.mark.parametrize('value', [0, 1, -5, '1', '0'])
def test_validate_options_truncate_strings_too_small(value):
    with pytest.raises(SQLParseError,
                       match='Invalid value for truncate_strings'):
        validate_options({'truncate_strings': value})


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_indent_columns_valid(value):
    opts = {'indent_columns': value}
    validate_options(opts)
    assert opts['indent_columns'] is value


def test_validate_options_indent_columns_enforces_reindent():
    opts = {'indent_columns': True}
    validate_options(opts)
    assert opts['reindent'] is True
    assert opts['strip_whitespace'] is True


def test_validate_options_indent_columns_invalid():
    with pytest.raises(SQLParseError,
                       match='Invalid value for indent_columns'):
        validate_options({'indent_columns': 'yes'})


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_reindent_valid(value):
    opts = {'reindent': value}
    validate_options(opts)
    assert opts['reindent'] is value


def test_validate_options_reindent_enforces_strip_whitespace():
    opts = {'reindent': True}
    validate_options(opts)
    assert opts['strip_whitespace'] is True


def test_validate_options_reindent_invalid():
    with pytest.raises(SQLParseError, match='Invalid value for reindent'):
        validate_options({'reindent': 'yes'})


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_reindent_aligned_valid(value):
    opts = {'reindent_aligned': value}
    validate_options(opts)
    assert opts['reindent_aligned'] is value


def test_validate_options_reindent_aligned_enforces_strip_whitespace():
    opts = {'reindent_aligned': True}
    validate_options(opts)
    assert opts['strip_whitespace'] is True


def test_validate_options_reindent_aligned_invalid():
    with pytest.raises(SQLParseError,
                       match='Invalid value for reindent_aligned'):
        validate_options({'reindent_aligned': 'yes'})


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_indent_after_first_valid(value):
    opts = {'indent_after_first': value}
    validate_options(opts)
    assert opts['indent_after_first'] is value


def test_validate_options_indent_after_first_invalid():
    with pytest.raises(SQLParseError,
                       match='Invalid value for indent_after_first'):
        validate_options({'indent_after_first': 'yes'})


def test_validate_options_indent_tabs_true_sets_tab():
    opts = {'indent_tabs': True}
    validate_options(opts)
    assert opts['indent_char'] == '\t'


def test_validate_options_indent_tabs_false_sets_space():
    opts = {'indent_tabs': False}
    validate_options(opts)
    assert opts['indent_char'] == ' '


def test_validate_options_indent_tabs_invalid():
    with pytest.raises(SQLParseError, match='Invalid value for indent_tabs'):
        validate_options({'indent_tabs': 'yes'})


@pytest.mark.parametrize('value,expected', [('4', 4), (4, 4), (True, 1)])
def test_validate_options_indent_width_converts(value, expected):
    opts = {'indent_width': value}
    validate_options(opts)
    assert opts['indent_width'] == expected


@pytest.mark.parametrize('value', ['abc', None, object()])
def test_validate_options_indent_width_non_integer(value):
    with pytest.raises(SQLParseError,
                       match='indent_width requires an integer'):
        validate_options({'indent_width': value})


@pytest.mark.parametrize('value', [0, -1, '-3'])
def test_validate_options_indent_width_non_positive(value):
    with pytest.raises(SQLParseError,
                       match='indent_width requires a positive integer'):
        validate_options({'indent_width': value})


@pytest.mark.parametrize('value,expected', [('5', 5), (5, 5), (0, 0)])
def test_validate_options_wrap_after_converts(value, expected):
    opts = {'wrap_after': value}
    validate_options(opts)
    assert opts['wrap_after'] == expected


@pytest.mark.parametrize('value', ['abc', None, object()])
def test_validate_options_wrap_after_non_integer(value):
    with pytest.raises(SQLParseError,
                       match='wrap_after requires an integer'):
        validate_options({'wrap_after': value})


@pytest.mark.parametrize('value', [-1, '-5'])
def test_validate_options_wrap_after_negative(value):
    with pytest.raises(SQLParseError,
                       match='wrap_after requires a positive integer'):
        validate_options({'wrap_after': value})


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_comma_first_valid(value):
    opts = {'comma_first': value}
    validate_options(opts)
    assert opts['comma_first'] is value


def test_validate_options_comma_first_invalid():
    with pytest.raises(SQLParseError,
                       match='comma_first requires a boolean value'):
        validate_options({'comma_first': 'yes'})


@pytest.mark.parametrize('value', [True, False])
def test_validate_options_compact_valid(value):
    opts = {'compact': value}
    validate_options(opts)
    assert opts['compact'] is value


def test_validate_options_compact_invalid():
    with pytest.raises(SQLParseError,
                       match='compact requires a boolean value'):
        validate_options({'compact': 'yes'})


def test_validate_options_right_margin_none():
    opts = {'right_margin': None}
    validate_options(opts)
    assert opts['right_margin'] is None


@pytest.mark.parametrize('value,expected', [('20', 20), (20, 20), (10, 10)])
def test_validate_options_right_margin_converts(value, expected):
    opts = {'right_margin': value}
    validate_options(opts)
    assert opts['right_margin'] == expected


@pytest.mark.parametrize('value', ['abc', object()])
def test_validate_options_right_margin_non_integer(value):
    with pytest.raises(SQLParseError,
                       match='right_margin requires an integer'):
        validate_options({'right_margin': value})


@pytest.mark.parametrize('value', [9, 0, -5, '9'])
def test_validate_options_right_margin_too_small(value):
    with pytest.raises(SQLParseError,
                       match='right_margin requires an integer > 10'):
        validate_options({'right_margin': value})


# ---------------------------------------------------------------------------
# build_filter_stack
# ---------------------------------------------------------------------------

class _FakeStack:
    def __init__(self):
        self.preprocess = []
        self.stmtprocess = []
        self.postprocess = []
        self.grouping_enabled = 0

    def enable_grouping(self):
        self.grouping_enabled += 1


def _make_stack():
    return _FakeStack()


def _validated(options=None):
    return validate_options(dict(options or {}))


def test_build_filter_stack_returns_same_stack():
    stack = _make_stack()
    result = build_filter_stack(stack, _validated())
    assert result is stack


def test_build_filter_stack_empty_options_no_filters():
    stack = _make_stack()
    build_filter_stack(stack, _validated())
    assert stack.preprocess == []
    assert stack.stmtprocess == []
    assert stack.postprocess == []
    assert stack.grouping_enabled == 0


def test_build_filter_stack_keyword_case():
    stack = _make_stack()
    build_filter_stack(stack, _validated({'keyword_case': 'upper'}))
    assert len(stack.preprocess) == 1
    f = stack.preprocess[0]
    assert isinstance(f, filters.KeywordCaseFilter)
    # The filter stores the bound method (str.upper), not the string name.
    assert f.convert == str.upper


def test_build_filter_stack_identifier_case():
    stack = _make_stack()
    build_filter_stack(stack, _validated({'identifier_case': 'lower'}))
    assert len(stack.preprocess) == 1
    f = stack.preprocess[0]
    assert isinstance(f, filters.IdentifierCaseFilter)
    # The filter stores the bound method (str.lower), not the string name.
    assert f.convert == str.lower


def test_build_filter_stack_truncate_strings():
    stack = _make_stack()
    build_filter_stack(stack, _validated({'truncate_strings': 5}))
    assert len(stack.preprocess) == 1
    f = stack.preprocess[0]
    assert isinstance(f, filters.TruncateStringFilter)
    assert f.width == 5
    assert f.char == '[...]'


def test_build_filter_stack_truncate_strings_custom_char():
    stack = _make_stack()
    build_filter_stack(
        stack, _validated({'truncate_strings': 3, 'truncate_char': '..'}))
    f = stack.preprocess[0]
    assert isinstance(f, filters.TruncateStringFilter)
    assert f.width == 3
    assert f.char == '..'


def test_build_filter_stack_preprocess_order():
    stack = _make_stack()
    build_filter_stack(stack, _validated({
        'keyword_case': 'upper',
        'identifier_case': 'lower',
        'truncate_strings': 4,
    }))
    assert len(stack.preprocess) == 3
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)
    assert isinstance(stack.preprocess[1], filters.IdentifierCaseFilter)
    assert isinstance(stack.preprocess[2], filters.TruncateStringFilter)


def test_build_filter_stack_spaces_around_operators():
    stack = _make_stack()
    build_filter_stack(stack, _validated({'use_space_around_operators': True}))
    assert stack.grouping_enabled == 1
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.SpacesAroundOperatorsFilter)


def test_build_filter_stack_strip_comments():
    stack = _make_stack()
    build_filter_stack(stack, _validated({'strip_comments': True}))
    assert stack.grouping_enabled == 1
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripCommentsFilter)


def test_build_filter_stack_strip_whitespace():
    stack = _make_stack()
    build_filter_stack(stack, _validated({'strip_whitespace': True}))
    assert stack.grouping_enabled == 1
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)


def test_build_filter_stack_reindent_implies_strip_whitespace():
    stack = _make_stack()
    build_filter_stack(stack, _validated({'reindent': True}))
    # strip_whitespace filter + reindent filter
    assert len(stack.stmtprocess) == 2
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)
    assert isinstance(stack.stmtprocess[1], filters.ReindentFilter)


def test_build_filter_stack_reindent_uses_options():
    opts = _validated({
        'reindent': True,
        'indent_tabs': True,
        'indent_width': 3,
        'indent_after_first': True,
        'indent_columns': True,
        'wrap_after': 7,
        'comma_first': True,
        'compact': True,
    })
    stack = _make_stack()
    build_filter_stack(stack, opts)
    reindent = stack.stmtprocess[-1]
    assert isinstance(reindent, filters.ReindentFilter)
    assert reindent.char == '\t'
    assert reindent.width == 3
    assert reindent.wrap_after == 7
    assert reindent.comma_first is True
    assert reindent.compact is True


def test_build_filter_stack_reindent_aligned():
    stack = _make_stack()
    build_filter_stack(stack, _validated({'reindent_aligned': True}))
    assert isinstance(stack.stmtprocess[-1], filters.AlignedIndentFilter)
    assert stack.stmtprocess[-1].char == ' '


def test_build_filter_stack_reindent_aligned_indent_char():
    stack = _make_stack()
    build_filter_stack(
        stack, _validated({'reindent_aligned': True, 'indent_tabs': True}))
    aligned = [f for f in stack.stmtprocess
               if isinstance(f, filters.AlignedIndentFilter)]
    assert len(aligned) == 1
    assert aligned[0].char == '\t'


def test_build_filter_stack_right_margin():
    stack = _make_stack()
    build_filter_stack(stack, _validated({'right_margin': 20}))
    assert stack.grouping_enabled == 1
    assert len(stack.stmtprocess) == 1
    f = stack.stmtprocess[0]
    assert isinstance(f, filters.RightMarginFilter)
    assert f.width == 20


def test_build_filter_stack_output_format_php():
    stack = _make_stack()
    build_filter_stack(stack, _validated({'output_format': 'php'}))
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPHPFilter)


def test_build_filter_stack_output_format_python():
    stack = _make_stack()
    build_filter_stack(stack, _validated({'output_format': 'python'}))
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)


def test_build_filter_stack_output_format_sql_no_postprocess():
    stack = _make_stack()
    build_filter_stack(stack, _validated({'output_format': 'sql'}))
    assert stack.postprocess == []


def test_build_filter_stack_output_format_case_insensitive_postprocess():
    # validate_options only accepts lowercase format names, but
    # build_filter_stack lowercases the value anyway. Test that path
    # directly by bypassing validate_options.
    stack = _make_stack()
    build_filter_stack(stack, {'output_format': 'PHP'})
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPHPFilter)


def test_build_filter_stack_output_format_validation_rejects_uppercase():
    with pytest.raises(SQLParseError, match='Unknown output format'):
        validate_options({'output_format': 'PHP'})


def test_build_filter_stack_combined():
    opts = _validated({
        'keyword_case': 'upper',
        'strip_comments': True,
        'reindent': True,
        'right_margin': 50,
        'output_format': 'python',
    })
    stack = _make_stack()
    result = build_filter_stack(stack, opts)
    assert result is stack

    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)

    kinds = [type(f) for f in stack.stmtprocess]
    assert filters.StripCommentsFilter in kinds
    assert filters.StripWhitespaceFilter in kinds
    assert filters.ReindentFilter in kinds
    assert filters.RightMarginFilter in kinds

    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)
