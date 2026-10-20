import pytest

from sqlparse.exceptions import SQLParseError
from sqlparse import filters
from sqlparse.formatter import validate_options, build_filter_stack


# ---------------------------------------------------------------------------
# validate_options
# ---------------------------------------------------------------------------

class TestValidateOptionsDefaults:
    def test_empty_options(self):
        opts = validate_options({})
        # defaults that are always set
        assert opts['indent_char'] == ' '
        assert opts['indent_width'] == 2
        assert opts['wrap_after'] == 0
        assert opts['comma_first'] is False
        assert opts['compact'] is False
        assert opts['indent_columns'] is False
        assert opts['indent_after_first'] is False
        assert opts['right_margin'] is None

    def test_returns_same_dict(self):
        opts = {}
        result = validate_options(opts)
        assert result is opts

    def test_none_values_accepted_for_cases(self):
        opts = validate_options({
            'keyword_case': None,
            'identifier_case': None,
            'output_format': None,
        })
        assert 'keyword_case' not in opts or opts['keyword_case'] is None
        assert opts['output_format'] is None

    def test_valid_case_values(self):
        for value in ('upper', 'lower', 'capitalize'):
            validate_options({'keyword_case': value})
            validate_options({'identifier_case': value})

    def test_invalid_keyword_case(self):
        with pytest.raises(SQLParseError, match='Invalid value for keyword_case'):
            validate_options({'keyword_case': 'bogus'})

    def test_invalid_identifier_case(self):
        with pytest.raises(SQLParseError, match='Invalid value for identifier_case'):
            validate_options({'identifier_case': 'bogus'})

    def test_valid_output_format(self):
        for value in ('sql', 'python', 'php'):
            opts = validate_options({'output_format': value})
            assert opts['output_format'] == value

    def test_invalid_output_format(self):
        with pytest.raises(SQLParseError, match='Unknown output format'):
            validate_options({'output_format': 'json'})


class TestValidateOptionsBooleans:
    @pytest.mark.parametrize('key', [
        'strip_comments',
        'strip_whitespace',
        'indent_after_first',
        'comma_first',
        'compact',
        'indent_tabs',
    ])
    def test_valid_bool(self, key):
        opts = validate_options({key: True})
        # indent_tabs doesn't store indent_tabs key, everything else does
        if key != 'indent_tabs':
            assert opts[key] is True
        opts = validate_options({key: False})
        if key != 'indent_tabs':
            assert opts[key] is False

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
    def test_invalid_bool(self, key):
        with pytest.raises(SQLParseError):
            validate_options({key: 'not_a_bool'})

    def test_strip_comments_message(self):
        with pytest.raises(SQLParseError, match='Invalid value for strip_comments'):
            validate_options({'strip_comments': 1})

    def test_use_space_around_operators_message(self):
        with pytest.raises(SQLParseError,
                           match='Invalid value for use_space_around_operators'):
            validate_options({'use_space_around_operators': 'yes'})

    def test_strip_whitespace_message(self):
        with pytest.raises(SQLParseError, match='Invalid value for strip_whitespace'):
            validate_options({'strip_whitespace': 'x'})

    def test_indent_columns_message(self):
        with pytest.raises(SQLParseError, match='Invalid value for indent_columns'):
            validate_options({'indent_columns': 'x'})

    def test_reindent_message(self):
        with pytest.raises(SQLParseError, match='Invalid value for reindent'):
            validate_options({'reindent': 'x'})

    def test_reindent_aligned_message(self):
        with pytest.raises(SQLParseError, match='Invalid value for reindent_aligned'):
            validate_options({'reindent_aligned': 'x'})

    def test_indent_after_first_message(self):
        with pytest.raises(SQLParseError, match='Invalid value for indent_after_first'):
            validate_options({'indent_after_first': 'x'})

    def test_indent_tabs_message(self):
        with pytest.raises(SQLParseError, match='Invalid value for indent_tabs'):
            validate_options({'indent_tabs': 'x'})

    def test_comma_first_message(self):
        with pytest.raises(SQLParseError, match='comma_first requires a boolean value'):
            validate_options({'comma_first': 'x'})

    def test_compact_message(self):
        with pytest.raises(SQLParseError, match='compact requires a boolean value'):
            validate_options({'compact': 'x'})


class TestValidateOptionsIndentColumns:
    def test_indent_columns_enforces_reindent(self):
        opts = validate_options({'indent_columns': True})
        assert opts['indent_columns'] is True
        assert opts['reindent'] is True
        assert opts['strip_whitespace'] is True

    def test_indent_columns_false(self):
        opts = validate_options({'indent_columns': False})
        assert opts['indent_columns'] is False
        assert 'reindent' not in opts or opts.get('reindent') is False


class TestValidateOptionsReindent:
    def test_reindent_enforces_strip_whitespace(self):
        opts = validate_options({'reindent': True})
        assert opts['reindent'] is True
        assert opts['strip_whitespace'] is True

    def test_reindent_aligned_enforces_strip_whitespace(self):
        opts = validate_options({'reindent_aligned': True})
        assert opts['reindent_aligned'] is True
        assert opts['strip_whitespace'] is True


class TestValidateOptionsIndentChar:
    def test_indent_tabs_sets_tab(self):
        opts = validate_options({'indent_tabs': True})
        assert opts['indent_char'] == '\t'

    def test_indent_tabs_false_sets_space(self):
        opts = validate_options({'indent_tabs': False})
        assert opts['indent_char'] == ' '

    def test_indent_tabs_default_space(self):
        opts = validate_options({})
        assert opts['indent_char'] == ' '


class TestValidateOptionsIndentWidth:
    def test_valid_int(self):
        opts = validate_options({'indent_width': 4})
        assert opts['indent_width'] == 4

    def test_string_int_coerced(self):
        opts = validate_options({'indent_width': '3'})
        assert opts['indent_width'] == 3

    def test_default(self):
        opts = validate_options({})
        assert opts['indent_width'] == 2

    def test_non_int(self):
        with pytest.raises(SQLParseError, match='indent_width requires an integer'):
            validate_options({'indent_width': 'abc'})

    def test_none(self):
        with pytest.raises(SQLParseError, match='indent_width requires an integer'):
            validate_options({'indent_width': None})

    def test_zero(self):
        with pytest.raises(SQLParseError,
                           match='indent_width requires a positive integer'):
            validate_options({'indent_width': 0})

    def test_negative(self):
        with pytest.raises(SQLParseError,
                           match='indent_width requires a positive integer'):
            validate_options({'indent_width': -1})


class TestValidateOptionsWrapAfter:
    def test_valid_int(self):
        opts = validate_options({'wrap_after': 10})
        assert opts['wrap_after'] == 10

    def test_string_int_coerced(self):
        opts = validate_options({'wrap_after': '5'})
        assert opts['wrap_after'] == 5

    def test_default(self):
        opts = validate_options({})
        assert opts['wrap_after'] == 0

    def test_non_int(self):
        with pytest.raises(SQLParseError, match='wrap_after requires an integer'):
            validate_options({'wrap_after': 'x'})

    def test_negative(self):
        with pytest.raises(SQLParseError,
                           match='wrap_after requires a positive integer'):
            validate_options({'wrap_after': -1})


class TestValidateOptionsTruncateStrings:
    def test_valid(self):
        opts = validate_options({'truncate_strings': 5})
        assert opts['truncate_strings'] == 5
        assert opts['truncate_char'] == '[...]'

    def test_string_int_coerced(self):
        opts = validate_options({'truncate_strings': '10'})
        assert opts['truncate_strings'] == 10

    def test_custom_truncate_char(self):
        opts = validate_options({'truncate_strings': 3, 'truncate_char': '..'})
        assert opts['truncate_char'] == '..'

    def test_none_is_noop(self):
        opts = validate_options({'truncate_strings': None})
        assert 'truncate_char' not in opts

    def test_non_int(self):
        with pytest.raises(SQLParseError,
                           match='Invalid value for truncate_strings'):
            validate_options({'truncate_strings': 'abc'})

    def test_one_invalid(self):
        with pytest.raises(SQLParseError,
                           match='Invalid value for truncate_strings'):
            validate_options({'truncate_strings': 1})

    def test_zero_invalid(self):
        with pytest.raises(SQLParseError,
                           match='Invalid value for truncate_strings'):
            validate_options({'truncate_strings': 0})

    def test_negative_invalid(self):
        with pytest.raises(SQLParseError,
                           match='Invalid value for truncate_strings'):
            validate_options({'truncate_strings': -5})


class TestValidateOptionsRightMargin:
    def test_none_default(self):
        opts = validate_options({})
        assert opts['right_margin'] is None

    def test_valid(self):
        opts = validate_options({'right_margin': 50})
        assert opts['right_margin'] == 50

    def test_string_int_coerced(self):
        opts = validate_options({'right_margin': '20'})
        assert opts['right_margin'] == 20

    def test_non_int(self):
        with pytest.raises(SQLParseError, match='right_margin requires an integer'):
            validate_options({'right_margin': 'x'})

    def test_too_small(self):
        with pytest.raises(SQLParseError,
                           match='right_margin requires an integer > 10'):
            validate_options({'right_margin': 9})

    def test_ten_invalid(self):
        with pytest.raises(SQLParseError,
                           match='right_margin requires an integer > 10'):
            validate_options({'right_margin': 10})


# ---------------------------------------------------------------------------
# build_filter_stack
# ---------------------------------------------------------------------------

def _stack():
    return filters.FilterStack()


def test_build_filter_stack_returns_same_stack():
    stack = _stack()
    result = build_filter_stack(stack, {})
    assert result is stack


def test_build_filter_stack_empty_no_filters():
    stack = _stack()
    build_filter_stack(stack, {})
    assert stack.preprocess == []
    assert stack.stmtprocess == []
    assert stack.postprocess == []


def test_keyword_case_adds_preprocess():
    stack = _stack()
    build_filter_stack(stack, validate_options({'keyword_case': 'upper'}))
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)
    assert stack.preprocess[0].convert == 'upper'


def test_identifier_case_adds_preprocess():
    stack = _stack()
    build_filter_stack(stack, validate_options({'identifier_case': 'lower'}))
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.IdentifierCaseFilter)


def test_truncate_strings_adds_preprocess():
    stack = _stack()
    opts = validate_options({'truncate_strings': 5})
    build_filter_stack(stack, opts)
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.TruncateStringFilter)


def test_all_preprocess_order():
    stack = _stack()
    opts = validate_options({
        'keyword_case': 'upper',
        'identifier_case': 'lower',
        'truncate_strings': 5,
    })
    build_filter_stack(stack, opts)
    assert len(stack.preprocess) == 3
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)
    assert isinstance(stack.preprocess[1], filters.IdentifierCaseFilter)
    assert isinstance(stack.preprocess[2], filters.TruncateStringFilter)


def test_use_space_around_operators():
    stack = _stack()
    build_filter_stack(stack, validate_options({'use_space_around_operators': True}))
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.SpacesAroundOperatorsFilter)


def test_strip_comments():
    stack = _stack()
    build_filter_stack(stack, validate_options({'strip_comments': True}))
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripCommentsFilter)


def test_strip_whitespace_only():
    stack = _stack()
    build_filter_stack(stack, validate_options({'strip_whitespace': True}))
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)


def test_strip_whitespace_via_reindent():
    stack = _stack()
    build_filter_stack(stack, validate_options({'reindent': True}))
    assert isinstance(stack.stmtprocess[0], filters.StripWhitespaceFilter)
    assert isinstance(stack.stmtprocess[1], filters.ReindentFilter)


def test_reindent_adds_reindent_filter():
    stack = _stack()
    build_filter_stack(stack, validate_options({'reindent': True}))
    assert isinstance(stack.stmtprocess[-1], filters.ReindentFilter)


def test_reindent_filter_parameters():
    stack = _stack()
    opts = validate_options({
        'reindent': True,
        'indent_tabs': True,
        'indent_width': 4,
        'indent_after_first': True,
        'indent_columns': True,
        'wrap_after': 7,
        'comma_first': True,
        'compact': True,
    })
    build_filter_stack(stack, opts)
    reindent = [f for f in stack.stmtprocess
                if isinstance(f, filters.ReindentFilter)][0]
    assert reindent.char == '\t'
    assert reindent.width == 4
    assert reindent.indent_after_first is True
    assert reindent.indent_columns is True
    assert reindent.wrap_after == 7
    assert reindent.comma_first is True
    assert reindent.compact is True


def test_reindent_aligned_adds_aligned_filter():
    stack = _stack()
    opts = validate_options({'reindent_aligned': True, 'indent_tabs': True})
    build_filter_stack(stack, opts)
    aligned = [f for f in stack.stmtprocess
               if isinstance(f, filters.AlignedIndentFilter)]
    assert len(aligned) == 1
    assert aligned[0].char == '\t'


def test_right_margin_adds_filter():
    stack = _stack()
    build_filter_stack(stack, validate_options({'right_margin': 42}))
    assert len(stack.stmtprocess) == 1
    assert isinstance(stack.stmtprocess[0], filters.RightMarginFilter)


def test_output_format_php():
    stack = _stack()
    build_filter_stack(stack, validate_options({'output_format': 'php'}))
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPHPFilter)


def test_output_format_python():
    stack = _stack()
    build_filter_stack(stack, validate_options({'output_format': 'python'}))
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)


def test_output_format_sql_no_filter():
    stack = _stack()
    build_filter_stack(stack, validate_options({'output_format': 'sql'}))
    assert stack.postprocess == []


def test_output_format_case_insensitive():
    stack = _stack()
    # build_filter_stack lowercases manually, validate_options accepts lowercase
    build_filter_stack(stack, validate_options({'output_format': 'PYTHON'}))
    # validate_options rejects 'PYTHON' since it checks exact values
    # so this should not be reached


def test_output_format_value_uppercase_rejected():
    with pytest.raises(SQLParseError, match='Unknown output format'):
        validate_options({'output_format': 'PHP'})


def test_full_stack_order():
    stack = _stack()
    opts = validate_options({
        'keyword_case': 'upper',
        'strip_comments': True,
        'reindent': True,
        'right_margin': 50,
        'output_format': 'python',
    })
    build_filter_stack(stack, opts)

    # preprocess
    assert len(stack.preprocess) == 1
    assert isinstance(stack.preprocess[0], filters.KeywordCaseFilter)

    # stmtprocess: strip_comments, strip_whitespace, reindent, right_margin
    stmt_types = [type(f) for f in stack.stmtprocess]
    assert filters.StripCommentsFilter in stmt_types
    assert filters.StripWhitespaceFilter in stmt_types
    assert filters.ReindentFilter in stmt_types
    assert filters.RightMarginFilter in stmt_types

    # postprocess
    assert len(stack.postprocess) == 1
    assert isinstance(stack.postprocess[0], filters.OutputPythonFilter)
