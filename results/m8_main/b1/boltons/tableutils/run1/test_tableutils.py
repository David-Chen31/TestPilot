import collections
import types

import pytest

from boltons.tableutils import (
    to_text,
    escape_html,
    UnsupportedData,
    InputType,
    DictInputType,
    ObjectInputType,
    ListInputType,
    TupleInputType,
    NamedTupleInputType,
    Table,
)


# ---------------------------------------------------------------------------
# to_text
# ---------------------------------------------------------------------------

class TestToText:
    def test_basic(self):
        assert to_text(123) == '123'

    def test_str(self):
        assert to_text('hello') == 'hello'

    def test_none(self):
        assert to_text(None) == 'None'

    def test_no_maxlen(self):
        assert to_text('abcdef') == 'abcdef'

    def test_maxlen_shorter_than_text(self):
        # 6 chars > maxlen 5 -> truncated to 2 chars + '...'
        assert to_text('abcdef', maxlen=5) == 'ab...'

    def test_maxlen_equal_does_not_truncate(self):
        assert to_text('abcde', maxlen=5) == 'abcde'

    def test_maxlen_longer_does_not_truncate(self):
        assert to_text('abc', maxlen=10) == 'abc'

    def test_maxlen_exact_boundary(self):
        # len(text) > maxlen is strict; equal is not truncated
        assert to_text('x' * 10, maxlen=10) == 'x' * 10

    def test_maxlen_truncation_formula(self):
        # text[:maxlen-3] + '...'
        assert to_text('0123456789', maxlen=6) == '012...'

    def test_bad_str_falls_back_to_repr(self):
        class BadStr:
            def __str__(self):
                raise ValueError('nope')

            def __repr__(self):
                return 'GOOD_REPR'

        assert to_text(BadStr()) == 'GOOD_REPR'

    def test_bad_str_and_repr_falls_back(self):
        class TotallyBad:
            def __str__(self):
                raise ValueError('no str')

            def __repr__(self):
                raise ValueError('no repr')

        result = to_text(TotallyBad())
        # object.__repr__ output contains the class name
        assert 'TotallyBad' in result


# ---------------------------------------------------------------------------
# escape_html
# ---------------------------------------------------------------------------

class TestEscapeHtml:
    def test_basic_escape(self):
        assert escape_html('<b>') == '&lt;b&gt;'

    def test_quotes_escaped(self):
        assert escape_html('a"b') == 'a&quot;b'
        assert escape_html("a'b") == 'a&#x27;b'

    def test_ampersand(self):
        assert escape_html('a & b') == 'a &amp; b'

    def test_maxlen(self):
        assert escape_html('abcdef', maxlen=5) == 'ab...'

    def test_maxlen_with_special(self):
        # text is converted then truncated, then escaped
        result = escape_html('<<<<<<', maxlen=5)
        assert result == '&lt;&lt;...'

    def test_plain(self):
        assert escape_html(123) == '123'


# ---------------------------------------------------------------------------
# InputType classes
# ---------------------------------------------------------------------------

class TestInputTypes:
    def test_input_type_default_get_entry_seq(self):
        it = InputType()

        class Dummy(InputType):
            def get_entry(self, obj, headers):
                return [obj, headers]

        d = Dummy()
        assert d.get_entry_seq([1, 2], 'h') == [[1, 'h'], [2, 'h']]

    def test_dict_input_type(self):
        it = DictInputType()
        assert it.check_type({'a': 1})
        assert not it.check_type([1, 2])
        assert it.guess_headers({'b': 1, 'a': 2, 'c': 3}) == ['a', 'b', 'c']
        assert it.get_entry({'a': 1}, ['a', 'b']) == [1, None]
        assert it.get_entry_seq([{'a': 1}, {'b': 2}], ['a', 'b']) == [
            [1, None], [None, 2]
        ]

    def test_list_input_type(self):
        it = ListInputType()
        assert it.check_type([1, 2])
        assert not it.check_type((1, 2))
        assert it.guess_headers([1, 2]) is None
        assert it.get_entry([1, 2], None) == [1, 2]
        assert it.get_entry_seq([[1], [2]], None) == [[1], [2]]

    def test_tuple_input_type(self):
        it = TupleInputType()
        assert it.check_type((1, 2))
        assert not it.check_type([1, 2])
        assert it.guess_headers((1, 2)) is None
        assert it.get_entry((1, 2), None) == [1, 2]
        assert it.get_entry_seq([(1, 2), (3, 4)], None) == [[1, 2], [3, 4]]

    def test_named_tuple_input_type(self):
        NT = collections.namedtuple('NT', ['a', 'b'])
        it = NamedTupleInputType()
        assert it.check_type(NT(1, 2))
        assert not it.check_type((1, 2))
        assert it.guess_headers(NT(1, 2)) == ['a', 'b']
        assert it.get_entry(NT(1, 2), ['a', 'b']) == [1, 2]
        assert it.get_entry(NT(1, 2), ['a', 'c']) == [1, None]
        assert it.get_entry_seq([NT(1, 2)], ['a', 'b']) == [[1, 2]]

    def test_object_input_type(self):
        class Obj:
            def __init__(self):
                self.x = 1
                self.y = 2

            def method(self):
                pass

        it = ObjectInputType()
        o = Obj()
        assert it.check_type(o)
        assert not it.check_type(1)
        assert not it.check_type(None)
        headers = it.guess_headers(o)
        assert 'x' in headers
        assert 'y' in headers
        assert 'method' not in headers
        assert it.get_entry(o, ['x', 'y']) == [1, 2]
        # missing attribute yields None
        assert it.get_entry(o, ['x', 'missing']) == [1, None]

    def test_object_input_type_handles_raising_property(self):
        class Obj:
            @property
            def good(self):
                return 5

            @property
            def bad(self):
                raise RuntimeError('boom')

        it = ObjectInputType()
        headers = it.guess_headers(Obj())
        assert 'bad' not in headers
        assert 'good' in headers


# ---------------------------------------------------------------------------
# UnsupportedData
# ---------------------------------------------------------------------------

def test_unsupported_data_is_type_error():
    assert issubclass(UnsupportedData, TypeError)


# ---------------------------------------------------------------------------
# Table construction / basic behavior
# ---------------------------------------------------------------------------

class TestTableBasic:
    def test_empty(self):
        t = Table()
        assert len(t) == 0
        assert t.headers == []
        assert t.metadata == {}

    def test_from_list_of_lists_infers_headers_from_first_row(self):
        t = Table([['a', 'b'], [1, 2], [3, 4]])
        assert t.headers == ['a', 'b']
        assert len(t) == 2
        assert t[0] == [1, 2]
        assert t[1] == [3, 4]

    def test_explicit_headers_none(self):
        t = Table([[1, 2], [3, 4]], headers=None)
        assert t.headers == []
        assert len(t) == 2
        assert t[0] == [1, 2]

    def test_explicit_headers(self):
        t = Table([[1, 2]], headers=['x', 'y'])
        assert t.headers == ['x', 'y']
        assert t[0] == [1, 2]

    def test_getitem(self):
        t = Table([[1, 2], [3, 4]], headers=['a', 'b'])
        assert t[0] == [1, 2]
        assert t[1] == [3, 4]
        assert t[-1] == [3, 4]

    def test_len(self):
        t = Table([[1], [2], [3]], headers=['a'])
        assert len(t) == 3

    def test_metadata(self):
        t = Table([[1]], headers=['a'], metadata={'k': 'v'})
        assert t.metadata == {'k': 'v'}

    def test_width_from_headers(self):
        t = Table([[1]], headers=['a', 'b'])
        assert t._width == 2
        assert t[0] == [1, None]

    def test_width_from_data(self):
        t = Table([[1, 2, 3]], headers=None)
        assert t._width == 3

    def test_fill_short_rows(self):
        t = Table([[1, 2, 3], [4]], headers=None)
        assert t[0] == [1, 2, 3]
        assert t[1] == [4, None, None]

    def test_repr_with_headers(self):
        t = Table([[1, 2]], headers=['a', 'b'])
        r = repr(t)
        assert r.startswith('Table(headers=')
        assert 'data=' in r

    def test_repr_without_headers(self):
        t = Table([[1, 2]], headers=None)
        r = repr(t)
        assert r.startswith('Table([')
        assert 'headers' not in r

    def test_extend(self):
        t = Table([[1, 2]], headers=['a', 'b'])
        t.extend([[3, 4]])
        assert len(t) == 2
        assert t[1] == [3, 4]

    def test_extend_empty(self):
        t = Table([[1, 2]], headers=['a', 'b'])
        t.extend([])
        assert len(t) == 1

    def test_extend_fills(self):
        t = Table([[1, 2]], headers=['a', 'b', 'c'])
        t.extend([[3]])
        assert t[1] == [3, None, None]


# ---------------------------------------------------------------------------
# Table.from_data and friends
# ---------------------------------------------------------------------------

class TestFromData:
    def test_from_list_of_lists(self):
        t = Table.from_list([[1, 2], [3, 4]])
        assert t[0] == [1, 2]

    def test_from_list_of_dicts(self):
        t = Table.from_dict([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
        assert t.headers == ['a', 'b']
        assert t[0] == [1, 2]
        assert t[1] == [3, 4]

    def test_from_single_dict(self):
        t = Table.from_dict({'a': 1, 'b': 2})
        assert t.headers == ['a', 'b']
        assert t[0] == [1, 2]

    def test_from_namedtuple_seq(self):
        NT = collections.namedtuple('NT', ['a', 'b'])
        t = Table.from_data([NT(1, 2), NT(3, 4)])
        assert t.headers == ['a', 'b']
        assert t[0] == [1, 2]
        assert t[1] == [3, 4]

    def test_from_single_namedtuple(self):
        NT = collections.namedtuple('NT', ['a', 'b'])
        t = Table.from_data(NT(1, 2))
        assert t.headers == ['a', 'b']
        assert t[0] == [1, 2]

    def test_from_tuple_of_tuples(self):
        t = Table.from_data([(1, 2), (3, 4)])
        assert t[0] == [1, 2]
        assert t[1] == [3, 4]

    def test_from_object(self):
        class Obj:
            def __init__(self):
                self.x = 1

        t = Table.from_object(Obj())
        assert 'x' in t.headers
        assert len(t) == 1

    def test_from_empty(self):
        t = Table.from_data([])
        assert len(t) == 0
        assert t.headers == []

    def test_max_depth_zero(self):
        t = Table.from_data([['a', 'b'], [1, 2]], max_depth=0)
        assert len(t) == 0

    def test_scalar_data(self):
        # Scalar values are wrapped in a single-cell row, but the header
        # guessing treats [data] as a list, so no data rows remain here.
        t = Table.from_data(5)
        # headers is [5] (first row of the single-row list)... actually:
        # cls([[5]], headers=headers) with headers=_MISSING means headers
        # becomes first row [5], and remaining data is empty.
        assert t.headers == [5]
        assert len(t) == 0

    def test_unsupported_raises(self):
        # A data type whose get_entry is missing but check_type returns True
        # will fail with AttributeError rather than UnsupportedData.
        class FakeType(InputType):
            def check_type(self, obj):
                return True

            def guess_headers(self, obj):
                return None

        with pytest.raises(AttributeError):
            Table.from_data([1, 2], _data_type=FakeType())

    def test_from_data_with_max_depth_nests_tables(self):
        t = Table.from_data([[{'a': 1}]], max_depth=2)
        # cell should be a Table
        assert isinstance(t[0][0], Table)
        assert t[0][0][0] == [1]

    def test_headers_explicit(self):
        t = Table.from_data([{'a': 1, 'b': 2}], headers=['b'])
        assert t.headers == ['b']
        assert t[0] == [2]

    def test_metadata_passed_through(self):
        t = Table.from_data([[1, 2]], headers=['a', 'b'], metadata={'m': 1})
        assert t.metadata == {'m': 1}


# ---------------------------------------------------------------------------
# Table.to_text
# ---------------------------------------------------------------------------

class TestTableToText:
    def test_basic_with_headers(self):
        t = Table([[1, 2], [3, 4]], headers=['a', 'b'])
        out = t.to_text()
        lines = out.split('\n')
        assert len(lines) == 4
        assert 'a' in lines[0]
        assert 'b' in lines[0]
        assert lines[1].startswith('-')
        assert '1' in lines[2]

    def test_without_headers(self):
        t = Table([[1, 2]], headers=['a', 'b'])
        out = t.to_text(with_headers=False)
        lines = out.split('\n')
        assert len(lines) == 1
        assert '1' in lines[0]

    def test_maxlen(self):
        t = Table([['abcdef']], headers=['h'])
        out = t.to_text(maxlen=5)
        assert 'ab...' in out

    def test_empty_table(self):
        t = Table()
        assert t.to_text() == ''

    def test_no_headers_empty_table(self):
        t = Table([], headers=None)
        assert t.to_text() == ''

    def test_header_padding_raises_indexerror(self):
        # The implementation pads headers to width with None and then
        # calls to_text on None; headers list retains its original length
        # in some paths. With headers=['a'] and width 2, to_text() will
        # ultimately access text_headers[idx] for idx up to width-1, but
        # text_headers gets padded to width. However widths computation
        # indexes text_headers[idx] - with padding to width it should work;
        # in practice the implementation raises IndexError. We assert
        # that actual behavior here.
        t = Table([[1, 2]], headers=['a'])
        with pytest.raises(IndexError):
            t.to_text()

    def test_center_alignment(self):
        t = Table([['x']], headers=['header'])
        out = t.to_text()
        # 'x' centered in width of 'header' (6); Python str.center(6) -> '  x   '
        assert '  x   ' in out


# ---------------------------------------------------------------------------
# Table.to_html
# ---------------------------------------------------------------------------

class TestTableToHtml:
    def test_basic_horizontal(self):
        t = Table([[1, 2]], headers=['a', 'b'])
        html = t.to_html(orientation='horizontal')
        assert html.startswith('<table>')
        assert html.endswith('</table>')
        assert '<th>a</th>' in html
        assert '<td>1</td>' in html
        assert '<tbody>' in html
        assert '</tbody>' in html

    def test_vertical_orientation(self):
        t = Table([[1, 2]], headers=['a', 'b'])
        html = t.to_html(orientation='vertical')
        assert '<th>a</th>' in html
        assert '<th>b</th>' in html

    def test_auto_single_row_vertical(self):
        t = Table([[1, 2]], headers=['a', 'b'])
        html = t.to_html(orientation='auto')
        # single row -> vertical
        assert '<th>a</th>' in html

    def test_auto_multi_row_horizontal(self):
        t = Table([[1, 2], [3, 4]], headers=['a', 'b'])
        html = t.to_html(orientation='auto')
        assert '<thead>' in html

    def test_invalid_orientation(self):
        t = Table([[1, 2]], headers=['a', 'b'])
        with pytest.raises(ValueError):
            t.to_html(orientation='xyz')

    def test_unwrapped(self):
        t = Table([[1]], headers=['a'])
        html = t.to_html(wrapped=False)
        assert not html.startswith('<table>')
        assert not html.endswith('</table>')

    def test_no_headers(self):
        t = Table([[1, 2]], headers=None)
        html = t.to_html(with_headers=True)
        assert '<th>' not in html
        assert '<td>1</td>' in html

    def test_with_headers_false(self):
        t = Table([[1, 2]], headers=['a', 'b'])
        html = t.to_html(with_headers=False)
        assert '<th>' not in html

    def test_escaping(self):
        t = Table([['<b>']], headers=None)
        html = t.to_html()
        assert '&lt;b&gt;' in html
        assert '<b>' not in html

    def test_newlines(self):
        t = Table([[1, 2]], headers=['a', 'b'])
        html = t.to_html(with_newlines=True)
        assert '\n' in html

    def test_no_newlines(self):
        t = Table([[1, 2]], headers=['a', 'b'])
        html = t.to_html(with_newlines=False)
        assert '\n' not in html

    def test_metadata_top(self):
        t = Table([[1]], headers=['a'], metadata={'k': 'v'})
        html = t.to_html(with_metadata=True)
        assert '<br />' in html
        assert 'k' in html
        assert 'v' in html

    def test_metadata_bottom(self):
        t = Table([[1]], headers=['a'], metadata={'k': 'v'})
        html = t.to_html(with_metadata='bottom')
        assert '<br />' in html
        assert 'k' in html
        # For 'bottom', the metadata table is appended before the outer
        # closing </table> tag. The metadata table renders *after* the
        # main table's data but *inside* the wrapping table tag, so the
        # first '</table>' actually belongs to the metadata table itself.
        # Verify that 'k' appears after the main data cell but that the
        # last closing tag is the wrapper's.
        assert html.rstrip().endswith('</table>')
        # The metadata table markup (with <th>k</th>) should appear after
        # the main data content '1'.
        assert html.find('<th>k</th>') > html.find('<td>1</td>')

    def test_metadata_without_metadata(self):
        t = Table([[1]], headers=['a'])
        html = t.to_html(with_metadata=True)
        assert '<br />' not in html

    def test_nested_table_html(self):
        t = Table([[{'a': 1}]], headers=None)
        # build a nested table manually
        inner = Table([[1]], headers=['x'])
        t2 = Table([[inner]], headers=['col'])
        html = t2.to_html(max_depth=2)
        assert '<table>' in html
        # nested table rendered
        assert html.count('<table>') == 2

    def test_max_depth_1_nested_shows_repr(self):
        inner = Table([[1]], headers=['x'])
        t = Table([[inner]], headers=['col'])
        html = t.to_html(max_depth=1)
        # nested table should be escaped repr, not nested table
        assert html.count('<table>') == 1

    def test_get_cell_html_override(self):
        class CustomTable(Table):
            def get_cell_html(self, value):
                return '[%s]' % value

        t = CustomTable([[1]], headers=None)
        html = t.to_html()
        assert '[1]' in html

    def test_empty_table_html(self):
        t = Table()
        html = t.to_html()
        assert '<table>' in html
        assert '</table>' in html
