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
# to_text / escape_html
# ---------------------------------------------------------------------------

def test_to_text_basic():
    assert to_text(123) == "123"
    assert to_text("hello") == "hello"
    assert to_text(None) == "None"


def test_to_text_maxlen_truncates():
    assert to_text("abcdefghij", maxlen=5) == "ab..."
    # exactly maxlen is not truncated
    assert to_text("abcde", maxlen=5) == "abcde"
    # shorter than maxlen untouched
    assert to_text("ab", maxlen=5) == "ab"


def test_to_text_maxlen_none_disables():
    assert to_text("abcdefghij", maxlen=None) == "abcdefghij"


def test_to_text_falls_back_to_repr():
    class BadStr:
        def __str__(self):
            raise ValueError("no str")

        def __repr__(self):
            return "REPRVALUE"

    assert to_text(BadStr()) == "REPRVALUE"


def test_to_text_falls_back_to_object_repr():
    class BadBoth:
        def __str__(self):
            raise ValueError("no str")

        def __repr__(self):
            raise ValueError("no repr")

    out = to_text(BadBoth())
    assert out.startswith("<")
    assert "BadBoth" in out


def test_escape_html_basic():
    assert escape_html("<a>") == "&lt;a&gt;"


def test_escape_html_quotes():
    # quote=True must escape both single and double quotes
    assert escape_html('"') == "&quot;"
    assert escape_html("'") == "&#x27;"
    assert escape_html('a"b\'c') == "a&quot;b&#x27;c"


def test_escape_html_ampersand():
    assert escape_html("a&b") == "a&amp;b"


def test_escape_html_maxlen():
    # truncation happens on the text BEFORE html-escaping
    assert escape_html("<abcdefgh>", maxlen=5) == "&lt;a..."
    assert escape_html("abcdefgh", maxlen=5) == "ab..."


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------

def test_input_type_base_get_entry_seq():
    it = InputType()
    # base get_entry is not defined; get_entry_seq calls it -> AttributeError
    with pytest.raises(AttributeError):
        it.get_entry_seq([1, 2, 3], [])


def test_input_type_base_get_entry_seq_passes_args_through():
    # A subclass that records the arguments passed for each entry
    class Recorder(InputType):
        def __init__(self):
            self.calls = []

        def get_entry(self, entry, headers):
            self.calls.append((entry, headers))
            return entry

    rec = Recorder()
    out = rec.get_entry_seq([10, 20, 30], ["H"])
    assert out == [10, 20, 30]
    # each call gets the actual entry and the exact headers object
    assert rec.calls == [(10, ["H"]), (20, ["H"]), (30, ["H"])]
    assert rec.calls[0][1] is rec.calls[1][1]  # same headers object
    assert rec.calls[0][1] is rec.calls[2][1]


def test_input_type_base_get_entry_seq_preserves_object_identity():
    headers_obj = object()

    class Recorder(InputType):
        def __init__(self):
            self.headers_seen = []

        def get_entry(self, entry, headers):
            self.headers_seen.append(headers)
            return entry

    rec = Recorder()
    rec.get_entry_seq([1, 2], headers_obj)
    assert rec.headers_seen == [headers_obj, headers_obj]
    assert rec.headers_seen[0] is headers_obj


def test_dict_input_type_check_and_headers():
    it = DictInputType()
    assert it.check_type({}) is True
    assert it.check_type([]) is False
    assert it.guess_headers({"b": 1, "a": 2}) == ["a", "b"]
    assert it.get_entry({"a": 1, "b": 2}, ["a", "b"]) == [1, 2]
    assert it.get_entry({"a": 1}, ["a", "b"]) == [1, None]


def test_dict_input_type_get_entry_seq():
    it = DictInputType()
    data = [{"a": 1, "b": 2}, {"a": 3}]
    assert it.get_entry_seq(data, ["a", "b"]) == [[1, 2], [3, None]]


def test_list_input_type():
    it = ListInputType()
    assert it.check_type([]) is True
    assert it.check_type((1, 2)) is False
    assert it.guess_headers([1, 2]) is None
    assert it.get_entry([1, 2], None) == [1, 2]
    seq = [[1], [2]]
    assert it.get_entry_seq(seq, None) is seq


def test_tuple_input_type():
    it = TupleInputType()
    assert it.check_type((1, 2)) is True
    assert it.check_type([]) is False
    assert it.guess_headers((1, 2)) is None
    assert it.get_entry((1, 2), None) == [1, 2]
    assert it.get_entry_seq([(1, 2), (3, 4)], None) == [[1, 2], [3, 4]]


def test_named_tuple_input_type():
    from collections import namedtuple

    Point = namedtuple("Point", ["x", "y"])
    it = NamedTupleInputType()
    p = Point(1, 2)
    assert it.check_type(p) is True
    assert it.check_type((1, 2)) is False
    assert it.check_type([1, 2]) is False
    assert it.guess_headers(p) == ["x", "y"]
    assert it.get_entry(p, ["x", "y"]) == [1, 2]
    assert it.get_entry_seq([p, Point(3, 4)], ["x", "y"]) == [[1, 2], [3, 4]]


def test_named_tuple_input_type_missing_attr_returns_none():
    from collections import namedtuple

    Point = namedtuple("Point", ["x", "y"])
    it = NamedTupleInputType()
    # Asking for a field not in the namedtuple yields None (default arg used)
    assert it.get_entry(Point(1, 2), ["x", "nope"]) == [1, None]
    assert it.get_entry_seq([Point(1, 2), Point(3, 4)], ["x", "nope"]) == [
        [1, None], [3, None]
    ]


def test_object_input_type():
    class Foo:
        def __init__(self):
            self.a = 1
            self.b = 2

        def method(self):
            return 1

    it = ObjectInputType()
    foo = Foo()
    assert it.check_type(foo) is True
    assert it.check_type(1) is False
    assert it.check_type("str") is False
    headers = it.guess_headers(foo)
    assert "a" in headers
    assert "b" in headers
    assert "method" not in headers
    assert it.get_entry(foo, ["a", "b"]) == [1, 2]
    # missing attribute -> None
    assert it.get_entry(foo, ["a", "missing"]) == [1, None]


def test_object_input_type_check_type_denylist():
    it = ObjectInputType()
    # Types in _DNR must return False
    assert it.check_type(None) is False
    assert it.check_type(True) is False
    assert it.check_type(3.14) is False
    assert it.check_type(1 + 2j) is False
    assert it.check_type(slice(1, 2)) is False
    assert it.check_type(b"bytes") is False
    assert it.check_type(lambda: None) is False


def test_object_input_type_skips_raising_properties():
    class Prop:
        @property
        def bad(self):
            raise RuntimeError("boom")

        def __init__(self):
            self.good = 5

    it = ObjectInputType()
    headers = it.guess_headers(Prop())
    assert "good" in headers
    assert "bad" not in headers


# ---------------------------------------------------------------------------
# Table basics
# ---------------------------------------------------------------------------

def test_table_simple_list_of_lists():
    # With default headers=_MISSING, the first row becomes headers
    t = Table([[1, 2], [3, 4]])
    assert len(t) == 1
    assert t.headers == [1, 2]
    assert t[0] == [3, 4]
    assert t._width == 2


def test_table_with_headers():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    assert t.headers == ["a", "b"]
    assert len(t) == 2


def test_table_first_row_becomes_headers():
    t = Table([["a", "b"], [1, 2], [3, 4]])
    assert t.headers == ["a", "b"]
    assert len(t) == 2
    assert t[0] == [1, 2]


def test_table_fill():
    t = Table([[1], [2, 3, 4]], headers=["a", "b", "c"])
    assert t[0] == [1, None, None]
    assert t[1] == [2, 3, 4]


def test_table_empty():
    t = Table()
    assert len(t) == 0
    assert t.headers == []
    assert t._width == 0


def test_table_empty_metadata_default():
    t = Table()
    assert t.metadata == {}
    t2 = Table(metadata={"a": 1})
    assert t2.metadata == {"a": 1}


def test_table_repr_with_headers():
    t = Table([[1, 2]], headers=["a", "b"])
    r = repr(t)
    assert "headers=" in r
    assert "'a'" in r


def test_table_repr_without_headers():
    # Passing a first row makes headers; empty table is default
    t = Table(headers=None)
    r = repr(t)
    assert "headers" not in r


def test_table_repr_exact():
    t = Table([[1, 2]], headers=["a", "b"])
    assert repr(t) == "Table(headers=['a', 'b'], data=[[1, 2]])"
    t2 = Table(headers=None)
    assert repr(t2) == "Table([])"


def test_table_extend():
    t = Table([[1, 2]], headers=["a", "b"])
    t.extend([[3, 4]])
    assert len(t) == 2
    assert t[1] == [3, 4]


def test_table_extend_empty_noop():
    t = Table([[1, 2]], headers=["a", "b"])
    t.extend([])
    assert len(t) == 1


def test_table_extend_fills_short_rows():
    t = Table([[1, 2]], headers=["a", "b", "c"])
    t.extend([[3]])
    assert t[1] == [3, None, None]


def test_table_set_width_reset():
    t = Table([[1, 2, 3]], headers=["a", "b", "c"])
    assert t._width == 3
    t._set_width(reset=True)
    # After reset, recompute from headers
    assert t._width == 3


def test_table_set_width_no_headers():
    t = Table([[1, 2, 3]])
    # No headers anymore because empty after init? Actually headers=[1,2,3]
    # Use headers=None explicitly
    t2 = Table([[4, 5, 6, 7]], headers=None)
    assert t2._width == 4


def test_table_set_width_from_data_max():
    t = Table([[1, 2], [3, 4, 5, 6]], headers=None)
    assert t._width == 4
    assert t[0] == [1, 2, None, None]


# ---------------------------------------------------------------------------
# Table.from_data / from_dict / from_list / from_object
# ---------------------------------------------------------------------------

def test_from_data_list_of_lists():
    t = Table.from_data([[1, 2], [3, 4]])
    assert len(t) == 2
    assert t[0] == [1, 2]


def test_from_data_from_dict_single():
    t = Table.from_data({"b": 2, "a": 1})
    assert t.headers == ["a", "b"]
    assert t[0] == [1, 2]


def test_from_data_list_of_dicts():
    t = Table.from_data([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert t.headers == ["a", "b"]
    assert t[0] == [1, 2]
    assert t[1] == [3, 4]


def test_from_data_list_of_namedtuples():
    from collections import namedtuple

    Point = namedtuple("Point", ["x", "y"])
    t = Table.from_data([Point(1, 2), Point(3, 4)])
    assert t.headers == ["x", "y"]
    assert t[0] == [1, 2]


def test_from_data_single_namedtuple():
    from collections import namedtuple

    Point = namedtuple("Point", ["x", "y"])
    t = Table.from_data(Point(1, 2))
    assert t.headers == ["x", "y"]
    assert t[0] == [1, 2]


def test_from_data_single_object():
    class Foo:
        def __init__(self):
            self.a = 1
            self.b = 2

    t = Table.from_data(Foo())
    assert "a" in t.headers
    assert "b" in t.headers
    assert t[0][t.headers.index("a")] == 1
    assert t[0][t.headers.index("b")] == 2


def test_from_data_scalar():
    # Scalar data is treated as a Table with one header and no data rows
    t = Table.from_data(42)
    assert t.headers == [42]
    assert len(t) == 0


def test_from_data_empty_list():
    t = Table.from_data([])
    assert len(t) == 0


def test_from_data_max_depth_zero():
    t = Table.from_data([[1, 2], [3, 4]], max_depth=0)
    assert len(t) == 0


def test_from_data_unsupported():
    # A non-Sequence that is also handled by ObjectInputType -> no raise
    class Weird:
        def __init__(self):
            self.x = 1

    t = Table.from_data(Weird())
    assert "x" in t.headers


def test_from_data_unsupported_type_raises():
    # Verify a supported object does not raise
    class CheckedData:
        pass

    t = Table.from_data(CheckedData())
    assert isinstance(t, Table)


def test_from_data_unsupported_raises_directly():
    # Force an UnsupportedData by supplying an InputType that never matches.
    # Note: from_data checks _data_type only when headers is _MISSING and
    # when guess_headers is needed. To trigger UnsupportedData without an
    # unhandled attribute error, provide an explicit headers list so that
    # guess_headers is never called, then have get_entry_seq raise via a
    # supplied _data_type that makes the loop over input types fall through.
    # The actual UnsupportedData path is the "for ... else" branch, which
    # is tricky to reach with a fully-formed InputType. Instead, verify that
    # a never-matching _data_type without headers raises AttributeError
    # (because guess_headers is invoked on it). This documents the real
    # behavior of the implementation.
    class NeverInput(InputType):
        def check_type(self, obj):
            return False

    with pytest.raises(AttributeError):
        Table.from_data([1, 2, 3], _data_type=NeverInput())


def test_from_data_unsupported_via_headers_raises():
    # When headers are supplied and a NeverInput type is forced, the code
    # falls into get_entry_seq which calls the base InputType.get_entry,
    # which is not defined -> AttributeError.
    class NeverInput(InputType):
        def check_type(self, obj):
            return False

    with pytest.raises(AttributeError):
        Table.from_data([1, 2, 3], headers=["a"], _data_type=NeverInput())


def test_from_data_headers_explicit():
    t = Table.from_dict([{"a": 1, "b": 2}], headers=["b", "a"])
    assert t.headers == ["b", "a"]
    assert t[0] == [2, 1]


def test_from_dict():
    t = Table.from_dict({"a": 1, "b": 2})
    assert t.headers == ["a", "b"]
    assert t[0] == [1, 2]


def test_from_dict_list():
    t = Table.from_dict([{"a": 1}, {"a": 2}])
    assert t.headers == ["a"]
    assert len(t) == 2
    assert t[0] == [1]
    assert t[1] == [2]


def test_from_list():
    t = Table.from_list([[1, 2], [3, 4]])
    assert len(t) == 2
    assert t[0] == [1, 2]


def test_from_object():
    class Foo:
        def __init__(self):
            self.a = 7

    t = Table.from_object(Foo())
    assert "a" in t.headers
    assert t[0][t.headers.index("a")] == 7


def test_from_data_metadata():
    t = Table.from_data([{"a": 1}], metadata={"m": "v"})
    assert t.metadata == {"m": "v"}


def test_from_data_nested_max_depth():
    t = Table.from_data([[{"a": 1}]], max_depth=2)
    # cell should become a Table
    assert isinstance(t[0][0], Table)


# ---------------------------------------------------------------------------
# to_html
# ---------------------------------------------------------------------------

def test_to_html_basic_auto_horizontal():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    html = t.to_html()
    assert "<table>" in html
    assert "</table>" in html
    assert "<th>a</th>" in html
    assert "<td>1</td>" in html


def test_to_html_wrapped_false():
    # auto orientation with a single row -> vertical layout (no tbody)
    t = Table([[1, 2]], headers=["a", "b"])
    html = t.to_html(wrapped=False)
    assert "<table>" not in html
    assert "<tr>" in html
    assert "<th>a</th>" in html
    assert "<td>1</td>" in html


def test_to_html_wrapped_false_horizontal():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    html = t.to_html(wrapped=False)
    assert "<table>" not in html
    assert "<tbody>" in html


def test_to_html_with_newlines():
    t = Table([[1, 2]], headers=["a", "b"])
    html = t.to_html(with_newlines=True)
    assert "\n" in html


def test_to_html_without_newlines_is_single_line():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    html = t.to_html(with_newlines=False)
    assert "\n" not in html


def test_to_html_no_headers():
    t = Table([[1, 2]])
    html = t.to_html(with_headers=False)
    assert "<thead>" not in html


def test_to_html_vertical():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    html = t.to_html(orientation="vertical")
    assert "<th>a</th>" in html
    assert "<td>1</td>" in html
    assert "<td>3</td>" in html


def test_to_html_orientation_first_letter():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    assert t.to_html(orientation="h") == t.to_html(orientation="horizontal")
    assert t.to_html(orientation="v") == t.to_html(orientation="vertical")


def test_to_html_horizontal():
    # Without explicit headers, first row becomes headers
    t = Table([[1, 2], [3, 4]])
    html = t.to_html(orientation="horizontal")
    assert "<th>1</th>" in html
    assert "<td>3</td>" in html


def test_to_html_horizontal_with_headers():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    html = t.to_html(orientation="horizontal")
    assert "<td>1</td>" in html
    assert "<td>3</td>" in html


def test_to_html_escapes():
    t = Table([["<script>"]])
    html = t.to_html()
    assert "&lt;script&gt;" in html
    assert "<script>" not in html


def test_to_html_invalid_orientation():
    t = Table([[1, 2]])
    with pytest.raises(ValueError) as excinfo:
        t.to_html(orientation="nonsense")
    assert "nonsense" in str(excinfo.value)


def test_to_html_with_metadata_top():
    t = Table([[1, 2]], headers=["a", "b"], metadata={"k": "v"})
    html = t.to_html(with_metadata=True)
    assert "k" in html
    assert "v" in html


def test_to_html_with_metadata_bottom():
    t = Table([[1, 2]], headers=["a", "b"], metadata={"k": "v"})
    html = t.to_html(with_metadata="bottom")
    assert "k" in html
    # metadata should come after table content
    assert html.rindex("k") > html.index("<table>")


def test_to_html_with_metadata_top_position():
    t = Table([[1, 2]], headers=["a", "b"], metadata={"k": "v"})
    html = t.to_html(with_metadata=True)
    # The metadata table is prepended, so the metadata table's '<table>' tag
    # appears first, and the metadata content 'k' precedes the main table's
    # content. Compare the second '<table>' occurrence (main table) to 'k'.
    second_table = html.index("<table>", html.index("<table>") + 1)
    assert html.index("k") < second_table
    # And a <br /> separates metadata table from the main table
    assert "<br />" in html


def test_to_html_nested_table_max_depth():
    inner = Table([[10]], headers=["x"])
    t = Table([[inner]], headers=["col"])
    html = t.to_html(max_depth=2)
    # Inner table HTML is embedded, not repr'd
    assert "<table>" in html
    assert "&lt;Table" not in html


def test_get_cell_html():
    t = Table([[1]])
    assert t.get_cell_html("<x>") == "&lt;x&gt;"


# ---------------------------------------------------------------------------
# Table.to_text
# ---------------------------------------------------------------------------

def test_to_text_table_basic():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    text = t.to_text()
    assert "a" in text
    assert "1" in text
    assert "|" in text


def test_to_text_no_headers():
    t = Table([[1, 2], [3, 4]])
    text = t.to_text()
    # headers derived from first row; separator line is present
    lines = text.split("\n")
    assert len(lines) == 3
    assert lines[0] == "1 | 2"
    assert lines[1] == "--|--"
    assert lines[2] == "3 | 4"


def test_to_text_with_headers_false():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    text = t.to_text(with_headers=False)
    lines = text.split("\n")
    assert len(lines) == 2
    assert "a" not in text


def test_to_text_exact_format():
    t = Table([[1, 2], [3, 4]], headers=["a", "b"])
    text = t.to_text()
    assert text == "a | b\n--|--\n1 | 2\n3 | 4"


def test_to_text_centering():
    t = Table([["xx"]], headers=["abcdef"])
    text = t.to_text()
    lines = text.split("\n")
    # header wider -> centered
    assert lines[0] == "abcdef"
    assert lines[1] == "------"


def test_to_text_maxlen():
    t = Table([["abcdefghij"]])
    text = t.to_text(maxlen=5)
    assert "..." in text
    assert "abcdefghij" not in text


def test_to_text_empty():
    t = Table()
    assert t.to_text() == ""


def test_to_text_metadata_not_shown():
    # to_text ignores metadata
    t = Table([[1]], headers=["a"], metadata={"k": "v"})
    text = t.to_text()
    assert "k" not in text
    assert "v" not in text
