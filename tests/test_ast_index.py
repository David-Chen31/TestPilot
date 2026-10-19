from testpilot.retrieval.ast_index import extract_symbols

SOURCE = '''
import functools


def helper(x):
    return x + 1


@functools.lru_cache
def cached(x: int) -> int:
    return helper(x) * 2


class Box:
    """A box."""

    def __init__(self, v):
        self.v = v

    @property
    def doubled(self):
        return cached(self.v)

    async def fetch(self, client):
        return await client.get(self.v)
'''


def _by_qualname():
    return {s.qualname: s for s in extract_symbols(SOURCE, "pkg/mod.py")}


def test_extracts_functions_classes_methods():
    syms = _by_qualname()
    assert set(syms) == {"helper", "cached", "Box", "Box.__init__", "Box.doubled", "Box.fetch"}
    assert syms["helper"].kind == "function"
    assert syms["Box"].kind == "class"
    assert syms["Box.fetch"].kind == "method"
    assert all(s.file == "pkg/mod.py" for s in syms.values())


def test_line_ranges_include_decorators():
    syms = _by_qualname()
    assert (syms["helper"].start_line, syms["helper"].end_line) == (5, 6)
    assert syms["cached"].start_line == 9  # 装饰器所在行
    assert syms["Box.doubled"].start_line == 20


def test_signatures():
    syms = _by_qualname()
    assert syms["cached"].signature == "def cached(x: int) -> int"
    assert syms["Box.fetch"].signature == "async def fetch(self, client)"
    assert syms["Box"].signature == "class Box"


def test_called_names():
    syms = _by_qualname()
    assert syms["cached"].calls == ["helper"]
    assert syms["Box.doubled"].calls == ["cached"]
    assert syms["Box.fetch"].calls == ["get"]
    assert syms["helper"].calls == []
