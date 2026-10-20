import os
import os.path
from pathlib import Path

import pytest

from boltons.pathutils import augpath, shrinkuser, expandpath


# ---------------------------------------------------------------------------
# augpath
# ---------------------------------------------------------------------------

def test_augpath_noop():
    assert augpath('foo.bar') == 'foo.bar'


def test_augpath_docstring_example():
    newpath = augpath('foo.bar', '_suff', 'pref_', ext='.baz', base='bar')
    assert newpath == 'pref_bar_suff.baz'


def test_augpath_replace_ext():
    assert augpath('foo.bar', ext='.BAZ') == 'foo.BAZ'


def test_augpath_suffix():
    assert augpath('foo.bar', suffix='_') == 'foo_.bar'


def test_augpath_prefix():
    assert augpath('foo.bar', prefix='_') == '_foo.bar'


def test_augpath_replace_base():
    assert augpath('foo.bar', base='baz') == 'baz.bar'


def test_augpath_multidot_true_replace_ext():
    assert augpath('foo.tar.gz', ext='.zip', multidot=True) == 'foo.zip'


def test_augpath_multidot_false_replace_ext():
    assert augpath('foo.tar.gz', ext='.zip', multidot=False) == 'foo.tar.zip'


def test_augpath_multidot_true_suffix():
    assert augpath('foo.tar.gz', suffix='_new', multidot=True) == 'foo_new.tar.gz'


def test_augpath_multidot_true_no_dot():
    # No dot at all: orig_base is whole name, orig_ext is ''
    assert augpath('foobar', ext='.txt', multidot=True) == 'foobar.txt'


def test_augpath_no_extension():
    assert augpath('foo', ext='.txt') == 'foo.txt'


def test_augpath_base_no_ext():
    assert augpath('foo.bar', base='baz', ext='') == 'baz'


def test_augpath_dpath_replacement():
    result = augpath(os.path.join('a', 'b', 'foo.bar'), dpath=os.path.join('x', 'y'))
    assert result == os.path.join('x', 'y', 'foo.bar')


def test_augpath_preserves_dpath():
    p = os.path.join('some', 'dir', 'foo.bar')
    assert augpath(p, ext='.baz') == os.path.join('some', 'dir', 'foo.baz')


def test_augpath_prefix_suffix_combined():
    assert augpath('foo.bar', prefix='p_', suffix='_s') == 'p_foo_s.bar'


def test_augpath_all_replacements():
    result = augpath(
        os.path.join('a', 'foo.bar'),
        suffix='_s',
        prefix='p_',
        ext='.X',
        base='baz',
        dpath='d',
    )
    assert result == os.path.join('d', 'p_baz_s.X')


def test_augpath_accepts_pathlike_input():
    result = augpath(Path('foo.bar'), ext='.baz')
    assert result == 'foo.baz'


def test_augpath_returns_str():
    assert isinstance(augpath('foo.bar', ext='.baz'), str)


def test_augpath_multidot_with_base_replacement():
    result = augpath('foo.tar.gz', base='bar', multidot=True)
    assert result == 'bar.tar.gz'


def test_augpath_hidden_file():
    # '.bashrc' -> splitext gives ('.bashrc', '')
    assert augpath('.bashrc', ext='.bak') == '.bashrc.bak'


def test_augpath_directory_input():
    # Trailing separator: basename is ''
    result = augpath(os.path.join('a', 'b', ''))
    assert result == os.path.join('a', 'b', '')


# ---------------------------------------------------------------------------
# shrinkuser
# ---------------------------------------------------------------------------

def test_shrinkuser_home():
    home = os.path.expanduser('~')
    assert shrinkuser(home) == '~'


def test_shrinkuser_custom_home():
    home = os.path.expanduser('~')
    assert shrinkuser(home, '$HOME') == '$HOME'


def test_shrinkuser_inside_home():
    home = os.path.expanduser('~')
    target = os.path.join(home, '1')
    assert shrinkuser(target) == os.path.join('~', '1')


def test_shrinkuser_inside_home_custom_symbol():
    home = os.path.expanduser('~')
    target = os.path.join(home, '1')
    assert shrinkuser(target, '$HOME') == os.path.join('$HOME', '1')


def test_shrinkuser_prefix_not_home():
    home = os.path.expanduser('~')
    # Similar prefix but not actually inside home
    target = home + '1'
    assert shrinkuser(target) == target


def test_shrinkuser_unrelated_path():
    assert shrinkuser('/some/other/path') == os.path.normpath('/some/other/path')


def test_shrinkuser_normalizes():
    home = os.path.expanduser('~')
    messy = os.path.join(home, 'a', '..', 'b')
    assert shrinkuser(messy) == os.path.join('~', 'b')


def test_shrinkuser_returns_str():
    assert isinstance(shrinkuser(os.path.expanduser('~')), str)


# ---------------------------------------------------------------------------
# expandpath
# ---------------------------------------------------------------------------

def test_expandpath_env_var(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    home = os.path.expanduser('~')
    assert expandpath('~/$SPAM') == os.path.join(home, 'eggs')


def test_expandpath_plain(monkeypatch):
    # Ensure any 'SPAM' variable present in the environment is irrelevant
    # by checking a path that doesn't use variable expansion.
    assert expandpath('foo') == 'foo'


def test_expandpath_expands_tilde(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    assert expandpath('~/$SPAM') == os.path.join(os.path.expanduser('~'), 'eggs')


def test_expandpath_multiple_vars(monkeypatch):
    monkeypatch.setenv('FOO', 'one')
    monkeypatch.setenv('BAR', 'two')
    assert expandpath('$FOO/$BAR') == os.path.join('one', 'two')


def test_expandpath_undefined_var(monkeypatch):
    monkeypatch.delenv('UNDEFINED_VAR_XYZ', raising=False)
    assert expandpath('$UNDEFINED_VAR_XYZ') == '$UNDEFINED_VAR_XYZ'


def test_expandpath_returns_str():
    assert isinstance(expandpath('foo'), str)


def test_expandpath_pathlike(monkeypatch):
    monkeypatch.delenv('UNDEFINED_VAR_XYZ', raising=False)
    result = expandpath(Path('foo'))
    assert result == 'foo'


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

def test_all_exports():
    import boltons.pathutils as mod
    assert set(mod.__all__) == {'augpath', 'shrinkuser', 'expandpath'}
