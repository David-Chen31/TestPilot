import os
from os.path import expanduser, join

import pytest

from boltons.pathutils import augpath, shrinkuser, expandpath


# ---------------------------------------------------------------------------
# augpath
# ---------------------------------------------------------------------------

def test_augpath_noop():
    assert augpath('foo.bar') == 'foo.bar'


def test_augpath_docstring_examples():
    assert augpath('foo.bar') == 'foo.bar'
    assert augpath('foo.bar', ext='.BAZ') == 'foo.BAZ'
    assert augpath('foo.bar', suffix='_') == 'foo_.bar'
    assert augpath('foo.bar', prefix='_') == '_foo.bar'
    assert augpath('foo.bar', base='baz') == 'baz.bar'
    assert augpath('foo.tar.gz', ext='.zip', multidot=True) == 'foo.zip'
    assert augpath('foo.tar.gz', ext='.zip', multidot=False) == 'foo.tar.zip'
    assert augpath('foo.tar.gz', suffix='_new', multidot=True) == 'foo_new.tar.gz'


def test_augpath_combined_example():
    newpath = augpath('foo.bar', suffix='_suff', prefix='pref_',
                      ext='.baz', base='bar')
    assert newpath == 'pref_bar_suff.baz'


def test_augpath_no_extension():
    assert augpath('foo') == 'foo'
    assert augpath('foo', suffix='_x') == 'foo_x'
    assert augpath('foo', prefix='p_') == 'p_foo'
    assert augpath('foo', ext='.txt') == 'foo.txt'
    assert augpath('foo', base='bar') == 'bar'


def test_augpath_dpath_replacement():
    assert augpath('/a/b/c.txt', dpath='/x/y') == join('/x/y', 'c.txt')


def test_augpath_dpath_preserved():
    assert augpath('/a/b/c.txt') == join('/a/b', 'c.txt')


def test_augpath_dpath_none_resets():
    # Passing dpath='' replaces the directory with empty
    assert augpath('/a/b/c.txt', dpath='') == 'c.txt'


def test_augpath_multidot_no_extension():
    assert augpath('foo', multidot=True) == 'foo'
    assert augpath('foo', multidot=True, ext='.txt') == 'foo.txt'
    assert augpath('foo', multidot=True, suffix='_s') == 'foo_s'


def test_augpath_multidot_with_single_dot():
    assert augpath('foo.bar', multidot=True) == 'foo.bar'
    assert augpath('foo.bar', multidot=True, ext='.baz') == 'foo.baz'


def test_augpath_multidot_base_keeps_dots():
    # multidot: first dot splits extension, so base is only before first dot
    assert augpath('foo.tar.gz', multidot=True, base='new') == 'new.tar.gz'


def test_augpath_multidot_suffix_only():
    assert augpath('foo.tar.gz', multidot=True, suffix='_s') == 'foo_s.tar.gz'


def test_augpath_multidot_prefix():
    assert augpath('foo.tar.gz', multidot=True, prefix='p_') == 'p_foo.tar.gz'


def test_augpath_hidden_file():
    # splitext treats '.bashrc' as having no extension
    assert augpath('.bashrc') == '.bashrc'
    assert augpath('.bashrc', ext='.txt') == '.bashrc.txt'


def test_augpath_pathlib_like():
    class FakePath:
        def __init__(self, s):
            self._s = s

        def __str__(self):
            return self._s

        def __fspath__(self):
            return self._s

    # os.path.split works with __fspath__
    result = augpath(FakePath('dir/foo.txt'), ext='.md')
    assert result == join('dir', 'foo.md')


def test_augpath_trailing_sep_dir():
    # A path ending in sep is just a dir with empty fname
    assert augpath('a/b/') == join('a/b', '')


# ---------------------------------------------------------------------------
# shrinkuser
# ---------------------------------------------------------------------------

def test_shrinkuser_home_exact():
    home = expanduser('~')
    assert shrinkuser(home) == '~'


def test_shrinkuser_home_custom_symbol():
    home = expanduser('~')
    assert shrinkuser(home, home='$HOME') == '$HOME'


def test_shrinkuser_home_subpath():
    home = expanduser('~')
    assert shrinkuser(home + '/1') == join('~', '1')
    assert shrinkuser(home + '/1/2/3') == join('~', '1/2/3')


def test_shrinkuser_custom_symbol_subpath():
    home = expanduser('~')
    assert shrinkuser(home + '/1', '$HOME') == join('$HOME', '1')


def test_shrinkuser_sibling_prefix_not_shortened():
    home = expanduser('~')
    # home + '1' shares prefix but is not inside home
    assert shrinkuser(home + '1') == home + '1'


def test_shrinkuser_unrelated_path():
    assert shrinkuser('/some/other/path') == '/some/other/path'


def test_shrinkuser_normalizes():
    home = expanduser('~')
    # redundant separators / dots are normalized first
    assert shrinkuser(home + '/./a/../b') == join('~', 'b')


def test_shrinkuser_dot_path_not_home():
    # '.' normalizes to '.', should be unrelated to home (unless home is cwd)
    result = shrinkuser('.')
    if os.path.normpath(expanduser('~')) != '.':
        assert result == '.'


# ---------------------------------------------------------------------------
# expandpath
# ---------------------------------------------------------------------------

def test_expandpath_env_var(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    assert expandpath('~/$SPAM') == expanduser('~/eggs')


def test_expandpath_plain(monkeypatch):
    monkeypatch.delenv('SPAM', raising=False)
    assert expandpath('nonexistentvar') == 'nonexistentvar'


def test_expandpath_tilde():
    assert expandpath('~') == expanduser('~')
    assert expandpath('~/foo') == expanduser('~/foo')


def test_expandpath_env_var_brace(monkeypatch):
    monkeypatch.setenv('MYVAR', 'value')
    assert expandpath('${MYVAR}/x') == 'value/x'


def test_expandpath_env_var_only(monkeypatch):
    monkeypatch.setenv('ONLYVAR', 'abc')
    assert expandpath('$ONLYVAR') == 'abc'


def test_expandpath_undefined_env_removed():
    # expandvars removes undefined variables
    assert expandpath('$THIS_VAR_SHOULD_NOT_EXIST_ANYWHERE') == ''


def test_expandpath_combined(monkeypatch):
    monkeypatch.setenv('FOODIR', 'things')
    result = expandpath('~/dir/$FOODIR/file')
    assert result == expanduser('~/dir/things/file')
