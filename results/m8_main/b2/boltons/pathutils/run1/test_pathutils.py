import os
import os.path
from pathlib import Path

import pytest

from boltons.pathutils import augpath, shrinkuser, expandpath


# ---------------------------------------------------------------------------
# augpath
# ---------------------------------------------------------------------------

def test_augpath_identity():
    assert augpath('foo.bar') == 'foo.bar'
    assert augpath('noext') == 'noext'


def test_augpath_docstring_examples():
    assert augpath('foo.bar', ext='.BAZ') == 'foo.BAZ'
    assert augpath('foo.bar', suffix='_') == 'foo_.bar'
    assert augpath('foo.bar', prefix='_') == '_foo.bar'
    assert augpath('foo.bar', base='baz') == 'baz.bar'
    assert augpath('foo.tar.gz', ext='.zip', multidot=True) == 'foo.zip'
    assert augpath('foo.tar.gz', ext='.zip', multidot=False) == 'foo.tar.zip'
    assert augpath('foo.tar.gz', suffix='_new', multidot=True) == 'foo_new.tar.gz'


def test_augpath_all_components():
    newpath = augpath('foo.bar', suffix='_suff', prefix='pref_', ext='.baz', base='bar')
    assert newpath == 'pref_bar_suff.baz'


def test_augpath_directory_replacement():
    assert augpath('/a/b/foo.txt', dpath='/x/y') == os.path.join('/x/y', 'foo.txt')
    # Default keeps original directory
    assert augpath('/a/b/foo.txt', suffix='_v2') == os.path.join('/a/b', 'foo_v2.txt')


def test_augpath_keeps_original_dpath():
    result = augpath('some/dir/file.py', ext='.txt')
    assert result == os.path.join('some/dir', 'file.txt')


def test_augpath_no_extension():
    # No extension: ext default is ''
    assert augpath('file', suffix='_s', prefix='p_') == 'p_file_s'
    # Adding an ext to a file with no extension
    assert augpath('file', ext='.txt') == 'file.txt'


def test_augpath_multidot_no_dot():
    assert augpath('nodot', multidot=True) == 'nodot'
    assert augpath('nodot', ext='.x', multidot=True) == 'nodot.x'


def test_augpath_multidot_single_dot():
    assert augpath('name.ext', multidot=True) == 'name.ext'


def test_augpath_multidot_replaces_whole_extension():
    assert augpath('a.b.c.d', ext='.z', multidot=True) == 'a.z'
    assert augpath('a.b.c.d', ext='.z', multidot=False) == 'a.b.c.z'


def test_augpath_multidot_replaces_base():
    # base replacement removes everything up to and including first dot
    result = augpath('a.b.c', base='X', multidot=True)
    assert result == 'X.b.c'


def test_augpath_empty_string():
    assert augpath('') == ''


def test_augpath_pathlike():
    result = augpath(Path('foo.bar'), ext='.baz')
    assert result == 'foo.baz'


def test_augpath_absolute_path():
    result = augpath('/tmp/foo.tar.gz', ext='.zip', multidot=True)
    assert result == os.path.join('/tmp', 'foo.zip')


def test_augpath_default_multidot_is_false():
    # The default for multidot is False, so the last dot defines the extension.
    # This distinguishes the default from multidot=True on a multi-dot name.
    assert augpath('foo.tar.gz', ext='.zip') == 'foo.tar.zip'
    assert augpath('a.b.c', suffix='_x') == 'a.b_x.c'
    assert augpath('a.b.c', base='X') == 'X.c'
    # Explicit multidot=False matches the default behavior
    assert augpath('a.b.c', ext='.z', multidot=False) == 'a.b.z'


# ---------------------------------------------------------------------------
# shrinkuser
# ---------------------------------------------------------------------------

def test_shrinkuser_docstring_examples():
    path = os.path.expanduser('~')
    assert path != '~'
    assert shrinkuser(path) == '~'
    assert shrinkuser(path + '1') == path + '1'
    assert shrinkuser(os.path.join(path, '1')) == os.path.join('~', '1')
    assert shrinkuser(os.path.join(path, '1'), '$HOME') == os.path.join('$HOME', '1')


def test_shrinkuser_custom_home():
    home = os.path.expanduser('~')
    assert shrinkuser(home, home='$HOME') == '$HOME'


def test_shrinkuser_non_home_path_untouched():
    assert shrinkuser('/some/other/path') == '/some/other/path'


def test_shrinkuser_normalizes():
    home = os.path.expanduser('~')
    # normpath applied; 'foo/../bar' becomes 'bar'
    assert shrinkuser('foo/../bar') == 'bar'
    weird = os.path.join(home, '1', '..', '2')
    assert shrinkuser(weird) == os.path.join('~', '2')


def test_shrinkuser_pathlike():
    home = os.path.expanduser('~')
    result = shrinkuser(Path(home) / '1')
    assert result == os.path.join('~', '1')


# ---------------------------------------------------------------------------
# expandpath
# ---------------------------------------------------------------------------

def test_expandpath_docstring_examples(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    assert expandpath('~/$SPAM') == os.path.expanduser('~/eggs')
    assert expandpath('foo') == 'foo'


def test_expandpath_env_var(monkeypatch):
    monkeypatch.setenv('MYVAR', 'myvalue')
    assert expandpath('$MYVAR/path') == 'myvalue/path'


def test_expandpath_missing_env_var(monkeypatch):
    monkeypatch.delenv('DEFINITELY_NOT_SET_VAR', raising=False)
    # expandvars leaves unset vars; on POSIX it empties them, on Windows it keeps
    result = expandpath('$DEFINITELY_NOT_SET_VAR')
    assert '$DEFINITELY_NOT_SET_VAR' not in result or result == '$DEFINITELY_NOT_SET_VAR'


def test_expandpath_tilde(monkeypatch):
    home = os.path.expanduser('~')
    assert expandpath('~') == home
    assert expandpath('~/sub') == os.path.join(home, 'sub')


def test_expandpath_pathlike():
    assert expandpath(Path('foo')) == 'foo'


def test_expandpath_plain():
    assert expandpath('no/expansion/here') == 'no/expansion/here'
