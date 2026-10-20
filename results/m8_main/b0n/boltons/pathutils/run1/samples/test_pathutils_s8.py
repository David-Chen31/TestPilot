import os
import sys

import pytest

from boltons.pathutils import augpath, shrinkuser, expandpath


# ---------------------------------------------------------------------------
# augpath
# ---------------------------------------------------------------------------

def test_augpath_identity():
    assert augpath('foo.bar') == 'foo.bar'


def test_augpath_docstring_examples():
    assert augpath('foo.bar', ext='.BAZ') == 'foo.BAZ'
    assert augpath('foo.bar', suffix='_') == 'foo_.bar'
    assert augpath('foo.bar', prefix='_') == '_foo.bar'
    assert augpath('foo.bar', base='baz') == 'baz.bar'
    assert augpath('foo.tar.gz', ext='.zip', multidot=True) == 'foo.zip'
    assert augpath('foo.tar.gz', ext='.zip', multidot=False) == 'foo.tar.zip'
    assert augpath('foo.tar.gz', suffix='_new', multidot=True) == 'foo_new.tar.gz'


def test_augpath_combined():
    newpath = augpath('foo.bar', suffix='_suff', prefix='pref_',
                      ext='.baz', base='bar')
    assert newpath == 'pref_bar_suff.baz'


def test_augpath_no_extension():
    assert augpath('foo') == 'foo'
    assert augpath('foo', ext='.bar') == 'foo.bar'
    assert augpath('foo', suffix='_s') == 'foo_s'
    assert augpath('foo', prefix='p_') == 'p_foo'
    assert augpath('foo', base='bar') == 'bar'


def test_augpath_multidot_no_extension():
    # multidot branch with no dot in the filename
    assert augpath('foo', multidot=True) == 'foo'
    assert augpath('foo', ext='.bar', multidot=True) == 'foo.bar'


def test_augpath_multidot_split():
    # The first dot defines the extension
    assert augpath('a.b.c.d', ext='.x', multidot=True) == 'a.x'
    assert augpath('a.b.c.d', suffix='_s', multidot=True) == 'a_s.b.c.d'
    assert augpath('a.b.c.d', base='z', multidot=True) == 'z.b.c.d'


def test_augpath_last_dot_split():
    assert augpath('a.b.c.d', ext='.x') == 'a.b.c.x'
    assert augpath('a.b.c.d', suffix='_s') == 'a.b.c_s.d'
    assert augpath('a.b.c.d', base='z') == 'z.d'


def test_augpath_dpath_replaces_directory(tmp_path):
    result = augpath('/some/dir/foo.bar', dpath='/other/dir')
    assert result == os.path.join('/other/dir', 'foo.bar')


def test_augpath_preserves_existing_directory():
    result = augpath('/some/dir/foo.bar')
    assert result == os.path.join('/some/dir', 'foo.bar')


def test_augpath_trailing_sep_in_dpath():
    result = augpath('dir/foo.bar', dpath='dir/')
    assert result == os.path.join('dir/', 'foo.bar')


def test_augpath_pathlike(tmp_path):
    from pathlib import Path
    result = augpath(Path('foo.bar'), ext='.baz')
    assert result == 'foo.baz'


def test_augpath_empty_string():
    assert augpath('') == ''
    assert augpath('', ext='.x') == '.x'
    assert augpath('', prefix='p') == 'p'
    assert augpath('', suffix='s') == 's'


def test_augpath_empty_suffix_prefix_defaults():
    # explicit empty strings behave like defaults
    assert augpath('foo.bar', suffix='', prefix='') == 'foo.bar'


# ---------------------------------------------------------------------------
# shrinkuser
# ---------------------------------------------------------------------------

def test_shrinkuser_home_exact():
    userhome = os.path.expanduser('~')
    assert shrinkuser(userhome) == '~'


def test_shrinkuser_custom_home_symbol():
    userhome = os.path.expanduser('~')
    assert shrinkuser(userhome, '$HOME') == '$HOME'


def test_shrinkuser_preserves_identical_prefix_not_dir():
    userhome = os.path.expanduser('~')
    path = userhome + '1'
    assert shrinkuser(path) == path


def test_shrinkuser_child_path():
    userhome = os.path.expanduser('~')
    result = shrinkuser(userhome + os.path.sep + '1')
    assert result == os.path.join('~', '1')


def test_shrinkuser_child_path_custom_home():
    userhome = os.path.expanduser('~')
    result = shrinkuser(userhome + os.path.sep + '1', '$HOME')
    assert result == os.path.join('$HOME', '1')


def test_shrinkuser_nested_child_path():
    userhome = os.path.expanduser('~')
    result = shrinkuser(userhome + os.path.sep + os.path.join('a', 'b'))
    assert result == os.path.join('~', 'a', 'b')


def test_shrinkuser_unrelated_path():
    assert shrinkuser('/some/random/path') == '/some/random/path'


def test_shrinkuser_normalizes_path():
    userhome = os.path.expanduser('~')
    messy = userhome + os.path.sep + '..' + os.path.sep + os.path.basename(userhome)
    # normpath collapses the ".." so we end up exactly at home
    assert shrinkuser(messy) == '~'


def test_shrinkuser_does_not_expand_tilde():
    assert shrinkuser('~') == '~'


def test_shrinkuser_pathlike():
    from pathlib import Path
    userhome = os.path.expanduser('~')
    result = shrinkuser(Path(userhome + os.path.sep + 'x'))
    assert result == os.path.join('~', 'x')


# ---------------------------------------------------------------------------
# expandpath
# ---------------------------------------------------------------------------

def test_expandpath_env_var(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    assert expandpath('~/$SPAM') == os.path.expanduser('~/eggs')


def test_expandpath_plain_path(monkeypatch):
    monkeypatch.setenv('SPAM', 'eggs')
    assert expandpath('foo') == 'foo'


def test_expandpath_no_env_var(monkeypatch):
    monkeypatch.delenv('BOLTONS_NONEXISTENT_VAR', raising=False)
    assert expandpath('$BOLTONS_NONEXISTENT_VAR') == '$BOLTONS_NONEXISTENT_VAR'


def test_expandpath_tilde_only():
    assert expandpath('~') == os.path.expanduser('~')


def test_expandpath_tilde_and_subpath():
    assert expandpath('~/foo') == os.path.expanduser('~/foo')


def test_expandpath_pathlike(tmp_path):
    from pathlib import Path
    result = expandpath(Path('foo'))
    assert result == 'foo'


def test_expandpath_env_var_inside(monkeypatch):
    monkeypatch.setenv('BOLTONS_DIR', 'mydir')
    assert expandpath('/base/$BOLTONS_DIR/file') == '/base/mydir/file'


def test_expandpath_combined_tilde_and_env(monkeypatch):
    monkeypatch.setenv('BOLTONS_SUB', 'sub')
    result = expandpath('~/$BOLTONS_SUB')
    assert result == os.path.expanduser('~/sub')
