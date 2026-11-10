"""Runs last: nothing under Code/ is newer than this test session's stamp (no bytecode, no writes). Read-only `find`.

Only the session stamp: Uzayr edits Code/ himself, so a fixed build-day stamp would flag his own work.

Also: no participant's name in any file that git would publish (participants appear by code, P1-P6, only), and
only the kinds of file the app is made of are published (no data file, model or archive).
"""

import os
import shutil
import subprocess

import pytest

import openday


def test_code_untouched():
    stamps = [s for s in [os.environ.get("OPENDAY_SESSION_STAMP", "")] if s and os.path.exists(s)]
    assert stamps, "no stamp file to compare against"
    for s in stamps:
        out = subprocess.run(["find", openday.CODE, "-newer", s], capture_output=True, text=True, timeout=120)
        assert out.stdout.strip() == "", out.stdout[:2000]


def test_no_pycache_in_openday():
    """No __pycache__ anywhere in OpenDay/. Laptop only: in a public copy (no thesis code beside it) a git-ignored
    __pycache__ harms nothing."""
    if not os.path.isdir(os.path.join(openday.CODE, "pipeline")):
        pytest.skip("no thesis code in ../Code (a public copy): a git-ignored __pycache__ is harmless there")
    found = [os.path.join(r, "__pycache__") for r, dirs, _ in os.walk(openday.OPENDAY) if "__pycache__" in dirs]
    assert not found, ("delete these folders and run with PYTHONDONTWRITEBYTECODE=1, as the README shows: "
                       + ", ".join(found))


def _git(*args):
    return subprocess.run(["git", "-C", openday.OPENDAY, *args], capture_output=True, text=True, timeout=60)


# The kinds of file this repo is made of. Anything else (a capture, an array, a pickle, an archive, a JSON state
# file) fails by its name, so a data file or the model cannot slip in with `git add -A`.
PUBLISHED_EXT = (".py", ".md", ".yaml", ".png", ".command", ".ini")
PUBLISHED_NAMES = (".gitignore", "LICENSE")


def test_no_participant_name_in_published_files():
    """Every file git would publish (tracked, or new and not ignored), by its path and its raw bytes (pictures too),
    against the laptop-only list of names (the six participants and the two recorded but never used): a plain
    substring in any case, so also inside a longer word or file name. Only the kinds of file above are published.
    The list itself, the trained model and its manifest must stay ignored. A failure names the file, never the name (unless the name is in the file's path)."""
    if not openday.have_people():
        pytest.skip("no configs/people.local.yaml on this machine")
    if not shutil.which("git") or _git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        pytest.skip("not a git work tree")
    people = openday._people()
    names = [str(x).casefold() for x in (*people["codes"].values(), *people.get("excluded", []))]
    files = [f for f in _git("ls-files", "--cached", "--others", "--exclude-standard", "-z").stdout.split("\0") if f]
    assert files
    hits, odd = [], []
    for f in files:
        base = os.path.basename(f)
        if base not in PUBLISHED_NAMES and os.path.splitext(base)[1] not in PUBLISHED_EXT:
            odd.append(f)
        if any(n in f.casefold() for n in names):
            hits.append(f"{f} (its path)")
        path = os.path.join(openday.OPENDAY, f)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as fh:
            raw = fh.read()
        data, text = raw.lower(), raw.decode("utf-8", "ignore").casefold()
        if any(n.encode("utf-8") in data or n in text for n in names):
            hits.append(f)
    assert not odd, f"not a kind of file this repo publishes (see PUBLISHED_EXT): {odd[:50]}"
    assert not hits, f"a participant's name is in these files: {hits[:50]}"
    for f in ("configs/people.local.yaml", "state/model/model.json", "state/model/manifest.json"):
        assert _git("check-ignore", "-q", f).returncode == 0, f"{f} is not git-ignored"
