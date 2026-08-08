import re
from pathlib import Path

import pytest

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def vcs_requirements() -> list[str]:
    lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip().startswith(("git+", "hg+", "svn+", "bzr+"))
    ]


def test_requirements_file_exists():
    assert REQUIREMENTS.is_file()


def test_there_is_at_least_one_vcs_requirement():
    """Sanity check: if this fails the other tests are vacuously passing."""
    assert vcs_requirements()


@pytest.mark.parametrize("requirement", vcs_requirements())
def test_vcs_requirement_is_pinned_to_a_full_sha(requirement):
    assert "@" in requirement, f"no revision pinned: {requirement}"
    revision = requirement.rsplit("@", 1)[1].split("#")[0].strip()
    assert SHA_RE.match(revision), (
        f"revision {revision!r} is not a full 40-char commit SHA. "
        "Branch and tag refs are mutable and make builds unreproducible."
    )
