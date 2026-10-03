"""
CI reproducibility guard: the lint policy must not depend on whatever ruff
version or default rule set happens to be current.

History: CI ran `pip install ruff` unpinned, and a newer ruff enabled many more
rules by default (hundreds of findings with no code change), turning CI red.
These checks keep ruff pinned and the rules explicit.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_ci_pins_the_ruff_version():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    installs = re.findall(r"pip install\s+(.*ruff.*)", ci)

    assert installs, "CI no longer installs ruff"
    for line in installs:
        assert re.search(r'ruff==\d+\.\d+\.\d+', line), f"ruff must be pinned in CI, found: {line!r}"


def test_ruff_rules_are_selected_explicitly():
    config = (ROOT / "ruff.toml").read_text(encoding="utf-8")

    assert re.search(r"^\[lint\]", config, re.M)
    selected = re.search(r"^select\s*=\s*\[(.*?)\]", config, re.M | re.S)
    assert selected, "ruff.toml must select rules explicitly instead of relying on defaults"
    rules = set(re.findall(r'"([A-Z]+\d*)"', selected.group(1)))
    assert rules and "F" in rules, f"unexpected rule selection: {rules}"


def test_ci_lints_the_same_paths_the_project_lints_locally():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "ruff check src api tests" in ci
