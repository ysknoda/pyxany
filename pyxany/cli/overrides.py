"""Parse ``--set SECTION.KEY=VALUE`` command-line overrides."""

from __future__ import annotations

from ..config import ConfigError


def parse_overrides(items: list[str] | None) -> dict:
    if not items:
        return {}
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"--set {item!r}: expected SECTION.KEY=VALUE")
        key, _, value = item.partition("=")
        key = key.strip()
        if "." not in key:
            raise ConfigError(f"--set {item!r}: key must be SECTION.KEY, e.g. scf.alpha")
        out[key] = value.strip()
    return out
