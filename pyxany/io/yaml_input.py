"""Load and validate the single YAML input file."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..config import Config, ConfigError


def load_config(path, overrides: dict | None = None) -> Config:
    """Parse a YAML input file into a validated :class:`~pyxany.config.Config`."""
    path = Path(path)
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise ConfigError(f"input file not found: {path}") from None
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: YAML syntax error -- {exc}") from None
    if raw is None:
        raise ConfigError(f"{path}: file is empty")

    cfg = Config.from_dict(raw, ctx=str(path), source_dir=path.parent)
    if overrides:
        cfg.apply_overrides(overrides)
    return cfg


def load_molecule(cfg: Config):
    """Read the structure referenced by ``cfg.molecule.structure``."""
    from .xyz import read_xyz

    return read_xyz(
        cfg.structure_path(),
        units=cfg.molecule.units,
        species=cfg.passivation.species,
        name=cfg.molecule.name,
        charge=cfg.molecule.charge,
        multiplicity=cfg.molecule.mult,
    )


def dump_config(cfg: Config) -> str:
    """Render the fully resolved configuration back to YAML."""
    return yaml.safe_dump(cfg.to_dict(), sort_keys=False, default_flow_style=False)
