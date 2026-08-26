"""Calculation settings: one YAML file, validated into typed dataclasses.

Unknown keys are an error rather than being ignored, so a typo in an input file
is reported instead of silently leaving a default in place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

_ANGSTROM = ("angstrom", "ang", "a")
_BOHR = ("bohr", "au", "a.u.")


class ConfigError(ValueError):
    """Raised for anything wrong with an input file."""


@dataclass
class MoleculeConfig:
    name: str = "molecule"
    structure: str = ""          # path to an .xyz file
    charge: int = 0
    mult: int = 1
    units: str = "angstrom"

    def validate(self, ctx: str) -> None:
        if not self.structure:
            raise ConfigError(f"{ctx}: molecule.structure (an .xyz path) is required")
        if self.units.lower() in _ANGSTROM:
            self.units = "angstrom"
        elif self.units.lower() in _BOHR:
            self.units = "bohr"
        else:
            raise ConfigError(f"{ctx}: molecule.units must be angstrom or bohr, "
                              f"got {self.units!r}")
        if self.mult < 1:
            raise ConfigError(f"{ctx}: molecule.mult must be >= 1, got {self.mult}")


@dataclass
class PassivationConfig:
    scheme: str = "none"         # none | pseudo_hydrogen | hydrogen
    species: dict = field(default_factory=dict)
    basis: str = "derived"

    def validate(self, ctx: str) -> None:
        if self.scheme not in ("none", "pseudo_hydrogen", "hydrogen"):
            raise ConfigError(f"{ctx}: passivation.scheme must be none, "
                              f"pseudo_hydrogen or hydrogen; got {self.scheme!r}")
        for label, spec in self.species.items():
            if "alpha" not in spec and "Z" not in spec:
                raise ConfigError(f"{ctx}: passivation.species['{label}'] needs an "
                                  f"'alpha' (fractional nuclear charge)")


@dataclass
class BasisConfig:
    set: str = "dz"
    nao_n_r: int = 2000

    def validate(self, ctx: str) -> None:
        from .basis.gto import GTO_KEYS
        from .basis.sto import STO_KEYS

        key = self.set.lower().strip()
        if key not in set(STO_KEYS) | set(GTO_KEYS) | {"nao"}:
            raise ConfigError(
                f"{ctx}: unknown basis.set {self.set!r}. "
                f"STO: {', '.join(STO_KEYS)}; GTO: {', '.join(sorted(GTO_KEYS))}; NAO: nao"
            )
        self.set = key


@dataclass
class GridConfig:
    n_r: int = 30
    n_a: int = 12
    partition: str = "becke"
    poisson_l_max: int = 6
    screening_tol: float = 1e-10
    pair_n_r: int = 40
    pair_n_a: int = 14
    chunk_size: Any = "auto"
    memory_budget_mb: float = 4096.0

    def validate(self, ctx: str) -> None:
        if self.n_r < 5 or self.n_a < 4:
            raise ConfigError(f"{ctx}: grid.n_r >= 5 and grid.n_a >= 4 required")
        if self.partition not in ("becke", "none"):
            raise ConfigError(f"{ctx}: grid.partition must be 'becke' or 'none'")
        if self.partition == "none":
            raise ConfigError(
                f"{ctx}: grid.partition='none' omits the Becke cell functions, which "
                f"overcounts overlapping regions by roughly the number of atoms. "
                f"It exists only for the demonstration in tests/test_grid.py."
            )
        if not (0 <= self.poisson_l_max <= 12):
            raise ConfigError(f"{ctx}: grid.poisson_l_max must be between 0 and 12")
        if isinstance(self.chunk_size, str) and self.chunk_size != "auto":
            raise ConfigError(f"{ctx}: grid.chunk_size must be an integer or 'auto'")


@dataclass
class SCFConfig:
    max_iter: int = 100
    e_tol: float = 1e-6
    d_tol: float = 1e-5
    alpha: float = 0.7
    unrestricted: bool = False
    smearing: float = 0.0
    s_threshold: float = 5e-4
    ediis_iters: int = 8
    mix_ediis: float = 0.25
    mix_diis: float = 0.4
    checkpoint_every: int = 0

    def validate(self, ctx: str) -> None:
        if not (0.0 < self.alpha <= 2.0):
            raise ConfigError(f"{ctx}: scf.alpha must be in (0, 2]; got {self.alpha}")
        if self.max_iter < 1:
            raise ConfigError(f"{ctx}: scf.max_iter must be >= 1")
        for name in ("mix_ediis", "mix_diis"):
            v = getattr(self, name)
            if not (0.0 < v <= 1.0):
                raise ConfigError(f"{ctx}: scf.{name} must be in (0, 1]; got {v}")
        if self.smearing < 0:
            raise ConfigError(f"{ctx}: scf.smearing must be >= 0")


@dataclass
class ForcesConfig:
    enabled: bool = True
    method: str = "analytic"
    fd_step: float = 0.02
    #: Enforce sum(F) = 0 by removing the net translation from analytic forces.
    project_translation: bool = True

    def validate(self, ctx: str) -> None:
        if self.method not in ("analytic", "finite_difference"):
            raise ConfigError(f"{ctx}: forces.method must be 'analytic' or "
                              f"'finite_difference'; got {self.method!r}")


@dataclass
class OptimiseConfig:
    enabled: bool = False
    method: str = "lbfgs"
    f_max: float = 1e-3
    max_steps: int = 100
    freeze: Any = "none"           # none | core | [indices]
    trajectory: bool = True

    def validate(self, ctx: str) -> None:
        if self.method not in ("lbfgs", "steepest_descent"):
            raise ConfigError(f"{ctx}: optimise.method must be 'lbfgs' or "
                              f"'steepest_descent'")
        if isinstance(self.freeze, str) and self.freeze not in ("none", "core"):
            raise ConfigError(f"{ctx}: optimise.freeze must be 'none', 'core' or a "
                              f"list of atom indices")


@dataclass
class PropertiesConfig:
    population: bool = True
    dipole: bool = True
    dos: dict = field(default_factory=lambda: {"sigma": 0.3, "n_points": 600})
    bde: dict = field(default_factory=lambda: {"enabled": False})
    esp: dict = field(default_factory=lambda: {"enabled": False, "n_pts": 20, "margin": 3.0})

    def validate(self, ctx: str) -> None:
        return


@dataclass
class OutputConfig:
    dir: str = "."
    prefix: str = ""
    figure: bool = True
    json: bool = True
    npz: bool = False
    timing: bool = True

    def validate(self, ctx: str) -> None:
        return


@dataclass
class RuntimeConfig:
    numba: str = "auto"
    mpi: str = "auto"
    verbosity: int = 1

    def validate(self, ctx: str) -> None:
        for name in ("numba", "mpi"):
            v = getattr(self, name)
            if v not in ("auto", "on", "off"):
                raise ConfigError(f"{ctx}: runtime.{name} must be auto, on or off")


_SECTIONS = {
    "molecule": MoleculeConfig,
    "passivation": PassivationConfig,
    "basis": BasisConfig,
    "grid": GridConfig,
    "scf": SCFConfig,
    "forces": ForcesConfig,
    "optimise": OptimiseConfig,
    "properties": PropertiesConfig,
    "output": OutputConfig,
    "runtime": RuntimeConfig,
}


@dataclass
class Config:
    molecule: MoleculeConfig = field(default_factory=MoleculeConfig)
    passivation: PassivationConfig = field(default_factory=PassivationConfig)
    basis: BasisConfig = field(default_factory=BasisConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    scf: SCFConfig = field(default_factory=SCFConfig)
    forces: ForcesConfig = field(default_factory=ForcesConfig)
    optimise: OptimiseConfig = field(default_factory=OptimiseConfig)
    properties: PropertiesConfig = field(default_factory=PropertiesConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    #: Directory the YAML file lives in, used to resolve relative paths.
    source_dir: Path = field(default_factory=Path)

    @classmethod
    def from_dict(cls, raw: dict, ctx: str = "<config>", source_dir: Path | None = None):
        if not isinstance(raw, dict):
            raise ConfigError(f"{ctx}: the top level of the input file must be a mapping")
        unknown = set(raw) - set(_SECTIONS)
        if unknown:
            raise ConfigError(
                f"{ctx}: unknown top-level section(s) {sorted(unknown)}. "
                f"Valid sections: {', '.join(sorted(_SECTIONS))}"
            )
        kwargs = {}
        for name, klass in _SECTIONS.items():
            section = raw.get(name) or {}
            if not isinstance(section, dict):
                raise ConfigError(f"{ctx}: section '{name}' must be a mapping")
            valid = {f.name for f in fields(klass)}
            bad = set(section) - valid
            if bad:
                raise ConfigError(
                    f"{ctx}: unknown key(s) {sorted(bad)} in section '{name}'. "
                    f"Valid keys: {', '.join(sorted(valid))}"
                )
            kwargs[name] = klass(**section)
        cfg = cls(**kwargs, source_dir=Path(source_dir or "."))
        cfg.validate(ctx)
        return cfg

    def validate(self, ctx: str = "<config>") -> None:
        for name in _SECTIONS:
            getattr(self, name).validate(ctx)
        if not self.output.prefix:
            self.output.prefix = self.molecule.name

    def structure_path(self) -> Path:
        p = Path(self.molecule.structure)
        return p if p.is_absolute() else (self.source_dir / p)

    def to_dict(self) -> dict:
        out = {name: asdict(getattr(self, name)) for name in _SECTIONS}
        return out

    def apply_overrides(self, overrides: dict) -> None:
        """Apply ``{"scf.alpha": 0.75}`` style overrides in place."""
        for dotted, value in overrides.items():
            section, _, key = dotted.partition(".")
            if section not in _SECTIONS or not key:
                raise ConfigError(f"--set {dotted}: expected SECTION.KEY with SECTION "
                                  f"one of {', '.join(sorted(_SECTIONS))}")
            obj = getattr(self, section)
            if not hasattr(obj, key):
                valid = ", ".join(sorted(f.name for f in fields(obj)))
                raise ConfigError(f"--set {dotted}: '{section}' has no key '{key}'. "
                                  f"Valid keys: {valid}")
            setattr(obj, key, _coerce(getattr(obj, key), value, dotted))
        self.validate("<overrides>")


def _coerce(current, value, dotted: str):
    """Convert a string override to the type of the existing value."""
    if not isinstance(value, str):
        return value
    if isinstance(current, bool):
        low = value.strip().lower()
        if low in ("true", "yes", "1", "on"):
            return True
        if low in ("false", "no", "0", "off"):
            return False
        raise ConfigError(f"--set {dotted}: expected a boolean, got {value!r}")
    for caster in (int, float) if not isinstance(current, str) else ():
        try:
            return caster(value)
        except ValueError:
            continue
    return value
