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


def DEFAULT_PRUNING(n_a: int) -> list:
    """Angular sizes of the inner regions for ``grid.pruning.regions: auto``.

    Region bounds are fractions of the Bragg-Slater radius; see ADR 0009 for the
    measurements behind them.
    """
    return [[0.25, max(4, n_a // 3)], [0.5, max(4, (2 * n_a) // 3)]]


class ConfigError(ValueError):
    """Raised for anything wrong with an input file."""


@dataclass
class MoleculeConfig:
    name: str = "molecule"
    structure: str = ""  # path to an .xyz file
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
            raise ConfigError(f"{ctx}: molecule.units must be angstrom or bohr, got {self.units!r}")
        if self.mult < 1:
            raise ConfigError(f"{ctx}: molecule.mult must be >= 1, got {self.mult}")


@dataclass
class PassivationConfig:
    scheme: str = "none"  # none | pseudo_hydrogen | hydrogen
    species: dict = field(default_factory=dict)
    basis: str = "derived"

    def validate(self, ctx: str) -> None:
        if self.scheme not in ("none", "pseudo_hydrogen", "hydrogen"):
            raise ConfigError(
                f"{ctx}: passivation.scheme must be none, "
                f"pseudo_hydrogen or hydrogen; got {self.scheme!r}"
            )
        for label, spec in self.species.items():
            if "alpha" not in spec and "Z" not in spec:
                raise ConfigError(
                    f"{ctx}: passivation.species['{label}'] needs an "
                    f"'alpha' (fractional nuclear charge)"
                )


@dataclass
class BasisConfig:
    set: str = "dz"

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
    #: Angular rule of the molecular grid: "product" (n_a x 2 n_a Gauss-Legendre
    #: x uniform) or "lebedev" (the Lebedev rule of the same degree, ~2/3 the points).
    angular: str = "product"
    #: 'auto' resolves to n_a - 1, the highest multipole the angular grid can
    #: resolve; anything lower leaves a force error set by the truncation.
    poisson_l_max: Any = "auto"
    screening_tol: float = 1e-10
    #: Radial points per element, overriding the ratio table (e.g. {Si: 40}).
    n_r_by_element: dict = field(default_factory=dict)
    #: Two-centre grids for the S and T matrices and their gradient.
    pair: dict = field(default_factory=lambda: {"n_r": 40, "n_a": 14})
    #: Logarithmic radial grid of the spherical atom solver (NAO basis, SAD
    #: initial guess, pseudo-hydrogen exponents).
    atom: dict = field(default_factory=lambda: {"n_points": 2000})
    #: One-atom DV grid on which the SAD guess is projected onto the basis.
    guess: dict = field(default_factory=lambda: {"n_r": 80, "n_a": 14})
    #: Radial quadrature for one-centre integrals of non-Slater functions.
    one_centre: dict = field(default_factory=lambda: {"n_r": 400})
    #: Fewer angular points on the inner shells of the molecular grid (ADR 0009).
    pruning: dict = field(default_factory=lambda: {"enabled": True, "regions": "auto"})
    #: Basis functions below this magnitude on a block of grid points are
    #: dropped from that block's matrix products; 0 disables (ADR 0009).
    #: 1e-6 changes Si47H60 by 1.3e-6 Ha and 2e-7 Ha/Bohr (Q35).
    basis_cutoff: float = 1e-6
    chunk_size: Any = "auto"
    memory_budget_mb: float = 4096.0
    #: Scale n_r per element (n_r is then the second-row size); see
    #: data/tables/radial_points.json and tools/radial_grid_sizes.py.
    radial_by_element: bool = True

    @property
    def l_max(self) -> int:
        """The multipole cut-off actually used ('auto' -> n_a - 1)."""
        return self.n_a - 1 if self.poisson_l_max == "auto" else int(self.poisson_l_max)

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
        # The radial solver is stable at any l (ratio-damped Green's function),
        # so the limit is the angular grid: an n_a x 2n_a Gauss-Legendre product
        # projects onto Y_lm orthonormally only while 2 l <= 2 n_a - 1.
        # --set coerces against the current value, which may be the string "auto"
        if isinstance(self.poisson_l_max, str) and self.poisson_l_max.strip().isdigit():
            self.poisson_l_max = int(self.poisson_l_max)
        if self.poisson_l_max != "auto" and (
                isinstance(self.poisson_l_max, bool) or not isinstance(self.poisson_l_max, int)):
            raise ConfigError(f"{ctx}: grid.poisson_l_max must be 'auto' or an integer; "
                              f"got {self.poisson_l_max!r}")
        if not (0 <= self.l_max <= self.n_a - 1):
            raise ConfigError(
                f"{ctx}: grid.poisson_l_max must be between 0 and n_a - 1 = {self.n_a - 1} "
                f"(the angular grid cannot resolve higher multipoles); "
                f"got {self.poisson_l_max}"
            )
        if isinstance(self.chunk_size, str) and self.chunk_size != "auto":
            raise ConfigError(f"{ctx}: grid.chunk_size must be an integer or 'auto'")
        if self.angular not in ("product", "lebedev"):
            raise ConfigError(f"{ctx}: grid.angular must be 'product' or 'lebedev'; "
                              f"got {self.angular!r}")
        if self.angular == "lebedev" and self.n_a > 66:
            raise ConfigError(f"{ctx}: grid.angular: lebedev supports n_a <= 66 "
                              f"(Lebedev order 131); got n_a = {self.n_a}")
        self._validate_sub_grids(ctx)

    #: defaults and lower bounds of the nested grid blocks
    _SUBGRIDS = {
        "pair": ({"n_r": 40, "n_a": 14}, {"n_r": 5, "n_a": 4}),
        "atom": ({"n_points": 2000}, {"n_points": 500}),
        "guess": ({"n_r": 80, "n_a": 14}, {"n_r": 5, "n_a": 4}),
        "one_centre": ({"n_r": 400}, {"n_r": 50}),
    }

    def _validate_sub_grids(self, ctx: str) -> None:
        from .data import elements as elem

        for name, (defaults, minima) in self._SUBGRIDS.items():
            block = getattr(self, name)
            if not isinstance(block, dict):
                raise ConfigError(f"{ctx}: grid.{name} must be a mapping")
            unknown = set(block) - set(defaults)
            if unknown:
                raise ConfigError(f"{ctx}: grid.{name} has unknown key(s) {sorted(unknown)}. "
                                  f"Valid keys: {sorted(defaults)}")
            merged = {**defaults, **block}
            for key, low in minima.items():
                value = merged[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < low:
                    raise ConfigError(f"{ctx}: grid.{name}.{key} must be an integer >= {low}; "
                                      f"got {value!r}")
            setattr(self, name, merged)

        if not isinstance(self.n_r_by_element, dict):
            raise ConfigError(f"{ctx}: grid.n_r_by_element must be a mapping like {{Si: 40}}")
        for symbol, value in self.n_r_by_element.items():
            try:
                elem.atomic_number(str(symbol))
            except (KeyError, ValueError):
                raise ConfigError(f"{ctx}: grid.n_r_by_element: unknown element {symbol!r}")
            if isinstance(value, bool) or not isinstance(value, int) or value < 5:
                raise ConfigError(f"{ctx}: grid.n_r_by_element.{symbol} must be an integer "
                                  f">= 5; got {value!r}")

        if not isinstance(self.pruning, dict):
            raise ConfigError(f"{ctx}: grid.pruning must be a mapping")
        unknown = set(self.pruning) - {"enabled", "regions"}
        if unknown:
            raise ConfigError(f"{ctx}: grid.pruning has unknown key(s) {sorted(unknown)}. "
                              f"Valid keys: ['enabled', 'regions']")
        self.pruning = {"enabled": True, "regions": "auto", **self.pruning}
        if not isinstance(self.pruning["enabled"], bool):
            raise ConfigError(f"{ctx}: grid.pruning.enabled must be true or false")
        regions = self.pruning["regions"]
        if regions != "auto":
            ok = isinstance(regions, list) and all(
                isinstance(r, (list, tuple)) and len(r) == 2 for r in regions)
            if ok:
                bounds = [float(r[0]) for r in regions]
                sizes = [r[1] for r in regions]
                ok = (all(b > 0 for b in bounds) and bounds == sorted(bounds)
                      and len(set(bounds)) == len(bounds)
                      and all(isinstance(n, int) and 4 <= n <= self.n_a for n in sizes))
            if not ok:
                raise ConfigError(
                    f"{ctx}: grid.pruning.regions must be 'auto' or a list of "
                    f"[r/R_Bragg upper bound, n_a] pairs with increasing bounds and "
                    f"4 <= n_a <= grid.n_a = {self.n_a}; got {regions!r}")

        if not (isinstance(self.basis_cutoff, (int, float)) and 0 <= self.basis_cutoff < 1e-3):
            raise ConfigError(f"{ctx}: grid.basis_cutoff must be in [0, 1e-3) (0 disables); "
                              f"got {self.basis_cutoff!r}")

    def pruning_regions(self) -> list:
        """``[(r/R upper bound, n_a), ...]`` actually used; [] when pruning is off."""
        if not self.pruning.get("enabled", True):
            return []
        regions = self.pruning.get("regions", "auto")
        if regions == "auto":
            regions = DEFAULT_PRUNING(self.n_a)
        return [(float(b), int(n)) for b, n in regions if int(n) < self.n_a]

    def n_r_for(self, symbol: str, scaled: int) -> int:
        """Radial points for one element: explicit override, else the table."""
        return int(self.n_r_by_element.get(symbol, scaled))


@dataclass
class SCFConfig:
    max_iter: int = 100
    e_tol: float = 1e-6
    d_tol: float = 1e-5
    #: X-alpha parameter: a number for every element (default 0.7, ADR 0002),
    #: "alpha_hf" for the recomputed Schwarz alpha_HF table, or a mapping
    #: ``{default: 0.7, Si: 0.73}``; between atoms the Becke-weighted mean
    #: (ADR 0011). Pseudo-hydrogens use hydrogen's value.
    alpha: Any = 0.7
    unrestricted: bool = False
    smearing: float = 0.0
    s_threshold: float = 5e-4
    ediis_iters: int = 8
    mix_ediis: float = 0.25
    mix_diis: float = 0.4
    #: Write the SCF state every N iterations (0: only at the end). The file is
    #: <output.dir>/<prefix>_scf.npz; resume with --restart. See ADR 0004.
    checkpoint_every: int = 0

    def validate(self, ctx: str) -> None:
        a = self.alpha
        if isinstance(a, str) and a.strip().replace(".", "", 1).isdigit():
            self.alpha = a = float(a)                  # --set scf.alpha=0.75
        if isinstance(a, str):
            if a not in ("alpha_hf", "alpha_hf_polarised"):
                raise ConfigError(f"{ctx}: scf.alpha must be a number, 'alpha_hf', "
                                  f"'alpha_hf_polarised' or a "
                                  f"mapping like {{default: 0.7, Si: 0.73}}; got {a!r}")
        elif isinstance(a, dict):
            from .data import elements as elem

            for key, value in a.items():
                if key != "default":
                    try:
                        elem.atomic_number(str(key))
                    except (KeyError, ValueError):
                        raise ConfigError(f"{ctx}: scf.alpha: unknown element {key!r}") from None
                if isinstance(value, bool) or not isinstance(value, (int, float)) \
                        or not 0.0 < value <= 2.0:
                    raise ConfigError(f"{ctx}: scf.alpha.{key} must be in (0, 2]; got {value!r}")
        elif isinstance(a, bool) or not isinstance(a, (int, float)) or not 0.0 < a <= 2.0:
            raise ConfigError(f"{ctx}: scf.alpha must be in (0, 2]; got {a!r}")
        if self.max_iter < 1:
            raise ConfigError(f"{ctx}: scf.max_iter must be >= 1")
        for name in ("mix_ediis", "mix_diis"):
            v = getattr(self, name)
            if not (0.0 < v <= 1.0):
                raise ConfigError(f"{ctx}: scf.{name} must be in (0, 1]; got {v}")
        if self.smearing < 0:
            raise ConfigError(f"{ctx}: scf.smearing must be >= 0")
        if self.checkpoint_every < 0:
            raise ConfigError(f"{ctx}: scf.checkpoint_every must be >= 0 (0 = only at the end)")

    def alpha_for(self, element: str) -> float:
        """The X-alpha parameter of one element (pseudo-hydrogens: pass 'H')."""
        a = self.alpha
        if isinstance(a, (int, float)):
            return float(a)
        if a in ("alpha_hf", "alpha_hf_polarised"):
            from .data import loaders

            try:
                table = (loaders.alpha_hf_table() if a == "alpha_hf"
                         else loaders.alpha_hf_polarised_table())
            except FileNotFoundError:
                raise ConfigError(f"scf.alpha: {a}: the table has not been built yet "
                                  f"(tools/alpha_hf.py, ADR 0011)") from None
            if element not in table:
                raise ConfigError(f"scf.alpha: {a}: no value for {element}")
            return float(table[element])
        return float(a.get(element, a.get("default", 0.7)))

    @property
    def uniform_alpha(self) -> float | None:
        """The single alpha of a uniform setting, or None when it varies by element."""
        a = self.alpha
        if isinstance(a, (int, float)):
            return float(a)
        if isinstance(a, dict) and set(a) <= {"default"}:
            return float(a.get("default", 0.7))
        return None

    @property
    def alpha_label(self) -> str:
        a = self.alpha
        if isinstance(a, (int, float)):
            return f"{a:g}"
        if a == "alpha_hf":
            return "alpha_HF (Schwarz, by element)"
        if a == "alpha_hf_polarised":
            return "alpha_HF spin-polarised (by element)"
        return "by element " + ", ".join(f"{k} {v:g}" for k, v in a.items())


@dataclass
class ForcesConfig:
    enabled: bool = True
    method: str = "analytic"
    fd_step: float = 0.02
    #: Enforce sum(F) = 0 by removing the net translation from analytic forces.
    project_translation: bool = True

    def validate(self, ctx: str) -> None:
        if self.method not in ("analytic", "finite_difference"):
            raise ConfigError(
                f"{ctx}: forces.method must be 'analytic' or "
                f"'finite_difference'; got {self.method!r}"
            )


@dataclass
class OptimiseConfig:
    enabled: bool = False
    method: str = "lbfgs"
    f_max: float = 1e-3
    max_steps: int = 100
    freeze: Any = "none"  # none | core | [indices]
    trajectory: bool = True

    def validate(self, ctx: str) -> None:
        if self.method == "steepest_descent":
            raise ConfigError(
                f"{ctx}: optimise.method 'steepest_descent' ran scipy's conjugate "
                f"gradient, never steepest descent. Use 'cg' for that solver, or "
                f"'lbfgs' (the default)."
            )
        if self.method not in ("lbfgs", "cg"):
            raise ConfigError(f"{ctx}: optimise.method must be 'lbfgs' or 'cg'")
        if isinstance(self.freeze, str) and self.freeze not in ("none", "core"):
            raise ConfigError(
                f"{ctx}: optimise.freeze must be 'none', 'core' or a list of atom indices"
            )


@dataclass
class PropertiesConfig:
    population: bool = True
    dipole: bool = True
    dos: dict = field(default_factory=lambda: {"sigma": 0.3, "n_points": 600})
    bde: dict = field(default_factory=lambda: {"enabled": False})
    esp: dict = field(default_factory=lambda: {"enabled": False, "n_pts": 20, "margin": 3.0})
    #: Bond overlap population (ADR 0005): pairs to analyse (default: Mayer bond
    #: order >= min_bond_order) and whether to draw the atom-pair heatmap.
    overlap_population: dict = field(default_factory=lambda: {
        "pairs": [], "min_bond_order": 0.2, "heatmap": False, "sigma": 0.3})
    #: An orbital whose Loewdin weight on pseudo-hydrogen terminators exceeds this
    #: counts as a terminator state: it triggers the passivation warning and is
    #: skipped by the core gap. 0.5 ("mostly on the terminators") is a heuristic.
    terminator_weight_max: float = 0.5

    #: allowed keys of each sub-dict, so a typo is an error here too
    _SUBKEYS = {
        "dos": {"sigma", "n_points"},
        "bde": {"enabled"},
        "esp": {"enabled", "n_pts", "margin"},
        "overlap_population": {"pairs", "min_bond_order", "heatmap", "sigma"},
    }

    def validate(self, ctx: str) -> None:
        # These sub-dicts used to skip validation entirely, so
        # `properties: {dos: {sigmaa: 0.3}}` was accepted in silence while
        # `scf: {alfa: 0.7}` was correctly rejected -- the opposite of what this
        # module's docstring promises.
        for name, allowed in self._SUBKEYS.items():
            values = getattr(self, name)
            if not isinstance(values, dict):
                raise ConfigError(f"{ctx}: properties.{name} must be a mapping")
            unknown = set(values) - allowed
            if unknown:
                raise ConfigError(
                    f"{ctx}: properties.{name} has unknown key(s) "
                    f"{sorted(unknown)}. Valid keys: {sorted(allowed)}"
                )
        if not (0.0 < self.terminator_weight_max <= 1.0):
            raise ConfigError(f"{ctx}: properties.terminator_weight_max must be in (0, 1]")
        sigma = self.dos.get("sigma", 0.3)
        if not (isinstance(sigma, (int, float)) and sigma > 0):
            raise ConfigError(f"{ctx}: properties.dos.sigma must be > 0; got {sigma}")


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


@dataclass
class EmbeddingConfig:
    """Point-charge embedding (ADR 0010)."""

    #: File of point charges (``x y z q`` or ``label x y z q``), relative to the
    #: input file; None: no embedding. ``pyxany embed`` writes one.
    charges: Any = None
    units: str = "angstrom"
    #: Gaussian width of every charge, Bohr (always Bohr, whatever ``units``).
    sigma: float = 0.5
    #: Electrons within this distance (Angstrom) of a positive charge are
    #: reported; above ``spill_warn`` electrons a warning is printed.
    spill_radius: float = 1.0
    spill_warn: float = 0.1

    def validate(self, ctx: str) -> None:
        if self.charges is not None and not isinstance(self.charges, str):
            raise ConfigError(f"{ctx}: embedding.charges must be a file name")
        if self.units not in ("angstrom", "bohr"):
            raise ConfigError(f"{ctx}: embedding.units must be 'angstrom' or 'bohr'")
        if not (isinstance(self.sigma, (int, float)) and self.sigma > 0):
            raise ConfigError(f"{ctx}: embedding.sigma must be > 0 (Bohr)")
        if not (self.spill_radius > 0 and self.spill_warn >= 0):
            raise ConfigError(f"{ctx}: embedding.spill_radius must be > 0 and "
                              f"spill_warn >= 0")


@dataclass
class TransitionStateConfig:
    """Slater transition state (ADR 0006). Set exactly one of ``remove``/``excite``.

    ``remove: homo`` | ``remove: 5`` (1-based MO) | ``remove: {atom: 3, shell: 1s}``
    ``excite: {from: homo, to: lumo}``
    """

    remove: Any = None
    excite: Any = None
    spin: str = "alpha"
    electrons: float = 0.5

    @property
    def enabled(self) -> bool:
        return self.remove is not None or self.excite is not None

    def validate(self, ctx: str) -> None:
        if self.remove is not None and self.excite is not None:
            raise ConfigError(f"{ctx}: transition_state takes either 'remove' or 'excite', not both")
        if self.excite is not None and (not isinstance(self.excite, dict)
                                        or set(self.excite) != {"from", "to"}):
            raise ConfigError(f"{ctx}: transition_state.excite needs exactly 'from' and 'to'")
        if self.spin not in ("alpha", "beta"):
            raise ConfigError(f"{ctx}: transition_state.spin must be 'alpha' or 'beta'")
        if not (0.0 < float(self.electrons) <= 1.0):
            raise ConfigError(f"{ctx}: transition_state.electrons must be in (0, 1]")
        if self.enabled:
            from .scf.transition_state import parse_target

            try:
                for spec in ([self.remove] if self.remove is not None
                             else [self.excite["from"], self.excite["to"]]):
                    parse_target(spec)
            except ValueError as exc:
                raise ConfigError(f"{ctx}: {exc}") from None


#: keys that were renamed; using the old name is an error that says where it went
_MOVED = {
    ("grid", "pair_n_r"): "grid: {pair: {n_r: ...}}",
    ("grid", "pair_n_a"): "grid: {pair: {n_a: ...}}",
    ("basis", "nao_n_r"): "grid: {atom: {n_points: ...}}",
}

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
    "transition_state": TransitionStateConfig,
    "embedding": EmbeddingConfig,
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
    transition_state: TransitionStateConfig = field(default_factory=TransitionStateConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
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
            for key in section:
                if (name, key) in _MOVED:
                    raise ConfigError(f"{ctx}: '{name}.{key}' has moved: write "
                                      f"{_MOVED[(name, key)]} instead")
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
        if self.transition_state.enabled and self.scf.smearing > 0:
            raise ConfigError(f"{ctx}: a transition state fixes occupations by hand; "
                              f"scf.smearing must be 0")
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
                raise ConfigError(
                    f"--set {dotted}: expected SECTION.KEY with SECTION "
                    f"one of {', '.join(sorted(_SECTIONS))}"
                )
            obj = getattr(self, section)
            key, _, sub = key.partition(".")
            if sub and isinstance(getattr(obj, key, None), dict):
                block = dict(getattr(obj, key))
                block[sub] = _coerce(block.get(sub, value), value, dotted)
                setattr(obj, key, block)
                continue
            if not hasattr(obj, key):
                valid = ", ".join(sorted(f.name for f in fields(obj)))
                raise ConfigError(
                    f"--set {dotted}: '{section}' has no key '{key}'. Valid keys: {valid}"
                )
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
