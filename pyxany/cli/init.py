"""``pyxany init`` -- write a template YAML + XYZ pair."""

from __future__ import annotations

from pathlib import Path

from . import EXIT_OK

TEMPLATES = {
    "molecule": {
        "xyz": ("3\nwater\nO   0.000000   0.000000   0.117300\n"
                "H   0.000000   0.757200  -0.469200\nH   0.000000  -0.757200  -0.469200\n"),
        "extra": "",
    },
    "radical": {
        "xyz": ("4\nmethyl radical (doublet)\nC   0.000000   0.000000   0.000000\n"
                "H   1.079000   0.000000   0.000000\nH  -0.539500   0.934500   0.000000\n"
                "H  -0.539500  -0.934500   0.000000\n"),
        "extra": "",
    },
    "cluster": {
        "xyz": ("5\nSi5 fragment -- run 'pyxany passivate' on this first\n"
                "Si  0.000000  0.000000  0.000000\nSi  1.357750  1.357750  1.357750\n"
                "Si  1.357750 -1.357750 -1.357750\nSi -1.357750  1.357750 -1.357750\n"
                "Si -1.357750 -1.357750  1.357750\n"),
        "extra": (
            "# This fragment has dangling bonds. Terminate it first:\n"
            "#   pyxany passivate {name}.xyz --coordination 4 \\\n"
            "#        -o {name}_h.xyz --emit-yaml {name}_h.yaml\n"
            "# then relax only the terminators:\n"
            "#   pyxany opt {name}_h.yaml --freeze-core\n"
        ),
    },
}

_YAML = """\
molecule:
  name:      {name}
  structure: {name}.xyz
  charge:    0
  mult:      {mult}
  units:     angstrom

basis:
  set: dz                # sz szp dz dzp tz tzp tzpp | sto3g 631g 631gd | nao

grid:
  n_r:           30      # radial points per atom
  n_a:           12      # polar points per atom (2*n_a azimuthal)
  poisson_l_max: auto    # multipole cut-off of the Hartree expansion (n_a - 1)
  screening_tol: 1.0e-10

scf:
  max_iter: 150
  e_tol:    1.0e-6       # total-energy convergence (Ha)
  d_tol:    1.0e-5       # density-matrix rms convergence
  alpha:    0.7          # Slater X-alpha exchange parameter
  smearing: 0.0          # Fermi smearing kT (Ha); > 0 for near-degenerate systems

forces:
  enabled: true
  method:  analytic      # analytic | finite_difference

optimise:
  enabled:   false
  f_max:     1.0e-3      # Ha/Bohr
  max_steps: 100
  freeze:    none        # none | core (relax only terminators) | [0, 1, 2]

properties:
  population: true
  dipole:     true

output:
  dir:    ./results
  prefix: {name}
  figure: true
  json:   true
  timing: true
"""


def execute(args) -> int:
    template = TEMPLATES[args.template]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = args.name

    xyz_path = outdir / f"{name}.xyz"
    yaml_path = outdir / f"{name}.yaml"
    for p in (xyz_path, yaml_path):
        if p.exists():
            print(f"  refusing to overwrite existing file {p}")
            return 1

    xyz_path.write_text(template["xyz"])
    mult = 2 if args.template == "radical" else 1
    yaml_path.write_text(template["extra"].format(name=name)
                         + _YAML.format(name=name, mult=mult))
    print(f"  wrote {xyz_path}")
    print(f"  wrote {yaml_path}")
    print(f"\n  run it with:  pyxany run {yaml_path}")
    return EXIT_OK
