"""``pyxany basis`` -- basis-set coverage."""

from __future__ import annotations

from ..data import elements as elem
from ..data.loaders import coverage_report
from . import EXIT_INPUT, EXIT_OK


def execute(args) -> int:
    if args.element and not elem.is_supported(args.element):
        print(f"  unknown element {args.element!r}")
        return EXIT_INPUT

    if args.element:
        return _element_report(args.element)

    print("  BASIS SET COVERAGE")
    print("  " + "-" * 74)
    print(f"  {'table':<14} {'elements':>9}   note")
    for name, n, note in coverage_report():
        print(f"  {name:<14} {n:>9}   {note[:52]}")
    print()
    print("  STO keys : sz szp dz dzp tz tzp tzpp   (all 86 elements H-Rn)")
    print("  GTO keys : sto3g 631g 631gd            (limited element coverage above)")
    print("  NAO key  : nao   (numerical orbitals from a spherical atomic solve)")
    print()
    print("  A pseudo-hydrogen terminator always gets a 1s function derived from its")
    print("  own fractional charge, independent of the basis key.")
    return EXIT_OK


def _element_report(symbol: str) -> int:
    from ..basis.sto import sto_exponents

    print(f"  {symbol}   Z = {elem.atomic_number(symbol)}   "
          f"group {elem.group(symbol)}   "
          f"valence electrons {elem.valence_electrons(symbol)}")
    print(f"  ground-state multiplicity {elem.ground_state_multiplicity(symbol)}"
          + ("" if elem.multiplicity_is_reliable(symbol)
             else "  [derived from Hund's rule; d/f-block exceptions not captured]"))
    print(f"  Bragg-Slater radius {elem.bragg_slater_radius(symbol):.3f} Bohr")
    print("  configuration: " + "  ".join(
        f"{k}={v:g}" for k, v in sorted(elem.occupations(symbol).items())))
    print()
    print(f"  {'basis':<8} {'functions':>10}   exponents")
    for key in ("sz", "dz", "dzp", "tz", "tzpp"):
        funcs = sto_exponents(key).get(symbol, [])
        zetas = sorted({round(z, 4) for _n, _l, _m, z in funcs}, reverse=True)
        print(f"  {key:<8} {len(funcs):>10}   {zetas[:9]}")
    return EXIT_OK
