"""``pyxany basis`` -- basis-set coverage."""

from __future__ import annotations

from ..data import elements as elem
from ..data.loaders import coverage_report
from . import EXIT_INPUT, EXIT_OK


def execute(args) -> int:
    if args.element and not elem.is_supported(args.element):
        print(f"  unknown element {args.element!r}")
        return EXIT_INPUT
    key = (getattr(args, "key", None) or "").lower() or None
    if key is not None and key not in _all_keys():
        # the key used to be ignored: 'pyxany basis nonsense' printed the
        # overview and exited 0 (ADR 0012 P0)
        print(f"  unknown basis key {args.key!r}; choose from {', '.join(_all_keys())}")
        return EXIT_INPUT
    if key is not None:
        return _key_report(key, args.element)
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


def _all_keys() -> list[str]:
    from ..basis.gto import GTO_KEYS
    from ..basis.sto import STO_KEYS

    return [*STO_KEYS, *GTO_KEYS, "nao"]


def _key_covers(key: str) -> list[str]:
    """Elements the key has functions for, in order of Z."""
    from ..basis.gto import GTO_KEYS
    from ..basis.sto import STO_KEYS, sto_exponents
    from ..data.loaders import gto_table

    if key in STO_KEYS:
        have = set(sto_exponents(key))
    elif key in GTO_KEYS:
        have = set(gto_table(GTO_KEYS[key][0]))
    else:  # nao: solved for every element
        have = set(elem.symbols())
    return sorted(have, key=elem.atomic_number)


def _key_report(key: str, element: str | None) -> int:
    covers = _key_covers(key)
    if element:
        ok = element in covers
        if ok:
            _element_report(element)
        print(f"  basis {key}: {element} {'is' if ok else 'is NOT'} covered")
        return EXIT_OK if ok else EXIT_INPUT
    print(f"  {key}: {len(covers)} elements")
    print("  " + " ".join(covers))
    return EXIT_OK
