"""Place hydrogen / pseudo-hydrogen terminators on a cluster."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import ANG_TO_BOHR, BOHR_TO_ANG
from ..data.radii import covalent_radius
from ..data.radii import source as radii_source
from ..molecule import Atom, Molecule
from .counting import assign_roles, terminator_charge_for
from .detect import (
    bond_report,
    directions_from_reference,
    ideal_coordination,
    missing_bond_directions,
    neighbor_lists,
)
from .species import PseudoSpecies, label_for


@dataclass
class PassivationResult:
    molecule: Molecule
    species: dict[str, PseudoSpecies]
    n_added: int
    bonds: list = field(default_factory=list)
    placements: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def species_yaml(self) -> dict:
        return {label: sp.to_yaml_entry() for label, sp in sorted(self.species.items())}

    def electron_balance(self) -> dict:
        """Whether the electron count matches the requested spin state.

        An even count is right for a closed shell, but an open-shell cluster
        (multiplicity > 1) should have an odd one; the general requirement is
        that N and ``multiplicity - 1`` share parity.
        """
        n = self.molecule.n_electrons
        n_unpaired = self.molecule.multiplicity - 1
        target = 2.0 * round((n - n_unpaired) / 2.0) + n_unpaired
        return {"n_electrons": float(n), "nearest_even": target,
                "target": target, "n_unpaired": n_unpaired,
                "deviation": float(abs(n - target))}


def terminator_distance(element: str, neighbor_element: str,
                        nn_distance_ang: float) -> float:
    """Initial X-H* distance in Angstrom.

    Combination rule, stated explicitly because a per-element radius is not
    automatically pair-sized::

        d(X-H*) = d_nn(X-Y) * [r_cov(X) + r_cov(H)] / [r_cov(X) + r_cov(Y)]

    i.e. the real X-Y bond of *this* structure, rescaled by the ratio of the
    two summed covalent radii.  It is an initial guess only: the terminator
    positions are meant to be relaxed (``pyxany opt --freeze-core``), which is
    what actually determines the distance.
    """
    r_x = covalent_radius(element)
    r_h = covalent_radius("H")
    r_y = covalent_radius(neighbor_element)
    return nn_distance_ang * (r_x + r_h) / max(r_x + r_y, 1e-6)


def passivate(molecule, scheme: str = "pseudo_hydrogen", tolerance: float = 1.20,
              coordination: int | None = None, reference=None,
              role_overrides: dict | None = None,
              bond_scale: float | None = None) -> PassivationResult:
    """Terminate every dangling bond of ``molecule``.

    ``scheme='pseudo_hydrogen'`` assigns each terminator the fractional charge
    from the electron-counting rule; ``scheme='hydrogen'`` uses ordinary
    hydrogen (alpha = 1) everywhere, which is the comparison
    Phys. Rev. B **85**, 195328 (2012) is about.
    """
    if scheme not in ("pseudo_hydrogen", "hydrogen"):
        raise ValueError(f"unknown passivation scheme {scheme!r}")

    neighbors = neighbor_lists(molecule, tolerance)
    bonds = bond_report(molecule, neighbors, tolerance)
    ideal, coord_warnings = ideal_coordination(molecule, neighbors, coordination)
    # `ideal`, not the observed neighbor count: the charge is computed with the
    # ideal coordination, and a surface site has fewer neighbors than that.
    roles = assign_roles(molecule, neighbors, role_overrides, ideal)

    pos_ang = molecule.positions * BOHR_TO_ANG
    new_atoms = list(molecule.atoms)
    species: dict[str, PseudoSpecies] = {}
    placements: list[dict] = []
    warnings: list[str] = list(coord_warnings)

    for i, atom in enumerate(molecule):
        if atom.is_pseudo:
            continue
        want = ideal.get(atom.element, 4)
        have = len(neighbors[i])
        n_missing = want - have
        if n_missing <= 0:
            continue
        if have == 0:
            warnings.append(
                f"{atom.symbol}{i + 1} has no detected neighbors; the terminator "
                f"directions are arbitrary. Check grid.tolerance ({tolerance})."
            )

        dirs = None
        if reference is not None:
            dirs = directions_from_reference(molecule, reference, i, tolerance)
            if len(dirs) < n_missing:
                dirs = None
        if dirs is None:
            dirs = missing_bond_directions(molecule, i, neighbors[i], n_missing)
        dirs = np.asarray(dirs)[:n_missing]

        role = roles.get(i, "cation")
        if scheme == "pseudo_hydrogen":
            alpha = terminator_charge_for(atom.element, role, want)
            label = label_for(atom.element)
        else:
            alpha = 1.0
            label = "H"

        from ..data import elements as elem

        species.setdefault(label, PseudoSpecies(
            label=label, passivates=atom.element, role=role,
            valence_electrons=elem.valence_electrons(atom.element),
            coordination=want, alpha=alpha))

        nn = _nearest_neighbor_distance(pos_ang, i, neighbors[i], molecule)
        d_ang = terminator_distance(atom.element, nn[1], nn[0])
        if bond_scale:
            d_ang *= bond_scale

        for direction in dirs:
            unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
            new_pos = molecule[i].pos + unit * d_ang * ANG_TO_BOHR
            new_atoms.append(
                Atom.pseudo_hydrogen(label, alpha, new_pos) if scheme == "pseudo_hydrogen"
                else Atom.from_element("H", new_pos)
            )
            placements.append({
                "host": f"{atom.symbol}{i + 1}", "host_index": i, "label": label,
                "alpha": float(alpha), "distance_ang": float(d_ang),
                "reference_bond": f"{atom.element}-{nn[1]}", "reference_distance": nn[0],
                "r_cov_sum_XH": covalent_radius(atom.element) + covalent_radius("H"),
                "r_cov_sum_XY": covalent_radius(atom.element) + covalent_radius(nn[1]),
            })

    result_mol = Molecule(atoms=new_atoms, charge=molecule.charge,
                          multiplicity=molecule.multiplicity,
                          name=molecule.name)
    # A closed shell needs an even count, but an open-shell cluster is supposed
    # to have an odd one: what must hold in general is that N and the number of
    # unpaired electrons (multiplicity - 1) share parity.
    n_unpaired = result_mol.multiplicity - 1
    target = 2.0 * round((result_mol.n_electrons - n_unpaired) / 2.0) + n_unpaired
    if abs(result_mol.n_electrons - target) > 1e-6:
        shell = "even" if n_unpaired == 0 else f"odd (multiplicity {result_mol.multiplicity})"
        warnings.append(
            f"total electron count is {result_mol.n_electrons:.6f}, which is not "
            f"compatible with a {shell} shell (nearest is {target:g}). The "
            f"electron-counting rule gives a consistent shell only when every "
            f"dangling bond is terminated and the coordination numbers are right."
        )

    return PassivationResult(molecule=result_mol, species=species,
                             n_added=len(new_atoms) - len(molecule),
                             bonds=bonds, placements=placements, warnings=warnings)


def _nearest_neighbor_distance(pos_ang, i, neighbors, molecule):
    """(distance, element) of the closest real neighbor, for the bond-length scale."""
    if not neighbors:
        return 2.35, molecule[i].element        # a typical covalent bond
    ds = [(float(np.linalg.norm(pos_ang[i] - pos_ang[j])), molecule[j].element)
          for j in neighbors]
    return min(ds)


def format_report(result: PassivationResult, scheme: str) -> str:
    """The text ``pyxany passivate`` prints."""
    out = ["=" * 78,
           f"  PASSIVATION  ({scheme})",
           "=" * 78, ""]
    out.append(f"  covalent radii: {radii_source()}")
    out.append("")
    out.append(f"  detected bonds ({len(result.bonds)}), with the cutoff used")
    out.append(f"  {'pair':>14} {'distance':>10} {'r_cov sum':>11} {'cutoff':>9}")
    for b in result.bonds[:24]:
        out.append(f"  {b['pair']:>14} {b['distance']:>10.4f} {b['r_cov_sum']:>11.4f} "
                   f"{b['cutoff']:>9.4f}")
    if len(result.bonds) > 24:
        out.append(f"  ... {len(result.bonds) - 24} more")
    out.append("")

    out.append(f"  terminator species ({len(result.species)})")
    for sp in sorted(result.species.values(), key=lambda s: s.label):
        out.append("    " + sp.describe())
    out.append("")

    out.append(f"  added {result.n_added} terminators")
    if result.placements:
        out.append(f"  {'host':>8} {'label':>7} {'alpha':>9} {'d(X-H*)':>9}  scale basis")
        seen = set()
        for p in result.placements:
            key = (p["host_index"], p["label"])
            if key in seen:
                continue
            seen.add(key)
            out.append(f"  {p['host']:>8} {p['label']:>7} {p['alpha']:>9.6f} "
                       f"{p['distance_ang']:>9.4f}  {p['reference_bond']} = "
                       f"{p['reference_distance']:.4f} A x "
                       f"{p['r_cov_sum_XH']:.3f}/{p['r_cov_sum_XY']:.3f}")
    out.append("")

    bal = result.electron_balance()
    verdict = "closed shell" if bal["deviation"] < 1e-6 else "NOT an even integer"
    out.append(f"  electron balance: {bal['n_electrons']:.6f} electrons -- {verdict}")
    out.append("")
    for warn in result.warnings:
        out.append(f"  [warning] {warn}")
    return "\n".join(out)
