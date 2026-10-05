"""Find under-coordinated atoms on a cluster surface and where their bonds went."""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..constants import BOHR_TO_ANG
from ..data.radii import bond_cutoff, covalent_radius

#: Ideal directions for the common coordination polyhedra (unit vectors).
IDEAL_POLYHEDRA: dict[int, np.ndarray] = {
    2: np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
    3: np.array([[1.0, 0.0, 0.0],
                 [-0.5, 0.8660254037844386, 0.0],
                 [-0.5, -0.8660254037844386, 0.0]]),
    4: np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                 [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]) / np.sqrt(3.0),
    6: np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0],
                 [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]]),
}


def neighbor_lists(molecule, tolerance: float = 1.20) -> dict[int, list[int]]:
    """``{atom: [neighbors]}`` using ``tolerance * (r_cov(A) + r_cov(B))``."""
    pos = molecule.positions * BOHR_TO_ANG
    n = len(molecule)
    out: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < bond_cutoff(molecule[i].element, molecule[j].element, tolerance):
                out[i].append(j)
                out[j].append(i)
    return out


def bond_report(molecule, neighbors, tolerance: float = 1.20) -> list[dict]:
    """Every detected bond with its cutoff, so the choice can be checked.

    The cutoff is *additive* in the covalent radii; printing the combined value
    next to the measured distance is what makes it verifiable for the specific
    pair rather than plausible per element.
    """
    pos = molecule.positions * BOHR_TO_ANG
    seen = set()
    rows = []
    for i, js in neighbors.items():
        for j in js:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            a, b = molecule[i].element, molecule[j].element
            rows.append({
                "i": i, "j": j, "pair": f"{molecule[i].symbol}{i + 1}-{molecule[j].symbol}{j + 1}",
                "distance": float(np.linalg.norm(pos[i] - pos[j])),
                "r_cov_sum": covalent_radius(a) + covalent_radius(b),
                "cutoff": bond_cutoff(a, b, tolerance),
            })
    return sorted(rows, key=lambda r: r["distance"])


def ideal_coordination(molecule, neighbors, override: int | None = None
                       ) -> tuple[dict[str, int], list[str]]:
    """Bulk coordination number per element, inferred from the cluster interior.

    The **maximum** coordination seen for an element is taken as its bulk value:
    a fragment cut from a crystal keeps fully coordinated atoms in its interior,
    while every surface atom is missing bonds.  (Taking the most *common* value
    instead breaks on small clusters, where under-coordinated surface atoms are
    the majority and every atom then looks saturated.)

    Returns ``(coordination, warnings)``; the warnings flag elements whose
    inferred value is below the chemically expected one, where an explicit
    ``--coordination`` is needed.
    """
    warnings: list[str] = []
    if override:
        return {a.element: override for a in molecule}, warnings

    counts: dict[str, Counter] = {}
    for i, atom in enumerate(molecule):
        if atom.is_pseudo:
            continue
        counts.setdefault(atom.element, Counter())[len(neighbors[i])] += 1

    out: dict[str, int] = {}
    for el, c in counts.items():
        best = max(c)
        out[el] = best
        n_at_max = c[best]
        if best < 3:
            warnings.append(
                f"{el}: the largest coordination found in this structure is {best}, "
                f"which is too low to be a bulk value -- pass --coordination "
                f"explicitly (4 for tetrahedral, 6 for octahedral)."
            )
        elif n_at_max == 1 and sum(c.values()) > 4:
            warnings.append(
                f"{el}: only one atom reaches coordination {best}; if that is not the "
                f"bulk coordination, pass --coordination explicitly."
            )
    return out, warnings


def missing_bond_directions(molecule, index: int, neighbors: list[int],
                            n_missing: int) -> np.ndarray:
    """Unit vectors along which ``n_missing`` bonds are absent.

    Existing bond directions are matched against the ideal polyhedron for the
    full coordination number by finding the rotation that best aligns them
    (an orthogonal Procrustes fit); the unmatched vertices are the missing
    bonds.  With no neighbors at all the ideal polyhedron is used unrotated.
    """
    pos = molecule.positions
    have = np.array([pos[j] - pos[index] for j in neighbors], dtype=float)
    n_total = len(neighbors) + n_missing
    ideal = IDEAL_POLYHEDRA.get(n_total)
    if ideal is None:
        return _spread_directions(have, n_missing)
    if len(have) == 0:
        return ideal[:n_missing]

    have = have / np.linalg.norm(have, axis=1)[:, None]
    rot = _best_rotation(ideal, have)
    rotated = ideal @ rot.T
    # greedily assign each existing bond to its closest ideal vertex
    unused = list(range(len(rotated)))
    for v in have:
        best = max(unused, key=lambda k: float(np.dot(rotated[k], v)))
        unused.remove(best)
    return rotated[unused[:n_missing]]


def _best_rotation(ideal: np.ndarray, have: np.ndarray) -> np.ndarray:
    """Rotation aligning a subset of ``ideal`` with ``have`` (Kabsch)."""
    best_rot = np.eye(3)
    best_score = -np.inf
    from itertools import permutations

    n = len(have)
    if n > len(ideal):
        return best_rot
    for subset in permutations(range(len(ideal)), n):
        A = ideal[list(subset)]
        U, _, Vt = np.linalg.svd(have.T @ A)
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1.0, 1.0, d]) @ Vt
        score = float(np.sum(have * (A @ R.T)))
        if score > best_score:
            best_score, best_rot = score, R
    return best_rot


def _spread_directions(have: np.ndarray, n_missing: int) -> np.ndarray:
    """Fallback: point away from the existing bonds, spread apart."""
    if len(have) == 0:
        return np.eye(3)[:n_missing]
    unit = have / np.linalg.norm(have, axis=1)[:, None]
    away = -unit.sum(axis=0)
    norm = np.linalg.norm(away)
    base = away / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])
    if n_missing == 1:
        return base[None, :]
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(helper, base))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    perp = np.cross(base, helper)
    perp /= np.linalg.norm(perp)
    perp2 = np.cross(base, perp)
    out = []
    for k in range(n_missing):
        phi = 2.0 * np.pi * k / n_missing
        v = base * 0.7 + 0.7 * (np.cos(phi) * perp + np.sin(phi) * perp2)
        out.append(v / np.linalg.norm(v))
    return np.array(out)


def directions_from_reference(molecule, reference, index: int, tolerance: float = 1.20):
    """Missing bond directions recovered from a bulk reference structure.

    The cluster atom is matched to its counterpart in ``reference`` by position;
    neighbors present in the bulk but absent from the cluster give the exact
    directions of the cut bonds.
    """
    ref_pos = reference.positions
    cluster_pos = molecule.positions
    site = int(np.argmin(np.linalg.norm(ref_pos - cluster_pos[index], axis=1)))
    ref_neigh = neighbor_lists(reference, tolerance)[site]
    kept = {tuple(np.round(cluster_pos[j] - cluster_pos[index], 4))
            for j in neighbor_lists(molecule, tolerance)[index]}
    out = []
    for j in ref_neigh:
        d = ref_pos[j] - ref_pos[site]
        if tuple(np.round(d, 4)) not in kept:
            out.append(d / np.linalg.norm(d))
    return np.array(out) if out else np.zeros((0, 3))
