"""Orbital occupations: aufbau filling and Fermi-Dirac smearing.

The single rule everywhere in this module: ``n_electrons`` is the number of
electrons *in the channel being filled*, and ``max_occ`` is how many electrons
one spatial orbital of that channel can hold -- 2 for a spin-restricted
calculation, 1 for one spin channel of an unrestricted one.

The original single-file version conflated the two (it passed ``2 * n_alpha``
electrons together with ``max_occ = 1``), which put twice as many electrons in
each spin channel as the molecule had; see docs/legacy_comparison.md.
"""

from __future__ import annotations

import numpy as np

_MAX_BISECTION = 200


def aufbau_occupations(eps: np.ndarray, n_electrons: float,
                       max_occ: float = 2.0) -> np.ndarray:
    """Fill orbitals in energy order.  Handles a fractional electron count."""
    occ = np.zeros(len(eps))
    order = np.argsort(np.where(np.isfinite(eps), eps, np.inf))
    remaining = float(n_electrons)
    for i in order:
        if remaining <= 1e-12 or not np.isfinite(eps[i]):
            break
        take = min(max_occ, remaining)
        occ[i] = take
        remaining -= take
    if remaining > 1e-8:
        raise ValueError(
            f"cannot place {n_electrons:g} electrons: only "
            f"{np.isfinite(eps).sum()} usable orbitals x max_occ {max_occ:g} available. "
            f"Use a larger basis set."
        )
    return occ


def fermi_occupations(eps: np.ndarray, n_electrons: float, sigma: float,
                      max_occ: float = 2.0) -> np.ndarray:
    """Fermi-Dirac occupations at smearing width ``sigma`` (Hartree).

    ``sigma <= 0`` falls back to :func:`aufbau_occupations`.
    """
    if sigma <= 0.0:
        return aufbau_occupations(eps, n_electrons, max_occ)

    finite = np.isfinite(eps)
    e = eps[finite]
    if len(e) == 0:  # pragma: no cover - degenerate basis
        return np.zeros(len(eps))

    def count(mu: float) -> float:
        x = np.clip((e - mu) / sigma, -500.0, 500.0)
        return float(np.sum(max_occ / (1.0 + np.exp(x))))

    lo, hi = e.min() - 50.0 * sigma, e.max() + 50.0 * sigma
    for _ in range(_MAX_BISECTION):
        mu = 0.5 * (lo + hi)
        n = count(mu)
        if abs(n - n_electrons) < 1e-12:
            break
        if n < n_electrons:
            lo = mu
        else:
            hi = mu

    occ = np.zeros(len(eps))
    x = np.clip((e - mu) / sigma, -500.0, 500.0)
    occ[finite] = max_occ / (1.0 + np.exp(x))
    # The bisection converges on mu whether or not the requested count actually
    # fits: with too few orbitals it just walks mu to `hi` and returns a smaller
    # total.  aufbau_occupations raises in that case, and the smeared path must
    # too, or the SCF converges tidily for the wrong number of electrons.
    placed = float(occ.sum())
    if abs(placed - n_electrons) > 1e-6 * max(1.0, abs(n_electrons)):
        raise ValueError(
            f"cannot place {n_electrons:g} electrons: only "
            f"{int(finite.sum())} usable orbitals x max_occ {max_occ:g} available "
            f"(placed {placed:g} at smearing {sigma:g}). Use a larger basis set."
        )
    return occ


def smearing_entropy(occ: np.ndarray, sigma: float, max_occ: float = 2.0) -> float:
    """Return ``sigma * S >= 0`` for a set of fractional occupations.

    With ``S = -max_occ sum_i [f ln f + (1-f) ln(1-f)]`` and ``f = occ/max_occ``,
    the smeared free energy is ``F = E - sigma S``, so callers **subtract** this
    value from the total energy.  It is zero for integer occupations.
    """
    if sigma <= 0.0:
        return 0.0
    f = np.clip(occ / max_occ, 1e-15, 1.0 - 1e-15)
    return -sigma * max_occ * float(np.sum(f * np.log(f) + (1.0 - f) * np.log(1.0 - f)))


def density_matrix(C: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """P_{mu nu} = sum_i occ_i C_{mu i} C_{nu i}."""
    active = occ > 0
    return (C[:, active] * occ[active]) @ C[:, active].T


def spin_electron_counts(n_electrons: float, multiplicity: int) -> tuple[float, float]:
    """(n_alpha, n_beta) for a given multiplicity, validated for consistency."""
    n_unpaired = multiplicity - 1
    n_alpha = (n_electrons + n_unpaired) / 2.0
    n_beta = (n_electrons - n_unpaired) / 2.0
    if n_beta < -1e-9:
        raise ValueError(
            f"multiplicity {multiplicity} needs {n_unpaired} unpaired electrons "
            f"but the molecule only has {n_electrons:g}"
        )
    # An integer electron count must split into integer alpha and beta counts,
    # so N and the number of unpaired electrons have to share parity.  Without
    # this check a radical left at the default mult=1 converges a half-occupied
    # state that is not any real spin state: CH3 (9 electrons) at mult=1 gives
    # 4.5/4.5 and an energy 26 mHa (0.71 eV) above the true doublet, silently.
    # A fractional N -- a pseudo-hydrogen terminated cluster -- has no parity to
    # satisfy and is exempt.
    if abs(n_electrons - round(n_electrons)) < 1e-9:
        total = int(round(n_electrons))
        if (total - n_unpaired) % 2 != 0:
            raise ValueError(
                f"multiplicity {multiplicity} is impossible for {total} electrons: "
                f"the alpha/beta split would be {n_alpha:g}/{n_beta:g}, which is not "
                f"a real spin state. An odd electron count needs an even "
                f"multiplicity and an even count an odd one; "
                f"try molecule.mult = {2 if total % 2 else 1}."
            )
    return n_alpha, max(n_beta, 0.0)
