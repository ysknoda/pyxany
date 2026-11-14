"""Slater transition-state occupations (ADR 0006).

Slater's transition state puts half an electron's worth of the change into
the SCF: for an ionization, 0.5 electron is removed from the orbital; for an
excitation i -> j, 0.5 electron is moved. By Janak's theorem
(dE/dn_i = eps_i) the orbital energy at that half-way point approximates the
energy difference to third order:

    ionization energy  ~ -eps_i(TS)
    excitation energy  ~  eps_j(TS) - eps_i(TS)

The SCF fills orbitals by aufbau as usual and then this module adjusts the
occupations of one spin channel every iteration, so the hole follows the
orbital it was asked for as the orbitals relax:

* ``homo`` / ``lumo`` -- chosen by energy in the unmodified aufbau filling;
* an MO number (1-based, energy order within the spin channel);
* ``{atom, shell}`` (1-based atom as in the .xyz) -- the occupied orbital with
  the largest Loewdin weight on that atom's functions of that shell. With
  several equivalent atoms the initially delocalized core orbitals localize
  on the chosen atom as the SCF proceeds: removing charge there binds its core
  more strongly. Shells need a principal quantum number, which STO and NAO
  functions carry; Gaussian sets do not, so they are refused for core holes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ConfigError


@dataclass
class Target:
    kind: str  # "frontier", "index", "shell"
    which: object  # "homo"/"lumo", int (0-based), or (atom0, n, l)


_L = {"s": 0, "p": 1, "d": 2, "f": 3}

#: Loewdin weight on the target shell above which an eigenvector counts as a
#: localized core orbital and the hole is taken from its occupation.
_LOCALIZED = 0.9


def parse_target(spec) -> Target:
    """``'homo'``, ``'lumo'``, an int (1-based MO), or ``{'atom': k, 'shell': '1s'}``."""
    if isinstance(spec, str) and spec.lower() in ("homo", "lumo"):
        return Target("frontier", spec.lower())
    if isinstance(spec, bool):
        raise ConfigError(f"transition_state: bad orbital {spec!r}")
    if isinstance(spec, int):
        if spec < 1:
            raise ConfigError("transition_state: MO numbers are 1-based")
        return Target("index", spec - 1)
    if isinstance(spec, dict) and set(spec) == {"atom", "shell"}:
        shell = str(spec["shell"]).strip().lower()
        if len(shell) != 2 or not shell[0].isdigit() or shell[1] not in _L:
            raise ConfigError(f"transition_state: shell must look like '1s' or '2p', got {shell!r}")
        n, l = int(shell[0]), _L[shell[1]]
        if l >= n:
            raise ConfigError(f"transition_state: shell {shell} does not exist")
        if int(spec["atom"]) < 1:
            raise ConfigError("transition_state: atom numbers are 1-based, as in the .xyz")
        return Target("shell", (int(spec["atom"]) - 1, n, l))
    raise ConfigError(f"transition_state: cannot interpret orbital {spec!r}")


def shell_mask(basis, atom: int, n: int, l: int) -> np.ndarray:
    """Basis functions of ``atom`` belonging to shell (n, l)."""
    mask = []
    for f in basis:
        if f.atom_idx != atom or f.l != l:
            mask.append(False)
            continue
        fn = getattr(f, "n", None)
        if fn is None and f.kind == "nao":
            # NAO labels carry the shell, e.g. 'O1_2p+1'
            tail = f.label.split("_")[-1]
            fn = int(tail[0]) if tail[:1].isdigit() else None
        if fn is None:
            raise ConfigError(
                "transition_state: a core hole needs a basis with principal "
                "quantum numbers (an STO set such as dz, or nao); Gaussian "
                "sets do not identify shells"
            )
        mask.append(fn == n)
    mask = np.array(mask, dtype=bool)
    if not mask.any():
        raise ConfigError(f"transition_state: atom {atom + 1} has no {n}{'spdf'[l]} functions")
    return mask


class HoleManager:
    """Applies the transition-state change to one spin channel every iteration."""

    def __init__(self, config, molecule, basis, S):
        ts = config.transition_state
        self.electrons = float(ts.electrons)
        self.spin = ts.spin
        self.mode = "remove" if ts.remove is not None else "excite"
        if self.mode == "remove":
            self.source = parse_target(ts.remove)
            self.dest = None
        else:
            self.source = parse_target(ts.excite["from"])
            self.dest = parse_target(ts.excite["to"])
        self._proj = None
        if self.source.kind == "shell":
            atom, n, l = self.source.which
            if atom >= len(molecule):
                raise ConfigError(f"transition_state: atom {atom + 1} > {len(molecule)} atoms")
            w, U = np.linalg.eigh(S)
            self._sqrtS = (U * np.sqrt(np.maximum(w, 0.0))) @ U.T
            self._proj = shell_mask(basis, atom, n, l)
            element = molecule[atom].element
            # the same shell on every atom of that element: the space a core
            # hole can delocalize over, and has to be localized within
            self._elem_proj = np.zeros(len(basis), dtype=bool)
            for k, a in enumerate(molecule):
                if a.element == element:
                    self._elem_proj |= shell_mask(basis, k, n, l)
        self.last = {}
        #: x * c_h c_h^T for a localized core hole, subtracted from the density
        self.correction = None

    def _pick(self, target, eps, occ, C, occupied: bool) -> int:
        finite = np.isfinite(eps)
        if target.kind == "frontier":
            pool = np.where(finite & ((occ > 1e-8) if target.which == "homo" else (occ <= 1e-8)))[0]
            return int(
                pool[np.argmax(eps[pool])] if target.which == "homo" else pool[np.argmin(eps[pool])]
            )
        if target.kind == "index":
            order = np.argsort(np.where(finite, eps, np.inf))
            if target.which >= len(order):
                raise ConfigError(f"transition_state: MO {target.which + 1} does not exist")
            return int(order[target.which])
        weights = np.sum((self._sqrtS @ C)[self._proj] ** 2, axis=0)
        pool = np.where(finite & (occ > 1e-8))[0]
        return int(pool[np.argmax(weights[pool])])

    def _localized_hole(self, eps, occ, C):
        """Hole vector of maximal weight on the target atom's shell.

        Equivalent atoms give delocalized core orbitals (N2: 1s sigma_g and
        sigma_u, each 0.50 on either atom) and an exactly symmetric start never
        breaks that symmetry by itself. Within the occupied orbitals that carry
        this shell of this element, the combination with the largest Loewdin
        weight on the target atom is the leading right singular vector of the
        target rows of S^1/2 C. Its energy is the Rayleigh quotient
        sum_k v_k^2 eps_k (C diagonalizes F).
        """
        LC = self._sqrtS @ C
        finite = np.isfinite(eps)
        on_element = np.sum(LC[self._elem_proj] ** 2, axis=0)
        sub = np.where(finite & (occ > 1e-8) & (on_element > 0.1))[0]
        if len(sub) == 0:
            raise ConfigError("transition_state: no occupied orbital carries the requested shell")
        _, _, vt = np.linalg.svd(LC[np.ix_(self._proj, sub)], full_matrices=False)
        v = vt[0]
        c_h = C[:, sub] @ v
        return c_h, float(np.dot(v * v, eps[sub])), int(sub[np.argmax(np.abs(v))])

    def apply(self, eps, occ, C) -> np.ndarray:
        """Return occupations for the channel this manager acts on.

        For an orbital hole the occupation itself is reduced. For a core hole
        the occupations stay integer and ``self.correction`` (x c_h c_h^T) is
        to be subtracted from that channel's density matrix instead.
        """
        occ = np.array(occ, dtype=float)
        self.correction = None
        if self.source.kind == "shell":
            # Once an eigenvector of F is itself localized on the target atom,
            # take the hole from that orbital's occupation: the density then
            # commutes with F and the SCF can converge. Until then (an exactly
            # symmetric start, N2: two 1s combinations at 0.50 each) subtract
            # the maximally localized combination to break the symmetry; that
            # hole is not an eigenvector, so it only serves as the kick.
            LC = self._sqrtS @ C
            on_target = np.sum(LC[self._proj] ** 2, axis=0)
            pool = np.where(np.isfinite(eps) & (occ > 1e-8))[0]
            best = int(pool[np.argmax(on_target[pool])])
            if on_target[best] < _LOCALIZED:
                c_h, e_h, i = self._localized_hole(eps, occ, C)
                self.correction = self.electrons * np.outer(c_h, c_h)
                self.last = {"source": i, "eps_source": e_h, "localized": False}
                if self.dest is not None:
                    j = self._pick(self.dest, eps, occ, C, occupied=False)
                    if occ[j] + self.electrons > 1.0 + 1e-9:   # as in the branch below
                        raise ConfigError(
                            f"transition_state: orbital {j + 1} cannot take "
                            f"{self.electrons} more electrons in one spin channel"
                        )
                    occ[j] += self.electrons
                    self.last.update({"dest": j, "eps_dest": float(eps[j])})
                return occ
            i = best
        else:
            i = self._pick(self.source, eps, occ, C, occupied=True)
        if occ[i] < self.electrons - 1e-9:
            raise ConfigError(
                f"transition_state: orbital {i + 1} holds {occ[i]:.3f} electrons, "
                f"cannot remove {self.electrons}"
            )
        occ[i] -= self.electrons
        self.last = {"source": i, "eps_source": float(eps[i])}
        if self.dest is not None:
            j = self._pick(self.dest, eps, occ, C, occupied=False)
            if occ[j] + self.electrons > 1.0 + 1e-9:
                raise ConfigError(
                    f"transition_state: orbital {j + 1} cannot take "
                    f"{self.electrons} more electrons in one spin channel"
                )
            occ[j] += self.electrons
            self.last.update({"dest": j, "eps_dest": float(eps[j])})
        return occ
