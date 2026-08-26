"""Occupation bookkeeping: the electron count must be conserved exactly."""

from __future__ import annotations

import numpy as np
import pytest

from dvxa.scf.occupations import (
    aufbau_occupations,
    density_matrix,
    fermi_occupations,
    smearing_entropy,
    spin_electron_counts,
)

EPS = np.array([-2.0, -1.5, -1.0, -0.2, 0.3, 0.9])


def test_restricted_filling():
    occ = aufbau_occupations(EPS, 10, max_occ=2.0)
    assert occ.sum() == pytest.approx(10.0)
    assert set(np.unique(occ)) <= {0.0, 2.0}


def test_unrestricted_channels_hold_the_right_counts():
    """The open-shell regression: each channel gets n_sigma, not 2 n_sigma.

    A version of this code passed ``2 * n_alpha`` electrons together with a
    per-orbital capacity of 1, doubling the electron count of every open-shell
    system.  CH3 (9 electrons) came out with 16.
    """
    n_alpha, n_beta = spin_electron_counts(9, multiplicity=2)
    assert (n_alpha, n_beta) == (5.0, 4.0)
    occ_a = aufbau_occupations(EPS, n_alpha, max_occ=1.0)
    occ_b = aufbau_occupations(EPS, n_beta, max_occ=1.0)
    assert occ_a.sum() == pytest.approx(5.0)
    assert occ_b.sum() == pytest.approx(4.0)
    assert occ_a.sum() + occ_b.sum() == pytest.approx(9.0)
    assert occ_a.max() <= 1.0


def test_fractional_electron_count():
    """Pseudo-hydrogen charges make non-integer totals legitimate."""
    occ = aufbau_occupations(EPS, 7.5, max_occ=2.0)
    assert occ.sum() == pytest.approx(7.5)


def test_too_few_orbitals_is_an_error():
    with pytest.raises(ValueError, match="larger basis"):
        aufbau_occupations(EPS, 40, max_occ=2.0)


@pytest.mark.parametrize("sigma", [0.5, 0.05, 0.005])
def test_smearing_conserves_the_electron_count(sigma):
    occ = fermi_occupations(EPS, 10, sigma, max_occ=2.0)
    assert occ.sum() == pytest.approx(10.0, abs=1e-9)


def test_smearing_tends_to_aufbau():
    hard = aufbau_occupations(EPS, 10, max_occ=2.0)
    soft = fermi_occupations(EPS, 10, 1e-4, max_occ=2.0)
    assert np.abs(soft - hard).max() < 1e-9


def test_smearing_entropy_is_non_negative_and_vanishes_for_integers():
    occ = fermi_occupations(EPS, 10, 0.2, max_occ=2.0)
    assert smearing_entropy(occ, 0.2, 2.0) > 0
    assert smearing_entropy(aufbau_occupations(EPS, 10, 2.0), 0.0, 2.0) == 0.0


def test_density_matrix_trace_in_an_orthonormal_basis():
    C = np.linalg.qr(np.random.default_rng(0).normal(size=(6, 6)))[0]
    occ = aufbau_occupations(EPS, 8, max_occ=2.0)
    P = density_matrix(C, occ)
    assert np.trace(P) == pytest.approx(8.0)


def test_impossible_multiplicity_is_rejected():
    with pytest.raises(ValueError, match="unpaired"):
        spin_electron_counts(2, multiplicity=6)
