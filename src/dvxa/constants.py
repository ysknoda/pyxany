"""Physical constants and unit conversions (atomic units throughout the code)."""

import numpy as np

# CODATA 2018 values
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
AU_TO_DEBYE = 2.541746473
EV_TO_KCAL_MOL = 23.060548

PI = np.pi

# Slater Xalpha exchange:
#     eps_x(rho) = -(9/8) * alpha * (3 rho / pi)^(1/3)
#     V_x(rho)   = (4/3) eps_x = -(3/2) * alpha * (3 rho / pi)^(1/3)
#                              = -3 alpha (3 rho / (8 pi))^(1/3)
# See dvxa.integrals.xc -- these are the single source of truth for the
# coefficients and are covered by test_xc.py::test_potential_is_functional_derivative.
XALPHA_ENERGY_COEFF = 9.0 / 8.0
XALPHA_POTENTIAL_COEFF = 3.0 / 2.0

DEFAULT_ALPHA = 0.7
