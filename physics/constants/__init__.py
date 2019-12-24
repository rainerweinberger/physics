#! /physics/constants/__init__.py
#
# This file is part of physics.
# Copyright (C) 2020  Rainer Weinberger (rainer.weinberger@cfa.harvard.edu)
#
# Physics is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Physics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with physics.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

FloatType = np.float64  # make sure all variables are represented with 64 bit floating point numbers

# often used physical constants (cgs units; NIST 2010)
GRAVITY = FloatType(6.6738e-8)
SOLAR_MASS = FloatType(1.989e33)
SOLAR_LUM = FloatType(3.826e33)
SOLAR_EFF_TEMP = FloatType(5.780e3)
RAD_CONST = FloatType(7.5657e-15)
AVOGADRO = FloatType(6.02214e23)
BOLTZMANN = FloatType(1.38065e-16)
GAS_CONST = FloatType(8.31446e7)
CLIGHT = FloatType(2.99792458e10)
PLANCK = FloatType(6.6260695e-27)
PROTONMASS = FloatType(1.67262178e-24)
ELECTRONMASS = FloatType(9.1093829e-28)
THOMPSON = FloatType(6.65245873e-25)
ELECTRONCHARGE = FloatType(4.8032042e-10)
LYMAN_ALPHA = FloatType(1215.6e-8)  # 1215.6 Angstroem
LYMAN_ALPHA_HeII = FloatType(303.8e-8)  # 303.8 Angstroem
OSCILLATOR_STRENGTH = FloatType(0.41615)
OSCILLATOR_STRENGTH_HeII = FloatType(0.41615)


