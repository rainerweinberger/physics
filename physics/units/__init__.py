#! /physics/units/__init__.py
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

"""
 all relative to cgs unit system
"""
# technically, this is a constant/property of the sun, however, often used as a unit in astrophysics
SOLAR_MASS = FloatType(1.989e33)
SOLAR_LUM = FloatType(3.826e33)
PARSEC = FloatType(3.085678e18)
KILOPARSEC = FloatType(3.085678e21)
MEGAPARSEC = FloatType(3.085678e24)
ASTRONOMICAL_UNIT = FloatType(1.49598e13)
HUBBLE = FloatType(3.2407789e-18)  # in h/sec
ELECTRONVOLT_IN_ERGS = FloatType(1.60217656e-12)
SEC_PER_GIGAYEAR = FloatType(3.15576e16)
SEC_PER_MEGAYEAR = FloatType(3.15576e13)
SEC_PER_YEAR = FloatType(3.15576e7)

# derived ones
SOLAR_MASS_PER_YEAR = SOLAR_MASS / SEC_PER_YEAR
SOLAR_MASS_PER_SQUARE_KPC = SOLAR_MASS / KILOPARSEC / KILOPARSEC
SOLAR_MASS_PER_CUBE_KPC = SOLAR_MASS / KILOPARSEC / KILOPARSEC / KILOPARSEC
