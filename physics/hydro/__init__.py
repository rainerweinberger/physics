#! /physics/hydro/__init__.py
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

FloatType = np.float64

class HydroState(object):
    """
    class for hydrodynamic state
    """
    def __init__(self,
                 mass=None,
                 density=None,
                 velocity=None,
                 specific_energy=None,
                 gamma: FloatType = FloatType(5.0 / 3.0),
                 unit_length_in_cm: FloatType = FloatType(1.0),
                 unit_mass_in_g: FloatType = FloatType(1.0),
                 unit_velocity_in_cm_per_s: FloatType = FloatType(1.0),
                 hubble_param: FloatType = FloatType(1.0),
                 scale_factor: FloatType = FloatType(1.0)
                 ):
        self._mass = mass
        self._density = density
        self._velocity = velocity
        self._specific_energy = specific_energy
        self._gamma = gamma
        self._unit_length_in_cm = unit_length_in_cm
        self._unit_mass_in_g = unit_mass_in_g
        self._unit_velocity_in_cm_per_s = unit_velocity_in_cm_per_s
        self._hubble_param = hubble_param
        self._scale_factor = scale_factor

    """ magic methods """
    def __str__(self):
        out = "hydro state:"
        out += "\n mass = " + str(self._mass)
        out += "\n density = " + str(self._density)
        out += "\n velocity = " + str(self._velocity)
        out += "\n specific energy = " + str(self._specific_energy)
        return out

    """ properties """
    @property
    def volume(self):
        return self._mass / self._density

