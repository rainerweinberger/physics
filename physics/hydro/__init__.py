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
                 specific_thermal_energy=None,
                 gamma: FloatType = FloatType(5.0 / 3.0),
                 unit_length_in_cm: FloatType = FloatType(1.0),
                 unit_mass_in_g: FloatType = FloatType(1.0),
                 unit_velocity_in_cm_per_s: FloatType = FloatType(1.0),
                 hubble_param: FloatType = FloatType(1.0),
                 scale_factor: FloatType = FloatType(1.0)
                 ):
        self._mass = np.array(mass, dtype=FloatType, ndmin=1)
        self._density = np.array(density, dtype=FloatType, ndmin=1)
        self._velocity = np.array(velocity, dtype=FloatType, ndmin=2)
        self._specific_thermal_energy = np.array(specific_thermal_energy, dtype=FloatType, ndmin=1)
        self._gamma = FloatType(gamma)
        self._unit_length_in_cm = FloatType(unit_length_in_cm)
        self._unit_mass_in_g = FloatType(unit_mass_in_g)
        self._unit_velocity_in_cm_per_s = FloatType(unit_velocity_in_cm_per_s)
        self._hubble_param = FloatType(hubble_param)
        self._scale_factor = FloatType(scale_factor)

    """ factories """

    @staticmethod
    def from_conserved_variables(volume=None,
                                 mass=None,
                                 momentum=None,
                                 thermal_energy=None,
                                 gamma: FloatType = FloatType(5.0 / 3.0),
                                 unit_length_in_cm: FloatType = FloatType(1.0),
                                 unit_mass_in_g: FloatType = FloatType(1.0),
                                 unit_velocity_in_cm_per_s: FloatType = FloatType(1.0),
                                 hubble_param: FloatType = FloatType(1.0),
                                 scale_factor: FloatType = FloatType(1.0)
                                 ):
        """ create HydroState from conserved variables """
        _mass = np.array(mass, dtype=FloatType, ndmin=1)
        _momentum = np.array(momentum, dtype=FloatType, ndmin=2)
        density = _mass / np.array(volume, dtype=FloatType, ndmin=1)
        velocity = np.zeros(_momentum.shape)
        for dim in np.arange(_momentum.shape[1]):
            velocity[:, dim] = _momentum[:, dim] / _mass
        specific_thermal_energy = thermal_energy / _mass
        return HydroState(mass=_mass, density=density, velocity=velocity,
                          specific_thermal_energy=specific_thermal_energy, gamma=gamma,
                          unit_length_in_cm=unit_length_in_cm, unit_mass_in_g=unit_mass_in_g,
                          unit_velocity_in_cm_per_s=unit_velocity_in_cm_per_s, hubble_param=hubble_param,
                          scale_factor=scale_factor)

    @staticmethod
    def from_primitive_variables(volume=None,
                                 density=None,
                                 velocity=None,
                                 specific_thermal_energy=None,
                                 gamma: FloatType = FloatType(5.0 / 3.0),
                                 unit_length_in_cm: FloatType = FloatType(1.0),
                                 unit_mass_in_g: FloatType = FloatType(1.0),
                                 unit_velocity_in_cm_per_s: FloatType = FloatType(1.0),
                                 hubble_param: FloatType = FloatType(1.0),
                                 scale_factor: FloatType = FloatType(1.0)
                                 ):
        """ create HydroState from primitive variables """
        mass = density * volume
        return HydroState(mass=mass, density=density, velocity=velocity,
                          specific_thermal_energy=specific_thermal_energy, gamma=gamma,
                          unit_length_in_cm=unit_length_in_cm, unit_mass_in_g=unit_mass_in_g,
                          unit_velocity_in_cm_per_s=unit_velocity_in_cm_per_s, hubble_param=hubble_param,
                          scale_factor=scale_factor)

    """ magic methods """

    def __str__(self):
        out = "hydro state:"
        out += "\n mass = " + str(self._mass)
        out += "\n density = " + str(self._density)
        out += "\n velocity = " + str(self._velocity)
        out += "\n specific thermal energy = " + str(self._specific_thermal_energy)
        return out

    def __repr__(self):
        out = "hydro state: < "
        out += "mass = " + str(self._mass)
        out += "; density = " + str(self._density)
        out += "; velocity = " + str(self._velocity)
        out += "; specific thermal energy = " + str(self._specific_thermal_energy) + " >"
        return out

    def __eq__(self, other):
        if np.all(self.mass == other.mass) \
                and np.all(self.density == other.density) \
                and np.all(self.velocity == other.velocity) \
                and np.all(self.specific_thermal_energy == other.specific_thermal_energy):
            return True
        return False

    """ properties """

    @property
    def mass(self):
        return self._mass

    @property
    def density(self):
        return self._density

    @property
    def volume(self):
        return self._mass / self._density

    @property
    def velocity(self):
        return self._velocity

    @property
    def momentum(self):
        return self._velocity * self._mass

    @property
    def kinetic_energy(self):
        return 0.5 * self._mass * (self._velocity[:, 0] ** 2 + self._velocity[:, 1] ** 2 + self._velocity[:, 2] ** 2)

    @property
    def specific_thermal_energy(self):
        return self._specific_thermal_energy

    @property
    def thermal_energy_density(self):
        return self._specific_thermal_energy * self._density

    @property
    def pressure(self):
        return (self._gamma - 1.0) * self._density * self._specific_thermal_energy

    @property
    def thermal_energy(self):
        return self._specific_thermal_energy * self._mass

    @property
    def sound_speed_squared(self):
        return self._gamma * (self._gamma - 1.0) * self._specific_thermal_energy

    @property
    def sound_speed(self):
        return np.sqrt(self.sound_speed_squared)

    @property
    def total_energy(self):
        return self.thermal_energy + self.kinetic_energy

    """ functions that use external information """
