#! physics/test/test_physics_hydro.py
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

import pytest
import numpy as np
from physics.hydro import HydroState
from physics.hydro import MhdState
from physics import constants


class TestHydro(object):
    def test_hydro_state_init(self):
        """
        initialization via constructor and factories

        comparison operation
        :return:
        """
        HydroState()

        second_state = HydroState.from_conserved_variables(volume=2.0, mass=1.0, momentum=[0.5, 0.5, 1],
                                                           thermal_energy=2.0)
        third_state = HydroState.from_primitive_variables(volume=2.0, density=0.5, velocity=[0.5, 0.5, 1],
                                                          specific_thermal_energy=2.0)
        assert second_state == third_state

    def test_hydro_state_magic_methods(self):
        """
        magic (dunder) methods of a hydro state;

        note that __eq__ is checked already in initialization test
        :return:
        """
        state = HydroState(density=1.0, mass=2.0, velocity=[1.0, 0.0, 0.5], specific_thermal_energy=0.5)

        # __str__
        print(state)

        # __repr__
        print(state.__repr__())

        state2 = HydroState(density=1.0, mass=2.0, velocity=[1.0, 0.0, 0.5], specific_thermal_energy=0.25)
        assert not state == state2

    def test_hydro_state_properties(self):
        """
        calculate properties
        :return:
        """
        state = HydroState(density=1.0, mass=2.0, velocity=[1.0, 0.0, 0.5], specific_thermal_energy=0.5)

        # mass
        assert state.mass == pytest.approx(2.0)
        # density
        assert state.density == pytest.approx(1.0)
        # volume
        assert state.volume == pytest.approx(2.0)

        # velocity
        assert state.velocity == pytest.approx(np.array([[1.0, 0.0, 0.5]]))
        # absolute velocity
        assert state.absolute_velocity == pytest.approx(np.sqrt(1.25))
        # momentum
        assert state.momentum == pytest.approx(np.array([[2.0, 0.0, 1.0]]))
        # kinetic energy
        assert state.kinetic_energy == pytest.approx(1.25)

        # specific thermal energy
        assert state.specific_thermal_energy == pytest.approx(0.5)
        # thermal energy density
        assert state.thermal_energy_density == pytest.approx(0.5)
        # pressure
        assert state.pressure == pytest.approx(1.0 / 3.0)
        # thermal energy
        assert state.thermal_energy == pytest.approx(1.0)
        # sound speed
        assert state.sound_speed == pytest.approx(np.sqrt(5. / 9.))
        # mach number
        assert state.mach_number == pytest.approx(np.sqrt(1.25 * 9. / 5.))

        # total energy
        assert state.total_energy == pytest.approx(2.25)

    def test_hydro_state_cooling_luminosity(self):
        """ cooling luminosity """
        cooling_rate = np.array([-8.7344289e-23, -9.4096156e-23, -9.4876307e-23, -1.0734809e-22,
                                 -9.4646093e-23, -9.4072881e-23, -1.2774670e-22, -9.3815756e-23,
                                 -8.9719915e-23, -8.7572850e-23])
        density = np.array([4.4664993e-08, 4.0362085e-08, 3.7757609e-08, 2.9720892e-08,
                            4.2373319e-08, 4.0239804e-08, 2.2796231e-08, 4.0245101e-08,
                            4.3894516e-08, 4.1929820e-08])
        mass = np.array([1.9719103e-08, 2.3327638e-08, 4.2077549e-08, 2.9918770e-08,
                         2.9821706e-08, 2.9963097e-08, 7.7762463e-09, 2.5147488e-08,
                         2.7254773e-08, 2.4179668e-08])
        unit_length_in_cm = np.float64(3.085678e+21)
        unit_mass_in_g = np.float64(1.989e+43)
        unit_velocity_in_cm_per_s = np.float64(100000.0)
        hubble_param = np.float64(0.6774)

        state = HydroState(density=density,
                           mass=mass,
                           unit_length_in_cm=unit_length_in_cm,
                           unit_mass_in_g=unit_mass_in_g,
                           unit_velocity_in_cm_per_s=unit_velocity_in_cm_per_s,
                           hubble_param=hubble_param)
        cooling_lum = state.cooling_luminosity(cooling_rate)

        assert np.sum(cooling_lum) == pytest.approx(-1.8049498952448396e+33)

    def test_hydo_state_temperature(self):
        """ calculate temperature using mu or Xh and fe """
        state = HydroState(density=1.0, mass=2.0, velocity=[1.0, 0.0, 0.5], specific_thermal_energy=0.5)

        temperature_ref = 2. / 3. * 0.5 * 0.5876821814762576 * constants.PROTONMASS / constants.BOLTZMANN
        temperature1 = state.temperature(mu=0.5876821814762576)
        temperature2 = state.temperature(hydrogen_fraction=0.76, electron_abundance=1.16)

        assert temperature1 == pytest.approx(temperature_ref)
        assert temperature2 == pytest.approx(temperature_ref)

    def test_hydro_state_electron_number_density(self):
        state = HydroState(density=1.0, mass=2.0, velocity=[1.0, 0.0, 0.5], specific_thermal_energy=0.5)
        electron_number_density_ref = 1.16 * 0.76 * 1.0 / constants.PROTONMASS
        electron_number_density = state.electron_number_density(hydrogen_fraction=0.76, electron_abundance=1.16)
        assert electron_number_density == pytest.approx(electron_number_density_ref)

    def test_hydro_state_pseudo_entropy(self):
        state = HydroState(density=1.0, mass=2.0, velocity=[1.0, 0.0, 0.5], specific_thermal_energy=0.5)
        temperature_ref = 2. / 3. * 0.5 * 0.5876821814762576 * constants.PROTONMASS / constants.BOLTZMANN
        electron_number_density_ref = 1.16 * 0.76 * 1.0 / constants.PROTONMASS
        entropy_ref = temperature_ref / electron_number_density_ref**(2./3.)
        entropy = state.pseudo_entropy(hydrogen_fraction=0.76, electron_abundance=1.16)
        assert entropy == pytest.approx(entropy_ref)


class TestMhdState(object):
    def test_mhd_state_init(self):
        MhdState()

        second_state = MhdState.from_conserved_variables(volume=2.0, mass=1.0, momentum=[0.5, 0.5, 1],
                                                         thermal_energy=2.0, conserved_magnetic_field=[0.6, 0.8, 0.2])
        third_state = MhdState.from_primitive_variables(volume=2.0, density=0.5, velocity=[0.5, 0.5, 1],
                                                        specific_thermal_energy=2.0, magnetic_field=[0.3, 0.4, 0.1])
        assert second_state == third_state

    def test_mhd_state_magic_methods(self):
        """
        magic (dunder) methods of a mhd state;

        note that __eq__ is checked already in initialization test
        :return:
        """
        state = MhdState(density=1.0, mass=2.0, velocity=[1.0, 0.0, 0.5],
                         specific_thermal_energy=0.5, magnetic_field=[2.0, 4.5, 1.0])

        # __str__
        print(state)

        # __repr__
        print(state.__repr__())

        state2 = MhdState(density=1.0, mass=2.0, velocity=[1.0, 0.0, 0.5],
                          specific_thermal_energy=0.25, magnetic_field=[2.0, 4.5, 1.0])
        assert not state == state2

    def test_mhd_state_properties(self):
        """
        calculate properties
        :return:
        """
        state = MhdState(density=1.0, mass=2.0, velocity=[1.0, 0.0, 0.5],
                         specific_thermal_energy=0.5, magnetic_field=[2.0, 1.0, 4.0],
                         lorentz_heaviside=True)

        # mass
        assert state.mass == pytest.approx(2.0)
        # density
        assert state.density == pytest.approx(1.0)
        # volume
        assert state.volume == pytest.approx(2.0)

        # velocity
        assert state.velocity == pytest.approx(np.array([[1.0, 0.0, 0.5]]))
        # absolute velocity
        assert state.absolute_velocity == pytest.approx(np.sqrt(1.25))
        # momentum
        assert state.momentum == pytest.approx(np.array([[2.0, 0.0, 1.0]]))
        # kinetic energy
        assert state.kinetic_energy == pytest.approx(1.25)

        # specific thermal energy
        assert state.specific_thermal_energy == pytest.approx(0.5)
        # thermal energy density
        assert state.thermal_energy_density == pytest.approx(0.5)
        # pressure
        assert state.thermal_pressure == pytest.approx(1.0 / 3.0)
        assert state.pressure == pytest.approx(1./3. + 10.5)
        # thermal energy
        assert state.thermal_energy == pytest.approx(1.0)
        # sound speed
        assert state.sound_speed == pytest.approx(np.sqrt(5. / 9. + 21.0))
        # mach number
        assert state.mach_number == pytest.approx(np.sqrt(1.25 / (5. / 9. + 21.0)))

        # magnetic field
        b_fld = np.sqrt(4.0 * np.pi) * np.array([[2.0, 1.0, 4.0]], ndmin=2)
        assert state.magnetic_field == pytest.approx(b_fld)

        assert state.magnetic_energy_density == pytest.approx(10.5)

        assert state.magnetic_energy == pytest.approx(21.0)

        assert state.alfven_velocity == pytest.approx(np.sqrt(21.0))

        assert state.alfvenic_mach_number == pytest.approx(np.sqrt(1.25 / 21.0))

        # total energy
        assert state.total_energy == pytest.approx(2.25 + 21)

    def test_mhd_state_projected_magnetic_field(self):
        state = MhdState(density=[0.5, 1.0], mass=[1.0, 2.0], velocity=[[1.0, 0.0, 0.5], [1.0, 0.0, 0.5]],
                         specific_thermal_energy=[0.1, 0.5], magnetic_field=[[2.0, 1.0, 2.0], [2.0, 1.0, 4.0]],
                         lorentz_heaviside=False)

        direction = np.array([0, 0, 1])
        assert state.projected_magnetic_field(direction=direction) == pytest.approx([2.0, 4.0])
