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


class TestHydro(object):
    def test_hydro_state_init(self):
        """
        initialization
        :return:
        """
        HydroState()

    def test_hydro_state_magic_methods(self):
        """
        magic (dunder) methods of a hydro state
        :return:
        """
        state = HydroState(density=1.0, mass=2.0, velocity=[1.0, 0.0, 0.5], specific_thermal_energy=0.5)

        # __str__
        print(state)

        # __repr__
        print(state.__repr__())

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
        # momentum
        assert state.momentum == pytest.approx(np.array([[2.0, 0.0, 1.0]]))
        # kinetic energy
        assert state.kinetic_energy == pytest.approx(1.25)

        # specific thermal energy
        assert state.specific_thermal_energy == pytest.approx(0.5)
        # thermal energy density
        assert state.thermal_energy_density == pytest.approx(0.5)
        # pressure
        assert state.pressure == pytest.approx(1.0/3.0)
        # thermal energy
        assert state.thermal_energy == pytest.approx(1.0)

        # total energy
        assert state.total_energy == pytest.approx(2.25)

    # ToDo: cooling luminosity