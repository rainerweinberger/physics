#! physics/test/test_physics_cosmology.py
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
from physics.cosmology import Cosmology
from physics.constants import CLIGHT, GRAVITY
from physics.units import MEGAPARSEC

class TestCosmology(object):
    def test_init(self):
        """
        initialization via constructor
        :return:
        """
        Cosmology(hubble_constant=0.7)

    def test_from_literature_factory(self):
        """
        ensure that all literature data sets are complete to create an instance
        :return:
        """
        for key in Cosmology.literature_values.keys():
            Cosmology.from_literature(key)

    def test_from_literature_does_not_exist(self):
        """
        call Cosmology.from_literature with invalid source
        :return:
        """
        with pytest.raises(NotImplementedError):
            Cosmology.from_literature(source="NonExistingData")

    def test_getattr(self):
        """
        get dictionary with literature values directly
        :return:
        """
        my_cosmology = Cosmology()
        print(my_cosmology.wmap9)
        print(my_cosmology._hubble_constant)  # make sure this is still possible
        print(my_cosmology['wmap9'])
        print(Cosmology().planck2016)

    def test_hubble_distance(self):
        """
        Hubble distance
        """
        h_0 = 1.0
        my_cosmology = Cosmology(hubble_constant=h_0)
        d_h = my_cosmology.hubble_distance

        H_0 = 100. * h_0 * 1.0e5 / MEGAPARSEC
        d_h_reference = CLIGHT / H_0

        assert d_h == pytest.approx(d_h_reference)

    def test_hubble_time(self):
        """
        Hubble time
        """
        h_0 = 1.0
        my_cosmology = Cosmology(hubble_constant=h_0)
        t_h = my_cosmology.hubble_time

        H_0 = 100. * h_0 * 1.0e5 / MEGAPARSEC
        t_h_reference = 1.0 / H_0

        assert t_h == pytest.approx(t_h_reference)

    def test_critical_density(self):
        """
        critical density
        """
        h_0 = 0.6774
        my_cosmology = Cosmology(hubble_constant=h_0)
        rho_c = my_cosmology.critical_density

        H_0 = 100. * h_0 * 1.0e5 / MEGAPARSEC
        rho_c_reference = 3.0 * H_0 * H_0 / 8.0 / np.pi / GRAVITY

        assert rho_c == pytest.approx(rho_c_reference)

        """
        ToDo: Omega curvature
        """

        """
        ToDo: peculiar velocity
        """

        """
        ToDo: redshift to scalefactor, scalefactor to redshift
        """

        """
        ToDo: E-factor
        """

        """
        ToDo: Comoving distance
        """

        """
        ToDo: Transverse comoving distance
        """

        """
        ToDo: Angular diameter distance
        """

        """
        ToDo: Luminosity distance
        """

        """
        ToDo: k-correction
        """

        """
        ToDo: distance modulus
        """

        """
        ToDo: Comoving volume
        """

        """
        ToDo: Lookback time
        """

        """
        ToDo: Probability of intersecting objects
        """
