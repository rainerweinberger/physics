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
from physics.units import MEGAPARSEC, SEC_PER_GIGAYEAR


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

    def test_omega_curvature(self):
        """
        Omega curvature
        """
        o_m = 0.20
        o_l = 0.72
        my_cosmology = Cosmology(omega_matter=o_m, omega_lambda=o_l)
        o_k = my_cosmology.omega_curvature

        o_k_reference = 1.0 - o_m - o_l
        assert o_k == pytest.approx(o_k_reference)

    def test_peculiar_velocity(self):
        """
        peculiar velocity
        """
        vpec = Cosmology.peculiar_velocity(redshift=1.0, redshift_observer=0.5)

        vpec_reference = CLIGHT * 0.250
        assert vpec == pytest.approx(vpec_reference)

    def test_redshift_scalefactor_conversion(self):
        """
        redshift to scalefactor, scalefactor to redshift
        """
        redshift_reference = 2.4633
        scalefactor = Cosmology.scalefactor(redshift=redshift_reference)
        redshift = Cosmology.redshift(scalefactor=scalefactor)

        scalefactor_reference = 1.0 / (1.0 + redshift_reference)
        assert scalefactor == pytest.approx(scalefactor_reference)
        assert redshift == pytest.approx(redshift_reference)

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

    def test_lookback_time(self):
        """
        Lookback time; compare to some pre-calculated ones (used yt to calculate it)
        """
        redshifts = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 2.25, 3.75, 4.34, 5.0, 7.0, 1100.0])
        lookback_ref_p16 = np.array(
            [0.0, 3.0351346358407234, 5.193918814993813, 6.7653096884483475, 7.93569077168917, 10.87878447278143,
             12.136004047592554, 12.403029538446914, 12.62666394724965, 13.03808995523595, 13.802244387905864])

        # # comparison with yt
        # import yt.utilities.cosmology as yt_cosmo
        # mycosmo = Cosmology()['planck2016']
        # cosmo = yt_cosmo.Cosmology(hubble_constant=mycosmo['hubble_constant'],
        #                            omega_matter=mycosmo['omega_matter'],
        #                            omega_lambda=mycosmo['omega_lambda'],
        #                            omega_curvature=0.0,
        #                            unit_registry=None,
        #                            unit_system="cgs",
        #                            use_dark_factor=False,
        #                            w_0=-1.0,
        #                            w_a=0.0)
        # for redshift in redshifts:
        #     print('redshift: ', redshift, 'lookback: ',
        #           (cosmo.t_from_z(0.0).in_units('Gyr') - cosmo.t_from_z(redshift).in_units('Gyr')))

        cosmo2 = Cosmology.from_literature(source="planck2016")
        lookback = []
        for redshift in redshifts:
            lookback.append(cosmo2.look_back_time(redshift)/SEC_PER_GIGAYEAR)

        assert np.array(lookback) == pytest.approx(lookback_ref_p16)

        """
        ToDo: Probability of intersecting objects
        """
