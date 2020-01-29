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
from physics.units import PARSEC, MEGAPARSEC, SEC_PER_GIGAYEAR

FloatType = np.float64


class TestCosmology(object):
    def test_init(self):
        """
        initialization via constructor
        :return:
        """
        Cosmology(hubble_constant=FloatType(0.7))

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
        h_0 = FloatType(1.0)
        my_cosmology = Cosmology(hubble_constant=h_0)
        d_h = my_cosmology.hubble_distance

        hubble_0 = 100. * h_0 * 1.0e5 / MEGAPARSEC
        d_h_reference = CLIGHT / hubble_0
        assert d_h == pytest.approx(d_h_reference)

    def test_hubble_time(self):
        """
        Hubble time
        """
        h_0 = FloatType(1.0)
        my_cosmology = Cosmology(hubble_constant=h_0)
        t_h = my_cosmology.hubble_time

        hubble_0 = 100. * h_0 * 1.0e5 / MEGAPARSEC
        t_h_reference = 1.0 / hubble_0
        assert t_h == pytest.approx(t_h_reference)

    def test_critical_density(self):
        """
        critical density
        """
        h_0 = FloatType(0.6774)
        my_cosmology = Cosmology(hubble_constant=h_0)
        rho_c = my_cosmology.critical_density

        hubble_0 = 100. * h_0 * 1.0e5 / MEGAPARSEC
        rho_c_reference = 3.0 * hubble_0 * hubble_0 / 8.0 / np.pi / GRAVITY
        assert rho_c == pytest.approx(rho_c_reference)

    def test_omega_curvature(self):
        """
        Omega curvature
        """
        o_m = FloatType(0.20)
        o_l = FloatType(0.72)
        my_cosmology = Cosmology(omega_matter=o_m, omega_lambda=o_l)
        o_k = my_cosmology.omega_curvature

        o_k_reference = 1.0 - o_m - o_l
        assert o_k == pytest.approx(o_k_reference)

    def test_peculiar_velocity(self):
        """
        peculiar velocity
        """
        v_pec = Cosmology.peculiar_velocity(redshift=FloatType(1.0),
                                            redshift_observer=FloatType(0.5))

        v_pec_reference = CLIGHT * 0.250
        assert v_pec == pytest.approx(v_pec_reference)

    def test_redshift_scalefactor_conversion(self):
        """
        redshift to scalefactor, scalefactor to redshift
        """
        redshift_reference = FloatType(2.4633)
        scalefactor = Cosmology.scalefactor(redshift=redshift_reference)
        redshift = Cosmology.redshift(scalefactor=scalefactor)

        scalefactor_reference = 1.0 / (1.0 + redshift_reference)
        assert scalefactor == pytest.approx(scalefactor_reference)
        assert redshift == pytest.approx(redshift_reference)

    def test_comoving_distance_line_of_sight(self):
        """
        Comoving distance
        """
        redshift = FloatType(4.69)
        ref_distance = FloatType(2.401637826144779e+28)  # cgs
        cosmo = Cosmology.from_literature('planck2016')
        assert cosmo.comoving_distance_line_of_sight(redshift) == pytest.approx(ref_distance, rel=0.001)

    def test_comoving_distance_transverse(self):
        """
        Transverse comoving distance
        """
        redshift = FloatType(4.69)
        ref_distance1 = FloatType(2.3774954250747463e+28)  # cgs # Om=0.3, Ol=0.6, Ok=0.1 h=0.7
        ref_distance2 = FloatType(2.3024772762163336e+28)  # cgs # Om=0.3, Ol=0.8, Ok=-0.1 h=0.7
        ref_distance3 = FloatType(2.3464085554485967e+28)  # cgs # Om=0.3, Ol=0.7, Ok=0.0 h=0.7

        cosmo1 = Cosmology(omega_lambda=FloatType(0.6),
                           omega_matter=FloatType(0.3),
                           hubble_constant=FloatType(0.7))
        cosmo2 = Cosmology(omega_lambda=FloatType(0.8),
                           omega_matter=FloatType(0.3),
                           hubble_constant=FloatType(0.7))
        cosmo3 = Cosmology(omega_lambda=FloatType(0.7),
                           omega_matter=FloatType(0.3),
                           hubble_constant=FloatType(0.7))
        assert cosmo1.comoving_distance_transverse(redshift) == pytest.approx(ref_distance1, rel=0.001)
        assert cosmo2.comoving_distance_transverse(redshift) == pytest.approx(ref_distance2, rel=0.001)
        assert cosmo3.comoving_distance_transverse(redshift) == pytest.approx(ref_distance3, rel=0.001)

    def test_angular_diameter_distance(self):
        """
        Angular diameter distance
        """
        redshift = FloatType(4.69)
        ref_distance = FloatType(4.2208046153686803e+27)  # cgs
        cosmo = Cosmology.from_literature('planck2016')
        assert cosmo.angular_diameter_distance(redshift) == pytest.approx(ref_distance, rel=0.001)

    def test_luminosity_distance(self):
        """
        Luminosity distance
        """
        redshift = FloatType(4.69)
        ref_distance = FloatType(1.3665319230763794e+29)  # cgs
        cosmo = Cosmology.from_literature('planck2016')
        assert cosmo.luminosity_distance(redshift) == pytest.approx(ref_distance, rel=0.001)

    def test_distance_modulus(self):
        """
        distance modulus
        """
        redshift = FloatType(4.69)
        ref_distance = FloatType(1.3665319230763794e+29)  # cgs
        ref_dm = 5. * np.log10(ref_distance / 10. / PARSEC)
        cosmo = Cosmology.from_literature('planck2016')
        assert cosmo.distance_modulus(redshift) == pytest.approx(ref_dm, rel=0.001)

    def test_comoving_volume(self):
        """
        Comoving volume
        note: the values with curvature (1 and 2) differ from yt solutions;
        implementation follows Hogg (1999)

        """
        redshift = FloatType(4.69)
        ref_volume1 = FloatType(5.162427665378852e+85)  # cgs # Om=0.3, Ol=0.6, Ok=0.1 h=0.7
        ref_volume2 = FloatType(5.674663557429024e+85)  # cgs # Om=0.3, Ol=0.8, Ok=-0.1 h=0.7
        ref_volume3 = FloatType(5.411273810433006e+85)  # cgs # Om=0.3, Ol=0.7, Ok=0.0 h=0.7

        cosmo1 = Cosmology(hubble_constant=0.7, omega_lambda=0.6, omega_matter=0.3)
        cosmo2 = Cosmology(hubble_constant=0.7, omega_lambda=0.8, omega_matter=0.3)
        cosmo3 = Cosmology(hubble_constant=0.7, omega_lambda=0.7, omega_matter=0.3)
        assert cosmo1.comoving_volume(redshift) == pytest.approx(ref_volume1, rel=0.001)
        assert cosmo2.comoving_volume(redshift) == pytest.approx(ref_volume2, rel=0.001)
        assert cosmo3.comoving_volume(redshift) == pytest.approx(ref_volume3, rel=0.001)

    def test_lookback_time(self):
        """
        Lookback time; compare to some pre-calculated ones (used yt to calculate it)
        """
        redshifts = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 2.25, 3.75, 4.34, 5.0, 7.0, 1100.0], dtype=FloatType)
        lookback_ref_p16 = np.array(
            [0.0, 3.0351346358407234, 5.193918814993813, 6.7653096884483475, 7.93569077168917, 10.87878447278143,
             12.136004047592554, 12.403029538446914, 12.62666394724965, 13.03808995523595, 13.802244387905864],
            dtype=FloatType)

        cosmo = Cosmology.from_literature(source="planck2016")
        lookback = []
        for redshift in redshifts:
            lookback.append(cosmo.lookback_time(redshift) / SEC_PER_GIGAYEAR)

        assert np.array(lookback) == pytest.approx(lookback_ref_p16)
