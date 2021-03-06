#! pyhysics/cosmology/__init__.py
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
from __future__ import annotations
import numpy as np
from physics import constants, units


class Cosmology(object):
    literature_values = {'planck2018': {'omega_matter': np.float64(0.315823),
                                        'omega_lambda': np.float64(0.684097),
                                        'omega_baryon': np.float64(0.04927),
                                        'hubble_constant': np.float64(0.6732117),
                                        'sigma_8': np.float64(0.8101),
                                        'spectral_index': np.float64(0.9652)
                                        },  # Planck+2018
                         'planck2016': {'omega_matter': np.float64(0.3089),
                                        'omega_lambda': np.float64(0.6911),
                                        'omega_baryon': np.float64(0.0486),
                                        'hubble_constant': np.float64(0.6774),
                                        'sigma_8': np.float64(0.8159),
                                        'spectral_index': np.float64(0.9667)
                                        },  # Springel+ 2018 (IllustrisTNG)
                         'wmap9': {'omega_matter': np.float64(0.2726),
                                   'omega_lambda': np.float64(0.7274),
                                   'omega_baryon': np.float64(0.0456),
                                   'hubble_constant': np.float64(0.6774),
                                   'sigma_8': np.float64(0.704),
                                   'spectral_index': np.float64(0.963)
                                   },  # Vogelsberger+ 2014 (Illustris)
                         'wmap7': {'omega_matter': np.float64(0.27),
                                   'omega_lambda': np.float64(0.73),
                                   'omega_baryon': np.float64(0.045),
                                   'hubble_constant': np.float64(0.7),
                                   'sigma_8': np.float64(0.8),
                                   'spectral_index': np.float64(0.95)
                                   },  # Vogelsberger+2012 (Moving-mesh cosmology)
                         'wmap1': {'omega_matter': np.float64(0.25),
                                   'omega_lambda': np.float64(0.75),
                                   'omega_baryon': np.float64(0.045),
                                   'hubble_constant': np.float64(0.73),
                                   'sigma_8': np.float64(0.9),
                                   'spectral_index': np.float64(1.0)
                                   },  # Springel+2005, 2008 (Millennium, Aquarius)
                         }
    """
        Class that handles all kind of cosmological unit conversions
        :type hubble_constant: numpy.float64
            Hubble parameter today in units of 100 km/s/Mpc
    """

    def __init__(self,
                 omega_matter: np.float64 = 0.3089,
                 omega_lambda: np.float64 = 0.6911,
                 omega_baryon: np.float64 = 0.0486,
                 hubble_constant: np.float64 = 0.6774,
                 sigma_8: np.float64 = 0.8159,
                 spectral_index: np.float64 = 0.9667
                 ):
        self._omega_matter = omega_matter
        self._omega_lambda = omega_lambda
        self._omega_baryon = omega_baryon
        self._hubble_constant = hubble_constant
        self._sigma_8 = sigma_8
        self._spectral_index = spectral_index

    @classmethod
    def from_literature(cls, source: str = 'Planck2016') -> Cosmology:
        """
            Factory for Cosmology class with one of the parameter sets in the literature
        :param source: string
            string describing the literature source;
        :return: class Cosmology
            Cosmology object with appropriate parameters
        """
        if source.lower() in cls.literature_values.keys():
            kwargs = cls.literature_values[source.lower()]
            return cls(**kwargs)
        else:
            print(f"{source} is unknown!")
            # convert keys object to list, to get a nicer display
            implemented = [key for key in cls.literature_values.keys()]
            print(f"Implemented literature values: {implemented}")
            raise NotImplementedError

    def __getattr__(self, item):
        """
        invoked only if normal lookup fails,
        requires __getitem__
        """
        return self.__getitem__(item)  # same as self[item]

    def __getitem__(self, item):
        return self.literature_values[item]

    @property
    def hubble_distance(self) -> np.float64:
        """
        :return: np.float64
            hubble distance in cm
        """
        return np.float64(0.1 * constants.CLIGHT / self._hubble_constant * units.PARSEC)

    @property
    def hubble_time(self) -> np.float64:
        """
        :return: np.float64
            hubble time in s
        """
        return np.float64(0.1 / self._hubble_constant * units.PARSEC)

    @property
    def critical_density(self) -> np.float64:
        """
        :return: np.float64
            critical density of the Universe
        """
        return np.float64(
            300.0 * self._hubble_constant * self._hubble_constant / 8.0 / np.pi / units.PARSEC / units.PARSEC / constants.GRAVITY)

    @property
    def omega_curvature(self) -> np.float64:
        """
        :return: np.float64
            curvature parameter
        """
        return np.float64(1.0 - self._omega_lambda - self._omega_matter)

    @staticmethod
    def peculiar_velocity(redshift: np.float64 = 1.0, redshift_observer: np.float64 = 0.0) -> np.float64:
        """
        :param redshift: np.float64
            redshift of emission
        :param redshift_observer: np.float64
            redshift of observer
        :return: np.float64
            peculiar velocity
        """
        return constants.CLIGHT * (redshift - redshift_observer) / (1.0 + redshift)

    @staticmethod
    def scalefactor(redshift: np.float64, scalefactor_reference: np.float64 = 1.0) -> np.float64:
        return np.float64(scalefactor_reference / (1.0 + redshift))

    @staticmethod
    def redshift(scalefactor: np.float64, scalefactor_reference: np.float64 = 1.0) -> np.float64:
        return np.float64(scalefactor_reference / scalefactor - 1.0)

    def _e_factor(self, redshift):
        redshift_plus_one = (redshift + 1.0)
        return np.sqrt(self._omega_matter * redshift_plus_one * redshift_plus_one * redshift_plus_one +
                       self.omega_curvature * redshift_plus_one * redshift_plus_one +
                       self._omega_lambda)

    def comoving_distance_line_of_sight(self, redshift: np.float64):
        redshift_array = np.linspace(0, redshift, 5000)
        integrand_array = 1. / self._e_factor(redshift_array)
        return self.hubble_distance * np.trapz(integrand_array, redshift_array)

    def comoving_distance_transverse(self, redshift: np.float64):
        if self.omega_curvature > 1e-12:
            sqrt_omega_c = np.sqrt(self.omega_curvature)
            fac = np.sinh(sqrt_omega_c * self.comoving_distance_line_of_sight(redshift) / self.hubble_distance)
            return self.hubble_distance * fac / sqrt_omega_c
        elif self.omega_curvature < -1e-12:
            sqrt_omega_c = np.sqrt(-self.omega_curvature)
            fac = np.sin(sqrt_omega_c * self.comoving_distance_line_of_sight(redshift) / self.hubble_distance)
            return self.hubble_distance * fac / sqrt_omega_c
        else:
            return self.comoving_distance_line_of_sight(redshift)

    def angular_diameter_distance(self, redshift: np.float64):
        return self.comoving_distance_transverse(redshift) / (1.0 + redshift)

    def luminosity_distance(self, redshift: np.float64):
        return (1.0 + redshift) * self.comoving_distance_transverse(redshift)

    def distance_modulus(self, redshift: np.float64):
        return 5.0 * np.log10(self.luminosity_distance(redshift) / 10. / units.PARSEC)

    def comoving_volume(self, redshift: np.float64):
        d_m = self.comoving_distance_transverse(redshift)
        if self.omega_curvature > 1e-12:
            d_h = self.hubble_distance
            sqrt_omega_c = np.sqrt(self.omega_curvature)
            prefac = 2.0 * np.pi * d_h * d_h * d_h / self.omega_curvature
            term1 = d_m / d_h * np.sqrt(1.0 + self.omega_curvature * d_m / d_h * d_m / d_h)
            term2 = 1.0 / sqrt_omega_c * np.arcsinh(sqrt_omega_c * d_m / d_h)
            return prefac * (term1 - term2)
        elif self.omega_curvature < -1e-12:
            d_h = self.hubble_distance
            sqrt_omega_c = np.sqrt(-self.omega_curvature)
            prefac = 2.0 * np.pi * self.hubble_distance ** 3 / self.omega_curvature
            term1 = d_m / d_h * np.sqrt(1.0 + self.omega_curvature * d_m / d_h * d_m / d_h)
            term2 = 1.0 / sqrt_omega_c * np.arcsin(sqrt_omega_c * d_m / d_h)
            return prefac * (term1 - term2)
        else:
            return 4.0 * np.pi * d_m * d_m * d_m / 3.0

    def _integrand_look_back_time(self, a):
        integrand = 1. / a / self._hubble_constant / units.HUBBLE
        integrand /= np.sqrt(self._omega_matter / a / a / a + self.omega_curvature / a / a + self._omega_lambda)
        return integrand

    def lookback_time(self, redshift: np.float64):
        """
        :param redshift: np.float64
        :return: np.float64
            time of a given redshift to redshift 0 in seconds
        """
        scalefactor = self.scalefactor(redshift)
        a_array = np.linspace(scalefactor, 1.0, 5000)
        integrand_array = self._integrand_look_back_time(a_array)
        return np.trapz(integrand_array, a_array)
