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


class Cosmology(object):
    literature_values = {'planck2016': {'omega_matter': np.float64(0.3089),
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