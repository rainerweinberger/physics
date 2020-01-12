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
from physics.cosmology import Cosmology


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