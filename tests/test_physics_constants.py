#! /tests/test_physics_constants.py
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


def test_PhysicsConstantsLoadable():
    from physics import constants
    assert constants.CLIGHT == pytest.approx(2.99792458e10)
