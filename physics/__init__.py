#! /physics/__init__.py
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

from . import constants
from . import cosmology
from . import hydro
from . import units


def print_copyright():
    """
    prints copyright for physics module
    :return:
    """
    cpyright = "This is a physics package for python\n" \
               "Copyright (C) 2020-2022  Rainer Weinberger\n" \
               "\n" \
               "This program is free software: you can redistribute it and/or modify\n"\
               "it under the terms of the GNU General Public License as published by\n"\
               "the Free Software Foundation, either version 3 of the License, or\n"\
               "(at your option) any later version.\n"\
               "\n"\
               "This program is distributed in the hope that it will be useful,\n"\
               "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"\
               "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"\
               "GNU General Public License for more details.\n"\
               "\n"\
               "You should have received a copy of the GNU General Public License\n"\
               "along with this program.  If not, see <https://www.gnu.org/licenses/>.\n"
    print(cpyright)

