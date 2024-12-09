# /biosed/masking.py
# Module for managing detector image masks.
#
#
# Copyright (C) 2024 Tine Kalac
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from biosed.config import config

import numpy as np


# Default mask for the Cheetah camera
default_mask = np.zeros((512, 512), dtype = 'bool')
default_mask[255:257,:] = True
default_mask[:,255:257] = True


def mask_data(sed_data):
    full_mask = np.tile(config.get("masking.detector_mask")[np.newaxis, :], (sed_data.shape[0], 1, 1))
    sed_data[full_mask] = config.get("masking.masking_value")
    sed_data = np.ma.masked_array(sed_data, full_mask)
    return sed_data
