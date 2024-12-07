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


# Set the default mask for the Cheetah camera
default_mask = np.zeros((512, 512), dtype = 'bool')
default_mask[255:257,:] = True
default_mask[:,255:257] = True

"""
class DetectorMask:
    # Masked values (excluded from analysis) are true
    
    if _mask == None:
    	_mask = default_mask

    @classmethod
    def set_detector_mask(cls, mask: np.ndarray):
        if not isinstance(mask, np.ndarray):
            raise TypeError("Detector mask must be a 2D numpy array.")
        if mask.ndim != 2:
            raise ValueError("Detector mask must be a 2D numpy array.")
        cls._mask = mask

    @classmethod
    def get_detector_mask(cls):
        if cls._mask is None:
            raise ValueError("No detector mask has been set.")
        return cls._mask
"""

def mask_data(sed_data):
    full_mask = np.tile(config.get("masking.detector_mask")[np.newaxis, :], (sed_data.shape[0], 1, 1))
    sed_data[full_mask] = config.get("masking.masking_value")
    return sed_data
