# /biosed/utilities.py
# Module for internal utility functions.
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

import numpy as np

from .config import config

class FormatDataShape():

	def __init__(self, data_shape):
		"""
		Helper class for linearizing and reformatting stacks of data. Linearization
		is benefitial for performance.
		
		Parameters
	    ----------
	    data_shape : tuple
	        The real-space dimensions of the data (Y, X). Can be 2D for formatted
	        data, or 1D for linear stack of data.

	    Returns
	    -------
	    None

	    Examples
	    --------
	    >>> format_shape = FormatDataShape(image_stack.shape[:-2])
	    >>> image_stack = format_shape.to_1D(image_stack)  # get linear stack
	    >>> image_stack = format_shape.to_2D(image_stack)  # get reformatted stack
		"""

		self.get_shape_2D = data_shape
		self.get_shape_1D = (int(np.prod(self.get_shape_2D)),)

	def to_1D(self, arr):
		if arr.ndim in [2, 3, 4]:	
			return arr.reshape((*self.get_shape_1D,
								*arr.shape[len(self.get_shape_2D):]))
		else:		
			raise Exception("Invalid array shape. Expected 2D, 3D, or 4D.")

	def to_2D(self, arr):
		if arr.ndim in [1, 2, 3]:
			return arr.reshape((*self.get_shape_2D, *arr.shape[1:]))
		else:		
			raise Exception("Invalid array shape. Expected 1D, 2D, or 3D.")
