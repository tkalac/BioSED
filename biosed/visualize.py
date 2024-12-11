# /biosed/visualize.py
# Module for the visualization.
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
import matplotlib.pyplot as plt
import matplotlib.colors as plt_col
from matplotlib.colors import Normalize
from cmcrameri import cm


def detector_plot(angles,
				  norm = config.get('visualize.detector_plot_norm'),
				  cmap = config.get('visualize.detector_plot_cmap'),
				  origin = config.get('visualize.detector_plot_origin'),
				  aspect = config.get('visualize.detector_plot_aspect')):

	plt.imshow(angles, norm = norm, cmap = cmap,
				origin = origin, aspect = aspect)


plot_orientation_kw = {
	"norm" : config.get('visualize.orientation_plot_norm'),
	"cmap" : config.get('visualize.orientation_plot_cmap'),
	"origin" : config.get('visualize.orientation_plot_origin'),
	"aspect" : config.get('visualize.orientation_plot_aspect')
}

def get_c_wheel(rotation = 0,
                resolution = 1024):
    c_wheel = np.arange(-resolution/2, resolution/2)
    c_wheel_X, c_wheel_Y = np.meshgrid(c_wheel, c_wheel, indexing = "xy")
    
    c_wheel_R = np.sqrt((c_wheel_X/2)**2 + (c_wheel_Y/2)**2)
    c_wheel_phi = np.arctan2(c_wheel_Y, c_wheel_X) + rotation
    c_wheel_phi[c_wheel_phi <= 0] += 2*np.pi
    c_wheel_phi = c_wheel_phi%np.pi
    
    c_wheel_phi[(c_wheel_R > resolution * 0.24)] = np.nan
    c_wheel_phi[(c_wheel_R < resolution * 0.08)] = np.nan

    return c_wheel_phi

def plot_orientation(angles,
					 alpha = 1,
					 cmap_rotation = 0,
					 orientation_offset = config.get("visualize.orientation_offset")):

	# Create the figure
	fig = plt.figure()

	# Add the main image on the left
	left, bottom, width, height = 0.1, 0.1, 0.6, 0.8
	ax = fig.add_axes([left, bottom, width, height])
	ax.imshow(angles, alpha = alpha, **plot_orientation_kw)
	ax.axis(False)

	ax_cm = fig.add_axes([left + width, bottom, 0.2, height * 0.3])
	ax_cm.imshow(get_c_wheel(), **plot_orientation_kw)
	ax_cm.axis(False)

# To do:
# def mean_detector_plot():
