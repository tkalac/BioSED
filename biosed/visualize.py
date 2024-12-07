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

def plot_orientation(angles,
					 cmap_offset = 0,
					 orientation_offset = config.get("visualize.orientation_offset")):

	fig, ax = plt.subplots(1, 2, figsize = (12, 8))

	ax[0].imshow((angles + cmap_offset)%np.pi, **plot_orientation_kw)

	ax[1].axis(False)

	# Plotting the scale ring
	ax[1] = fig.add_subplot(122, projection='polar')
	col_map = cm.romaO(np.linspace(0, 1, 200))	
	upper_lobe = (np.linspace(0, np.pi, 200) + orientation_offset + cmap_offset) %np.pi
	lower_lobe = upper_lobe + np.pi

	ax[1].vlines(upper_lobe, 0.5*np.ones(200), np.ones(200), color = col_map, lw = 5)
	ax[1].vlines(lower_lobe, 0.5*np.ones(200), np.ones(200), color = col_map, lw = 5)
	ax[1].axis(False)
	ax[1].set(ylim = (0, 2))

# To do:
# def mean_detector_plot():
