# /biosed/config.py
# Configuration.
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

import copy
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from cmcrameri import cm


# Default masks
no_mask = np.zeros((512, 512), dtype = 'bool')

cheetah_mask = np.copy(no_mask)
cheetah_mask[255:257, :]=True
cheetah_mask[:, 255:257]=True


class Config:
    """
    A class to manage the global configuration settings for the biosed package.
    """

    _defaults = {
        "preprocess": {
            "direct_beam_threshold": 100,
            "trim_radius": 100,
        },

        "masking": {
            "detector_mask": cheetah_mask,
            "masking_value" : -1,
        },

        "integration": {
            "n_phi_bins": 120,              # Binning factor for radial/sector integration
            "q_range": (1.2, 2.7),			# Q range of the crown integration
            "q_callibration": 2.55/70		# nm⁻1/pixel
        },

        "orientation": {
            "weighing_exponent": 1.0,
            "fit_eta_limits": (0.005, 0.95),
            "method": "harmonic_analysis",
        },

        "visualize": {
            "detector_plot_norm": Normalize(vmin=0, vmax=40),
            "detector_plot_cmap": "turbo",
            "detector_plot_origin": "lower",
            "detector_plot_aspect": "equal",

            "orientation_plot_norm": Normalize(vmin=0, vmax=np.pi),
            "orientation_plot_cmap": cm.romaO,
            "orientation_plot_aspect": "equal",

            "orientation_offset": 0.5 * np.pi, # valid for cellulose
        },

        "analyze": {
            "orientation_method" : "harmonic_analysis",  # "harmonic_analysis", "argmax", "model_fitting"
        },
    }

    def __init__(self):
        # Store the current configuration, initially set to defaults
        self._config = copy.deepcopy(self._defaults)

    def get(self, key):
        """
        Get the current value of a configuration key.

        Parameters
        ----------
        key : str
            Dot-separated key to retrieve a configuration value
            (e.g., "integration.default_method").

        Returns
        -------
        value : any
            The current value of the requested configuration key.

        Raises
        ------
        KeyError
            If the key does not exist.
        """
        keys = key.split(".")
        config = self._config
        for k in keys:
            if k not in config:
                raise KeyError(f"Configuration key '{key}' not found.")
            config = config[k]
        return config

    def set(self, key, value):
        """
        Set the value of a configuration key.

        Parameters
        ----------
        key : str
            Dot-separated key to set a configuration value
            (e.g., "integration.default_method").
        value : any
            The new value for the configuration key.

        Raises
        ------
        KeyError
            If the key does not exist.
        """
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                raise KeyError(f"Configuration key '{key}' not found.")
            config = config[k]
        if keys[-1] not in config:
            raise KeyError(f"Configuration key '{key}' not found.")
        config[keys[-1]] = value

    def reset(self, key=None):
        """
        Reset a configuration key or all configurations to their default values.

        Parameters
        ----------
        key : str, optional
            Dot-separated key to reset a specific configuration value.
            If not provided, resets all configurations.
        """
        if key is None:
            self._config = copy.deepcopy(self._defaults)
        else:
            keys = key.split(".")
            config = self._config
            default_config = self._defaults
            for k in keys[:-1]:
                if k not in config or k not in default_config:
                    raise KeyError(f"Configuration key '{key}' not found.")
                config = config[k]
                default_config = default_config[k]
            if keys[-1] not in default_config:
                raise KeyError(f"Configuration key '{key}' not found.")
            config[keys[-1]] = default_config[keys[-1]]

    def get_all(self):
        """
        Get the entire configuration dictionary.

        Returns
        -------
        dict
            A deep copy of the current configuration.
        """
        return copy.deepcopy(self._config)

# Create a global instance of Config
config = Config()
