# /biosed/analyze.py
# Analysis pipelines for orientation mapping.
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

from biosed import io, masking, preprocess, integration, orientation, visualize, utilities
from .config import config


class AnalysisPipeline:

    def __init__(self):
        # Data
        self.data = None
        self.data_masked = None
        self.beam_centers = None
        self.valid_frames = None
        self.scan_shape = None
        self.data_trimmed = None
        self.format_shape = None
        self.beam_centers_valid = None
        self.azi_intensity_profiles = None
        self.orientation_image = None

        # Analysis parameters
        self.scan_limits = None
        self.orientation_method = config.get("analyze.orientation_method")
        self.azi_resolution = config.get("integration.n_phi_bins")

    def load_data(self, data_directory):
        print("Loading data...")
        self.data = io.load_data(data_directory)
        print("...done!\n")

        print("Masking data...")
        self.data = masking.mask_data(self.data)  # Set invalid pixels to -1 and return masked array
        print("...done!\n")

        print("Finding beam centers...")
        self.beam_centers = preprocess.find_beam_centers(self.data)
        print("...done!\n")

    def map_orientation(self, data_directory = None):

        # Step 1: Load and mask data
        if (self.data is None) and (data_directory is None):
            raise Exception("""Please load data or specify data_directory""")        
        elif (self.data is None) and (data_directory is not None):
            self.load_data(data_directory)
        else:
            pass

        # Step 3: Determine the shape of the scan
        if self.scan_limits is None:
            raise Exception("""Scan limits not set. They can be determined using .get("beam centers")""")

        print("Computing scan shape...")
        self.valid_frames, self.scan_shape = preprocess.get_scan_shape(self.beam_centers,
                                                                           self.scan_limits)
        self.format_shape = utilities.FormatDataShape(self.scan_shape)

        print("...done!\n")

        # Step 5: Trim the data
        print("Trimming data...")
        self.data_centered = preprocess.center_images(self.data[self.valid_frames],
                                                      self.beam_centers[self.valid_frames])
        print("...done!\n")

        # Step 6: Integrate
        print("Integrating data...")
        self.azi_intensity, self.phi_values = integration.crown_integration(self.data_centered,
                                                                            n_phi_bins = self.azi_resolution)
        print("...done!\n")

        # Step 7: Fit model
        print("Computing orientation...")
        if self.orientation_method == "harmonic_analysis":
            self.orientation_map = orientation.harmonic_analysis(self.azi_intensity,
                                                                        self.phi_values)
            print("...done!\n")
            visualize.plot_orientation(self.format_shape.to_2D(self.orientation_map))

        elif self.orientation_method == "model_fitting":
            self.orientation_map = orientation.fit_poisson_odf(self.azi_intensity,
                                                                        self.phi_values)
            print("...done!\n")
            visualize.plot_orientation(self.format_shape.to_2D(self.orientation_map[:,0]))
        
        elif self.orientation_method == "argmax":
            self.orientation_map = orientation.find_orientation_peaks(self.azi_intensity,
                                                                        self.phi_values)
            print("...done!\n")
            visualize.plot_orientation(self.format_shape.to_2D(self.orientation_map))
        else:
            raise ValueError(f"Unknown orientation method: {self.orientation_method}")

        return self


    def get(self, step_name):
        if step_name == "data":
            return self.data
        elif step_name == "beam centers":
            return self.beam_centers
        elif step_name == "scan limits":
            return self.scan_limits
        elif step_name == "valid frames":
            return self.valid_frames
        elif step_name == "scan shape":
            return self.scan_shape
        elif step_name == "centered data":
            return self.format_shape.to_2D(self.data_centered)
        elif step_name == "azint profiles":
            return self.format_shape.to_2D(self.azi_intensity)
        elif step_name == "orientation map":
            return self.format_shape.to_2D(self.orientation_map)
        else:
            raise ValueError(f"Unknown step: {step_name}")
