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

from biosed import io, masking, preprocess, integration, orientation, visualize
from .config import config

class AnalysisPipeline:

    def __init__(self):
        # Data
        self.data = None
        self.data_masked = None
        self.beam_centers = None
        self.valid_frames = None
        self.scan_shape = None
        self.data_valid = None
        self.data_trimmed = None
        self.beam_centers_valid = None
        self.azi_intensity_profiles = None
        self.orientation_image = None

        # Analysis parameters
        self.scan_lims = None
        self.orientation_method = config.get("analyze.orientation_method")
        self.azi_resolution = config.get("integration.n_phi_bins")

    def map_orientation(self, data_path, params=None):
        # Sanity checks
        if self.scan_lims == None:
            raise Exception("""Scan limits not determined. Please set 
                _.scan_lims tuple to (index_first_frame, index_last_frame).""")

        # Step 1: Load data
        print("Loading data...")
        self.data = io.load_data(data_path)
        print("...done!\n")

        # Step 2: Mask data
        print("Masking data...")
        self.data = masking.mask_data(self.data)  # Set invalid pixels to -1 and return masked array
        print("...done!\n")

        # Step 2: Find beam centers:
        print("Finding beam centers...")
        self.beam_centers = preprocess.find_beam_centers(self.data)
        print("...done!\n")

        # Step 3: Determine the shape of the scan
        print("Computing scan shape and reshaping data...")
        self.valid_frames, self.scan_shape = preprocess.get_scan_shape(self.beam_centers,
                                                                           self.scan_lims)
        print("...done!\n")

        # Step 4: Get stack of valid frames
        print("Determining scan shape...")
        self.data_valid_frames = self.data[self.valid_frames]
        self.beam_centers_valid = self.beam_centers[self.valid_frames]
        print("...done!\n")

        # Step 5: Trim the data
        print("Trimming data...")
        self.data_centered = preprocess.center_images(self.data_valid_frames,
                                                      self.beam_centers_valid)
        print("...done!\n")

        # Step 6: Integrate
        print("Integrating data...")
        self.azi_intensity_profiles, self.phi_values = integration.crown_integration(self.data_centered,
                                                                                    n_phi_bins = self.azi_resolution)
        print("...done!\n")

        # Step 7: Fit model
        print("Computing orientation...")
        if self.orientation_method == "harmonic_analysis":
            self.orientation_image = orientation.harmonic_analysis(self.azi_intensity_profiles,
                                                                        self.phi_values)
            print("...done!\n")
            visualize.plot_orientation(self.orientation_image.reshape(self.scan_shape))

        elif self.orientation_method == "model_fitting":
            self.orientation_image = orientation.fit_poisson_odf(self.azi_intensity_profiles,
                                                                        self.phi_values)
            print("...done!\n")
            visualize.plot_orientation(self.orientation_image[:,0].reshape(self.scan_shape))
        
        elif self.orientation_method == "argmax":
            self.orientation_image = orientation.find_orientation_peaks(self.azi_intensity_profiles,
                                                                        self.phi_values)
            print("...done!\n")
            visualize.plot_orientation(self.orientation_image.reshape(self.scan_shape))
        else:
            raise ValueError(f"Unknown orientation method: {self.orientation_method}")

        return self


    def get(self, step_name):
        if step_name == "data":
            return self.data
        elif step_name == "beam_centers":
            return self.beam_centers
        elif step_name == "scan_lims":
            return self.scan_lims
        elif step_name == "valid_frames":
            return self.valid_frames
        elif step_name == "scan_shape":
            return self.scan_shape
        elif step_name == "data_valid":
            return self.data_valid
        elif step_name == "data_centered":
            return self.data_centered
        elif step_name == "beam_centers_valid":
            return self.beam_centers_valid
        elif step_name == "azi_intensity_profiles":
            return self.azi_intensity_profiles
        elif step_name == "orientation_image":
            return self.orientation_image
        else:
            raise ValueError(f"Unknown step: {step_name}")
