# /biosed/io.py
# Module handles the import and output of data.
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
import h5py
import cv2 as cv
import os

def load_data(image_directory):
    """
    Loads the image data using OpenCV.
    Parameters
    ----------
    image_directory : string
        Directory of the detector images.

    Returns
    -------
    return_type
        Array containing the raw image data stack.

    Examples
    --------
    >>> data_directory = r"/home/tine/test_data/scan_01"
    >>> data = biosed.load_data(data_directory)
    """
    file_names = os.listdir(image_directory)
    file_names.sort()
    return np.array([cv.imread(os.path.join(image_directory, i),
        cv.IMREAD_UNCHANGED) for i in file_names], dtype = 'int16')


def save_to_hdf5(dataset, dataset_label, filename):
    """
    Save a dataset to an HDF5 file. If the file already exists, the dataset will be added.

    Parameters:
        dataset (numpy.ndarray): The data to save.
        dataset_label (str): The label to use for the dataset in the HDF5 file.
        filename (str): The name of the HDF5 file.

    Examples
    --------
    >>> data = biosed.load_data
    >>> intensity = biosed.poisson_model(phi, np.pi/4, 0.7, 2):
    >>> plt.plot(phi, intensity)
    """

    with h5py.File(filename, 'a') as h5file:
        if dataset_label in h5file:
            print(f"Dataset '{dataset_label}' already exists in {filename}. Overwriting it.")
            del h5file[dataset_label]  # Delete existing dataset with the same label
        # Create the dataset with the given label
        h5file.create_dataset(dataset_label, data=dataset)
        print(f"Dataset '{dataset_label}' saved to {filename}.")


def load_from_hdf5(filename, dataset_label = None):
    """
    Open an HDF5 file and retrieve dataset(s).

    Parameters:
        filename (str): The name of the HDF5 file to open.
        dataset_label (str, optional): The label of the specific dataset to retrieve. 
                                       If None, lists all available datasets.

    Returns:
        dict or numpy.ndarray: If `dataset_label` is provided, returns the dataset as a NumPy array.
                               If `dataset_label` is None, returns a dictionary of all datasets.
    """
    with h5py.File(filename, 'r') as h5file:
        if dataset_label:
            if dataset_label in h5file:
                print(f"Dataset '{dataset_label}' found in {filename}.")
                return h5file[dataset_label][...]  # Read the dataset as a NumPy array
            else:
                print(f"Dataset '{dataset_label}' not found in {filename}.")
                return None
        else:
            print(f"Datasets in {filename}:")
            dataset_dict = {}
            for key in h5file.keys():
                print(f" - {key}")
                dataset_dict[key] = h5file[key][...]  # Load all datasets
            return dataset_dict

