# /biosed/preprocess.py
# Centering and trimming of SED datasets.
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

from biosed.config import config
from biosed.utilities import FormatDataShape
from ._cpp.center_of_mass import compute_centers_of_mass    # C++ extension

def find_beam_centers(sed_data,
					  direct_beam_threshold = config.get("preprocess.direct_beam_threshold")):
    """
    Find the beam centers for each detector image in a 1D stack.

    Parameters
    ----------
    sed_data : NumPy NDArray.
        1D stack of detector images (NumPy array). The shape is taken as
        (n_images, detector_QY, detector_QX).
    direct_beam_threshold : int, optional.
        Only pixels with intensity above this threshold will be considered
        for the calculation of the beam centers.

    Returns
    -------
    NumPy NDArray
        Returns the coordinates of the center of the beam. The array is
        indexed as (n_images, beam_center_QY, beam_center_QX)

    Raises
    ------
    ExceptionType
        Description of conditions when this exception is raised.

    Notes
    -----
    The function calls the C++ extension for calculating the centers of mass
    of the stack of images for memory and speed efficiency. If the mask is
    not set, all pixel values will be used in the calculation.

    Examples
    --------
    >>> data_beamCenters = find_beam_centers(data, threshold = 150)
    >>> print(data_beamCenters)
    """

    if isinstance(sed_data, np.ma.MaskedArray):
        return compute_centers_of_mass(sed_data.data, direct_beam_threshold)
    else:
        return compute_centers_of_mass(sed_data, direct_beam_threshold)

    return compute_centers_of_mass(sed_data, direct_beam_threshold)


def get_scan_shape(beam_centers, scan_limits = None):
    """
    Function accepts the beam center data, computes which images are valid parts
    of the scan and determines the usable size of the scan. The outputs are used
    to turn the stack of images into a 2D array of images used for further
    processing.

    Parameters
    ----------
    beam_centers : NumPy Array (2D)
        A stack of beam position coordinates. Size is
        (n_images, 2).
    set_scan_limits : tuple
        Tuple of the first and last indicies of the scan. This is determined 
        by plotting the beam positions.

    Returns
    -------
    tuple (valid_frames, scan_shape)
        Returns a boolean array valid_frames, which specifies
        the indicies of the valid scan frames, and the scan_shape of the final
        indexed as (scan_dimension_Y, scan_dimension_X).

    Notes
    -----
    It determines which frames are valid from the gradient of the beam position.


    Examples
    --------
    Examples of how to use the function.

    >>> valid_frames, scan_shape = get_scan_shape(data_beamCenters, (371, 3212))
    >>> data_reshaped = data[valid_frames].reshape(scan_shape)
    >>> print(data_reshaped.shape)
    """
    
    # Sanity checks
    if (scan_limits == None):
        raise ValueError("Please specify scan_limits.")
    elif (len(scan_limits) != 2):
        raise ValueError("""scan_limits should be a tuple of length 2 
            (indicies of the first and last valid frame).""")
    elif (beam_centers.ndim != 2) or (beam_centers.shape[1] != 2):
        raise ValueError("""beam_centers should be an array of beam center coordinates. 
            Indexing is [image_index, coordinate]. Coordinate is (center_y, center_x)""")

    valid_frames = np.zeros(len(beam_centers), dtype = 'bool')
    valid_frames[scan_limits[0]:scan_limits[1]] = True

    # The beam shifts are peaks.
    CoM_gradient = np.gradient(beam_centers[:,1])

    # We want to exclude the points where the beam is going backwards. The
    # threshold is 0.5 instead of 0 bc there is some noise.
    valid_frames[np.where(CoM_gradient > 2)[0]] = False

    # The next part determines the lengths of the consecutive valid frames
    # which count as one row of scan images.
    changes = np.diff(valid_frames.astype(int))

    # Start indices of True segments (+1 accounts for the shift due to diff)
    starts = np.where(changes == 1)[0] + 1  
    # End indices of True segments (directly where changes go -1)
    ends = np.where(changes == -1)[0] + 1

    # Handle if the array starts or ends with True
    if valid_frames[0]:
        starts = np.insert(starts, 0, 0)
    if valid_frames[-1]:
        ends = np.append(ends, len(valid_frames))

    # Compute lengths of True segments
    lengths = ends - starts

    # Determine the minimum length of True segments
    min_length = lengths.min()

    # Truncate segments longer than the minimum length
    for start, end in zip(starts, ends):
        if end - start > min_length:
            valid_frames[start + min_length:end] = False

    return valid_frames, (len(lengths), int(min_length))


def center_images(sed_data, beam_centers,
				trimming_radius = config.get("preprocess.trim_radius")):
    """
    Function that trims the data array around the beam centres.

    Parameters
    ----------
    sed_data : NumPy Array (3D or 4D)
        Detector image stack. Indexing: (image_index, QY, QX)
    beam_centers : NumPy Array (2D or 3D)
        Array with beam center coordinates.

    Returns
    -------
    return_type
        Description of the returned value(s).

    Raises
    ------
    Exception
        Is raised when the dimensions of the image stack and beam center
        stacks do not match. The image stack and beam center coordinate
        arrays should have the same sized and should correspond to each
        other index-wise.

    Notes
    -----
    It can fail if the trimmed images include indicies outside of the detector.

    Examples
    --------
    Examples of how to use the function.

    >>> data_trimmed = trim_images(data, data_beamCenters)
    >>> biosed.plot_mean(data_trimmed[5:10, 8:18])
    """

    format_data_shape = FormatDataShape(sed_data.shape[:-2])
    sed_data = format_data_shape.to_1D(sed_data)

    format_beam_center_shape = FormatDataShape(beam_centers.shape[:-1])
    beam_centers = format_beam_center_shape.to_1D(beam_centers)

    if sed_data.shape[0] != beam_centers.shape[0]:
        raise Exception("Number of images does not match number of beam center coordinates.")
    elif beam_centers.shape[1] != 2:
        raise Exception("Beam center array should have shape (n_images, 2).")
        
    trimmed_edge_width = 2*trimming_radius+1

    trimmed_data = np.zeros((sed_data.shape[0],
                             trimmed_edge_width,
                             trimmed_edge_width))

    trimmed_data_mask = np.zeros((sed_data.shape[0],
                             trimmed_edge_width,
                             trimmed_edge_width), dtype = "bool")     
    
    # Trimmed data
    for image_index, img in enumerate(sed_data):

        beam_center_QX = beam_centers[image_index][1].astype(int)
        beam_center_QY = beam_centers[image_index][0].astype(int)       

        # QX direction
        trim_QX_upper = beam_center_QX + trimming_radius +1 
        trim_QX_lower = beam_center_QX - trimming_radius
        
        # QY direction
        trim_QY_upper = beam_center_QY + trimming_radius +1
        trim_QY_lower = beam_center_QY - trimming_radius
        
        trimmed_data[image_index] = img[trim_QY_lower:trim_QY_upper,
                                        trim_QX_lower:trim_QX_upper].data

        trimmed_data_mask[image_index] = img[trim_QY_lower:trim_QY_upper,
                                             trim_QX_lower:trim_QX_upper].mask

    return format_data_shape.to_2D(np.ma.masked_array(trimmed_data, trimmed_data_mask).astype("int16"))
    