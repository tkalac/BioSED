"""
/biosed/integration.py

Module for managing detector image masks.

Created on Fri Nov 15 09:40:24 2024

@author: Tine Kalac
"""
import numpy as np

from .config import config
from biosed.utilities import FormatDataShape
from ._cpp.crown_integration import compute_crown_integral

def crown_integration(sed_data,
    n_phi_bins = config.get("integration.n_phi_bins"),
    q_range = config.get("integration.q_range"),
    q_callibration = config.get("integration.q_callibration")):
    """
    Performs crown reduction on a detector image array.

    Parameters
    ----------
    sed_data : NumPy Array (3D or 4D)
        Detector image array. Beams should be centered.
    n_phi_bins : int, optional
        The number of phi bins in the crown. A higher number means
        better angular resolution but slows processing time.
    q_range : tuple, optional
        The q range of the studied Bragg reflection.
    q_callibration : float, optional
        Units: nm-1 / pixel.

    Returns
    -------
    tuple
        (azi_intensities, phi_vals) - the azimuthal intensity arrays and
        corresponding phi values. The azimuthal intensity profiles are returned
        in the same shape as the input data.

    Raises
    ------
    ExceptionType
        The sed_data should be a 1D or 2D stack of centered detector images.

    Notes
    -----
    The function uses a C++ extension. Computation mostly scales with number of
    images and is not affected much by the other parameters.

    Examples
    --------
    >>> data_azi_intensities, data_phi_vals = crown_integration(data_centered,
                                                                n_phi_bins = 60)
    >>> plt.plot(data_phi_vals, data_azi_intensities[0])
    """

    # Sanity checks
    if (sed_data.ndim < 3):
        raise Exception("""The data should be a 3D or 4D array, corresponding
            to the shape (image_indicies, QY, QX)""")
        
    # The image array is linearized so the c++ extension can work with it.
    # It is reshaped again in the return statement.
    format_shape = FormatDataShape(sed_data.shape[:-2])
    sed_data = format_shape.to_1D(sed_data)

    # The centers of each bin is returned
    phi_bin_edges = np.linspace(0, 360, n_phi_bins + 1)
    phi_vals = np.array([0.5*(phi_bin_edges[i] + phi_bin_edges[i+1]) for i in range(n_phi_bins)])

    if isinstance(sed_data, np.ma.MaskedArray):
        azi_intensities = compute_crown_integral(sed_data.data, n_phi_bins, q_range, q_callibration)
    else:
        azi_intensities = compute_crown_integral(sed_data, n_phi_bins, q_range, q_callibration)

    return format_shape.to_2D(azi_intensities), phi_vals
    