# /biosed/orientation.py
# Module for computing preferential orientation.
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
from biosed.utilities import FormatDataShape

import numpy as np
from lmfit import Model, Parameters

def poisson_odf(phi, phi_0, eta, C):
    """
    Orientation distribution function model based on the Poisson kernel.

    Parameters
    ----------
    phi : NumPy Array (1D)
        Array of azimuthal angles in radians.
    phi_0 : float
        Preferential orientation
    eta : float
        Degree of alignment. 0 < eta < 1.
    C : float
        Scaling constant.

    Returns
    -------
    intensity
        NumPy array with the intensity values for each azimuthal angle.

    Examples
    --------

    >>> phi = np.linspace(0, 2*np.pi, 100)
    >>> intensity = biosed.poisson_model(phi, np.pi/4, 0.7, 2):
    >>> plt.plot(phi, intensity)
    """

    return C * (1 - eta**2) / ((1 + eta)**2 - 4*eta*np.cos(phi - phi_0)**2)


# Defines the model and parameters
poisson_odf_model = Model(poisson_odf)

poisson_odf_params = Parameters()

poisson_odf_params.add('phi_0', value=np.pi/2,
                                min=0,
                                max=np.pi)

poisson_odf_params.add('eta', value=0.5,
                              min=config.get("orientation.fit_eta_limits")[0],
                              max=config.get("orientation.fit_eta_limits")[1])

poisson_odf_params.add('C', value=1.0, min=0)


def fit_poisson_odf(azi_intensities,
                    phi,
                    weight_exponent = config.get("orientation.weighing_exponent"),
                    ):
    """
    Fits the Poisson model to azimuthal intensity data array.

    Parameters
    ----------
    intensities : NumPy Array (2D)
        Azimuthal intensity profiles of the SED data, obtained from crown
        reduction. Indexing is (image_index, phi).
    phi : NumPy Array (1D)
        phi values corresponding to the third dimension of the
        intensities array. In degrees.

    Returns
    -------
    NumPy NDArray
        Array containing the fitting results. Shape is (image_index, fitted_params).
        The fitted parameters (2nd dimension) are indexed [phi_0, eta, C].

    Examples
    --------
    >>> fits = biosed.fit_poisson_model(data_azi_intensities, phi_vals)
    >>> biosed.orientation_plot(fits[:, :, 0])
    """

    format_shape = FormatDataShape(azi_intensities.shape[:-1])
    azi_intensities = format_shape.to_1D(azi_intensities)

    fitting_results = np.zeros((*format_shape.get_shape_1D, 3))

    for index, azi_int in enumerate(azi_intensities):
        fit_i = poisson_odf_model.fit(azi_int, phi = phi * (np.pi/180),
            params = poisson_odf_params,
            weights = azi_int**weight_exponent,
            nan_policy = 'omit')
        
        fitting_results[index] = [fit_i.params['phi_0'].value,
                                  fit_i.params['eta'].value,
                                  fit_i.params['C'].value]

    #print(fitting_results.shape)

    return format_shape.to_2D(fitting_results)



def find_orientation_peaks(azi_intensities, phi):
    """
    Determines the preferential orientation from an azimuthal intensity
    profile through taking the highest intensity value. Should only be
    used with a high number of phi bins.

    Parameters
    ----------
    intensities : NumPy Array (2D)
        Azimuthal intensity profiles of the SED data, obtained from crown
        reduction. Indexing is (image_index, phi).
    phi : NumPy Array (1D)
        phi values corresponding to the third dimension of the
        intensities array. In degrees.

    Returns
    -------
    NumPy NDArray
        Array with the phi values of the highest intensity
        (between 0 and 180 degrees).

    Examples
    --------
    Examples of how to use the function.

    >>> angles = biosed.find_orientation_peaks(data_azi_intensities, phi_vals)
    >>> biosed.orientation_plot(angles)
    """

    format_shape = FormatDataShape(azi_intensities.shape[:-1])

    azi_intensities = format_shape.to_1D(azi_intensities)

    # Averages through the point symmetry around the center
    azi_intensities_overlapped = 0.5 * azi_intensities[:, :int(phi.shape[0]/2)] + 0.5 * azi_intensities[:, int(phi.shape[0]/2):]
    
    peaks = np.zeros(format_shape.get_shape_1D)

    for index, azi_int in enumerate(azi_intensities_overlapped):
            peaks[index] = phi[azi_int.argmax()] * (np.pi/180)
    
    return format_shape.to_2D(peaks)


def harmonic_analysis(azi_intensities, phi):
    """
    Determines the preferential orientation from the phase of the second
    harmonic, computed from the DFS of the azimuthal intensity data.

    Parameters
    ----------
    intensities : NumPy Array (2D)
        Azimuthal intensity profiles of the SED data, obtained from crown
        reduction. Indexing is (image_index, phi).
    phi : NumPy Array (1D)
        phi values corresponding to the third dimension of the
        intensities array. In degrees.

    Returns
    -------
    NumPy NDArray
        Array with the phi values of the highest intensity
        (between 0 and 180 degrees).

    Examples
    --------

    >>> angles = biosed.find_orientation_peaks(data_azi_intensities, phi_vals)
    >>> biosed.orientation_plot(angles)
    """

    format_shape = FormatDataShape(azi_intensities.shape[:-1])
 
    azi_intensities = format_shape.to_1D(azi_intensities)

    azi_intensities_fft = np.fft.fft(azi_intensities)
    phase = np.angle(azi_intensities_fft[:,2])
    orientation = (-0.5 * phase) %np.pi
    
    return format_shape.to_2D(orientation)
