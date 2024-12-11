/* crown_integration.cpp
 *
 *
 * Copyright (C) [YEAR] [YOUR NAME]
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

// Python binding libraries
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Math stuff
#define M_PI 3.14159265358979323846
#include <cmath>
#include <vector>
#include <tuple>

// Namespace required by Pybind11
namespace py = pybind11;

/// Function for computing the crown integral.
py::array_t<double> compute_crown_integral(
    // SED data as a stack of 2D arrays (int16)
    py::array_t<int16_t> sedDataArray, 
    int nPhiBins,                        // Number of phi bins
    std::tuple<double, double> QRange,   // q range of integration (incl.)
    double qCallibration                 // q/pixel value
){
    // Retrieve the array data and information through the buffer
    py::buffer_info bufSedData = sedDataArray.request();

    // Data checks
    if (bufSedData.ndim != 3) {
        throw std::runtime_error("Input should be a 3D NumPy array. The shape "
            "should be (nImages, nQY, nQX)");
    }
    
    py::ssize_t nImages = bufSedData.shape[0];    // Y scan dimension
    py::ssize_t nQY = bufSedData.shape[1];    // X scan dimension
    py::ssize_t nQX = bufSedData.shape[2];  // QY dimension size of the detector

    // Calculate beam center
    int beamCenterQY = static_cast<int>(nQY / 2);  // Integer division
    int beamCenterQX = static_cast<int>(nQX / 2);

    // Compute q and phi values for the entire detector
    auto pixelsPhiArray = py::array_t<double>({nQY, nQX});
    auto pixelsPhi = pixelsPhiArray.mutable_unchecked<2>();
    auto pixelsQArray = py::array_t<double>({nQY, nQX});
    auto pixelsQ = pixelsQArray.mutable_unchecked<2>();

    auto validPixelsArray = py::array_t<bool>({nQY, nQX});
    auto validPixels = validPixelsArray.mutable_unchecked<2>();

    auto binIndiciesArray = py::array_t<int>({nQY, nQX});
    auto binIndicies = binIndiciesArray.mutable_unchecked<2>();

    // This is for valid for the centered detectors. It excludes the masked.
    for (py::ssize_t iQY = 0; iQY < nQY; ++iQY) {
        int QY = iQY - beamCenterQY;
        for (py::ssize_t iQX = 0; iQX < nQX; ++iQX) {
            int QX = iQX - beamCenterQX;
            // Compute phi values
            pixelsQ(iQY, iQX) = std::sqrt(QY*QY + QX*QX) * qCallibration;
            pixelsPhi(iQY, iQX) = std::atan2(QY, QX) * 180.0 / M_PI;
            if (pixelsPhi(iQY, iQX) < 0) pixelsPhi(iQY, iQX) += 360.0;

            // Determine valid pixels for the given q range
            validPixels(iQY, iQX) = (pixelsQ(iQY, iQX) >= std::get<0>(QRange)) &&
                            (pixelsQ(iQY, iQX) <= std::get<1>(QRange));

            // Compute bin assignations
            if (validPixels(iQY, iQX)) {
                binIndicies(iQY, iQX)
                = static_cast<int>(pixelsPhi(iQY, iQX) /(360.0/nPhiBins));
                binIndicies(iQY, iQX) = std::clamp(binIndicies(iQY, iQX), 0, nPhiBins - 1);
            }
        }
    }

    //// IMAGE STACK PROCESSING ////

    // Initialize the output
    auto aziIntensityProfilesArray
        = py::array_t<double>(std::vector<py::ssize_t>{nImages, nPhiBins});
    auto aziIntensityProfiles
        = aziIntensityProfilesArray.mutable_unchecked<2>();

    for (py::ssize_t iImage = 0; iImage < nImages; ++iImage) {
        for (py::ssize_t iPhiBin = 0; iPhiBin < nPhiBins; ++iPhiBin) {
            aziIntensityProfiles(iImage, iPhiBin) = 0.0;
        }
    }
    auto sedData = sedDataArray.mutable_unchecked<3>();

    for (py::ssize_t iImage = 0; iImage < nImages; ++iImage) {
        std::vector<double> phiBinsSum(nPhiBins, 0.0);
        std::vector<int> pixelCount(nPhiBins, 0);

        for (py::ssize_t iQY = 0; iQY < nQY; ++iQY) {
            for (py::ssize_t iQX = 0; iQX < nQX; ++iQX) {
                if(!validPixels(iQY, iQX)) continue;
                if(sedData(iImage, iQY, iQX) < 0) continue;
                phiBinsSum[binIndicies(iQY, iQX)] += sedData(iImage, iQY, iQX);
                pixelCount[binIndicies(iQY, iQX)]++;
            }
        }

        // Average the intensities
        for (int iPhiBins = 0; iPhiBins < nPhiBins; ++iPhiBins) {
            if (pixelCount[iPhiBins] > 0) {
                aziIntensityProfiles(iImage, iPhiBins) = 0.0;
                aziIntensityProfiles(iImage, iPhiBins)
                = phiBinsSum[iPhiBins] / pixelCount[iPhiBins];
            }
        }
    }

    return aziIntensityProfilesArray;
}


// Pybind11 module definition
PYBIND11_MODULE(crown_integration, m) {
    // Optional docstring for the module
    m.doc() = "Crown integration algorithm."; 

    // Bind the 4D masked function
    m.def("compute_crown_integral", &compute_crown_integral,
        "Performs crown integration on a stack 2D detector images.");
}
