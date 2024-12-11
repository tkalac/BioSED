/* center_of_mass.cpp
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

#include <vector>
#include <tuple>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

///// FOR MASKED ARRAYS /////

// Function to calculate centers of mass for each image in the stack with a threshold
py::array_t<double>
    compute_centers_of_mass(py::array_t<int16_t> image_stack,
                            int16_t threshold) {
    
    // Get the buffers for the arrays                        
    auto bufData = image_stack.request();
    
    if (bufData.ndim != 3) {
        throw std::runtime_error("Data must be a 3D NumPy array: (num_images, height, width).");
    }

    if (!(image_stack.flags() & py::array::c_style)) {
        throw std::runtime_error("Input arrays must be C-contiguous.");
    }

    py::ssize_t nImages = bufData.shape[0];
    py::ssize_t nQY = bufData.shape[1];
    py::ssize_t nQX = bufData.shape[2];
    
    auto centersOfMass = py::array_t<double>(std::vector<py::ssize_t>{nImages, 2});
    auto centersOfMass_mutable = centersOfMass.mutable_unchecked<2>();
    

    // The memory should be overwritten to avoid weird things happening
    for (py::ssize_t imageIndex = 0; imageIndex < nImages; imageIndex++) {
        
        // Overwrites the results array with 0 to avoid weird things.
        for (py::ssize_t comIndex = 0; comIndex < 2; ++comIndex) {
            centersOfMass_mutable(imageIndex, comIndex) = 0.0;
        }
            
        // Initialize the temporary sums
        double sumQY = 0.0, sumQX = 0.0, totalSum = 0.0;

        // Computes the center of mass
        for (py::ssize_t indexQY = 0; indexQY < nQY; ++indexQY) {
            for (py::ssize_t indexQX = 0; indexQX < nQX; ++indexQX) {
                
                auto pixel = static_cast<int16_t*>(bufData.ptr)
                                                    [imageIndex  * nQY * nQX
                                                             + indexQY * nQX
                                                                   + indexQX];
                                                                   

                // Skip certain pixels
                if (pixel < threshold) continue;
                
                // 
                sumQY += pixel * indexQY;
                sumQX += pixel * indexQX;
                totalSum += pixel;       
                
            }
        }
        
        // If no pixels in the image are above the threshold it returns this
        if (totalSum == 0.0) {
            centersOfMass_mutable(imageIndex, 0) = -1.0;
            centersOfMass_mutable(imageIndex, 1) = -1.0;
            continue;
        }
        
        // write the results into the results array
        centersOfMass_mutable(imageIndex, 0) = sumQY / totalSum;
        centersOfMass_mutable(imageIndex, 1) = sumQX / totalSum;
    }
    
    return centersOfMass;
}

PYBIND11_MODULE(center_of_mass, m) {
    m.def("compute_centers_of_mass", &compute_centers_of_mass,
          "Compute centers of mass for a masked stack of images with a threshold.",
          py::arg("image_stack"), py::arg("threshold"));
}
