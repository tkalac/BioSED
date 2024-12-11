# /biosed/setup.py
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



from setuptools import setup, find_packages
import sys

# Ensure Pybind11 is installed
try:
    import pybind11

except ImportError:
    print("Please install pybind11 using pip: `pip install pybind11`")
    sys.exit(1)


from pybind11.setup_helpers import Pybind11Extension, build_ext


# C++ extensions
ext_modules = [
    # Center of mass extension
    Pybind11Extension(
        "biosed._cpp.center_of_mass",
        ["biosed/_cpp/center_of_mass.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
    
    # Integration extension
    Pybind11Extension(
        "biosed._cpp.crown_integration",
        ["biosed/_cpp/crown_integration.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
]


# Package requirements
reqs = [
    "numpy",       # Specify minimum version if needed
    "matplotlib",
    "lmfit",
    "cmcrameri",
    "opencv-python",
    "h5py",
],


# Package setup
setup(
    name="biosed",
    version="0.1.1",
    author="Tine Kalac",
    author_email="tine.kalac@mmk.su.se",
    description="A utility package for processing SED data to study biomaterials.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="GPL-3.0-or-later",  # Specify GPL v3
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=reqs,
)