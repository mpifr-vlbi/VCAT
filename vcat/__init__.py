# VCAT/__init__.py

from vcat.image_data import ImageData
from vcat.graph_generator import FitsImage, MultiFitsImage, KinematicPlot  #probably not needed
from vcat.image_cube import ImageCube
import os
import platform

""" Organizing, analyzing and plotting of multi-frequency, multi-epoch VLBI data.

Examples for running the modules are in the folder examples.

Example data for running the modules is in examples/example_data.
"""


print("\rThank you for using VCAT. Have fun with VLBI!", end="\n")
print("\rIf you are using this package please cite VCAT Team et al. 2025 ....")
