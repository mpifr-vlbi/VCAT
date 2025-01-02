# VCAT/__init__.py

from vcat.image_data import ImageData
from vcat.graph_generator import FitsImage
import os
import platform
""" Organizing, analyzing and plotting of multifrequency VLBI data.

Modules exported by this package:

- `alignmenti/align_imagesEHTim.py`: Align two VLBI maps using 2D crosscorrelation.
- `graph_generator_fei/graph_generator.py`: Generate plots for polarization VLBI data.
- `px_analysis`: Analysie multi-frequency maps on a px-basis.
- `px_analysis/Jets_analysis_v1.1_100123.py`: ?
- `px_analysis/Jets_analysis_v1_1001.py`: ?
- `px_analysis/Turnover_pixel.py`: ?
- `VLBI_map_analysis`: Plot and analyse clean maps and model components.
- `VLBI_map_analysis/cleanMap.py`: Provides the cleanMap class to plot VLBI clean map and models.
- `VLBI_map_analysis/modelComps.py`: Provides the cleanMap class to plot VLBI clean map and models.
- `VLBI_map_analysis/derive_beta_theta.py`: Derive and plot allowed parameter space for Theta and Beta.
- `VLBI_map_analysis/stackedImages.py`: Derive and plot a stacked mape out of several VLBI epochs.
- `VLBI_map_analysis/modules`: Helper functions for fitting and plotting.
- `dataset_examples`: example multi-frequency polarization dataset.
- `examples`: example scripts for running parts of this module.

Examples for running the modules are in the folder examples.

Example data for running the modules is in examples/example_data.
"""

# This is to remove the "Welcome to eht_imaging" message
print("\033[F\033[K", end="")  # Clears the second line
print("\033[F\033[K", end="")  # Clears the first line

print("\rThank you for using VCAT. Have fun with VLBI!", end="\n")
