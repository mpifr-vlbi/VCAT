from vcat import ImageData,ImageCube
from vcat.graph_generator import KinematicPlot
import matplotlib.pyplot as plt
import numpy as np
import glob

uvf_files=glob.glob("../dataset_example/0506+056_kinematic/*.uvf")
model_files=glob.glob("../dataset_example/0506+056_kinematic/*fits")

print(uvf_files)
print(model_files)

#Import as ImageCube
im_cube=ImageCube().import_files(uvf_files=uvf_files,model_fits_files=model_files)

#Import associations from GUI file
im_cube.import_component_association("../dataset_example/0506+056_kinematic/component_info.csv")

#do kinematic fit
im_cube.get_speed(show_plot=True)

