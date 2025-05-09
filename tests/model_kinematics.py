from vcat import ImageData,ImageCube
from vcat.plots.kinematic_plot import KinematicPlot
import matplotlib.pyplot as plt
import numpy as np
import glob

uvf_files=glob.glob("../dataset_example/0506+056_kinematic/*.uvf")
model_files=glob.glob("../dataset_example/0506+056_kinematic/*fits")

print(uvf_files)
print(model_files)

#Import as ImageCube
im_cube=ImageCube().import_files(uvf_files=uvf_files,model_fits_files=model_files,fit_comp_polarization=True)

im_cube.plot_components(show=True)

#Import associations from GUI file
im_cube.import_component_association("../dataset_example/0506+056_kinematic/component_info.csv")

#make a plot of the component evolution
im_cube.plot_component_evolution("tb")

#do kinematic fit
im_cube.get_speed(order=3,show_plot=True)

im_cube.movie(plot_components=True,fill_components=True,n_frames=100,interval=300,xlim=[5,-5],ylim=[-7,3])

