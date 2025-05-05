from vcat import ImageData
from vcat.kinematics import ComponentCollection
from vcat.plots.kinematic_plot import KinematicPlot
import matplotlib.pyplot as plt
import numpy as np

#Import Data
dataC=ImageData(model="../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.modelfits")

dataC.plot(plot_model=True,plot_comp_ids=True)

dataC.remove_component(0)
dataC.remove_component(1)
dataC.remove_component(2)
dataC.remove_component(3)
dataC.remove_component(5)

dataC.plot()
