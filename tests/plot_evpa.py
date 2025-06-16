from vcat import EvolutionPlot
import matplotlib.pyplot as plt

time=[2001,2002,2003,2004]
values=[1,1.5,2,1.5]
evpa=[0,45,60,-60]


plot=EvolutionPlot()

plot.plotEvolutionWithEVPA(time,values,evpa)
plt.show()
