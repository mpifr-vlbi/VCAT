from vcat import ImageData
import matplotlib.pyplot as plt

data=ImageData("../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.fits",
              uvf_file="../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.uvf",
               model="../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.modelfits",
               stokes_q="../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.qcln",
               stokes_u="../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.ucln",
              difmap_path="/usr/local/difmap/uvf_difmap_2.5g/")

#data.plot()

#data=data.rotate(-64,useDIFMAP=True)

data.plot(plot_mode="lin_pol",overplot_clean=True,plot_mask=True)

#do ridgeline fit
ridgeline,counter_ridgeline=data.get_ridgeline(method="polar",counterjet=True,start_radius=2)

#do alternative ridgeline fit
ridgeline,counter_ridgeline=data.get_ridgeline(method="slices",j_len=200,counterjet=True)


#rotate the image back for plotting
#data_new.rotate(+64,useDIFMAP=True)
data.plot(plot_mode="stokes_i",plot_ridgeline=True,plot_counter_ridgeline=True,counter_ridgeline_color="yellow")

ridgeline.plot(mode="open_angle")
ridgeline.plot(mode="intensity")
ridgeline.plot(mode="width")
ridgeline.plot(mode="ridgeline")
counter_ridgeline.plot(mode="ridgeline")

#get jet to counterjet profile:
data.jet_to_counterjet_profile()

