from vcat import ImageData
import matplotlib.pyplot as plt

data=ImageData("../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.fits",
              uvf_file="../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.uvf",
               model="../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.modelfits",
               stokes_q="../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.qcln",
               stokes_u="../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.ucln",
              difmap_path="/usr/local/difmap/uvf_difmap_2.5g/")

data.plot()

data_new=data.rotate(-64,useDIFMAP=True)

data_new.plot(plot_mode="lin_pol",overplot_clean=True,plot_mask=True)

#do ridgeline fit
ridgeline=data_new.get_ridgeline(j_len=200)

#rotate the image back for plotting
data_new.rotate(+64,useDIFMAP=True)
data_new.plot(plot_mode="stokes_i",plot_ridgeline=True)

ridgeline.plot(mode="open_angle")
ridgeline.plot(mode="intensity")
ridgeline.plot(mode="width")
ridgeline.plot(mode="ridgeline")

