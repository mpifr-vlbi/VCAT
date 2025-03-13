from vcat import ImageData
import matplotlib.pyplot as plt

data=ImageData("../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.fits",
              uvf_file="../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.uvf",
               model="../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.modelfits",
               stokes_q="../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.qcln",
               stokes_u="../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.ucln",
              difmap_path="/usr/local/difmap/uvf_difmap_2.5g/")

peak_dist=data.get_peak_distance()

data.plot(plot_mode="lin_pol",plot_line=[(0,0),(peak_dist[0],peak_dist[1])])


