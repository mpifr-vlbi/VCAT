from vcat import ImageData
from vcat.graph_generator import FitsImage
import matplotlib.pyplot as plt

data=ImageData("../dataset_example/0215_pol/Ku/0215+015.icn.fits",
        model="../dataset_example/0215_pol/Ku/0215+015.icn.fits",
        stokes_q="../dataset_example/0215_pol/Ku/0215+015.qcn.fits",
        stokes_u="../dataset_example/0215_pol/Ku/0215+015.ucn.fits",   
        uvf_file="../dataset_example/0215_pol/Ku/0215+015.uvf",
        difmap_path="/usr/local/difmap/uvf_difmap_2.5g/")

#create mask in the center
data.masking(mask_type="npix_x",args=[100,200])
FitsImage(data,plot_mode="lin_pol",plot_evpa=True,overplot_clean=True,plot_mask=True)
plt.show()

#reset mask
data.masking(mask_type="reset")

#cut left
data.masking(mask_type="cut_left",args=200)
FitsImage(data,plot_mode="lin_pol",plot_evpa=True,overplot_clean=True,plot_mask=True)
plt.show()

#reset mask
data.masking(mask_type="reset")

#cut right
data.masking(mask_type="cut_right",args=200)
FitsImage(data,plot_mode="lin_pol",plot_evpa=True,overplot_clean=True,plot_mask=True)
plt.show()

#reset mask
data.masking(mask_type="reset")

#cut circle
data.masking(mask_type="radius",args=100)
FitsImage(data,plot_mode="lin_pol",plot_evpa=True,overplot_clean=True,plot_mask=True)
plt.show()

#reset mask
data.masking(mask_type="reset")

#cut ellipse
data.masking(mask_type="ellipse",args={'e_args':[100,50,30],'e_xoffset':300,'e_yoffset':-50})
FitsImage(data,plot_mode="lin_pol",plot_evpa=True,overplot_clean=True,plot_mask=True)
plt.show()

#additional flux cut
data.masking(mask_type="flux_cut",args=0.2)
FitsImage(data,plot_mode="lin_pol",plot_evpa=True,overplot_clean=True,plot_mask=True)
plt.show()



#Test shift
data_shifted=data.shift(10,5)
FitsImage(data_shifted,plot_mode="lin_pol",plot_evpa=True,overplot_clean=True,plot_mask=True)
plt.show()

