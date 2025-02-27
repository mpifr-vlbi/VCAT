from vcat import ImageData,FitsImage
import matplotlib.pyplot as plt


#Import Data
data=ImageData("../dataset_example/0215_pol/0215+015.icn.fits",
        model="../dataset_example/0215_pol/0215+015.icn.fits",
        stokes_q="../dataset_example/0215_pol/0215+015.qcn.fits",
        stokes_u="../dataset_example/0215_pol/0215+015.ucn.fits",   
        uvf_file="../dataset_example/0215_pol/0215+015.uvf",
        difmap_path="/usr/local/difmap/uvf_difmap_2.5g/")


data_shifted=data.shift(5,5)

print("Plotting Image")
FitsImage(data,plot_mode="stokes_i")
plt.show()

print("Plotting Shifted Image")
FitsImage(data_shifted,plot_mode="stokes_i",plot_evpa=True)
plt.show()


print("Aligning shifted image with image before")
data_aligned=data_shifted.align(data)

FitsImage(data_aligned,plot_mode="stokes_i")
plt.show()


