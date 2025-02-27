from vcat import ImageCube, ImageData,FitsImage
import matplotlib.pyplot as plt
import numpy as np

data=ImageData("../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.fits",
        uvf_file="../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.uvf")

print(data.noise)


data=ImageData("../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.fits",
        uvf_file="../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.uvf",noise_method="box")

print(data.noise)

#testing noise from phase center shifting
for shift in np.linspace(0,100,101):
    print(data.get_noise_from_shift(shift_factor=shift))
    plt.scatter(shift,data.get_noise_from_shift(shift_factor=shift)*1000)
plt.xlabel("Shift Factor")
plt.ylabel("Noise Value [mJy]")
plt.show()
