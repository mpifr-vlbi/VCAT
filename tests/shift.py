from vcat import ImageData

data_shifted=ImageData("../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.fits",
        model="../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.mfit",
        uvf_file="../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.uvf")

for comp in data_shifted.components:
    print(comp.snr)

data_shifted=data_shifted.shift(10,10)

for comp in data_shifted.components:
    print(comp.snr)

