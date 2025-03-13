from vcat import ImageData

data_shifted=ImageData("../dataset_example/3C111_U_shifted/3C111_2014-05-08_15GHz.fits",
        uvf_file="../dataset_example/3C111_U_shifted/3C111_2014-05-08_15GHz.uvf")

data_shifted=data_shifted.center()

data_shifted.plot()
