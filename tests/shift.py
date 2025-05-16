from vcat import ImageData

data_shifted=ImageData("../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.fits",
        model="../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.mfit",
        stokes_q="../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.qcln",
        stokes_u="../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.ucln",
        uvf_file="../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.uvf",fit_comp_polarization=True)


data_shifted=data_shifted.shift(10,10)

print(data_shifted.uvf_file,data_shifted.stokes_q_mod_file,data_shifted.stokes_i_mod_file)
