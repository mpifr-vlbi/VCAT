from vcat import ImageData


#Import Data
dataC=ImageData("../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.modelfits")
dataQ=ImageData("../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.modelfits")


dataC.align(dataQ)
