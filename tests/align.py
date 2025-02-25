from vcat import ImageData


#Import Data
dataC=ImageData("../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.fits")
dataQ=ImageData("../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.fits")

dataC.shift(5,5)

print(dataC)
dataC.align(dataQ)
