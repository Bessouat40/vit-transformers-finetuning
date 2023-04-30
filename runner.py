from src.fineTunning import Training

training = Training(dataPath = "/Users/labess40/dev/keras-sklearn-medical-images-classification/chest_xray/", outdir = "./test")

training.trainModel()