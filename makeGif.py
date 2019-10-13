import matplotlib.pyplot as plt
import imageio,os

basePath = "./images/ROC/"
gifFname = "rocs.gif"
images = []
filenames=sorted(os.listdir(basePath))
for filename in filenames:
    images.append(imageio.imread(basePath+filename))
imageio.mimsave('./images/gifs/'+gifFname, images,duration=1)