import numpy as np
from config import Config

def posteriorHandling_v0(modelOutput):
    confidence = np.zeros(shape=(modelOutput.shape[0]))
    # confidence[0] = (modelOutput[0][1] * modelOutpsut[0][2]) ** 0.5
    confidence[0] = (modelOutput[0][1] + modelOutput[0][2])# ** 0.5
    for i in range(2, modelOutput.shape[0] + 1):
        h_smooth = max(1, i - Config.w_smooth + 1)
        modelOutput[i - 1] = modelOutput[h_smooth-1:i].sum(axis=0) / (i - h_smooth + 1)
        h_max = max(1, i - Config.w_max + 1)
        windowMax = np.max(modelOutput[h_max:i], axis=0)
        confidence[i - 1] = (windowMax[1] + windowMax[2]) # ** 0.5
    return confidence[:]

def posteriorHandling(modelOutput):
    confidence = np.zeros(shape=(modelOutput.shape[0]))
    # confidence[0] = (modelOutput[0][1] * modelOutpsut[0][2]) ** 0.5
    confidence[0] = (modelOutput[0][1] + modelOutput[0][2])# ** 0.5
    for i in range(2, modelOutput.shape[0] + 1):
        h_smooth = max(1, i - Config.w_smooth + 1)
        modelOutput[i - 1] = modelOutput[h_smooth-1:i].sum(axis=0) / (i - h_smooth + 1)
        h_max = max(1, i - Config.w_max + 1)
        windowMax = np.max(modelOutput[h_max:i], axis=0)
        confidence[i - 1] = (windowMax[1] + windowMax[2]) # ** 0.5
    return confidence[:]