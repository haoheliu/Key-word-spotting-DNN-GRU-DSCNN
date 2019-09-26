from util import Util
import os
import numpy as np
import tensorflow as tf
from config import Config

class dataLoader:
    def __init__(self,dataFileBatchSize = 30,shuffle = True):
        self.util = Util()
        self.shuffle = shuffle
        self.dataFileBatchSize = dataFileBatchSize
        self.left = Config.leftFrames
        self.right = Config.rightFrames
        self.trainData = {"data": np.empty(shape=[0,(self.left+self.right+1)*40]), "label": np.empty(shape=[0])}
        self.testData = {"data": np.empty(shape=[0,(self.left+self.right+1)*40]), "label": np.empty(shape=[0])}
        self.currentTestDataFile, self.currentTrainDataFile = 0,0
        # self.positiveTestPath = "/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/data/positive/test/"
        # self.positiveTrainPath = "/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/data/positive/train/"
        self.positiveTestPath = "./data/test/"
        self.positiveTrainPath = "./data/train/"
        self.testDataFiles, self.trainDataFiles = self.constructPositiveDataSet()

    # List All the Positive test&train files
    def constructPositiveDataSet(self):
        testDataFiles, trainDataFiles = os.listdir(self.positiveTestPath), os.listdir(self.positiveTrainPath)
        for i, testDataFile in enumerate(testDataFiles):
            if (testDataFile.split('.')[-1] != 'fbank'):
                testDataFiles.remove(testDataFile)
        for i, trainDataFile in enumerate(trainDataFiles):
            if (trainDataFile.split('.')[-1] != 'fbank'):
                trainDataFiles.remove(trainDataFile)
        return testDataFiles, trainDataFiles

    # Get a batch of positive training example
    def getTrainPositiveNextBatch(self):
        self.resetDataDict()
        for i in range(self.dataFileBatchSize):
            if(self.currentTrainDataFile >= len(self.trainDataFiles)):
                self.currentTrainDataFile = 0 # repeat the hole dataset again
                return np.empty(shape=[0]),np.empty(shape=[0])
            result,label = self.util.fbankTransform(self.positiveTrainPath + self.trainDataFiles[self.currentTrainDataFile])
            self.currentTrainDataFile += 1
            self.trainData['data'] = np.append(self.trainData['data'],result,axis=0)
            self.trainData['label'] = np.append(self.trainData['label'],label,axis=0)
        return self.trainData['data'],self.trainData['label']

    # Get a batch of positive test example
    def getTestPositiveNextBatch(self):
        self.resetDataDict()
        for i in range(self.dataFileBatchSize):
            if(self.currentTestDataFile >= len(self.testDataFiles)):
                self.currentTestDataFile = 0 # repeat the hole dataset again
            result,label = self.util.fbankTransform(self.positiveTestPath + self.testDataFiles[self.currentTestDataFile])
            self.currentTestDataFile += 1
            self.testData['data'] = np.append(self.testData['data'],result,axis=0)
            self.testData['label'] = np.append(self.testData['label'],label,axis=0)
        return self.testData['data'],self.testData['label']

    def resetDataDict(self):
        self.trainData = {"data": np.empty(shape=[0,(self.left+self.right+1)*40]), "label": np.empty(shape=[0,3])}
        self.testData = {"data": np.empty(shape=[0,(self.left+self.right+1)*40]), "label": np.empty(shape=[0,3])}

if __name__ == "__main__":
    dataloader = dataLoader(dataFileBatchSize=5)
    data,label = dataloader.getTrainPositiveNextBatch()
    print(data.shape)
    print(label.shape)