from util import Util
import os
import numpy as np
import tensorflow as tf
from config import Config
import random

class dataLoader:
    def __init__(self,dataFileBatchSize = 30):
        self.util = Util()
        self.dataFileBatchSize = dataFileBatchSize
        self.totalEpoches = Config.numEpochs
        self.left = Config.leftFrames
        self.right = Config.rightFrames
        self.currentTestDataFile, self.currentTrainDataFile = 0,0
        self.counter = 0
        self.testDataFiles, self.trainDataFiles = self.util.testDataFiles,self.util.trainDataFiles
        Config.testBatchSize = int(len(self.util.testDataFiles)*0.1)
        self.maxTestCacheSize = Config.testBatchSize * 700
        self.maxTrainCacheSize = Config.batchSize * 700
        self.trainData = {"data": np.empty(shape=[self.maxTrainCacheSize,(self.left+self.right+1)*40]), "label": np.empty(shape=[self.maxTrainCacheSize,3])}
        self.testData = {"data": np.zeros(shape=[self.maxTestCacheSize,(self.left+self.right+1)*40]), "label": np.zeros(shape=[self.maxTestCacheSize,3])}
        self.constructTestPositive()

    # Get a batch of positive training example
    def getTrainPositiveNextBatch(self):
        self.resetDataDict()
        counter,currentRow = 0,0
        if(self.currentTrainDataFile % 500 == 0):
            print(str(self.currentTrainDataFile)+" training files finished!")
        for i in range(Config.batchSize):
            if(self.currentTrainDataFile >= len(self.trainDataFiles)):
                self.currentTrainDataFile = 0 # repeat the hole dataset again
                Config.numEpochs -= 1
                if(Config.shuffle == True):
                    print("Shuffle training data ...")
                    random.shuffle(self.trainDataFiles)
                return np.empty(shape=[0]),np.empty(shape=[0])
            fname = self.util.splitFileName(self.trainDataFiles[self.currentTrainDataFile])
            result = np.load("./offlineData/"+fname+"_data.npy")
            label = np.load("./offlineData/"+fname+"_label.npy")
            self.currentTrainDataFile += 1
            for data,label in zip(result,label):
                self.trainData['data'][currentRow] = data
                self.trainData['label'][currentRow] = label
                currentRow += 1
        self.trainData['data'] = self.trainData['data'][:currentRow]
        self.trainData['label'] = self.trainData['label'][:currentRow]
        return self.trainData['data'],self.trainData['label']

    # Get a batch of positive test example
    def constructTestPositive(self):
        self.resetDataDict()
        self.currentTestDataFile = 0  # repeat the hole dataset again
        currentRow,counter = 0,0
        for i in range(Config.testBatchSize): # All the test files
            if(counter % 100 == 0):
                print("Loading "+str(counter)+" testfiles finished!")
            counter += 1
            fname = self.util.splitFileName(self.testDataFiles[self.currentTestDataFile])
            result = np.load("./offlineData/"+fname+"_data.npy")
            label = np.load("./offlineData/"+fname+"_label.npy")
            self.currentTestDataFile += 1
            for data,label in zip(result,label):
                self.testData['data'][currentRow] = data
                self.testData['label'][currentRow] = label
                currentRow += 1
        self.testData['data'] = self.testData['data'][:currentRow]
        self.testData['label'] = self.testData['label'][:currentRow]

    def getTestPositive(self):
        return self.testData['data'],self.testData['label']

    def resetDataDict(self):
        self.trainData = {"data": np.empty(shape=[self.maxTrainCacheSize,(self.left+self.right+1)*40]), "label": np.empty(shape=[self.maxTrainCacheSize,3])}

if __name__ == "__main__":
    dataloader = dataLoader(dataFileBatchSize=5)
    data,label = dataloader.getTestPositive()
    print(data.shape)
    print(label.shape)